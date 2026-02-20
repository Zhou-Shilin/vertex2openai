"""Translate OpenAI Responses request payload into Gemini/Vertex payload."""

from __future__ import annotations

import json
import mimetypes
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.errors import APIError

DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.IGNORECASE)
URI_PATTERN = re.compile(r"^(https?://|gs://)", re.IGNORECASE)

REASONING_BUDGET_BY_EFFORT = {
    "none": 0,
    "minimal": 0,
    "low": 1024,
    "medium": 4096,
    "high": 8192,
    "xhigh": 16384,
}

VERTEX_RESPONSE_MODALITY_BY_OPENAI = {
    "text": "TEXT",
    "audio": "AUDIO",
}

VALID_PROMPT_CACHE_RETENTION = {"in-memory", "24h"}


@dataclass
class TranslateToGoogleResult:
    payload: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    call_id_name_map: dict[str, str] = field(default_factory=dict)
    passthrough_fields: dict[str, Any] = field(default_factory=dict)
    include: list[str] = field(default_factory=list)


def translate_openai_request_to_google(
    request_payload: dict[str, Any],
    *,
    previous_response: Optional[dict[str, Any]] = None,
) -> TranslateToGoogleResult:
    warnings: list[str] = []
    _validate_top_level(request_payload, warnings=warnings)
    call_map = _collect_call_map(previous_response)
    system_texts: list[str] = []
    contents: list[dict[str, Any]] = []
    include_values = _parse_include(request_payload.get("include"))
    file_id_map = _parse_file_id_map(request_payload)

    if previous_response:
        contents.extend(_previous_response_to_contents(previous_response))

    input_value = request_payload.get("input", "")
    new_contents, discovered_system_texts = _input_to_contents(
        input_value,
        call_map=call_map,
        file_id_map=file_id_map,
        warnings=warnings,
    )
    contents.extend(new_contents)
    system_texts.extend(discovered_system_texts)

    explicit_instructions = request_payload.get("instructions")
    if explicit_instructions:
        if isinstance(explicit_instructions, str):
            system_texts.insert(0, explicit_instructions)
        else:
            raise APIError(
                "`instructions` must be a string.",
                400,
                param="instructions",
                error_type="invalid_request_error",
                code="invalid_type",
            )

    if not contents:
        contents.append({"role": "user", "parts": [{"text": ""}]})

    google_payload: dict[str, Any] = {"contents": contents}
    if system_texts:
        merged = "\n".join(text for text in system_texts if text.strip())
        if merged.strip():
            google_payload["systemInstruction"] = {"parts": [{"text": merged}]}

    tools_payload, function_tools, builtin_function_names_by_type = _translate_tools(
        request_payload.get("tools"),
        warnings=warnings,
    )
    if tools_payload:
        google_payload["tools"] = tools_payload
    tool_config = _translate_tool_choice(
        request_payload.get("tool_choice"),
        function_tools,
        builtin_function_names_by_type,
        warnings=warnings,
    )
    if tool_config:
        google_payload["toolConfig"] = tool_config

    generation_config = _translate_generation_config(request_payload, warnings=warnings)
    if generation_config:
        google_payload["generationConfig"] = generation_config

    _apply_vertex_extensions(request_payload, google_payload, warnings=warnings)
    passthrough_fields = _collect_passthrough_fields(request_payload, include_values)

    return TranslateToGoogleResult(
        payload=google_payload,
        warnings=warnings,
        call_id_name_map=call_map,
        passthrough_fields=passthrough_fields,
        include=include_values,
    )


def _validate_top_level(payload: dict[str, Any], *, warnings: list[str]) -> None:
    for boolean_field in ("background", "parallel_tool_calls"):
        if boolean_field in payload and not isinstance(payload.get(boolean_field), bool):
            raise APIError(
                f"`{boolean_field}` must be a boolean.",
                400,
                param=boolean_field,
                error_type="invalid_request_error",
                code="invalid_type",
            )

    truncation = payload.get("truncation")
    if truncation is not None and truncation not in {"auto", "disabled"}:
        raise APIError(
            "`truncation` must be `auto` or `disabled`.",
            400,
            param="truncation",
            error_type="invalid_request_error",
            code="invalid_type",
        )
    if truncation == "auto":
        warnings.append("`truncation=auto` is accepted but handled by upstream defaults.")

    service_tier = payload.get("service_tier")
    if service_tier is not None and service_tier not in {"auto", "default", "flex"}:
        raise APIError(
            "`service_tier` must be one of `auto`, `default`, or `flex`.",
            400,
            param="service_tier",
            error_type="invalid_request_error",
            code="invalid_type",
        )
    if service_tier is not None:
        warnings.append("`service_tier` has no direct Vertex equivalent and is preserved in response metadata only.")

    modalities = payload.get("modalities")
    if modalities is not None:
        if not isinstance(modalities, list) or not all(isinstance(item, str) for item in modalities):
            raise APIError(
                "`modalities` must be an array of strings.",
                400,
                param="modalities",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        invalid_modalities = [
            item for item in modalities if item.strip().lower() not in VERTEX_RESPONSE_MODALITY_BY_OPENAI
        ]
        if invalid_modalities:
            raise APIError(
                "Unsupported modality. Allowed values are `text` and `audio`.",
                400,
                param="modalities",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        if "audio" in modalities:
            warnings.append("Audio output modality is accepted but may be downgraded to text-only by upstream model.")

    audio_config = payload.get("audio")
    if audio_config is not None and not isinstance(audio_config, dict):
        raise APIError(
            "`audio` must be an object when provided.",
            400,
            param="audio",
            error_type="invalid_request_error",
            code="invalid_type",
        )
    if isinstance(audio_config, dict):
        audio_format = audio_config.get("format")
        if audio_format is not None and not isinstance(audio_format, str):
            raise APIError(
                "`audio.format` must be a string.",
                400,
                param="audio.format",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        audio_voice = audio_config.get("voice")
        if audio_voice is not None and not isinstance(audio_voice, str):
            raise APIError(
                "`audio.voice` must be a string.",
                400,
                param="audio.voice",
                error_type="invalid_request_error",
                code="invalid_type",
            )

    metadata = payload.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise APIError(
                "`metadata` must be an object.",
                400,
                param="metadata",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        if len(metadata) > 16:
            raise APIError(
                "`metadata` supports up to 16 key-value pairs.",
                400,
                param="metadata",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        for key, value in metadata.items():
            if not isinstance(key, str) or not isinstance(value, (str, int, float, bool)):
                raise APIError(
                    "`metadata` keys must be strings and values must be string/number/boolean.",
                    400,
                    param="metadata",
                    error_type="invalid_request_error",
                    code="invalid_type",
                )

    user = payload.get("user")
    if user is not None and not isinstance(user, str):
        raise APIError(
            "`user` must be a string.",
            400,
            param="user",
            error_type="invalid_request_error",
            code="invalid_type",
        )

    max_tool_calls = payload.get("max_tool_calls")
    if max_tool_calls is not None:
        if not isinstance(max_tool_calls, int) or isinstance(max_tool_calls, bool) or max_tool_calls <= 0:
            raise APIError(
                "`max_tool_calls` must be a positive integer.",
                400,
                param="max_tool_calls",
                error_type="invalid_request_error",
                code="invalid_type",
            )

    prompt_cache_key = payload.get("prompt_cache_key")
    if prompt_cache_key is not None and not isinstance(prompt_cache_key, str):
        raise APIError(
            "`prompt_cache_key` must be a string.",
            400,
            param="prompt_cache_key",
            error_type="invalid_request_error",
            code="invalid_type",
        )

    prompt_cache_retention = payload.get("prompt_cache_retention")
    if prompt_cache_retention is not None:
        if not isinstance(prompt_cache_retention, str) or prompt_cache_retention not in VALID_PROMPT_CACHE_RETENTION:
            raise APIError(
                "`prompt_cache_retention` must be `in-memory` or `24h`.",
                400,
                param="prompt_cache_retention",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        warnings.append(
            "`prompt_cache_retention` has no direct Vertex field and is preserved in response metadata only."
        )

    safety_identifier = payload.get("safety_identifier")
    if safety_identifier is not None and not isinstance(safety_identifier, str):
        raise APIError(
            "`safety_identifier` must be a string.",
            400,
            param="safety_identifier",
            error_type="invalid_request_error",
            code="invalid_type",
        )

    prompt = payload.get("prompt")
    if prompt is not None and not isinstance(prompt, (str, dict)):
        raise APIError(
            "`prompt` must be a string or object.",
            400,
            param="prompt",
            error_type="invalid_request_error",
            code="invalid_type",
        )

    stream_options = payload.get("stream_options")
    if stream_options is not None and not isinstance(stream_options, dict):
        raise APIError(
            "`stream_options` must be an object.",
            400,
            param="stream_options",
            error_type="invalid_request_error",
            code="invalid_type",
        )
    if isinstance(stream_options, dict):
        include_obfuscation = stream_options.get("include_obfuscation")
        if include_obfuscation is not None and not isinstance(include_obfuscation, bool):
            raise APIError(
                "`stream_options.include_obfuscation` must be a boolean.",
                400,
                param="stream_options.include_obfuscation",
                error_type="invalid_request_error",
                code="invalid_type",
            )

    include = payload.get("include")
    if include is not None:
        _parse_include(include)


def _parse_include(include_value: Any) -> list[str]:
    if include_value is None:
        return []
    if not isinstance(include_value, list):
        raise APIError(
            "`include` must be an array of strings.",
            400,
            param="include",
            error_type="invalid_request_error",
            code="invalid_type",
        )
    values: list[str] = []
    for value in include_value:
        if not isinstance(value, str):
            raise APIError(
                "`include` must contain strings only.",
                400,
                param="include",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        values.append(value)
    return values


def _collect_passthrough_fields(request_payload: dict[str, Any], include_values: list[str]) -> dict[str, Any]:
    passthrough: dict[str, Any] = {}
    for key in (
        "metadata",
        "service_tier",
        "user",
        "truncation",
        "parallel_tool_calls",
        "background",
        "prompt_cache_key",
        "prompt_cache_retention",
        "safety_identifier",
        "max_tool_calls",
        "stream_options",
        "prompt",
    ):
        if key in request_payload:
            passthrough[key] = request_payload.get(key)
    if include_values:
        passthrough["include"] = include_values
    if isinstance(request_payload.get("reasoning"), dict):
        passthrough["reasoning"] = request_payload.get("reasoning")
    if isinstance(request_payload.get("text"), dict):
        passthrough["text"] = request_payload.get("text")
    return passthrough


def _parse_file_id_map(request_payload: dict[str, Any]) -> dict[str, str]:
    sources = []
    direct = request_payload.get("file_id_map")
    if isinstance(direct, dict):
        sources.append(direct)
    vertex = request_payload.get("vertex")
    if isinstance(vertex, dict) and isinstance(vertex.get("file_id_map"), dict):
        sources.append(vertex.get("file_id_map"))

    merged: dict[str, str] = {}
    for source in sources:
        for key, value in source.items():
            if not isinstance(key, str) or not key.strip():
                continue
            if not isinstance(value, str) or not value.strip():
                continue
            merged[key.strip()] = value.strip()
    return merged


def _collect_call_map(previous_response: Optional[dict[str, Any]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not previous_response:
        return mapping
    output = previous_response.get("output")
    if not isinstance(output, list):
        return mapping
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        call_id = item.get("call_id")
        name = item.get("name")
        if isinstance(call_id, str) and call_id and isinstance(name, str) and name:
            mapping[call_id] = name
    return mapping


def _previous_response_to_contents(previous_response: dict[str, Any]) -> list[dict[str, Any]]:
    output = previous_response.get("output")
    if not isinstance(output, list):
        return []

    model_parts: list[dict[str, Any]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "message":
            for content in item.get("content", []):
                if isinstance(content, dict) and content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str):
                        model_parts.append({"text": text})
        elif item_type == "function_call":
            name = item.get("name")
            arguments = item.get("arguments")
            if isinstance(name, str) and name:
                parsed_args = _parse_json_or_string(arguments)
                model_parts.append({"functionCall": {"name": name, "args": parsed_args}})

    if not model_parts:
        return []
    return [{"role": "model", "parts": model_parts}]


def _input_to_contents(
    input_value: Any,
    *,
    call_map: dict[str, str],
    file_id_map: dict[str, str],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    contents: list[dict[str, Any]] = []
    system_texts: list[str] = []

    if isinstance(input_value, str):
        return ([{"role": "user", "parts": [{"text": input_value}]}], system_texts)

    if input_value is None:
        return ([], system_texts)

    if not isinstance(input_value, list):
        raise APIError(
            "`input` must be a string or an array.",
            400,
            param="input",
            error_type="invalid_request_error",
            code="invalid_type",
        )

    for item in input_value:
        if isinstance(item, str):
            contents.append({"role": "user", "parts": [{"text": item}]})
            continue

        if not isinstance(item, dict):
            warnings.append("Ignored non-object item inside `input`.")
            continue

        item_type = item.get("type")
        if item_type == "function_call_output":
            contents.append(_function_call_output_to_content(item, call_map=call_map))
            continue
        if item_type == "message":
            _append_message_item(
                item,
                contents=contents,
                system_texts=system_texts,
                file_id_map=file_id_map,
                warnings=warnings,
            )
            continue
        if item_type == "input_file":
            file_part = _input_file_to_part(item, file_id_map=file_id_map)
            contents.append({"role": "user", "parts": [file_part]})
            continue
        if item_type == "item_reference":
            warnings.append("`item_reference` is not directly supported and was ignored.")
            continue

        if "role" in item:
            _append_message_item(
                item,
                contents=contents,
                system_texts=system_texts,
                file_id_map=file_id_map,
                warnings=warnings,
            )
            continue

        parts = _content_to_parts(item, file_id_map=file_id_map, warnings=warnings)
        if parts:
            contents.append({"role": "user", "parts": parts})

    return contents, system_texts


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)
        return "\n".join(texts)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
    return ""


def _append_message_item(
    item: dict[str, Any],
    *,
    contents: list[dict[str, Any]],
    system_texts: list[str],
    file_id_map: dict[str, str],
    warnings: list[str],
) -> None:
    role = item.get("role")
    content_value = item.get("content")
    if role in {"system", "developer"}:
        system_text = _extract_text_from_content(content_value)
        if system_text:
            system_texts.append(system_text)
        return

    if role in {"assistant", "model"}:
        mapped_role = "model"
    else:
        mapped_role = "user"

    parts = _content_to_parts(content_value, file_id_map=file_id_map, warnings=warnings)
    if parts:
        contents.append({"role": mapped_role, "parts": parts})


def _content_to_parts(content: Any, *, file_id_map: dict[str, str], warnings: list[str]) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, dict):
        part = _content_block_to_part(content, file_id_map=file_id_map, warnings=warnings)
        return [part] if part else []
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for block in content:
            if isinstance(block, str):
                parts.append({"text": block})
                continue
            if not isinstance(block, dict):
                warnings.append("Ignored invalid content block.")
                continue
            part = _content_block_to_part(block, file_id_map=file_id_map, warnings=warnings)
            if part:
                parts.append(part)
        return parts
    return []


def _content_block_to_part(
    block: dict[str, Any],
    *,
    file_id_map: dict[str, str],
    warnings: list[str],
) -> Optional[dict[str, Any]]:
    block_type = block.get("type")
    if block_type in {None, "input_text", "text", "output_text"}:
        text = block.get("text")
        if isinstance(text, str):
            return {"text": text}
        return None

    if block_type in {"input_image", "image_url"}:
        return _image_block_to_part(block, file_id_map=file_id_map)

    if block_type == "input_file":
        return _input_file_to_part(block, file_id_map=file_id_map)

    if block_type in {"input_audio", "audio"}:
        return _audio_block_to_part(block)

    warnings.append(f"Ignored unsupported content block type: {block_type!r}.")
    return None


def _image_block_to_part(block: dict[str, Any], *, file_id_map: dict[str, str]) -> dict[str, Any]:
    file_id = block.get("file_id")
    if isinstance(file_id, str) and file_id:
        resolved = _resolve_file_id_to_part_source(
            file_id=file_id,
            file_id_map=file_id_map,
            param_name="input.image.file_id",
        )
        return _resolved_source_to_part(resolved, default_mime_type="image/jpeg")

    image_url = block.get("image_url")
    url_value: Optional[str] = None
    if isinstance(image_url, str):
        url_value = image_url
    elif isinstance(image_url, dict):
        candidate = image_url.get("url")
        if isinstance(candidate, str):
            url_value = candidate
    elif isinstance(block.get("url"), str):
        url_value = block.get("url")

    if not url_value:
        raise APIError(
            "Image block is missing `image_url`.",
            400,
            param="input",
            error_type="invalid_request_error",
            code="invalid_image",
        )

    match = DATA_URL_PATTERN.match(url_value)
    if match:
        return {
            "inlineData": {
                "mimeType": match.group("mime"),
                "data": match.group("data"),
            }
        }
    return {
        "fileData": {
            "fileUri": url_value,
            "mimeType": _guess_mime_type(url_value, default_mime_type="image/jpeg"),
        }
    }


def _input_file_to_part(block: dict[str, Any], *, file_id_map: dict[str, str]) -> dict[str, Any]:
    file_id = block.get("file_id")
    if isinstance(file_id, str) and file_id:
        resolved = _resolve_file_id_to_part_source(
            file_id=file_id,
            file_id_map=file_id_map,
            param_name="input.file_id",
        )
        return _resolved_source_to_part(resolved, default_mime_type="application/octet-stream")

    file_data = block.get("file_data")
    if not isinstance(file_data, str) or not file_data.strip():
        raise APIError(
            "`input_file` requires non-empty `file_data`.",
            400,
            param="input.file_data",
            error_type="invalid_request_error",
            code="invalid_type",
        )

    normalized = file_data.strip()
    match = DATA_URL_PATTERN.match(normalized)
    if match:
        return {
            "inlineData": {
                "mimeType": match.group("mime"),
                "data": match.group("data"),
            }
        }
    if URI_PATTERN.match(normalized):
        return {"fileData": {"fileUri": normalized}}

    filename = block.get("filename")
    if isinstance(filename, str) and filename.strip():
        return {"text": f"[file:{filename.strip()}]\n{normalized}"}
    return {"text": normalized}


def _audio_block_to_part(block: dict[str, Any]) -> dict[str, Any]:
    audio_url = block.get("audio_url")
    if isinstance(audio_url, str) and audio_url.strip():
        source = audio_url.strip()
        match = DATA_URL_PATTERN.match(source)
        if match:
            return {
                "inlineData": {
                    "mimeType": match.group("mime"),
                    "data": match.group("data"),
                }
            }
        if URI_PATTERN.match(source):
            return {"fileData": {"fileUri": source}}
        raise APIError(
            "`input_audio.audio_url` must be a data URL or URI.",
            400,
            param="input_audio.audio_url",
            error_type="invalid_request_error",
            code="invalid_type",
        )

    nested = block.get("input_audio") if isinstance(block.get("input_audio"), dict) else block.get("audio")
    if isinstance(nested, dict):
        data = nested.get("data")
        fmt = nested.get("format")
        if isinstance(data, str) and data.strip():
            mime = "audio/wav"
            if isinstance(fmt, str) and fmt.strip():
                normalized_fmt = fmt.strip().lower()
                if "/" in normalized_fmt:
                    mime = normalized_fmt
                else:
                    mime = f"audio/{normalized_fmt}"
            return {"inlineData": {"mimeType": mime, "data": data.strip()}}

    raise APIError(
        "Unsupported `input_audio` payload. Provide `audio_url` or `input_audio.data`.",
        400,
        param="input_audio",
        error_type="invalid_request_error",
        code="invalid_type",
    )


def _resolve_file_id_to_part_source(*, file_id: str, file_id_map: dict[str, str], param_name: str) -> str:
    resolved = file_id_map.get(file_id)
    if not resolved:
        raise APIError(
            f"`{param_name}` `{file_id}` cannot be resolved. Provide `file_id_map` or `vertex.file_id_map`.",
            400,
            param=param_name,
            error_type="invalid_request_error",
            code="invalid_type",
        )
    return resolved


def _resolved_source_to_part(source: str, *, default_mime_type: str) -> dict[str, Any]:
    match = DATA_URL_PATTERN.match(source)
    if match:
        return {
            "inlineData": {
                "mimeType": match.group("mime"),
                "data": match.group("data"),
            }
        }
    if URI_PATTERN.match(source):
        return {
            "fileData": {
                "fileUri": source,
                "mimeType": _guess_mime_type(source, default_mime_type=default_mime_type),
            }
        }
    return {"text": source}


def _guess_mime_type(source: str, *, default_mime_type: str) -> str:
    guessed, _ = mimetypes.guess_type(source)
    if isinstance(guessed, str) and guessed.strip():
        return guessed.strip()
    return default_mime_type


def _function_call_output_to_content(item: dict[str, Any], *, call_map: dict[str, str]) -> dict[str, Any]:
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        raise APIError(
            "function_call_output requires a non-empty `call_id`.",
            400,
            param="input.call_id",
            error_type="invalid_request_error",
            code="invalid_tool_output",
        )
    explicit_name = item.get("name")
    call_name = explicit_name if isinstance(explicit_name, str) and explicit_name else call_map.get(call_id)
    if not call_name:
        raise APIError(
            f"Unable to resolve function name for call_id `{call_id}`. Provide `previous_response_id` or include `name`.",
            400,
            param="input.call_id",
            error_type="invalid_request_error",
            code="invalid_tool_output",
        )

    output_value = item.get("output")
    parsed_output = _parse_json_or_string(output_value)
    return {
        "role": "user",
        "parts": [
            {
                "functionResponse": {
                    "name": call_name,
                    "response": parsed_output if isinstance(parsed_output, dict) else {"output": parsed_output},
                }
            }
        ],
    }


def _parse_json_or_string(value: Any) -> Any:
    if isinstance(value, (dict, list, int, float, bool)) or value is None:
        return value
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return ""
        try:
            return json.loads(trimmed)
        except json.JSONDecodeError:
            return value
    return str(value)


def _translate_tools(
    tools_value: Any,
    *,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[str]]]:
    if tools_value is None:
        return [], [], {}
    if not isinstance(tools_value, list):
        raise APIError(
            "`tools` must be an array.",
            400,
            param="tools",
            error_type="invalid_request_error",
            code="invalid_type",
        )

    declarations: list[dict[str, Any]] = []
    tools_payload: list[dict[str, Any]] = []
    has_google_search = False
    has_code_execution = False
    builtin_function_names_by_type: dict[str, list[str]] = {}

    for tool in tools_value:
        if not isinstance(tool, dict):
            raise APIError(
                "Each tool must be an object.",
                400,
                param="tools",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        tool_type = tool.get("type")
        if tool_type == "function":
            function_obj = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            declaration = _function_declaration_from_tool(function_obj)
            declarations.append(declaration)
            continue
        if tool_type in {"web_search_preview", "web_search_preview_2025_03_11", "web_search"}:
            has_google_search = True
            continue
        if tool_type == "code_interpreter":
            has_code_execution = True
            continue
        if tool_type == "vertex_tool":
            raw_vertex_tool = tool.get("vertex")
            if not isinstance(raw_vertex_tool, dict):
                raise APIError(
                    "`tools[].vertex` must be an object for `vertex_tool`.",
                    400,
                    param="tools",
                    error_type="invalid_request_error",
                    code="invalid_type",
                )
            tools_payload.append(raw_vertex_tool)
            continue
        if tool_type in {
            "file_search",
            "computer_use_preview",
            "image_generation",
            "mcp",
            "local_shell",
            "custom",
            "shell",
            "apply_patch",
        }:
            synthetic_declaration = _synthetic_function_declaration_for_builtin_tool(tool, warnings=warnings)
            declarations.append(synthetic_declaration)
            builtin_function_names_by_type.setdefault(str(tool_type), []).append(synthetic_declaration["name"])
            continue

        raise APIError(
            f"Unsupported tool type `{tool_type}`.",
            400,
            param="tools",
            error_type="invalid_request_error",
            code="unsupported_feature",
        )

    if declarations:
        tools_payload.insert(0, {"functionDeclarations": declarations})
    if has_google_search:
        tools_payload.append({"googleSearch": {}})
    if has_code_execution:
        tools_payload.append({"codeExecution": {}})

    if has_google_search and has_code_execution:
        warnings.append("Both web search and code interpreter tools were enabled for Vertex.")
    return tools_payload, declarations, builtin_function_names_by_type


def _synthetic_function_declaration_for_builtin_tool(
    tool: dict[str, Any],
    *,
    warnings: list[str],
) -> dict[str, Any]:
    tool_type = str(tool.get("type"))
    if tool_type == "file_search":
        warnings.append("`file_search` mapped to synthetic function tool.")
        return {
            "name": _safe_function_name(tool.get("name"), fallback="file_search"),
            "description": "Search indexed files and return relevant snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "User search query"},
                    "max_results": {"type": "integer", "description": "Maximum snippets to return"},
                },
                "required": ["query"],
            },
        }
    if tool_type == "computer_use_preview":
        warnings.append("`computer_use_preview` mapped to synthetic function tool.")
        return {
            "name": _safe_function_name(tool.get("name"), fallback="computer_use_preview"),
            "description": "Execute one computer-use action and return the observation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "text": {"type": "string"},
                    "key": {"type": "string"},
                },
                "required": ["action"],
            },
        }
    if tool_type == "image_generation":
        warnings.append("`image_generation` mapped to synthetic function tool.")
        return {
            "name": _safe_function_name(tool.get("name"), fallback="image_generation"),
            "description": "Generate an image and return URI/base64 data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "size": {"type": "string"},
                    "quality": {"type": "string"},
                },
                "required": ["prompt"],
            },
        }
    if tool_type == "mcp":
        server_label = tool.get("server_label") if isinstance(tool.get("server_label"), str) else "server"
        fallback_name = f"mcp_{server_label}"
        warnings.append("`mcp` mapped to synthetic function tool.")
        return {
            "name": _safe_function_name(tool.get("name"), fallback=fallback_name),
            "description": "Invoke one MCP server tool call.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string"},
                    "arguments": {"type": "object"},
                },
                "required": ["tool_name"],
            },
        }
    if tool_type == "local_shell":
        warnings.append("`local_shell` mapped to synthetic function tool.")
        return {
            "name": _safe_function_name(tool.get("name"), fallback="local_shell"),
            "description": "Execute a shell command and return stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "array", "items": {"type": "string"}},
                    "timeout_ms": {"type": "integer"},
                    "working_directory": {"type": "string"},
                },
                "required": ["command"],
            },
        }
    if tool_type == "custom":
        warnings.append("`custom` tool mapped to synthetic function tool.")
        return {
            "name": _safe_function_name(tool.get("name"), fallback="custom_tool"),
            "description": "Invoke a custom tool payload and return structured result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "payload": {"type": "object"},
                },
                "required": ["payload"],
            },
        }
    if tool_type == "shell":
        warnings.append("`shell` tool mapped to synthetic function tool.")
        return {
            "name": _safe_function_name(tool.get("name"), fallback="shell"),
            "description": "Execute shell command and return stdout/stderr/exit code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "array", "items": {"type": "string"}},
                    "timeout_ms": {"type": "integer"},
                    "working_directory": {"type": "string"},
                },
                "required": ["command"],
            },
        }
    if tool_type == "apply_patch":
        warnings.append("`apply_patch` tool mapped to synthetic function tool.")
        return {
            "name": _safe_function_name(tool.get("name"), fallback="apply_patch"),
            "description": "Apply unified patch to workspace and return patch result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {"type": "string"},
                },
                "required": ["patch"],
            },
        }
    raise APIError(
        f"Unsupported tool type `{tool_type}`.",
        400,
        param="tools",
        error_type="invalid_request_error",
        code="unsupported_feature",
    )


def _safe_function_name(candidate: Any, *, fallback: str) -> str:
    if isinstance(candidate, str) and candidate.strip():
        value = candidate.strip()
    else:
        value = fallback
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", value).strip("_")
    if not safe:
        safe = "tool"
    return safe[:64]


def _function_declaration_from_tool(function_obj: dict[str, Any]) -> dict[str, Any]:
    name = function_obj.get("name")
    if not isinstance(name, str) or not name.strip():
        raise APIError(
            "Function tool requires a non-empty `name`.",
            400,
            param="tools.name",
            error_type="invalid_request_error",
            code="invalid_tool",
        )
    declaration: dict[str, Any] = {"name": name.strip()}
    description = function_obj.get("description")
    if isinstance(description, str) and description.strip():
        declaration["description"] = description.strip()

    parameters = function_obj.get("parameters")
    if parameters is not None:
        if not isinstance(parameters, dict):
            raise APIError(
                "Function `parameters` must be a JSON schema object.",
                400,
                param="tools.parameters",
                error_type="invalid_request_error",
                code="invalid_tool",
            )
        declaration["parameters"] = parameters
    return declaration


def _translate_tool_choice(
    tool_choice: Any,
    function_tools: list[dict[str, Any]],
    builtin_function_names_by_type: dict[str, list[str]],
    *,
    warnings: list[str],
) -> Optional[dict[str, Any]]:
    if tool_choice is None:
        return None

    if not function_tools:
        if tool_choice in {"none", "auto", "required"}:
            if tool_choice == "required":
                warnings.append("`tool_choice=required` ignored because no function tools are declared.")
            return None
        if isinstance(tool_choice, dict):
            warnings.append("`tool_choice` for non-function tools has no Vertex equivalent and was ignored.")
            return None
        raise APIError(
            "Invalid `tool_choice`.",
            400,
            param="tool_choice",
            error_type="invalid_request_error",
            code="invalid_tool_choice",
        )

    mode = "AUTO"
    allowed_names: Optional[list[str]] = None

    if tool_choice == "auto":
        mode = "AUTO"
    elif tool_choice == "none":
        mode = "NONE"
    elif tool_choice == "required":
        mode = "ANY"
    elif isinstance(tool_choice, dict):
        tool_choice_type = tool_choice.get("type")
        if tool_choice_type == "function":
            name = tool_choice.get("name")
            if not isinstance(name, str) and isinstance(tool_choice.get("function"), dict):
                candidate = tool_choice["function"].get("name")
                name = candidate if isinstance(candidate, str) else None
            if not isinstance(name, str) or not name:
                raise APIError(
                    "Function tool_choice requires `name`.",
                    400,
                    param="tool_choice.name",
                    error_type="invalid_request_error",
                    code="invalid_tool_choice",
                )
            mode = "ANY"
            allowed_names = [name]
        elif tool_choice_type == "allowed_tools":
            mode_value = tool_choice.get("mode")
            mode = _tool_choice_mode_to_vertex(mode_value)
            allowed_names = _resolve_allowed_tool_choice_names(tool_choice.get("tools"), function_tools, builtin_function_names_by_type)
        elif isinstance(tool_choice_type, str):
            mapped_names = builtin_function_names_by_type.get(tool_choice_type)
            if mapped_names:
                mode = "ANY"
                allowed_names = mapped_names
            else:
                warnings.append("Non-function `tool_choice` has no direct Vertex equivalent and was ignored.")
                return None
        else:
            warnings.append("Non-function `tool_choice` has no direct Vertex equivalent and was ignored.")
            return None
    else:
        raise APIError(
            "Invalid `tool_choice`.",
            400,
            param="tool_choice",
            error_type="invalid_request_error",
            code="invalid_tool_choice",
        )

    config: dict[str, Any] = {"functionCallingConfig": {"mode": mode}}
    if allowed_names:
        config["functionCallingConfig"]["allowedFunctionNames"] = allowed_names
    return config


def _tool_choice_mode_to_vertex(mode: Any) -> str:
    if mode in {None, "auto"}:
        return "AUTO"
    if mode == "required":
        return "ANY"
    if mode == "none":
        return "NONE"
    raise APIError(
        "Invalid `tool_choice.mode`.",
        400,
        param="tool_choice.mode",
        error_type="invalid_request_error",
        code="invalid_tool_choice",
    )


def _resolve_allowed_tool_choice_names(
    tools: Any,
    function_tools: list[dict[str, Any]],
    builtin_function_names_by_type: dict[str, list[str]],
) -> Optional[list[str]]:
    if tools is None:
        return [tool["name"] for tool in function_tools if isinstance(tool.get("name"), str)]
    if not isinstance(tools, list):
        raise APIError(
            "`tool_choice.tools` must be an array.",
            400,
            param="tool_choice.tools",
            error_type="invalid_request_error",
            code="invalid_tool_choice",
        )

    allowed_names: list[str] = []
    for tool in tools:
        if not isinstance(tool, dict):
            raise APIError(
                "Each item in `tool_choice.tools` must be an object.",
                400,
                param="tool_choice.tools",
                error_type="invalid_request_error",
                code="invalid_tool_choice",
            )
        tool_type = tool.get("type")
        if tool_type == "function":
            name = tool.get("name")
            if not isinstance(name, str) and isinstance(tool.get("function"), dict):
                nested_name = tool["function"].get("name")
                name = nested_name if isinstance(nested_name, str) else None
            if not isinstance(name, str) or not name.strip():
                raise APIError(
                    "Function item in `tool_choice.tools` requires `name`.",
                    400,
                    param="tool_choice.tools",
                    error_type="invalid_request_error",
                    code="invalid_tool_choice",
                )
            allowed_names.append(name.strip())
            continue
        if isinstance(tool_type, str):
            mapped_names = builtin_function_names_by_type.get(tool_type)
            if mapped_names:
                allowed_names.extend(mapped_names)
                continue
        raise APIError(
            "Unsupported `tool_choice.tools` item.",
            400,
            param="tool_choice.tools",
            error_type="invalid_request_error",
            code="invalid_tool_choice",
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for name in allowed_names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped or None


def _translate_generation_config(request_payload: dict[str, Any], *, warnings: list[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    mapping = {
        "temperature": "temperature",
        "top_p": "topP",
        "top_k": "topK",
        "max_output_tokens": "maxOutputTokens",
        "seed": "seed",
        "presence_penalty": "presencePenalty",
        "frequency_penalty": "frequencyPenalty",
    }
    for openai_key, google_key in mapping.items():
        value = request_payload.get(openai_key)
        if isinstance(value, (int, float)):
            config[google_key] = value

    max_tokens = request_payload.get("max_tokens")
    if isinstance(max_tokens, (int, float)) and "maxOutputTokens" not in config:
        config["maxOutputTokens"] = int(max_tokens)

    stop_value = request_payload.get("stop")
    if isinstance(stop_value, str) and stop_value:
        config["stopSequences"] = [stop_value]
    elif isinstance(stop_value, list):
        stop_sequences = [value for value in stop_value if isinstance(value, str) and value]
        if stop_sequences:
            config["stopSequences"] = stop_sequences

    top_logprobs = request_payload.get("top_logprobs")
    if isinstance(top_logprobs, int) and not isinstance(top_logprobs, bool):
        if top_logprobs < 0:
            raise APIError(
                "`top_logprobs` must be >= 0.",
                400,
                param="top_logprobs",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        if top_logprobs > 0:
            config["responseLogprobs"] = True
            config["logprobs"] = top_logprobs

    _apply_modalities_and_audio_config(request_payload, config, warnings=warnings)

    if isinstance(request_payload.get("parallel_tool_calls"), bool):
        warnings.append("`parallel_tool_calls` has no direct Vertex equivalent and is preserved only in response metadata.")

    reasoning = request_payload.get("reasoning")
    if reasoning is not None:
        if not isinstance(reasoning, dict):
            raise APIError(
                "`reasoning` must be an object.",
                400,
                param="reasoning",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        thinking_config: dict[str, Any] = {}
        effort = reasoning.get("effort")
        if isinstance(effort, str):
            budget = REASONING_BUDGET_BY_EFFORT.get(effort)
            if budget is None:
                warnings.append(f"Unsupported reasoning effort `{effort}` ignored.")
            else:
                thinking_config["thinkingBudget"] = budget
        summary = reasoning.get("summary")
        if summary not in {None, "none"}:
            thinking_config["includeThoughts"] = True
        generate_summary = reasoning.get("generate_summary")
        if generate_summary not in {None, False, "none"}:
            thinking_config["includeThoughts"] = True
        if thinking_config:
            config["thinkingConfig"] = thinking_config

    text_config = request_payload.get("text")
    response_format = request_payload.get("response_format")
    if text_config is not None and response_format is not None:
        warnings.append("Both `text` and legacy `response_format` were provided. `text` takes precedence.")

    if text_config is not None:
        _apply_text_config(text_config, config, warnings=warnings)
    elif response_format is not None:
        warnings.append("`response_format` is legacy compatibility input; prefer `text` for Responses API.")
        _apply_legacy_response_format(response_format, config)

    return config


def _apply_modalities_and_audio_config(
    request_payload: dict[str, Any],
    generation_config: dict[str, Any],
    *,
    warnings: list[str],
) -> None:
    modalities = request_payload.get("modalities")
    if isinstance(modalities, list):
        vertex_modalities: list[str] = []
        for modality in modalities:
            if not isinstance(modality, str):
                continue
            mapped = VERTEX_RESPONSE_MODALITY_BY_OPENAI.get(modality.strip().lower())
            if mapped and mapped not in vertex_modalities:
                vertex_modalities.append(mapped)
        if vertex_modalities:
            generation_config["responseModalities"] = vertex_modalities

    audio_config = request_payload.get("audio")
    if isinstance(audio_config, dict):
        speech_config: dict[str, Any] = {}
        voice = audio_config.get("voice")
        if isinstance(voice, str) and voice.strip():
            speech_config["voiceConfig"] = {"prebuiltVoiceConfig": {"voiceName": voice.strip()}}
        if speech_config:
            generation_config["speechConfig"] = speech_config

        audio_format = audio_config.get("format")
        if isinstance(audio_format, str) and audio_format.strip():
            warnings.append(
                "`audio.format` is accepted but not directly mapped; configure exact format via `vertex.generation_config`."
            )


def _apply_text_config(text_config: Any, generation_config: dict[str, Any], *, warnings: list[str]) -> None:
    if not isinstance(text_config, dict):
        raise APIError(
            "`text` must be an object.",
            400,
            param="text",
            error_type="invalid_request_error",
            code="invalid_type",
        )
    verbosity = text_config.get("verbosity")
    if verbosity is not None:
        if verbosity not in {"low", "medium", "high"}:
            raise APIError(
                "`text.verbosity` must be `low`, `medium`, or `high`.",
                400,
                param="text.verbosity",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        warnings.append("`text.verbosity` has no direct Vertex equivalent and is preserved in response metadata only.")
    format_config = text_config.get("format")
    if format_config is None:
        return
    _apply_structured_format(format_config, generation_config, param_name="text.format")


def _apply_legacy_response_format(response_format: Any, generation_config: dict[str, Any]) -> None:
    if not isinstance(response_format, dict):
        raise APIError(
            "`response_format` must be an object.",
            400,
            param="response_format",
            error_type="invalid_request_error",
            code="invalid_type",
        )
    _apply_structured_format(response_format, generation_config, param_name="response_format")


def _apply_structured_format(format_config: Any, generation_config: dict[str, Any], *, param_name: str) -> None:
    if not isinstance(format_config, dict):
        raise APIError(
            f"`{param_name}` must be an object.",
            400,
            param=param_name,
            error_type="invalid_request_error",
            code="invalid_type",
        )

    format_type = format_config.get("type")
    if format_type in {None, "text"}:
        return
    if format_type == "json_object":
        generation_config["responseMimeType"] = "application/json"
        return
    if format_type == "json_schema":
        schema = None
        nested_schema = format_config.get("json_schema")
        if isinstance(nested_schema, dict):
            schema = nested_schema.get("schema")
        if schema is None:
            schema = format_config.get("schema")
        if not isinstance(schema, dict):
            raise APIError(
                f"`{param_name}.schema` must be an object for json_schema format.",
                400,
                param=param_name,
                error_type="invalid_request_error",
                code="invalid_schema",
            )
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = schema
        return
    raise APIError(
        f"Unsupported format type `{format_type}` for `{param_name}`.",
        400,
        param=param_name,
        error_type="invalid_request_error",
        code="unsupported_feature",
    )


def _apply_vertex_extensions(request_payload: dict[str, Any], google_payload: dict[str, Any], *, warnings: list[str]) -> None:
    vertex = request_payload.get("vertex")
    if vertex is not None and not isinstance(vertex, dict):
        raise APIError(
            "`vertex` extension must be an object.",
            400,
            param="vertex",
            error_type="invalid_request_error",
            code="invalid_type",
        )
    vertex = vertex if isinstance(vertex, dict) else {}

    safety_settings = vertex.get("safety_settings", request_payload.get("safety_settings"))
    if safety_settings is not None:
        if not isinstance(safety_settings, list):
            raise APIError(
                "`safety_settings` must be an array.",
                400,
                param="safety_settings",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        google_payload["safetySettings"] = safety_settings

    labels = vertex.get("labels", request_payload.get("labels"))
    if labels is not None:
        if not isinstance(labels, dict):
            raise APIError(
                "`labels` must be an object.",
                400,
                param="labels",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        google_payload["labels"] = labels

    cached_content = vertex.get("cached_content", request_payload.get("cached_content"))
    if cached_content is not None:
        if not isinstance(cached_content, str):
            raise APIError(
                "`cached_content` must be a string.",
                400,
                param="cached_content",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        google_payload["cachedContent"] = cached_content

    model_armor_config = vertex.get("model_armor_config", request_payload.get("model_armor_config"))
    if model_armor_config is not None:
        if not isinstance(model_armor_config, dict):
            raise APIError(
                "`model_armor_config` must be an object.",
                400,
                param="model_armor_config",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        google_payload["modelArmorConfig"] = model_armor_config

    raw_generation_config = vertex.get("generation_config")
    if raw_generation_config is not None:
        if not isinstance(raw_generation_config, dict):
            raise APIError(
                "`vertex.generation_config` must be an object.",
                400,
                param="vertex.generation_config",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        merged_generation = dict(google_payload.get("generationConfig") or {})
        merged_generation.update(raw_generation_config)
        google_payload["generationConfig"] = merged_generation

    raw_tool_config = vertex.get("tool_config")
    if raw_tool_config is not None:
        if not isinstance(raw_tool_config, dict):
            raise APIError(
                "`vertex.tool_config` must be an object.",
                400,
                param="vertex.tool_config",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        merged_tool_config = dict(google_payload.get("toolConfig") or {})
        merged_tool_config.update(raw_tool_config)
        google_payload["toolConfig"] = merged_tool_config

    raw_tools = vertex.get("tools")
    if raw_tools is not None:
        if not isinstance(raw_tools, list):
            raise APIError(
                "`vertex.tools` must be an array.",
                400,
                param="vertex.tools",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        existing_tools = list(google_payload.get("tools") or [])
        existing_tools.extend(tool for tool in raw_tools if isinstance(tool, dict))
        google_payload["tools"] = existing_tools

    system_instruction = vertex.get("system_instruction")
    if system_instruction is not None:
        if isinstance(system_instruction, str):
            google_payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        elif isinstance(system_instruction, dict):
            google_payload["systemInstruction"] = system_instruction
        else:
            raise APIError(
                "`vertex.system_instruction` must be a string or object.",
                400,
                param="vertex.system_instruction",
                error_type="invalid_request_error",
                code="invalid_type",
            )

    response_modalities = vertex.get("response_modalities")
    if response_modalities is not None:
        if not isinstance(response_modalities, list) or not all(isinstance(item, str) for item in response_modalities):
            raise APIError(
                "`vertex.response_modalities` must be an array of strings.",
                400,
                param="vertex.response_modalities",
                error_type="invalid_request_error",
                code="invalid_type",
            )
        merged_generation = dict(google_payload.get("generationConfig") or {})
        merged_generation["responseModalities"] = response_modalities
        google_payload["generationConfig"] = merged_generation
