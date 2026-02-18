"""Translate OpenAI Responses request payload into Gemini REST payload."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.errors import APIError

DATA_URL_PATTERN = re.compile(r"^data:(?P<mime>[^;]+);base64,(?P<data>.+)$", re.IGNORECASE)


@dataclass
class TranslateToGoogleResult:
    payload: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    call_id_name_map: dict[str, str] = field(default_factory=dict)


def translate_openai_request_to_google(
    request_payload: dict[str, Any],
    *,
    previous_response: Optional[dict[str, Any]] = None,
) -> TranslateToGoogleResult:
    warnings: list[str] = []
    _validate_top_level(request_payload)
    call_map = _collect_call_map(previous_response)
    system_texts: list[str] = []
    contents: list[dict[str, Any]] = []

    if previous_response:
        contents.extend(_previous_response_to_contents(previous_response))

    input_value = request_payload.get("input", "")
    new_contents, discovered_system_texts = _input_to_contents(
        input_value,
        call_map=call_map,
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

    tools_payload = _translate_tools(request_payload.get("tools"))
    if tools_payload:
        google_payload["tools"] = [{"functionDeclarations": tools_payload}]
    tool_config = _translate_tool_choice(request_payload.get("tool_choice"), tools_payload)
    if tool_config:
        google_payload["toolConfig"] = tool_config

    generation_config = _translate_generation_config(request_payload, warnings)
    if generation_config:
        google_payload["generationConfig"] = generation_config

    return TranslateToGoogleResult(
        payload=google_payload,
        warnings=warnings,
        call_id_name_map=call_map,
    )


def _validate_top_level(payload: dict[str, Any]) -> None:
    modalities = payload.get("modalities")
    if isinstance(modalities, list) and "audio" in modalities:
        raise APIError(
            "Audio modality is not supported in this proxy version.",
            400,
            param="modalities",
            error_type="invalid_request_error",
            code="unsupported_feature",
        )
    if payload.get("audio") is not None:
        raise APIError(
            "`audio` is not supported in this proxy version.",
            400,
            param="audio",
            error_type="invalid_request_error",
            code="unsupported_feature",
        )


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

        if item.get("type") == "function_call_output":
            contents.append(_function_call_output_to_content(item, call_map=call_map))
            continue

        if "role" in item:
            role = item.get("role")
            content_value = item.get("content")
            if role == "system":
                system_text = _extract_text_from_content(content_value)
                if system_text:
                    system_texts.append(system_text)
                continue
            mapped_role = "model" if role == "assistant" else "user"
            parts = _content_to_parts(content_value, warnings=warnings)
            if parts:
                contents.append({"role": mapped_role, "parts": parts})
            continue

        parts = _content_to_parts(item, warnings=warnings)
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


def _content_to_parts(content: Any, *, warnings: list[str]) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"text": content}]
    if isinstance(content, dict):
        part = _content_block_to_part(content, warnings=warnings)
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
            part = _content_block_to_part(block, warnings=warnings)
            if part:
                parts.append(part)
        return parts
    return []


def _content_block_to_part(block: dict[str, Any], *, warnings: list[str]) -> Optional[dict[str, Any]]:
    block_type = block.get("type")
    if block_type in {None, "input_text", "text", "output_text"}:
        text = block.get("text")
        if isinstance(text, str):
            return {"text": text}
        return None

    if block_type in {"input_image", "image_url"}:
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
        return {"fileData": {"fileUri": url_value}}

    if block_type in {"input_audio", "audio"}:
        raise APIError(
            "Audio content is not supported in this proxy version.",
            400,
            param="input",
            error_type="invalid_request_error",
            code="unsupported_feature",
        )

    warnings.append(f"Ignored unsupported content block type: {block_type!r}.")
    return None


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


def _translate_tools(tools_value: Any) -> list[dict[str, Any]]:
    if tools_value is None:
        return []
    if not isinstance(tools_value, list):
        raise APIError(
            "`tools` must be an array.",
            400,
            param="tools",
            error_type="invalid_request_error",
            code="invalid_type",
        )

    declarations: list[dict[str, Any]] = []
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
        if tool_type != "function":
            raise APIError(
                f"Unsupported tool type `{tool_type}`. Only `function` is supported.",
                400,
                param="tools",
                error_type="invalid_request_error",
                code="unsupported_feature",
            )

        function_obj = tool.get("function") if isinstance(tool.get("function"), dict) else tool
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
        declarations.append(declaration)
    return declarations


def _translate_tool_choice(tool_choice: Any, tools: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not tools:
        if tool_choice in (None, "none", "auto"):
            return None
        raise APIError(
            "`tool_choice` was provided but no function tools exist.",
            400,
            param="tool_choice",
            error_type="invalid_request_error",
            code="invalid_tool_choice",
        )

    mode = "AUTO"
    allowed_names: Optional[list[str]] = None

    if tool_choice is None or tool_choice == "auto":
        mode = "AUTO"
    elif tool_choice == "none":
        mode = "NONE"
    elif tool_choice == "required":
        mode = "ANY"
    elif isinstance(tool_choice, dict):
        if tool_choice.get("type") != "function":
            raise APIError(
                "Only function tool_choice is supported.",
                400,
                param="tool_choice",
                error_type="invalid_request_error",
                code="invalid_tool_choice",
            )
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


def _translate_generation_config(request_payload: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    mapping = {
        "temperature": "temperature",
        "top_p": "topP",
        "max_output_tokens": "maxOutputTokens",
    }
    for openai_key, google_key in mapping.items():
        value = request_payload.get(openai_key)
        if isinstance(value, (int, float)):
            config[google_key] = value

    max_tokens = request_payload.get("max_tokens")
    if isinstance(max_tokens, (int, float)) and "maxOutputTokens" not in config:
        config["maxOutputTokens"] = max_tokens

    response_format = request_payload.get("response_format")
    if response_format is None:
        return config
    if not isinstance(response_format, dict):
        raise APIError(
            "`response_format` must be an object.",
            400,
            param="response_format",
            error_type="invalid_request_error",
            code="invalid_type",
        )

    format_type = response_format.get("type")
    if format_type in {None, "text"}:
        return config

    if format_type == "json_object":
        config["responseMimeType"] = "application/json"
        return config

    if format_type == "json_schema":
        schema = None
        schema_container = response_format.get("json_schema")
        if isinstance(schema_container, dict):
            schema = schema_container.get("schema")
        if schema is None:
            schema = response_format.get("schema")
        if not isinstance(schema, dict):
            raise APIError(
                "`response_format.json_schema.schema` must be an object.",
                400,
                param="response_format",
                error_type="invalid_request_error",
                code="invalid_schema",
            )
        config["responseMimeType"] = "application/json"
        config["responseSchema"] = schema
        return config

    raise APIError(
        f"Unsupported response_format type `{format_type}`.",
        400,
        param="response_format",
        error_type="invalid_request_error",
        code="unsupported_feature",
    )
