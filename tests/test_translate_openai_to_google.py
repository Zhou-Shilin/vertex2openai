from app.translate.openai_to_google import translate_openai_request_to_google


def test_translate_text_and_function_tools() -> None:
    payload = {
        "model": "gemini-2.5-flash",
        "input": "hello",
        "tools": [
            {
                "type": "function",
                "name": "lookup_weather",
                "description": "Lookup weather.",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "tool_choice": "auto",
        "response_format": {"type": "json_object"},
    }

    result = translate_openai_request_to_google(payload)
    assert result.payload["contents"][0]["parts"][0]["text"] == "hello"
    assert result.payload["tools"][0]["functionDeclarations"][0]["name"] == "lookup_weather"
    assert result.payload["toolConfig"]["functionCallingConfig"]["mode"] == "AUTO"
    assert result.payload["generationConfig"]["responseMimeType"] == "application/json"


def test_translate_responses_text_format_and_reasoning() -> None:
    payload = {
        "model": "gemini-2.5-flash",
        "input": "hello",
        "text": {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            }
        },
        "reasoning": {"effort": "medium", "summary": "concise"},
        "metadata": {"trace_id": "abc"},
        "include": ["reasoning.encrypted_content"],
    }
    result = translate_openai_request_to_google(payload)
    generation_config = result.payload["generationConfig"]
    assert generation_config["responseMimeType"] == "application/json"
    assert generation_config["responseSchema"]["type"] == "object"
    assert generation_config["thinkingConfig"]["thinkingBudget"] == 4096
    assert generation_config["thinkingConfig"]["includeThoughts"] is True
    assert result.passthrough_fields["metadata"]["trace_id"] == "abc"
    assert "reasoning.encrypted_content" in result.include


def test_translate_function_call_output_with_previous_response_map() -> None:
    previous_response = {
        "id": "resp_prev",
        "output": [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "lookup_weather",
                "arguments": '{"city":"Paris"}',
            }
        ],
    }
    payload = {
        "model": "gemini-2.5-flash",
        "previous_response_id": "resp_prev",
        "input": [{"type": "function_call_output", "call_id": "call_123", "output": '{"temp":23}'}],
    }

    result = translate_openai_request_to_google(payload, previous_response=previous_response)
    last_content = result.payload["contents"][-1]
    function_response = last_content["parts"][0]["functionResponse"]
    assert function_response["name"] == "lookup_weather"
    assert function_response["response"]["temp"] == 23


def test_translate_builtin_tools_to_vertex_tools() -> None:
    payload = {
        "model": "gemini-2.5-flash",
        "input": "hello",
        "tools": [{"type": "code_interpreter"}, {"type": "web_search_preview"}],
    }
    result = translate_openai_request_to_google(payload)
    assert {"codeExecution": {}} in result.payload["tools"]
    assert {"googleSearch": {}} in result.payload["tools"]


def test_translate_input_file_data_url() -> None:
    payload = {
        "model": "gemini-2.5-flash",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_data": "data:text/plain;base64,aGVsbG8=",
                    }
                ],
            }
        ],
    }
    result = translate_openai_request_to_google(payload)
    part = result.payload["contents"][0]["parts"][0]
    assert part["inlineData"]["mimeType"] == "text/plain"
    assert part["inlineData"]["data"] == "aGVsbG8="


def test_translate_file_search_as_synthetic_function_tool() -> None:
    payload = {
        "model": "gemini-2.5-flash",
        "input": "hello",
        "tools": [{"type": "file_search"}],
    }
    result = translate_openai_request_to_google(payload)
    decls = result.payload["tools"][0]["functionDeclarations"]
    assert any(decl["name"] == "file_search" for decl in decls)


def test_translate_file_id_map_for_image_and_file() -> None:
    payload = {
        "model": "gemini-2.5-flash",
        "file_id_map": {
            "file_img": "https://example.com/test.png",
            "file_doc": "data:text/plain;base64,aGVsbG8=",
        },
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_image", "file_id": "file_img"},
                    {"type": "input_file", "file_id": "file_doc"},
                ],
            }
        ],
    }
    result = translate_openai_request_to_google(payload)
    parts = result.payload["contents"][0]["parts"]
    assert parts[0]["fileData"]["fileUri"] == "https://example.com/test.png"
    assert parts[0]["fileData"]["mimeType"] == "image/png"
    assert parts[1]["inlineData"]["mimeType"] == "text/plain"


def test_translate_input_audio_data_url() -> None:
    payload = {
        "model": "gemini-2.5-flash",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_audio", "audio_url": "data:audio/wav;base64,AAAA"},
                ],
            }
        ],
    }
    result = translate_openai_request_to_google(payload)
    part = result.payload["contents"][0]["parts"][0]
    assert part["inlineData"]["mimeType"] == "audio/wav"
    assert part["inlineData"]["data"] == "AAAA"


def test_translate_top_logprobs_and_modalities_audio() -> None:
    payload = {
        "model": "gemini-2.5-flash",
        "input": "hello",
        "top_logprobs": 3,
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": "wav"},
    }
    result = translate_openai_request_to_google(payload)
    generation = result.payload["generationConfig"]
    assert generation["responseLogprobs"] is True
    assert generation["logprobs"] == 3
    assert generation["responseModalities"] == ["TEXT", "AUDIO"]
    assert generation["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "alloy"


def test_translate_tool_choice_allowed_tools_for_synthetic_tool() -> None:
    payload = {
        "model": "gemini-2.5-flash",
        "input": "hello",
        "tools": [{"type": "shell", "name": "run_shell"}],
        "tool_choice": {"type": "allowed_tools", "mode": "required", "tools": [{"type": "shell"}]},
    }
    result = translate_openai_request_to_google(payload)
    config = result.payload["toolConfig"]["functionCallingConfig"]
    assert config["mode"] == "ANY"
    assert "run_shell" in config["allowedFunctionNames"]


def test_translate_reasoning_xhigh_and_generate_summary() -> None:
    payload = {
        "model": "gemini-2.5-flash",
        "input": "hello",
        "reasoning": {"effort": "xhigh", "generate_summary": "concise"},
    }
    result = translate_openai_request_to_google(payload)
    thinking = result.payload["generationConfig"]["thinkingConfig"]
    assert thinking["thinkingBudget"] == 16384
    assert thinking["includeThoughts"] is True


def test_translate_passthrough_prompt_cache_and_safety_identifier() -> None:
    payload = {
        "model": "gemini-2.5-flash",
        "input": "hello",
        "prompt_cache_key": "cache-key-1",
        "prompt_cache_retention": "24h",
        "safety_identifier": "user-123",
        "max_tool_calls": 2,
        "metadata": {"trace_id": "abc", "attempt": 1, "safe": True},
    }
    result = translate_openai_request_to_google(payload)
    passthrough = result.passthrough_fields
    assert passthrough["prompt_cache_key"] == "cache-key-1"
    assert passthrough["prompt_cache_retention"] == "24h"
    assert passthrough["safety_identifier"] == "user-123"
    assert passthrough["max_tool_calls"] == 2
    assert passthrough["metadata"]["attempt"] == 1
