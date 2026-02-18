from app.errors import APIError
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


def test_rejects_non_function_tool() -> None:
    payload = {
        "model": "gemini-2.5-flash",
        "input": "hello",
        "tools": [{"type": "code_interpreter"}],
    }
    try:
        translate_openai_request_to_google(payload)
    except APIError as exc:
        assert exc.status_code == 400
        assert exc.code == "unsupported_feature"
    else:
        assert False, "Expected APIError for unsupported tool type"

