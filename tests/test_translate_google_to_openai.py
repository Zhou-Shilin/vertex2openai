from app.translate.google_to_openai import translate_google_response_to_openai, usage_from_google


def test_usage_mapping() -> None:
    usage = usage_from_google(
        {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
    )
    assert usage == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_translate_google_response_with_text_and_function_call() -> None:
    google_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Hello "},
                        {"text": "world"},
                        {"functionCall": {"name": "lookup_weather", "args": {"city": "Paris"}}},
                    ]
                }
            }
        ],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 2, "totalTokenCount": 3},
    }
    response = translate_google_response_to_openai(
        google_payload=google_payload,
        model="gemini-2.5-flash",
        response_id="resp_test",
    )

    assert response["id"] == "resp_test"
    assert response["object"] == "response"
    assert response["model"] == "gemini-2.5-flash"
    assert response["usage"]["total_tokens"] == 3
    assert response["output"][0]["type"] == "message"
    assert response["output"][0]["content"][0]["text"] == "Hello world"
    assert response["output"][1]["type"] == "function_call"
    assert response["output"][1]["name"] == "lookup_weather"


def test_translate_google_response_with_logprobs() -> None:
    google_payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "hello"},
                    ]
                },
                "logprobsResult": {
                    "topCandidates": [
                        {"candidates": [{"token": "hello", "logProbability": -0.1}]},
                    ]
                },
            }
        ]
    }
    response = translate_google_response_to_openai(
        google_payload=google_payload,
        model="gemini-2.5-flash",
        response_id="resp_logprobs",
    )
    output_text = response["output"][0]["content"][0]
    assert "logprobs" in output_text
    assert output_text["logprobs"]["topCandidates"][0]["candidates"][0]["token"] == "hello"
