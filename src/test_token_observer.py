import os
from language_model_interface import LanguageModelInterface
from token_observer import llm_token_observer
from typed_dicts import LanguageModelConfig

config = LanguageModelConfig(
    api_type=os.getenv("API_TYPE", ""),
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    aws_access_key_id=os.getenv("GOOGLE_API_KEY", ""),  # Note: variable name mismatch
    aws_secret_access_key=os.getenv("AWS_ACCESS_KEY_ID", ""),
    aws_region=os.getenv("AWS_REGION", ""),
    google_api_key=os.getenv("GOOGLE_API_KEY", ""),
    local_model_name=os.getenv("LOCAL_MODEL_NAME", ""),
    embedding_api_type=os.getenv("EMBEDDING_API_TYPE", "")
)

def test_token_observer():
    # Test the singleton instance
    observer1 = llm_token_observer
    observer2 = llm_token_observer
    assert observer1 is observer2, "LLMTokenObserver is not a singleton instance."

    config.api_type = "google"

    lm = LanguageModelInterface(config)

    lm.get_response("Hi there, how are you doing today?", 0, False)
    print(observer1.get_total_tokens())

    lm.get_text_response("Hi there, how are you doing today?")
    print(observer1.get_total_tokens())

    assert observer1.get_total_tokens() == observer2.get_total_tokens(), "Token counts do not match between instances."
    assert observer1.get_total_tokens()["google"]["prompt"] > 0, "Tokens consumed"
    assert observer1.get_total_tokens()["google"]["completion"] > 0, "Tokens consumed"
    assert observer1.get_total_tokens()["google"]["total"] > 0, "Tokens consumed"

    observer1.reset()
    assert observer1.get_total_tokens()["google"]["prompt"] == 0, "Tokens not reset"
    assert observer1.get_total_tokens()["google"]["completion"] == 0, "Tokens not reset"
    assert observer1.get_total_tokens()["google"]["total"] == 0, "Tokens consumed"

    config.api_type = "openai"

    lm = LanguageModelInterface(config)

    lm.get_response("Hi there, how are you doing today? return a json with {'answer': '<your answer>'}", 0, False)
    print(observer1.get_total_tokens())

    lm.get_text_response("Hi there, how are you doing today?")
    print(observer1.get_total_tokens())

    assert observer1.get_total_tokens()["openai"]["prompt"] > 0, "Tokens consumed"
    assert observer1.get_total_tokens()["openai"]["completion"] > 0, "Tokens consumed"
    assert observer1.get_total_tokens()["openai"]["total"] > 0, "Tokens consumed"

    config.api_type = "local"

    lm = LanguageModelInterface(config)

    lm.get_response("Hi there, how are you doing today?", 0, False)
    print(observer1.get_total_tokens())

    lm.get_text_response("Hi there, how are you doing today?")
    print(observer1.get_total_tokens())

    assert observer1.get_total_tokens()["local"]["prompt"] > 0, "Tokens consumed"
    assert observer1.get_total_tokens()["local"]["completion"] > 0, "Tokens consumed"
    assert observer1.get_total_tokens()["local"]["total"] > 0, "Tokens consumed"

test_token_observer()