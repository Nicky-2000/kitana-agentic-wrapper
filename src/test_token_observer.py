import os
from src.language_model_interface import LanguageModelInterface
from src.token_observer import llm_token_observer
from src.typed_dicts import LanguageModelConfig

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
    print(observer1.get_api_token_counts())

    lm.get_text_response("Hi there, how are you doing today?")
    print(observer1.get_api_token_counts())

    assert observer1.get_api_token_counts() == observer2.get_api_token_counts(), "Token counts do not match between instances."
    assert observer1.get_api_token_counts()["google"]["prompt"] > 0, "Tokens consumed"
    assert observer1.get_api_token_counts()["google"]["completion"] > 0, "Tokens consumed"
    assert observer1.get_api_token_counts()["google"]["total"] > 0, "Tokens consumed"

    observer1.reset()
    assert observer1.get_api_token_counts()["google"]["prompt"] == 0, "Tokens not reset"
    assert observer1.get_api_token_counts()["google"]["completion"] == 0, "Tokens not reset"
    assert observer1.get_api_token_counts()["google"]["total"] == 0, "Tokens consumed"

    config.api_type = "openai"

    lm = LanguageModelInterface(config)

    lm.get_response("Hi there, how are you doing today? return a json with {'answer': '<your answer>'}", 0, False)
    print(observer1.get_api_token_counts())

    lm.get_text_response("Hi there, how are you doing today?")
    print(observer1.get_api_token_counts())

    assert observer1.get_api_token_counts()["openai"]["prompt"] > 0, "Tokens consumed"
    assert observer1.get_api_token_counts()["openai"]["completion"] > 0, "Tokens consumed"
    assert observer1.get_api_token_counts()["openai"]["total"] > 0, "Tokens consumed"

    config.api_type = "local"

    lm = LanguageModelInterface(config)

    lm.get_response("Hi there, how are you doing today?", 0, False)
    print(observer1.get_api_token_counts())

    lm.get_text_response("Hi there, how are you doing today?")
    print(observer1.get_api_token_counts())

    assert observer1.get_api_token_counts()["local"]["prompt"] > 0, "Tokens consumed"
    assert observer1.get_api_token_counts()["local"]["completion"] > 0, "Tokens consumed"
    assert observer1.get_api_token_counts()["local"]["total"] > 0, "Tokens consumed"
    
    print(observer1.get_overall_totals())

    assert observer1.get_overall_totals()["prompt"] > 0, "Tokens consumed"

test_token_observer()