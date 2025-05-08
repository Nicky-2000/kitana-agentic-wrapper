import tiktoken

def estimate_token_count(text: str, model: str = "gpt-4") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to default encoding if model is unknown
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))