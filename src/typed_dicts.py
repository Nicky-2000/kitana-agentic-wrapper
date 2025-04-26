from dataclasses import dataclass

@dataclass
class VectorDBConfig:
    task_type: str

    def __getitem__(self, key):
        return getattr(self, key)

@dataclass
class LanguageModelConfig:
    api_type: str | None
    openai_api_key: str | None
    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    aws_region: str | None
    google_api_key: str | None
    embedding_api_type: str | None
    local_model_name: str | None

    def __getitem__(self, key):
        return getattr(self, key)
    
@dataclass
class EvalConfig:
    optmial_selection: list[str]
    original_table_name: str
    query_table_name: str
    target_column_name: str

    def __getitem__(self, key):
        return getattr(self, key)