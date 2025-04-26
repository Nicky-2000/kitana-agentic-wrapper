from dataclasses import dataclass

@dataclass
class VectorDBConfig:
    task_type: str

    def __getitem__(self, key):
        return getattr(self, key)

@dataclass
class LanguageModelConfig:
    api_type: str
    openai_api_key: str
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str
    google_api_key: str
    embedding_api_type: str
    local_model_name: str

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