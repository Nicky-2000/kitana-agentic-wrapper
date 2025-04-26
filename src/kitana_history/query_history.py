
from dataclasses import dataclass

@dataclass
class Augplan:
    table_id: int
    iteration: int
    table_name: str
    column_name: str # {join_key}_table_name_{Column_name}

@dataclass
class KitanaResults:
    augplan: list[Augplan]
    accuracy: list[int]
    time_taken: float 
    num_iterations: int

@dataclass
class KitanaHistory:
    kitana_results: list[KitanaResults]
    files_cleaned_after: list[list[str]]