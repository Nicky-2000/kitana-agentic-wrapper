
from dataclasses import dataclass, field

@dataclass
class Augplan:
    table_id: int
    iteration: int
    table_name: str
    column_name: str # {join_key}_table_name_{Column_name}

@dataclass
class KitanaResults:
    augplan: list[Augplan]
    accuracy: list[float]
    time_taken: float 
    num_iterations: int

    @classmethod
    def from_dict(cls, data: dict) -> "KitanaResults":
        augplans = [Augplan(*entry) for entry in data["augplan"]]
        return cls(
            augplan=augplans,
            accuracy=data["accuracy"],
            time_taken=data["time_taken"],
            num_iterations=len(augplans),
        )

@dataclass
class KitanaHistory:
    kitana_results: list[KitanaResults] = field(default_factory=list)
    files_cleaned: list[list[str]] = field(default_factory=list)