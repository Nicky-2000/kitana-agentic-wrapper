
from dataclasses import dataclass, field
import json

@dataclass
class Augplan:
    table_id: int
    iteration: int
    table_name: str
    column_name: str # {join_key}_table_name_{Column_name}
    
    def __str__(self):
        return f"[{self.iteration}] {self.table_name}.{self.column_name}"

@dataclass
class KitanaResults:
    augplan: list[Augplan]
    accuracy: list[float]
    time_taken: float 
    num_iterations: int
    
    def __str__(self):
        acc_summary = f"Start Acc: {self.accuracy[0]:.3f} → End Acc: {self.accuracy[-1]:.3f}" if self.accuracy else "No accuracy recorded"
        return f"KitanaResults: {self.num_iterations} augments, {acc_summary}, Time: {self.time_taken:.2f}s"


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
    
    def __str__(self):
        s = f"KitanaHistory with {len(self.kitana_results)} runs:\n"
        for i, result in enumerate(self.kitana_results):
            s += f"  Run {i}: {result}\n"
            if i < len(self.files_cleaned):
                s += f"    Files added after: {self.files_cleaned[i]}\n"
        return s

    def to_dict(self):
        return {
            "kitana_results": [
                {
                    "augplan": [(a.table_id, a.iteration, a.table_name, a.column_name) for a in result.augplan if a != None],
                    "accuracy": result.accuracy,
                    "time_taken": result.time_taken,
                    "num_iterations": result.num_iterations,
                }
                for result in self.kitana_results if result != None
            ],
            "files_cleaned": self.files_cleaned,
        }

    def save(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)