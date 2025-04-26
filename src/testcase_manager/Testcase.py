from dataclasses import dataclass
from pathlib import Path
from typing import List
import shutil

@dataclass
class TestCase:
    name: str
    buyer_csv_path: str
    seller_original_folder_path: str
    seller_augmented_folder_path: str
    target_feature: str
    join_keys: List[List[str]]

    @classmethod
    def from_name(cls, name: str, master_csv_file: str, target_feature: str, join_keys: List[List[str]]):
        base = Path("data") / name
        return cls(
            name=name,
            buyer_csv_path=str(base / "buyer" / master_csv_file),
            seller_original_folder_path=str(base / "seller"),
            seller_augmented_folder_path=str(base / "seller_augmented"),
            target_feature=target_feature,
            join_keys=join_keys,
        )

    def prepare_augmented_folder(self):
        """Creates and prepares the seller_augmented_folder."""
        print(f"[INFO] Preparing augmented folder for {self.name}...")

        aug_folder = Path(self.seller_augmented_folder_path)
        orig_folder = Path(self.seller_original_folder_path)

        # Create if doesn't exist
        aug_folder.mkdir(parents=True, exist_ok=True)

        # Clean existing files
        for file_path in aug_folder.glob("*"):
            if file_path.is_file():
                file_path.unlink()

        # Copy original seller files
        for file_path in orig_folder.glob("*"):
            if file_path.is_file():
                shutil.copy2(file_path, aug_folder)

        print(f"[INFO] Prepared {aug_folder} with original seller data.")
