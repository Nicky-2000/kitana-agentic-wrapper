from pathlib import Path
import shutil
import pandas as pd
from typing import List, Callable

class DataLake:
    def __init__(self, datalake_folder: str):
        self.datalake_folder = Path(datalake_folder)
        if not self.datalake_folder.exists():
            raise ValueError(f"Datalake folder {datalake_folder} does not exist.")
        self.refresh_files()

    def refresh_files(self):
        """Refresh the internal cache of datalake files."""
        self.files = [f.name for f in self.datalake_folder.glob("*.csv")]
    
    def list_files(self) -> List[str]:
        return list(self.files)

    def search(self, search_fn: Callable[[str], bool]) -> List[str]:
        """
        Search datalake using a custom function.
        Args:
            search_fn: function that takes filename and returns True if wanted.
        Returns:
            List of matching filenames.
        """
        return [fname for fname in self.files if search_fn(fname)]

    def copy_files(self, files: List[str], dest_folder: str):
        """Copy selected files to another folder."""
        dest_folder = Path(dest_folder)
        dest_folder.mkdir(parents=True, exist_ok=True)

        for fname in files:
            src_path = self.datalake_folder / fname
            dst_path = dest_folder / fname

            if src_path.exists():
                shutil.copy2(src_path, dst_path)
            else:
                print(f"[WARNING] File {fname} does not exist in datalake.")

    def load_file(self, filename: str) -> pd.DataFrame:
        """Load a single CSV as a pandas DataFrame."""
        file_path = self.datalake_folder / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File {filename} not found in datalake.")
        return pd.read_csv(file_path)

