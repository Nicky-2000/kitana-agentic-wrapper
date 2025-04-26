import gdown
import zipfile
import os

def load_in_datalake(g_drive_url, extract_folder="data/", zip_filename="datalake.zip", delete_zip_after_extract=True):
    """
    Downloads a zip file from Google Drive, extracts it, and (optionally) deletes the zip file.

    Args:
        file_id (str): The Google Drive file ID of the zip file.
        extract_folder (str): The folder where the contents will be extracted.
        zip_filename (str): The local filename for the downloaded zip.
        delete_zip_after_extract (bool): Whether to delete the zip file after extraction.
    """
    final_folder_path = f"{extract_folder}/datalake"
    # Skip download if data is already extracted
    if os.path.exists(final_folder_path):
        print(f"[INFO] Folder '{final_folder_path}' already exists. Skipping download.")
        return

    # Step 1: Download the zip file
    if not os.path.exists(zip_filename):
        print("[INFO] Downloading dataset...")
        gdown.download(g_drive_url, zip_filename, quiet=False, fuzzy=True)
    else:
        print(f"[INFO] Zip file '{zip_filename}' already downloaded. Skipping download.")

    # Step 2: Extract the zip file
    print("[INFO] Unzipping dataset...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)

    # Step 3: Optionally delete the zip
    if delete_zip_after_extract:
        os.remove(zip_filename)
        print(f"[INFO] Deleted zip file '{zip_filename}' after extraction.")

    print(f"[INFO] Dataset ready in '{extract_folder}/'.")

