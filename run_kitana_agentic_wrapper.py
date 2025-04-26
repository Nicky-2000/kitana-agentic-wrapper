import shutil
import sys
from pathlib import Path
import os

from src.datalake_src.load_datalake import load_in_datalake
from src.embedding_datalake_search import embedding_datalake_search
from src.MCTS_datalake_search import MCTS_datalake_search

# This is needed if you don't have the vscode settings "python.autoComplete.extraPaths" set to "kitana-e2e"
sys.path.insert(0, str(Path(__file__).resolve().parent / "kitana-e2e"))

import pandas as pd
import numpy as np
from search_engine.experiment import ScaledExperiment
from search_engine.config import (
    Config,
    DataConfig,
    SearchConfig,
    ExperimentConfig,
    LoggingConfig,
)


def run_kitana(
    seller_data_folder_path: str,
    buyer_csv_path: str,
    join_keys: list,
    target_feature: str,
):

    config = Config(
        search=SearchConfig(iterations=12),
        data=DataConfig(
            directory_path=seller_data_folder_path,
            buyer_csv=buyer_csv_path,
            join_keys=join_keys,
            target_feature=target_feature,
            one_target_feature=False,
            need_to_clean_data=True,
        ),
        experiment=ExperimentConfig(plot_results=True, results_dir="results/"),
        logging=LoggingConfig(level="ERROR", file="logs/original_sample_execution.log"),
    )

    # Run exps
    company = ScaledExperiment(config)
    company_experiment_result = company.run()
    # Return as list
    return company_experiment_result


def copy_files_to_folder(src_folder: str, dest_folder: str, files: list):
    """
    Copy files from src_folder to dest_folder.
    NOTE: This assumes that all the files belong to the same folder.
    """
    for filename in files:
        src_file = os.path.join(src_folder, filename)
        dst_file = os.path.join(dest_folder, filename)

        if os.path.isfile(src_file):
            shutil.copy2(src_file, dst_file)
        else:
            print(f"Warning: {src_file} does not exist or is not a file.")


def clean_data_folder(folder_path: str):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def init_augmented_seller_folder(test_case: str):
    original_seller_data_folder_path = f"data/{test_case}/seller"
    # Augmented seller data folder path (The folder we will iteratively add data to and perform kitana experiments on)
    augmented_seller_data_folder_path = f"data/{test_case}/seller_augmented"

    # Create the target directory if it doesn't exist
    os.makedirs(augmented_seller_data_folder_path, exist_ok=True)

    # Make sure the folder is clean
    clean_data_folder(augmented_seller_data_folder_path)

    # Copy the files from the original seller data folder to the augmented folder
    copy_files_to_folder(
        src_folder=original_seller_data_folder_path,
        dest_folder=augmented_seller_data_folder_path,
        files=os.listdir(original_seller_data_folder_path),
    )
    return augmented_seller_data_folder_path, buyer_csv_path


if __name__ == "__main__":
    # Load in the datalake files
    load_in_datalake(
        g_drive_url="https://drive.google.com/file/d/1I9nEDN0_mlxz2NoioHhqRNpOMViJFjRc/view?usp=sharing",
        extract_folder="data",
        zip_filename="datalake.zip",
        delete_zip_after_extract=True,
    )
    # High level flow of what we can do here:
    # In a loop:
    # 1. Run the experiment and get the results (augplan and accuracy)
    # 2. Use the results_history to search for data from the data lake
    # Copy the files we want to run the experiment to a folder where kitana can access it.
    # 3. Run the experiment again with the new data
    # 4. Repeat until we get the results we want

    # Pseudo code:

    test_case = "test_case_1"
    buyer_csv_path = f"data/{test_case}/buyer/master.csv"
    seller_data_folder_path, buyer_csv_path = init_augmented_seller_folder(test_case)

    datalake_path = "data/datalake"
    results_history = []
    files_added = []

    # Step 1: Run Kitana (Initial Run with original data)
    results = run_kitana(
        seller_data_folder_path=seller_data_folder_path,
        buyer_csv_path=buyer_csv_path,
        join_keys=[["Country"], ["year"]],
        target_feature="suicides_no",
    )
    results_history.append(results)

    try:
        for i in range(1):
            # Step 2: Use results to search the "datalake"
            # This is where our methods come into play

            # Examples:
            files_to_use = embedding_datalake_search(results_history, datalake_path)
            # or
            files_to_use = MCTS_datalake_search(results_history, datalake_path)

            print(f"Adding Files in datalake: {files_to_use}")
            files_added.append(files_to_use)

            # Step 3: Copy the files to a folder where kitana can access it
            # TODO: Can decide to modify / make this more genric
            copy_files_to_folder(
                src_folder=datalake_path,  # Path where we found new data to use
                dest_folder=seller_data_folder_path,  # Path we use to run the experiement
                files=files_to_use,  # The files we found!
            )
            # Step 4: Run the experiment again with the new data (Repeat)
            new_results = run_kitana(
                seller_data_folder_path=seller_data_folder_path,
                buyer_csv_path=buyer_csv_path,
                join_keys=[["Country"], ["year"]],
                target_feature="suicides_no",
            )
            results_history.append(new_results)

    finally:
        clean_data_folder(seller_data_folder_path)
        print(f"Cleaned up augmented folder: {seller_data_folder_path}")
