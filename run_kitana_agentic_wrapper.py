import shutil
import sys
from pathlib import Path
import os

from src.datalake_src.load_datalake import load_in_datalake
from src.embedding_datalake_search import embedding_datalake_search
from src.MCTS_datalake_search import MCTS_datalake_search
from src.datalake_src.Datalake import DataLake
from src.testcase_manager.Testcase import TestCase

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

    test_cases = [
        TestCase.from_name(
            "test_case_1", "master.csv", "suicides_no", [["Country"], ["year"]]
        ),
        TestCase.from_name(
            "test_case_2",
            "Life Expectancy Data.csv",
            "Life expectancy",
            [["Country"], ["year"]],
        ),
        TestCase.from_name(
            "test_case_3",
            "Cost_of_Living_Index_by_Country_2024.csv",
            "Groceries Index",
            [["Country"]],
        ),
    ]
    
    for test_case in test_cases:
        test_case.prepare_augmented_folder()
        datalake = DataLake(datalake_folder="data/datalake")

        results_history = []
        files_added = []

        # Step 1: Run Kitana (Initial Run with original data)
        results = run_kitana(
            seller_data_folder_path=test_case.seller_augmented_folder_path,
            buyer_csv_path=test_case.buyer_csv_path,
            join_keys=test_case.join_keys,
            target_feature=test_case.target_feature,
        )
        results_history.append(results)

        try:
            for i in range(3):
                # Step 2: Use results to search the "datalake"
                # This is where our methods come into play

                # Examples:
                files_to_use = embedding_datalake_search(results_history, datalake, test_case, top_k=5)
                # or
                # files_to_use = MCTS_datalake_search(results_history, datalake)

                print(f"Adding Files in datalake: {files_to_use}")
                files_added.append(files_to_use)

                # Step 3: Copy the files to a folder where kitana can access it
                datalake.copy_files(
                    files=files_to_use,
                    dest_folder=test_case.seller_augmented_folder_path,
                )
                # Step 4: Run the experiment again with the new data (Repeat)
                new_results = run_kitana(
                    seller_data_folder_path=test_case.seller_augmented_folder_path,
                    buyer_csv_path=test_case.buyer_csv_path,
                    join_keys=test_case.join_keys,
                    target_feature=test_case.target_feature,
                )
                results_history.append(new_results)

        finally:
            clean_data_folder(test_case.seller_augmented_folder_path)
            print(f"Cleaned up augmented folder: {test_case.seller_augmented_folder_path}")
