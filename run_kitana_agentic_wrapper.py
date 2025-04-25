import sys
from pathlib import Path

from src import MCTS_datalake_search, embedding_datalake_search

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


if __name__ == "__main__":
    # Sample Run
    results = run_kitana(
        seller_data_folder_path="data/original_sample_execution/country_extend_table_search/seller",
        buyer_csv_path="data/original_sample_execution/country_extend_table_search/buyer/master.csv",
        join_keys=[["Country"], ["year"]],
        target_feature="suicides_no",
    )
    
    print(results['augplan'])
    print(results['accuracy'])
    
    # High level flow of what we can do here: 
    
    # In a loop: 
    # 1. Run the experiment and get the results (augplan and accuracy)
    # 2. Use the results_history to search for data from the data lake
    # Copy the files we want to run the experiment to a folder where kitana can access it.
    # 3. Run the experiment again with the new data
    # 4. Repeat until we get the results we want
    
    # Pseudo code:
    
    results_history = []
    for i in range(10):
        # Step 1: Run Kitana
        results = run_kitana(
            seller_data_folder_path="data/original_sample_execution/country_extend_table_search/seller",
            buyer_csv_path="data/original_sample_execution/country_extend_table_search/buyer/master.csv",
            join_keys=[["Country"], ["year"]],
            target_feature="suicides_no",
        )
        results_history.append(results)
        
        
        # Step 2: Use results to search the "datalake"
        # This is where our methods come into play
        
        # Examples: 
        files_to_use = embedding_datalake_search(results_history)
        # or
        files_to_use = MCTS_datalake_search(results_history)
        
        # Step 3: Copy the files to a folder where kitana can access it
        # TODO: Decide on this logic. I think a "Temp Folder" is probably sufficient.
        
        # Step 4: Run the experiment again with the new data
    
