import shutil
import sys
from pathlib import Path
from rich import print  # Fancy output for terminal
import os
from src.kitana_history.query_history import KitanaResults, KitanaHistory
from src.datalake.load_datalake import load_in_datalake
from src.embedding_datalake_search import embedding_datalake_search
from src.llm_enrich_search import llm_enrich_search_func
from src.llm_table_rank_search import llm_selector_search_func
from src.datalake.Datalake import DataLake
from src.testcase_manager.Testcase import TestCase
from src.utils import get_file_count
from src.token_observer import llm_token_observer
from test_cases import test_cases
import json

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

    if get_file_count(seller_data_folder_path) == 0: 
        print(f"Warning: {seller_data_folder_path} is empty. No files to process. (This is expected for some test cases)")
        return None

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
    
    if company_experiment_result is None:
        print(f"Warning: No valid results found for {seller_data_folder_path}.")
        return None
    kitana_results = KitanaResults.from_dict(company_experiment_result)
    
    return kitana_results


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
            

def save_token_usage_summary(token_usage_summary: dict, token_summary_path: str = "kitana_logs/token_summary.json"):
    # 1. Load existing token log (if any)
    if os.path.exists(token_summary_path):
        with open(token_summary_path, "r") as f:
            existing_summary = json.load(f)
    else:
        existing_summary = {}

    # 2. Merge with current run
    existing_summary.update(token_usage_summary)

    # 3. Save the updated log
    with open(token_summary_path, "w") as f:
        json.dump(existing_summary, f, indent=2)

    print(f"[bold green]✅ Saved updated token summary to:[/bold green] {token_summary_path}")



if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # Load in the datalake files
    load_in_datalake(
        g_drive_url="https://drive.google.com/file/d/1gSqBzDnqHmBvHVekADfHlILxH3t6_SST/view?usp=sharing",
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

    method = "embedding" #emb
    top_k_param = 2
    token_budget = 2000
    token_usage_summary = {}
    token_summary_path = "kitana_logs/token_summary.json"
    
    for test_case_dic in test_cases:
        test_case = TestCase.from_name(name=test_case_dic["test_name"],
                                      master_csv_file=test_case_dic["table"],
                                      target_feature=test_case_dic["target_col"],
                                      join_keys=test_case_dic["join_keys"])
        
        print("\n" + "=" * 80)
        print(f"[bold cyan]🚀 Running Test Case: [green]{test_case.name}[/green][/bold cyan]")
        print("=" * 80)
        
        test_case.prepare_augmented_folder()
        datalake = DataLake(datalake_folder="data/datalake")

        kitana_history = KitanaHistory()

        # Step 1: Run Kitana (Initial Run with original data)
        print("[bold]🧪 Step 1: Running Kitana baseline experiment...[/bold]")
        kitana_results = run_kitana(
            seller_data_folder_path=test_case.seller_augmented_folder_path,
            buyer_csv_path=test_case.buyer_csv_path,
            join_keys=test_case.join_keys,
            target_feature=test_case.target_feature,
        )
        
        kitana_history.kitana_results.append(kitana_results)

        try:
            for i in range(1):
                # Step 2: Use results to search the "datalake"
                print(f"[bold]🔍 Step 2: Searching Datalake using method: [yellow]{method}[/yellow][/bold]")

                # This is where our methods come into play

                # Examples:
                if method == "llm_enrich":
                    files_to_use = llm_enrich_search_func(kitana_history, 
                                                          datalake,
                                                          test_case, 
                                                          top_k=top_k_param,
                                                          token_budget=token_budget, 
                                                          budget_filter="greedy")
                elif method == "llm_selector":
                    files_to_use = llm_selector_search_func(kitana_history, 
                                                            datalake, 
                                                            test_case,
                                                            top_k=top_k_param,
                                                            token_budget=token_budget)
                else: 
                    files_to_use = embedding_datalake_search(kitana_history, datalake, test_case, top_k=top_k_param)
                    method = "embedding"

                print(f"[bold green]📂 Files selected from datalake:[/bold green] {files_to_use}")
                kitana_history.files_cleaned.append(files_to_use)

                # Step 3: Copy the files to a folder where kitana can access it
                print("[bold]📥 Step 3: Copying files into test case folder...[/bold]")

                datalake.copy_files(
                    files=files_to_use,
                    dest_folder=test_case.seller_augmented_folder_path,
                )
                
                # Step 4: Run the experiment again with the new data (Repeat)
                print("[bold]⚙️ Step 4: Re-running Kitana with new data...[/bold]")

                new_kitana_results = run_kitana(
                    seller_data_folder_path=test_case.seller_augmented_folder_path,
                    buyer_csv_path=test_case.buyer_csv_path,
                    join_keys=test_case.join_keys,
                    target_feature=test_case.target_feature,
                )
                kitana_history.kitana_results.append(new_kitana_results)
        except Exception as e:
            print(f"[bold red]❌ Test case {test_case.name} failed[/bold red]")
            print(f"[bold red]🛑 Error:[/bold red] {e}")
        finally:
            clean_data_folder(test_case.seller_augmented_folder_path)
            print(f"[bold]🧹 Cleaned up folder:[/bold] {test_case.seller_augmented_folder_path}")
            
            # ✨ Save history at the end!
            history_save_path = f"kitana_logs/{test_case.name}_history_{method}_{top_k_param}.json"
            os.makedirs("kitana_logs", exist_ok=True)
            kitana_history.save(history_save_path)
            print(f"[bold blue]🧾 Saved history to:[/bold blue] {history_save_path}")

        # Print the overall token counts
        print("[bold magenta]📊 Token Usage Summary:[/bold magenta]")
        token_totals = llm_token_observer.get_overall_totals()
        print(token_totals)
        token_usage_summary[test_case.name] = token_totals
        llm_token_observer.reset()
    
    print(f"[bold magenta]📊 Saving token usuage data to file: {token_summary_path}:[/bold magenta]")
    save_token_usage_summary(token_usage_summary, token_summary_path)
    