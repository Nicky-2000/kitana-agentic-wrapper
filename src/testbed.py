import json
import os
from typing import Dict, List

import pandas as pd
from typed_dicts import EvalConfig

from filterTables import FilterByEmbeddings, FilterTables

# change it to root of the project
root_dir = "./"

augmented_data_dir = f"{root_dir}/augmented-data"
data_dir = f"{root_dir}/data"

def get_eval() -> Dict[int, EvalConfig]:
    eval_config:Dict[int, EvalConfig]
    try:
        with open(f"{root_dir}/eval.json", 'r') as file:
            eval_config = json.load(file)
    except Exception as e:
        print("eval.json not found.")
        return {}
    
    if eval_config is None:
        print("Unable to load eval.json.")
        return {}
    
    return eval_config

def read_file_names(augmented_data_dir):
    try:
        files = os.listdir(augmented_data_dir)
        return files
    except Exception as e:
        print(f"Error reading directory {augmented_data_dir}: {e}")
        return []

def move_files_to_data_folder(files, data_dir):
    for file in files:
        try:
            with open(f"{augmented_data_dir}/{file}", 'r') as f:
                content = f.read()
            with open(f"{data_dir}/{file}", 'w') as f:
                f.write(content)
            print(f"Moved {file} to {data_dir}")
        except Exception as e:
            print(f"Error moving {file}: {e}")

def remove_files_from_data_folder(files, data_dir):
    for file in files:
        try:
            os.remove(f"{data_dir}/{file}")
            print(f"Removed {file} from {data_dir}")
        except Exception as e:
            print(f"Error removing {file}: {e}")

def sanity_check_test_case(eval: EvalConfig, data_dir:str) -> bool:
    """
    Check if the original table and query table exist in the data directory.
    """
    query_table_path = os.path.join(data_dir, f"{eval["query_table_name"]}.csv")
    if not os.path.exists(query_table_path):
        print(f"Query table {eval['query_table_name']} not found in data directory.")
        return False
    
    table = pd.read_csv(query_table_path)

    target_columb_name = eval["target_column_name"]

    if target_columb_name not in table.columns:
        print(f"Target column {target_columb_name} not found in query table {eval['query_table_name']}.")
        return False
    
    optimal_selections = eval["optmial_selection"]
    for optimal in optimal_selections:
        if not os.path.exists(os.path.join(data_dir, f"{optimal}.csv")):
            print(f"Optimal selection {optimal} not found in data directory.")
            return False

    return True



def test_script(filter_table: FilterTables):
    eval_config:Dict[int, EvalConfig] = get_eval()

    if len(eval_config) == 0:
        return
    
    files = read_file_names(augmented_data_dir)

    move_files_to_data_folder(files, data_dir)

    tables = read_file_names(data_dir)


    filter_table.set_tables(tables.copy())

    for eval in eval_config.values():
        if not sanity_check_test_case(eval, data_dir):
            print(f"Sanity check failed for eval {eval["original_table_name"]}.")
            continue
            
        query_table_name = f"{eval["query_table_name"]}.csv"
        
        tables.remove(query_table_name)
        filter_table.remove_table(query_table_name)


        top_selections = filter_table.filterByQuery(eval["query_table_name"], eval["target_column_name"], 5)
        optimal_tables = eval["optmial_selection"]
        optimal_tables = [optimal + ".csv" for optimal in optimal_tables]

        score = 0
        for selection in top_selections:
            if selection in optimal_tables:
                score += 1
        
        print(f"Score for eval {eval["original_table_name"]}: {score}/{len(optimal_tables)}")

        filter_table.add_table(query_table_name)
        tables.append(eval["query_table_name"])
    
    remove_files_from_data_folder(files, data_dir)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    # Example usage
    # files = read_file_names(augmented_data_dir)
    # remove_files_from_data_folder(files, data_dir)

    filter_table = FilterByEmbeddings([])
    test_script(filter_table)