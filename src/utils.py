import os
import logging
from typing import Dict, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


def get_pdataframe_from_csv(dataset_file: str, demo_folder="housing_data") -> pd.DataFrame|None:
    """
    Reads a CSV file and returns a pandas DataFrame.
    
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the data from the CSV file.
    """
    file_path = os.path.join(demo_folder, dataset_file)
    df = None

    try:
        if dataset_file.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='utf-8')
        elif dataset_file.endswith('.json'):
            df = pd.read_json(file_path)
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
    
    return df


def check_if_valid_json(string_input:str) -> Tuple[bool, Dict|None]:
    is_valid = False
    json_obj = None
    try:
        # Attempt to parse the string as JSON
        import json
        json_obj = json.loads(string_input)
        is_valid = True
    except:
        pass

    return is_valid, json_obj

def gemini_json_cleaner(output):
    """
    clean common formatting issues with gemini output
    """
    start_index = 0
    end_index = len(output)
    for i in range(len(output)):
        if output[i] == '{':
            start_index = i
            break
    for i in range(len(output)-1, -1, -1):
        if output[i] == '}':
            end_index = i + 1
            break
    return output[start_index:end_index]