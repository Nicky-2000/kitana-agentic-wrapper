import logging
import os
from typing import Tuple
import pandas as pd
from src.language_model_interface import LanguageModelInterface, Config
import json as json_package
import numpy as np


def add_csv_to_table_name_if_missing(table):
    return table if table.endswith(".csv") else table + ".csv"

def get_simple_table_description_from_col_names(table, cols, lm):
    prompt_describe_table = f"Give a one sentence description detailing what data is in table '{table}' with columns '{cols}' "
    table_description = lm.get_text_response(prompt_describe_table)
    return table_description

def rank_all_tables(query_table, query_column, kitana_results, num_tables = 5, window_size = 10, datalake_folder = "data/datalake", buyer_folder = "data/test_case_1/buyer", seller_folder ="data/test_case_1/seller", filtered_tables = []):
    


    assert num_tables < window_size, "num_tables must be smaller than window_size"

    full_table_list = []
    if len(filtered_tables) == 0: #we didn't filter tables out previously, get all tables

        for file in os.listdir(datalake_folder):
            if file.endswith(".csv"): 
                full_table_list.append(file)

    else:
        for table in filtered_tables:
            clean_table = add_csv_to_table_name_if_missing(table)
            full_table_list.append(clean_table)



    query_table_full = add_csv_to_table_name_if_missing(query_table)
    query_table_columns = pd.read_csv(os.path.join(buyer_folder, query_table_full)).columns.to_list()

    config = Config()
    lm = LanguageModelInterface(config)

    try:

        aug_plan = kitana_results.to_dict()["kitana_results"][0]

        sorted_aug_tables = get_sorted_accuracy_tables(aug_plan=aug_plan)

        top_aug_table = add_csv_to_table_name_if_missing(sorted_aug_tables[0][0])
        top_aug_table_columns =  pd.read_csv(os.path.join(seller_folder, top_aug_table)).columns.to_list()

        bottom_aug_table = add_csv_to_table_name_if_missing(sorted_aug_tables[-1][0])
        bottom_aug_table_columns =  pd.read_csv(os.path.join(seller_folder, bottom_aug_table)).columns.to_list()

        top_descript = get_simple_table_description_from_col_names(top_aug_table, top_aug_table_columns, lm)
        bottom_descript = get_simple_table_description_from_col_names(bottom_aug_table, bottom_aug_table_columns, lm)

    except:
        top_descript = ""
        bottom_descript = ""



    query_descript = get_simple_table_description_from_col_names(query_table_full, query_table_columns, lm)
    

    c = 0 
    top_table_candidates = []
    for i in range(0, len(full_table_list), window_size):
        c+= 1
        end_index = min(i+window_size, len(full_table_list))
        tables_to_list = full_table_list[i:end_index]
        tables_to_feed_to_llm = top_table_candidates + tables_to_list

        top_table_candidates = rank_tables(
            query_table = query_table_full,
            query_column = query_column,
            table_list=tables_to_feed_to_llm,
            num_tables = num_tables,
            table_description=query_descript,
            top_descript = top_descript,
            bottom_descript = bottom_descript,
        )

    #     print(f"Eval {c}: {top_table_candidates}")

    # print("END OF EVAL")
    # print()
    return top_table_candidates

def rank_tables(query_table, query_column, table_list, table_description = "", top_descript = "",bottom_descript = "", num_tables = 5, demo_folder = "data/datalake"):

    

    config = Config()
    lm = LanguageModelInterface(config)

    table_string = ""
    for table in table_list:

        try:
            table_df = pd.read_csv(os.path.join(demo_folder, table))
            table_cols = table_df.columns.to_list()
            table_string+= f"{table}: {table_cols}\n"
        except:
            continue

    additional_tables_string = f"Additionally, you have found that the following table {top_descript} has helped the most to predict {query_column}, while {bottom_descript} has helped the least"
    if top_descript == "" or bottom_descript == "":
        additional_tables_string = ""


    prompt = f"""
    You are a data scientist. You need to build a machine learning model to predict the following column {query_column} from table {query_table}. 

    You have the following description of table {query_table}: {table_description}

    {additional_tables_string}

    Given the following table: {query_table} and column: {query_column}, return the top {num_tables} that could be used to predict {query_column} in {query_table}. 
    Your answer should be a list of the top {num_tables} tables BEST related to table: {query_table} and column: {query_column} only. 
    Return you answer seperated by commas.

    The following is structured in <table>:<column> format. Which of the following tables are most similar to table: {query_table} and column: {query_column}:
    {table_string}

    """

    # print("compare prompt")
    # print(prompt)

    chain_of_thought_response = lm.get_text_response(prompt)

    # print()
    # print("#######   Response   ########")
    # print(chain_of_thought_response)
    # print()
    return parse_list_rankings(chain_of_thought_response, num_tables, table_list)

def parse_list_rankings(llm_response, num_tables, table_list, demo_folder = "data"):
 
    """
    For a given llm ranking reponses, returns the top x tables mentioned
    """
    initial_list = []
    c = 0 
    for table in table_list: 
        if table in llm_response:
            initial_list.append(table)
            c+=1 
        
        if c == num_tables: break

    return initial_list


def get_sorted_accuracy_tables(aug_plan):

    accuracy_tuples = []
    #print(len(aug_plan[0]["accuracy"]))
    for x in range(0, len(aug_plan["accuracy"])-1):
        tuples_add = (aug_plan["augplan"][x][2], aug_plan["accuracy"][x+1] - aug_plan["accuracy"][x])
        accuracy_tuples.append(tuples_add)

    return sorted(accuracy_tuples, key=lambda x: x[1])












    #print( rank_all_tables(query_table="crime-rate-by-country-2023.csv", query_column="crimeIndex",  num_tables=5,  window_size= 10) )

#parse_list_rankings("Urbanization rate.csv, Population growth.csv, military expenditure.csv, fake_crime_data.csv, who_suicide_statistics.csv")

# rank_list = rank_tables(query_table="crime-rate-by-country-2023.csv", 
#             query_column="crimeIndex", 
#             table_list=["Urbanization rate.csv", "Population growth.csv", "military expenditure.csv", "fake_crime_data.csv", "who_suicide_statistics.csv", "world_risk_index.csv", "ramen-ratings.csv", "Petrol Dataset June 20 2022.csv"],
#             num_tables=3)

# print(rank_list)



