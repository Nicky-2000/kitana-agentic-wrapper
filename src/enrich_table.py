
import os
import pandas as pd
from src.language_model_interface import LanguageModelInterface, Config

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_join_columns(lm:LanguageModelInterface, input_table:str, data_folder:str = "data"):
    """
    Prompts LLM to identify join columns 
    Parses out join columns from 
    """

    df = pd.read_csv(os.path.join(data_folder, input_table))
    table_columns = df.columns.to_list()
    #print(table_columns)

    find_join_key_prompt = f"""
    Given the following table name {input_table} and the following columns:
    {table_columns}

    List all of the potential join keys of the table. Response as briefly as possible.
    """

    lm_join_columns = lm.get_text_response(find_join_key_prompt)

    join_columns = []
    for column in table_columns:
        if column in lm_join_columns: 
            join_columns.append(column)

    sample_join_key_data = df[join_columns].sample(5).to_string()

    fin_prompt = f"""

    Describe what the following join_key columns from the table {input_table} entail:

    {sample_join_key_data}

    Please keep each description of the join key to one sentence.
    """

    join_key_descript = lm.get_text_response(fin_prompt)

    return join_columns, join_key_descript


def get_table_description(lm:LanguageModelInterface, input_table:str, table_join_keys_description:str, data_folder:str = "data"):

    

    df = pd.read_csv(os.path.join(data_folder, input_table))
    table_columns = df.columns.to_list()

    sample_data = df.sample(5).to_string()

    enrich_table_prompt = f"""
    Given the following sample from a table:
    {sample_data}
    Provide a one sentence description of the overall table
    """

    enrich_table_description = lm.get_text_response(enrich_table_prompt)

    return table_join_keys_description + "\n" + enrich_table_description

def embed_descriptions(lm:LanguageModelInterface, description_dict:dict, query_table:str, pct_return_rate = 0.5, num_tables = None):
    """
    For a given description dictionary, return the top % of descriptions that match the query 
    """
    assert query_table in description_dict.keys(), "Query table must be in description_list"

    #this is yucky code im sorry :(
    keys_list = list(description_dict.keys())
    query_index = keys_list.index(query_table)

    if num_tables is None:
        number_of_tables_to_return = int(len(description_dict)*pct_return_rate)
    else: 
        number_of_tables_to_return = num_tables
    embed_list = []
    for key in description_dict:
        description_embedding = lm.create_embedding(description_dict[key])
        embed_list.append(description_embedding)

    embeddings_np = np.array(embed_list)

    #TODO: Make this more efficent 
    similarity_matrix = cosine_similarity(embeddings_np)

    #get similarity vs query 
    similarity_query = similarity_matrix[query_index]

    top_n_indices = sorted(range(len(similarity_query)), key=lambda i: similarity_query[i], reverse=True)[:max(2, min(number_of_tables_to_return+1, len(similarity_query)))]

    dict_to_return = {keys_list[i]: description_dict[keys_list[i]] for i in top_n_indices}

    return dict_to_return



def enrich_and_filter_table_list(table_list:list[str], query_table, query_column, query_location, num_tables = 5, data_folder = "data/datalake"):


    config = Config()
    lm = LanguageModelInterface(config)

    #get the potential join keys for each table
    #probably should do something about the join key
    table_dict = {}
    for table in table_list:

        try:
            table_join_key, table_join_description = get_join_columns(lm, table, data_folder = data_folder)
            table_dict[table] = table_join_description
        except:
            continue

    #get embedded query keys 
    query_join_key, query_table_join_description = get_join_columns(lm, query_table, data_folder = query_location)
    table_dict[query_table] = query_table_join_description

    #now filter them out based on embedding similarity
    filtered_table_dict = embed_descriptions(lm, table_dict, query_table)

    filtered_table_dict_part_2 = {}
    for table in filtered_table_dict:
        if table != query_table:
            table_description = get_table_description(lm, table, filtered_table_dict[table], data_folder = data_folder)
            filtered_table_dict_part_2[table] = table_description

    query_table_description = get_table_description(lm, query_table, filtered_table_dict[query_table], data_folder = query_location)
    filtered_table_dict_part_2[query_table] = query_table_description

    final_dict = embed_descriptions(lm, filtered_table_dict_part_2, query_table, num_tables=num_tables) 
    
    return final_dict

if __name__ == "__main__":


    table_list = [
                    "chinese_gp_circuits.csv",
                    "users.csv",
                    "world-education-data.csv",
                    "world-education-data.csv",
                    "Top_Influencers.csv",
                    "tourism.csv",
                    "transactions.csv",
                    "unique-categories.csv",
                    "unemployment.csv",
                    "unique-categories.sorted-by-count.csv",
                    "upcoming_city.csv",
                    "oral_cancer_prediction_dataset.csv",
                    "La_Veranda_Reviews-2023-01-16.csv",
                  ]
    
    fin_dict = enrich_and_filter_table_list(table_list=table_list, query_table="master.csv",  query_column=None, query_location = "data/test_case_1/buyer/", data_folder="data/datalake")

    for x in fin_dict:
        print()
        print(x)
        print(fin_dict[x])
    