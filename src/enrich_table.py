
import os
import pandas as pd
from src.language_model_interface import LanguageModelInterface

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



def enrich_and_filter_table_list(table_list:list[str], query_table, query_column, num_tables = 5, data_folder = "data"):


    config = Config()
    lm = LanguageModelInterface(config)

    #get the potential join keys for each table
    #probably should do something about the join key
    table_dict = {}
    for table in table_list:
        table_join_key, table_join_description = get_join_columns(lm, table, data_folder = data_folder)
        table_dict[table] = table_join_description
    
    #now filter them out based on embedding similarity
    filtered_table_dict = embed_descriptions(lm, table_dict, query_table)

    filtered_table_dict_part_2 = {}
    for table in filtered_table_dict:
        table_description = get_table_description(lm, table, filtered_table_dict[table], data_folder = data_folder)
        filtered_table_dict_part_2[table] = table_description

    final_dict = embed_descriptions(lm, filtered_table_dict_part_2, query_table, num_tables=num_tables) 
    
    return final_dict


def enrich_table(input_table, demo_folder = "data"):

#     #get columns to enrich 
#     df = pd.read_csv(os.path.join(demo_folder, input_table))
#     table_columns = df.columns.to_list()
#     #print(table_columns)

#     find_join_key_prompt = f"""
#     Given the following table name {input_table} and the following columns:
#     {table_columns}

#     List all of the potential join keys of the table. Response in one sentence.
#     """

#     config = Config()
#     lm = LanguageModelInterface(config)

#     lm_join_columns = lm.get_text_response(find_join_key_prompt)

#     join_columns = []
#     for column in table_columns:
#         if column in lm_join_columns: 
#             join_columns.append(column)

#     #print("Parsed Join Keys")
#     #print(join_columns)

#     sample_join_key_data = df[join_columns].sample(5).to_string()

#     enrich_join_key_prompt = f"""
#     Given the following join key columns & sample data
#     {sample_join_key_data}
#     Provide a one sentence description of each column

#     """

#     enrich_join_key = lm.get_text_response(enrich_join_key_prompt)

#     sample_data = df.sample(5).to_string()

#     enrich_table_prompt = f"""
#     Given the following sample from a table:
#     {sample_data}
#     Provide a two sentence description of the overall table
#     """

#     enrich_table_description = lm.get_text_response(enrich_table_prompt)



#     full_table_description = f""" 

# Table name: {input_table}

# Table description: {enrich_table_description}

# Join keys: 
    
# {enrich_join_key}


#     """

#     detailed_table_embedding = lm._get_openai_embedding(full_table_description)

#     return full_table_description, detailed_table_embedding
    pass

if __name__ == "__main__":


    table_list = ['housing_pricing_data.csv', '2025-housing-dataset-alldata.csv', 
                  'mars_housing_data_500.csv', 'gotham_city_housing_16.csv', 'BostonHousing.csv', 'real_estate_info_by_location.csv', 
                  'california_housing_multiplier_by_ocean_proximity.csv', 'harry_potter_housing_6.csv', 'simpsons_housing_28.csv', 
                  'marvel_universe_housing_26.csv', 'miami-housing.csv', 'fortnite_housing_7.csv', 'minecraft_housing_23.csv', 
                  'Family Households_with_Married_Couples_Data.csv', 
                  'loan-prediction.csv']
    
    fin_dict = enrich_and_filter_table_list(table_list=table_list, query_table="housing_pricing_data.csv", query_column=None, data_folder="housing_data")

    for x in fin_dict:
        print()
        print(x)
        print(fin_dict[x])
    # for table in table_list:
    #     print(table)

    #     full_table_description, detailed_table_embedding = enrich_table(table)
    #     embedding_list.append(detailed_table_embedding)



    # embeddings_np = np.array(embedding_list)
    # similarity_matrix_pre = cosine_similarity(embeddings_np)
    # print(similarity_matrix_pre)
    # similarity_matrix_pre = np.array([
    #     [1.0, 0.83148714, 0.89471622, 0.84066556, 0.93834938, 0.86027314, 0.91079488, 0.86672887, 0.84423694, 0.86329856, 0.83962419],
    #     [0.83148714, 1.0, 0.85723294, 0.82134156, 0.82838194, 0.82001917, 0.84195256, 0.83253064, 0.83377984, 0.83832354, 0.83786642],
    #     [0.89471622, 0.85723294, 1.0, 0.86665496, 0.87479521, 0.85301044, 0.89657329, 0.8735298, 0.88602136, 0.88260463, 0.85740889],
    #     [0.84066556, 0.82134156, 0.86665496, 1.0, 0.86782073, 0.83209463, 0.86742065, 0.84436454, 0.86199878, 0.87245436, 0.83524153],
    #     [0.93834938, 0.82838194, 0.87479521, 0.86782073, 1.0, 0.86691373, 0.90344566, 0.84848972, 0.85110714, 0.85910645, 0.83540884],
    #     [0.86027314, 0.82001917, 0.85301044, 0.83209463, 0.86691373, 1.0, 0.87375057, 0.85189983, 0.8473908, 0.85168483, 0.83940858],
    #     [0.91079488, 0.84195256, 0.89657329, 0.86742065, 0.90344566, 0.87375057, 1.0, 0.86246852, 0.87906336, 0.87826256, 0.85256713],
    #     [0.86672887, 0.83253064, 0.8735298, 0.84436454, 0.84848972, 0.85189983, 0.86246852, 1.0, 0.85965542, 0.86465238, 0.8462111],
    #     [0.84423694, 0.83377984, 0.88602136, 0.86199878, 0.85110714, 0.8473908, 0.87906336, 0.85965542, 1.0, 0.86405006, 0.84244196],
    #     [0.86329856, 0.83832354, 0.88260463, 0.87245436, 0.85910645, 0.85168483, 0.87826256, 0.86465238, 0.86405006, 1.0, 0.88734699],
    #     [0.83962419, 0.83786642, 0.85740889, 0.83524153, 0.83540884, 0.83940858, 0.85256713, 0.8462111, 0.84244196, 0.88734699, 1.0]
    # ])

    # similarity_matrix = (similarity_matrix_pre - similarity_matrix_pre.min()) / (similarity_matrix_pre.max() - similarity_matrix_pre.min())

    # print(similarity_matrix)

    # G = nx.Graph()

    # # Add nodes
    # for table in table_list:
    #     G.add_node(table)

    # # Add edges with similarity weight
    # threshold = 0.3  # adjust as needed
    # for i in range(len(table_list)):
    #     for j in range(i + 1, len(table_list)):
    #         sim = similarity_matrix[i, j]
    #         if sim > threshold:
    #             G.add_edge(table_list[i], table_list[j], weight=sim)


    # pos = nx.spring_layout(G, seed=42)  # positions for layout
    # nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=1500)
    # edge_labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()})
    # plt.show()


