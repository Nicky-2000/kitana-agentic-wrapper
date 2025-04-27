from src.embedding_datalake_search import embedding_datalake_search
from src.filterTables import FilterByEmbeddings, FilterTables
from src.kitana_history.query_history import KitanaHistory
from src.llm_table_rank import rank_all_tables
from src.utils import read_file_names
from pathlib import Path



def llm_selector_search_func(kitana_results:KitanaHistory, datalake, test_case, top_k:int = 5):
    """
    Reads in the tables in the data lake
    For a given query table and query column, return to
    """
    full_path = Path(test_case.buyer_csv_path)

    query_location = str(full_path.parent)  
    query_table = full_path.stem + ".csv" #i am so sorry for this
    query_column = test_case.target_feature

    top_selections = embedding_datalake_search(kitana_results, datalake, test_case, top_k = 20)

    llm_selections = rank_all_tables(query_table, query_column, 
                    kitana_results.kitana_results[0], #assumes 1 history exists?
                    num_tables = top_k, 
                    window_size = 10, 
                    datalake_folder = "data/datalake", 
                    buyer_folder = query_location, 
                    seller_folder = test_case.seller_original_folder_path, 
                    filtered_tables = top_selections
                    )
    
    return llm_selections

    # fin_list = top_llm_selections_dict.keys()
    # return [table for table in fin_list if table != query_table]
