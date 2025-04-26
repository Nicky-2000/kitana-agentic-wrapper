from src.embedding_datalake_search import embedding_datalake_search
from src.filterTables import FilterByEmbeddings, FilterTables
from src.kitana_history.query_history import KitanaHistory
from src.utils import read_file_names
from pathlib import Path



def llm_enrich_search_func(kitana_results:KitanaHistory, datalake, test_case, top_k:int = 5):
    """
    Reads in the tables in the data lake
    For a given query table and query column, return to
    """
    full_path = Path(test_case.buyer_csv_path)

    query_location = str(full_path.parent)  
    query_table = full_path.stem + ".csv" #i am so sorry for this
    query_column = test_case.target_feature

    top_selections = embedding_datalake_search(kitana_results, datalake, test_case, top_k = 20)

    llm_filter_table = FilterTables(top_selections)
    top_llm_selections_dict = llm_filter_table.filterByQuery(query=query_column, target_table=query_table, query_location=query_location, data_folder="data/datalake", n = top_k)

    fin_list = top_llm_selections_dict.keys()
    return [table for table in fin_list if table != query_table]
