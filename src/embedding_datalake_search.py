from itertools import chain
from src.filterTables import FilterByEmbeddings
from src.datalake import Datalake
from src.kitana_history.query_history import KitanaHistory
from src.testcase_manager import Testcase


def embedding_datalake_search(kitana_history: KitanaHistory, datalake: Datalake, testcase: Testcase, top_k:int = 5, return_vec_db:bool = False):
    """
    Reads in the tables in the data lake
    For a given query table and query column, return to
    """
    
    filter_table = FilterByEmbeddings([])
    datalake_files = datalake.list_files()

    # print(f"Files in datalake: {datalake_files}")
    filter_table.set_tables(datalake_files)
    
    if len(kitana_history.files_cleaned) > 0:
        # Remove files we have already cleaned from the embeddingDB 
        all_cleaned_files = list(chain.from_iterable(kitana_history.files_cleaned))
        filter_table.remove_batch_of_tables(all_cleaned_files)

    top_selections = filter_table.filterByQuery(testcase.target_feature, testcase.buyer_csv_path, top_k)
    
    # Crazy gross hack but we in a time crunch
    if return_vec_db:
        return top_selections, filter_table.db.get_vector_db()
    
    return top_selections

if __name__ == "__main__":

    top_selections = embedding_datalake_search(kitana_results=None, datalake_path="data/datalake", query_column="suicides_no", query_table="master.csv")
    print(top_selections)
