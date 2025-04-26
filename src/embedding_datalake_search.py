from src.filterTables import FilterByEmbeddings
from src.datalake_src import Datalake
from src.testcase_manager import Testcase
from src.utils import read_file_names


def embedding_datalake_search(kitana_results: list[dict], datalake: Datalake, testcase: Testcase, top_k:int = 5):
    """
    Reads in the tables in the data lake
    For a given query table and query column, return to
    """
    
    filter_table = FilterByEmbeddings([])
    datalake_files = datalake.list_files()
    print(f"Files in datalake: {datalake_files}")
    filter_table.set_tables(read_file_names(datalake_path).copy())

    top_selections = filter_table.filterByQuery(query_column, query_table, top_k)

    return top_selections

if __name__ == "__main__":

    top_selections = embedding_datalake_search(kitana_results=None, datalake_path="data/datalake", query_column="suicides_no", query_table="master.csv")
    print(top_selections)
