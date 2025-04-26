from filterTables import FilterByEmbeddings
from utils import read_file_names


def embedding_datalake_search(kitana_results: list[dict], datalake_path: str, query_table:str, query_column:str, top_k:int = 5):
    """
    Reads in the tables in the data lake
    For a given query table and query column, return to
    """
    
    filter_table = FilterByEmbeddings([])
    filter_table.set_tables(read_file_names(datalake_path).copy())

    top_selections = filter_table.filterByQuery(query_column, query_table, top_k)

    return top_selections

if __name__ == "__main__":

    top_selections = embedding_datalake_search(kitana_results=None, datalake_path="data/datalake", query_column="suicides_no", query_table="master.csv")
    print(top_selections)
