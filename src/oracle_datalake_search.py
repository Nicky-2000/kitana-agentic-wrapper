from src.filterTables import FilterByEmbeddings
from src.datalake import Datalake
from src.kitana_history.query_history import KitanaHistory
from src.testcase_manager import Testcase

def oracle_datalake_search(kitana_history: KitanaHistory, datalake: Datalake, testcase: Testcase, top_k:int = 5):
    """
    Reads in the tables in the data lake
    For a given query table and query column, return to
    """

    if testcase.name == "test_case_4":
        return [
            "housing_location_data.csv",
            "location_income_ocean.csv",
            "housing_data_latitude.csv",
            "population_by_latitude.csv"]
    elif testcase.name == "test_case_5":
        return [
            "property_location_data.csv",
            "property_details_1.csv",
            "property_sales_data.csv",
            "property_details_2.csv"]
    else:
        return []

