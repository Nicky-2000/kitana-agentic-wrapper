test_cases = [
    {
        "test_name": "test_case_1",
        "table": "master.csv",
        "target_col": "suicides_no",
        "join_keys": [["Country"], ["year"]],
    },
    {
        "test_name": "test_case_2",
        "table": "raw_data.csv",
        "target_col": "human_development_index",
        "join_keys": [["Country"]],
    },
    {
        "test_name": "test_case_3",
        "table": "Cost_of_Living_Index_by_Country_2024.csv",
        "target_col": "Groceries Index",
        "join_keys": [["Country"]],
    },
    {
        "test_name": "test_case_4",
        "table": "housing_geo_data.csv",
        "target_col": "median_house_value",
        "join_keys": [["latitude"], ["longitude"]],
    },
    {
        "test_name": "test_case_5",
        "table": "house_details.csv",
        "target_col": "Price",
        "join_keys": [["Address"]],
    },

    #this one doesnt work for some reason even tho it has good test case ;(
        {
        "test_name": "test_case_6",
        "table": "crime-rate-by-country-2023-base.csv",
        "target_col": 'crimeIndex',
        "join_keys": [['country']],
    },
    {
        "test_name": "test_case_7",
        "table": "quality_of_life.csv",
        "target_col": "stability",
        "join_keys": [["country"]],
    },
    {
        "test_name": "test_case_8",
        "table": "quality_of_life.csv",
        "target_col": "rights",
        "join_keys": [["country"]],
    },
        {
        "test_name": "test_case_9",
        "table": "quality_of_life.csv",
        "target_col": "health",
        "join_keys": [["country"]],
    },
    {
        "test_name": "test_case_10",
        "table": "quality_of_life.csv",
        "target_col": "safety",
        "join_keys": [["country"]],
    },
    {
        "test_name": "test_case_11",
        "table": "quality_of_life.csv",
        "target_col": "climate",
        "join_keys": [["country"]],
    },

    {
        "test_name": "test_case_12",
        "table": "quality_of_life.csv",
        "target_col": "costs",
        "join_keys": [["country"]],
    },
    {
        "test_name": "test_case_13",
        "table": "quality_of_life.csv",
        "target_col": "popularity",
        "join_keys": [["country"]],
    },
    {
        "test_name": "test_case_14",
        "table": "corruption.csv",
        "target_col": "corruption_index",
        "join_keys": [["country"]],
    },
    {
        "test_name": "test_case_15",
        "table": "corruption.csv",
        "target_col": "annual_income",
        "join_keys": [["country"]],
    },
    {
        "test_name": "test_case_16",
        "table": "Petrol Dataset June 20 2022.csv",
        "target_col": "Daily Oil Consumption (Barrels)",
        "join_keys": [["country"]],
    },
    {
        "test_name": "test_case_17",
        "table": "most-polluted-countries.csv",
        "target_col": "pollution_2023",
        "join_keys": [["country"]],
    },
    ]