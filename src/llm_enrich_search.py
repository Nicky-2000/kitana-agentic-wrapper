from src.budget_handler.filter_by_budget import (
    TableCostMetadata,
    filter_tables_by_budget,
)
from src.budget_handler.token_count import estimate_token_count
from src.embedding_datalake_search import embedding_datalake_search
from src.filterTables import FilterByEmbeddings, FilterTables
from src.kitana_history.query_history import KitanaHistory
from src.utils import read_file_names
from pathlib import Path
from src.enrich_table import estimate_join_columns_prompt, get_table_description_prompt
from src.budget_handler.value_functions import VALUE_FUNCTIONS


def llm_enrich_search_func(
    kitana_results: KitanaHistory,
    datalake,
    test_case,
    top_k: int = 5,
    token_budget: int = 10000,
    budget_filter: str = "greedy",  # Options: "greedy", "random", or "value_per_token", "knapsack_brute_force", "knapsack_dp"
    value_function: str = "cosine_similarity",  # Options: "cosine_similarity", "historical_positive_similarity", "historical_negative_similarity", "combined_signal"
):
    """
    Reads in the tables in the data lake
    For a given query table and query column, return to
    """
    full_path = Path(test_case.buyer_csv_path)

    query_location = str(full_path.parent)
    query_table = full_path.stem + ".csv"  # i am so sorry for this
    query_column = test_case.target_feature

    top_selections, vec_db = embedding_datalake_search(
        kitana_results, datalake, test_case, top_k=20, return_vec_db=True
    )

    if token_budget is not None:
        # Apply the budget filtering to top selections
        # This will filter down the tables in top selections based on the budget
        top_selections = apply_budget_filtering(
            test_case,
            top_selections,
            token_budget,
            budget_filter,
            value_function,
            vec_db,
        )

    llm_filter_table = FilterTables(top_selections)
    top_llm_selections_dict = llm_filter_table.filterByQuery(
        query=query_column,
        target_table=query_table,
        query_location=query_location,
        data_folder="data/datalake",
        n=top_k,
    )

    fin_list = top_llm_selections_dict.keys()
    return [table for table in fin_list if table != query_table]


def apply_budget_filtering(test_case, top_selections, token_budget, budget_filter, value_function, vec_db):
    # value_function signature: (table: TableCostMetadata, query_embedding: np.ndarray, embedding_db: EmbeddingDB,)
    value_function = VALUE_FUNCTIONS.get(value_function, None)

    
    if value_function is None:
        raise ValueError(f"Value function '{value_function}' not found.")

    # We will not filter by budget
    # Get the tokens required for each table
    # want (table, Tokens Required)
    tokens_required_per_table_estimate: list[TableCostMetadata] = []
    
    for table in top_selections:
        # Get the estimated tokens that will be passed to LLM for this table
        # This is a rough estimate, but should be close enough
        total_prompt = estimate_join_columns_prompt(
            table, data_folder="data/datalake"
        ) + get_table_description_prompt(table, data_folder="data/datalake")
        num_tokens = estimate_token_count(total_prompt)
        
        # Calculate a "value" for the table based on the value function
        value = value_function(table=table, vec_db=vec_db, test_case=test_case)
        
        table_cost = TableCostMetadata(
            name=table,
            tokens=num_tokens,
            value=value,
        )
        tokens_required_per_table_estimate.append(table_cost)

    tables_filtered_by_budget = filter_tables_by_budget(
        tables=tokens_required_per_table_estimate,
        token_budget=token_budget,
        budget_filter=budget_filter,  # Can be "greedy", "random", or "value_per_token"
    )

    return tables_filtered_by_budget
