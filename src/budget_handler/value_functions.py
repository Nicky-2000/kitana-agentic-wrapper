

## We want a seleciton of "value functions" to use in the budget handler
    # These functions should return an float in the range [0, 1] based on what we think the value of the table is
    # 1. Just use cosine similarity between the table and the query
    
    
    # using the history to come up with combined value functions
    # Things we can use:
    # 1. Cosine similarity of the table and the query
    # 2. Cosine similarity of the table and the tables that were good for the query in the history
    # 3. Cosine similarity of the table and the tables that WERE BADDDD for the query in the history
    # 4. We could do some clustering of sorts and select from clusters that were good for the query. 
    
from pathlib import Path
from src.kitana_history.query_history import KitanaHistory
from src.testcase_manager import Testcase
import numpy as np

from src.vectorDB import VectorDB


def cosine_similarity_value(table_name: str, vec_db: VectorDB, test_case: Testcase) -> float:
    query_name = Path(test_case.buyer_csv_path).stem + ".csv"
    return vec_db.get_cosine_similarity(table_name, query_name)


# def historical_positive_similarity(
#     table: TableCostMetadata,
#     query_embedding: np.ndarray,
#     embedding_db: EmbeddingDB,
#     kitana: KitanaHistory,
#     testcase: Testcase,
# ) -> float:
#     good_tables = kitana.get_good_tables_for_query(testcase.buyer_csv_path)

#     if not good_tables:
#         return 0.0

#     sim_sum = 0.0
#     count = 0

#     for good_table in good_tables:
#         emb = embedding_db.get_embedding(good_table)
#         if emb is not None:
#             sim_sum += np.dot(embedding_db.get_embedding(table.name), emb)
#             count += 1

#     return sim_sum / count if count > 0 else 0.0


# def historical_negative_similarity(
#     table: TableCostMetadata,
#     query_embedding: np.ndarray,
#     embedding_db: EmbeddingDB,
#     kitana: KitanaHistory,
#     testcase: Testcase,
# ) -> float:
#     bad_tables = kitana.get_bad_tables_for_query(testcase.buyer_csv_path)

#     if not bad_tables:
#         return 0.0

#     sim_sum = 0.0
#     count = 0

#     for bad_table in bad_tables:
#         emb = embedding_db.get_embedding(bad_table)
#         if emb is not None:
#             sim_sum += np.dot(embedding_db.get_embedding(table.name), emb)
#             count += 1

#     return sim_sum / count if count > 0 else 0.0


# def combined_signal(
#     table: TableCostMetadata,
#     query_embedding: np.ndarray,
#     embedding_db: EmbeddingDB,
#     kitana: KitanaHistory,
#     testcase: Testcase,
# ) -> float:
#     sim = cosine_similarity_value(table, query_embedding, embedding_db, kitana, testcase)
#     pos = historical_positive_similarity(table, query_embedding, embedding_db, kitana, testcase)
#     neg = historical_negative_similarity(table, query_embedding, embedding_db, kitana, testcase)

#     return max(0.0, sim + 0.5 * pos - 0.5 * neg)  # weighted blend, clipped to [0, ∞)


# Registry of value functions
VALUE_FUNCTIONS = {
    "cosine_similarity": cosine_similarity_value,
    # "historical_positive": historical_positive_similarity,
    # "historical_negative": historical_negative_similarity,
    # "combined_signal": combined_signal,
}