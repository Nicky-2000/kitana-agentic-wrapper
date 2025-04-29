import os
import time
from src.typed_dicts import LanguageModelConfig, VectorDBConfig
from src.vectorDB import VectorDB
from src.utils import get_pdataframe_from_csv
import hashlib
from tqdm import tqdm


class EmbeddingDB:
    def __init__(self, tables: list[str]):
        self.tables = tables

        # db_name:str = (os.getenv("API_TYPE") if os.getenv("API_TYPE")!=None else "undefined")  + "_similarity"#type:ignore
        # table_name:str = "_".join(tables)
        # combined_name:str = db_name + table_name
        # self.name_hash = hashlib.sha1(combined_name.encode('utf-8')).hexdigest() #type:ignore
        self.name_hash = "constant_name"  # type:ignore

        config = LanguageModelConfig(
            api_type=os.getenv("API_TYPE"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region=os.getenv("AWS_REGION"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            local_model_name=os.getenv("LOCAL_MODEL_NAME"),
            embedding_api_type=None
        )

        vdb_config = VectorDBConfig(
            task_type="SEMANTIC_SIMILARITY"
        )

        self.db = VectorDB(self.name_hash, vdb_config, config)

        existing_ids = self.db.collection.get().get('ids', [])

        # for table in tqdm(tables, desc="Embedding tables"):
        #     pd = get_pdataframe_from_csv(table)
        #     if pd is not None:
        #         column_names = ", ".join(pd.columns)
        #         document = table + " " + column_names
        #         if table not in existing_ids:
        #             time.sleep(0.27)  # Rate limit for embedding API
        #             self.db.add_document(document, id=table)


    def get_vector_db(self) -> VectorDB:
        return self.db
    
    def remove_from_db(self, table_name):
        """
        Remove a table from the vector database.
        :param table_name: Name of the table to remove.
        """
        self.db.remove_document(table_name)

    def add_to_db(self, table_name):
        """
        Add a table to the vector database.
        :param table_name: Name of the table to add.
        """
        self.db.add_document(table_name, id=table_name)