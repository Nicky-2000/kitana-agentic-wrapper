from src.embedding_db import EmbeddingDB
from src.enrich_table import enrich_and_filter_table_list


class FilterTables:
    def __init__(self, tables:list[str]):
        self.tables = tables
    
    def set_tables(self, tables):
        self.tables = tables
    
    def remove_table(self, table):
        self.tables.remove(table)
    
    def add_table(self, table):
        self.tables.append(table)

    #@abstractmethod
    def filterByQuery(self, query, target_table, data_folder, n:int) -> list[str]:
        if target_table not in self.tables:
            self.add_table(target_table)

        fin_dict = enrich_and_filter_table_list(table_list=self.tables, query_table=target_table, query_column=query, data_folder = data_folder)

        return fin_dict


class FilterByEmbeddings(FilterTables):
    def __init__(self, tables:list[str]):
        super().__init__(tables)
        self.set_tables(tables)
    
    def set_tables(self, tables:list[str]):
        super().set_tables(tables)
        if len(tables) == 0:
            return
        self.db: EmbeddingDB = EmbeddingDB(tables)
    
    def remove_table(self, table):
        super().remove_table(table)
        self.db.remove_from_db(table)
    
    def remove_batch_of_tables(self, tables:list[str]):
        for table in tables:
            self.remove_table(table)

    def add_table(self, table):
        super().add_table(table)
        self.db.add_to_db(table)

    def filterByQuery(self, query:str, target_table:str, n:int) -> list[str]:
        results = self.db.get_vector_db().get_n_closest(query + target_table, n)
        
        return results