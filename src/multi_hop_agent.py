import os
from typing import Dict, Generator, ItemsView, List, Optional, Set, Tuple
from src.datalake.Datalake import DataLake
from src.embedding_datalake_search import embedding_datalake_search
from src.embedding_db import EmbeddingDB
from src.filterTables import FilterByEmbeddings, FilterTables
from src.kitana_history.query_history import KitanaHistory
from src.language_model_interface import LanguageModelInterface
from src.testcase_manager.Testcase import TestCase
from src.utils import read_file_names
from pathlib import Path
from itertools import chain
from src.utils import gemini_json_cleaner, check_if_valid_json
import pandas as pd

filter_tables = None

class Config:
    def __init__(self):
        self.api_type = os.getenv("API_TYPE", "openai")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.embedding_api_type = os.getenv("EMBEDDING_API_TYPE", "openai")
        self.local_model_name = os.getenv("LOCAL_MODEL_NAME", "gemma3:12b")

config = Config()
lm = LanguageModelInterface(config)


class Node:
    """
    Represents a node in the graph, identified by a table name and
    containing a set of column names associated with that table.
    Each node maintains a dictionary of its children, mapping child nodes
    to the column used for the join.
    """
    def __init__(self, table_name: str, columns: Optional[Set[str]] = None):
        """
        Initializes a Node instance.

        Args:
            table_name (str): The name of the table this node represents.
                              Must be a non-empty string.
            columns (Optional[Set[str]]): A set of column names belonging to this table.
                                          Defaults to an empty set if None.

        Raises:
            ValueError: If table_name is not a non-empty string.
        """
        if not isinstance(table_name, str) or not table_name:
            raise ValueError("table_name must be a non-empty string")
        self.table_name: str = table_name
        # Stores column names belonging to this table node
        self.columns: Set[str] = columns if columns is not None else set()
        # Stores nodes that this node has a directed edge towards (children).
        # Key: Child Node object, Value: join_column (str) used for the edge
        # Assumption: The join_column name exists in BOTH parent and child DataFrames for the join.
        self.children: Dict['Node', str] = {} # Use forward reference for Node type hint

    def add_child(self, child_node: 'Node', join_column: str):
        """
        Adds a child node and the joining column to this node's children dict,
        representing a directed edge from self to child_node using join_column.
        If the child already exists, it updates the join_column.

        Args:
            child_node (Node): The node to add as a child.
            join_column (str): The name of the column used to join this node to the child node.
                               It's assumed this column name exists in both tables involved in the join.
        """
        if not isinstance(join_column, str) or not join_column:
             raise ValueError("join_column must be a non-empty string")
        # Add or update the child and its associated join column
        self.children[child_node] = join_column

    def get_children_nodes(self) -> List['Node']:
        """Returns a list of the actual child Node objects."""
        return list(self.children.keys())

    def get_children_with_joins(self) -> ItemsView['Node', str]:
        """Returns an items view of the children dictionary (Node, join_column)."""
        return self.children.items()

    def __repr__(self) -> str:
        """
        Provides a developer-friendly string representation of the Node,
        including its table name, columns, and children with join columns.
        """
        # Sort children by table name for consistent output
        child_reprs = sorted([f"{child.table_name}(on:{join_col})"
                              for child, join_col in self.children.items()])
        # Sort columns for consistent output
        col_reprs = sorted(list(self.columns))
        return (f"Node(table_name='{self.table_name}', "
                f"columns={{{', '.join(col_reprs)}}}, "
                f"children=[{', '.join(child_reprs)}])")

    def __eq__(self, other) -> bool:
        """
        Checks if two Node objects are equal based on their table_name.
        """
        if not isinstance(other, Node):
            return NotImplemented
        return self.table_name == other.table_name

    def __hash__(self) -> int:
        """
        Computes a hash based on the table_name. Allows Node objects in sets/dict keys.
        """
        return hash(self.table_name)

class Graph:
    """
    Represents a directed graph using an adjacency list approach.
    Nodes store table names and columns. Edges store the join column.
    Nodes are stored in a dictionary mapping table names to Node objects.
    Edges are represented by the `children` dictionary within each `Node`.
    Includes functionality to join associated pandas DataFrames based on graph structure.
    """
    def __init__(self):
        """Initializes an empty Graph."""
        self.nodes: Dict[str, Node] = {} # Key: table_name (str), Value: Node object

    def add_node(self, table_name: str, columns: Optional[Set[str]] = None) -> Node:
        """
        Adds a new node to the graph or updates columns if it already exists.

        Args:
            table_name (str): The unique identifier (table name) for the node.
            columns (Optional[Set[str]]): A set of column names for the table.
                                          If the node exists, these columns are added
                                          to the existing set.

        Returns:
            Node: The Node object corresponding to the table_name.

        Raises:
            ValueError: If table_name is not a non-empty string.
        """
        if not isinstance(table_name, str) or not table_name:
            raise ValueError("table_name must be a non-empty string")

        if table_name not in self.nodes:
            # Create a new Node if it doesn't exist
            new_node = Node(table_name, columns)
            self.nodes[table_name] = new_node
            # Automatically add columns from node name if columns aren't provided explicitly
            # This helps ensure join columns are present in the Node's column set
            # even if not passed during add_node/add_edge creation.
            # You might adjust this logic based on how columns are managed.
            # if columns is None:
            #    # Example: Infer potential keys - adjust as needed
            #    if 'id' in table_name.lower(): new_node.columns.add(table_name.lower())
            #    if 'fk' in table_name.lower(): new_node.columns.add(table_name.lower())
            return new_node
        else:
            # Node exists, update its columns if new ones are provided
            existing_node = self.nodes[table_name]
            if columns:
                existing_node.columns.update(columns) # Add new columns
            return existing_node

    def get_node(self, table_name: str) -> Optional[Node]:
        """
        Retrieves a node from the graph by its table name.

        Args:
            table_name (str): The name of the table (node identifier).

        Returns:
            Optional[Node]: The Node object if found, otherwise None.
        """
        return self.nodes.get(table_name)

    def add_edge(self, parent_table_name: str, child_table_name: str, join_column: str):
        """
        Adds a directed edge from a parent node to a child node with the joining column.
        If either node does not exist, it will be created (with empty columns initially).
        The join column is added to the parent node's column set if not already present.

        Args:
            parent_table_name (str): The table name of the node where the edge originates.
            child_table_name (str): The table name of the node where the edge terminates.
            join_column (str): The name of the column used for the join. Assumed to exist
                               in both parent and child DataFrames.

        Raises:
            ValueError: If table names or join_column are invalid.
        """
        if not isinstance(join_column, str) or not join_column:
             raise ValueError("join_column must be a non-empty string")

        parent_node = self.add_node(parent_table_name, set())
        child_node = self.add_node(child_table_name, set())

        # Ensure the join column is listed in the parent node's columns
        parent_node.columns.add(join_column)
        # Optionally, add to child node columns as well if it represents the PK/target
        # child_node.columns.add(join_column) # Uncomment if needed

        # Add the child node and join column to the parent node's children dict
        parent_node.add_child(child_node, join_column)

    def get_neighbors(self, table_name: str) -> List[Node]:
        """
        Gets the direct neighbors (children nodes) of a node specified by its table name.

        Args:
            table_name (str): The table name of the node whose neighbors are requested.

        Returns:
            List[Node]: A list of neighboring Node objects (children). Returns an empty list
                        if the node doesn't exist or has no children.
        """
        node = self.get_node(table_name)
        if node:
            return node.get_children_nodes()
        else:
            return []

    def get_neighbors_with_joins(self, table_name: str) -> List[Tuple[Node, str]]:
        """
        Gets the direct neighbors (children) and the corresponding join columns.

        Args:
            table_name (str): The table name of the node.

        Returns:
            List[Tuple[Node, str]]: A list of tuples, where each tuple contains
                                     a child Node and the join column string.
                                     Returns an empty list if the node doesn't exist.
        """
        node = self.get_node(table_name)
        if node:
            return list(node.get_children_with_joins())
        else:
            return []

    def __repr__(self) -> str:
        """
        Provides a developer-friendly string representation of the entire Graph.
        """
        if not self.nodes:
            return "Graph(empty)"
        repr_str = "Graph:\n"
        for table_name in sorted(self.nodes.keys()):
            node = self.nodes[table_name]
            repr_str += f"  - {node}\n"
        return repr_str.strip()

    def bfs(self, start_table_name: str) -> Generator[Node, None, None]:
        """
        Performs a Breadth-First Search (BFS) traversal starting from a given node.
        Yields nodes level by level. (Traversal logic only considers node reachability).

        Args:
            start_table_name (str): The table name of the node to start the BFS from.

        Yields:
            Node: Nodes are yielded one by one in BFS order.

        Raises:
            ValueError: If the start node doesn't exist.
        """
        start_node = self.get_node(start_table_name)
        if not start_node:
            raise ValueError(f"Start node '{start_table_name}' not found in graph.")

        visited: Set[Node] = set()
        queue: List[Node] = [start_node]
        visited.add(start_node)

        while queue:
            current_node = queue.pop(0)
            yield current_node

            for child_node in current_node.get_children_nodes():
                if child_node not in visited:
                    visited.add(child_node)
                    queue.append(child_node)

    def dfs(self, start_table_name: str) -> Generator[Node, None, None]:
        """
        Performs a Depth-First Search (DFS) traversal (iterative) starting from a given node.
        Yields nodes in DFS order (pre-order). (Traversal logic only considers node reachability).

        Args:
            start_table_name (str): The table name of the node to start the DFS from.

        Yields:
            Node: Nodes are yielded one by one in DFS order.

        Raises:
            ValueError: If the start node doesn't exist.
        """
        start_node = self.get_node(start_table_name)
        if not start_node:
            raise ValueError(f"Start node '{start_table_name}' not found in graph.")

        visited: Set[Node] = set()
        stack: List[Node] = [start_node]

        while stack:
            current_node = stack.pop()

            if current_node not in visited:
                visited.add(current_node)
                yield current_node

                for child_node in reversed(current_node.get_children_nodes()):
                     if child_node not in visited:
                        stack.append(child_node)

    def has_cycle(self) -> bool:
        """
        Checks if the directed graph contains at least one cycle using DFS.
        (Cycle detection logic only considers node reachability).

        Returns:
            bool: True if a cycle is detected, False otherwise.
        """
        visiting: Set[Node] = set()
        visited: Set[Node] = set()

        def _dfs_cycle_check(node: Node) -> bool:
            visiting.add(node)
            visited.add(node)

            for neighbor_node in node.get_children_nodes():
                if neighbor_node not in visited:
                    if _dfs_cycle_check(neighbor_node):
                        return True
                elif neighbor_node in visiting: # Back edge found
                    return True

            visiting.remove(node) # Backtrack
            return False

        for node in self.nodes.values():
            if node not in visited:
                if _dfs_cycle_check(node):
                    return True
        return False

    def get_all_nodes(self) -> List[Node]:
        """Returns a list containing all Node objects currently in the graph."""
        return list(self.nodes.values())

    def get_all_nodes_str(self) -> List[str]:
        """Returns a list containing all Node objects currently in the graph."""
        return [node.table_name for node in self.nodes.values()]

    def get_all_edges(self) -> List[Tuple[Node, Node, str]]:
        """
        Returns a list of all directed edges in the graph.
        Each edge is represented as a tuple (parent_node, child_node, join_column).
        """
        edges = []
        for parent_node in self.nodes.values():
            for child_node, join_column in parent_node.get_children_with_joins():
                edges.append((parent_node, child_node, join_column))
        return edges

    def find_bottom_most_node_by_column(self, column_name: str) -> Optional[Node]:
        """
        Finds a node that contains the specified column but none of its children do.
        This identifies a node where the column "stops" propagating downwards in a branch.
        Note: In graphs with cycles or multiple paths, this might return the first
              such node encountered during iteration.

        Args:
            column_name (str): The name of the column to search for.

        Returns:
            Optional[Node]: The first "bottom-most" node found for that column in any branch,
                            or None if no such node exists.

        Raises:
            ValueError: If column_name is not a non-empty string.
        """
        if not isinstance(column_name, str) or not column_name:
            raise ValueError("column_name must be a non-empty string")

        for node in self.nodes.values():
            if column_name in node.columns:
                is_bottom_most = True
                for child_node in node.get_children_nodes():
                    if column_name in child_node.columns:
                        is_bottom_most = False
                        break
                if is_bottom_most:
                    return node
        return None

    def get_all_unique_column_names(self) -> List[str]:
        """
        Collects all unique column names from all nodes in the graph.

        Returns:
            List[str]: A sorted list of unique column names found across all nodes.
        """
        all_columns: Set[str] = set()
        for node in self.nodes.values():
            all_columns.update(node.columns)
        return sorted(list(all_columns))

    def join_tables_to_dataframe(self, root_table_name: str, dataframes: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Joins pandas DataFrames together based on the graph structure, starting from a root table.
        Performs a traversal (BFS-like) and uses left joins. Handles column name conflicts with suffixes.

        Args:
            root_table_name (str): The name of the table to start the joins from.
            dataframes (Dict[str, pd.DataFrame]): A dictionary mapping table names (node names)
                                                  to their corresponding pandas DataFrame objects.

        Returns:
            pd.DataFrame: A single DataFrame containing the result of all left joins
                          performed according to the graph structure starting from the root.

        Raises:
            ValueError: If the root table name is not found in the graph or in the dataframes dict.
            KeyError: If a table needed for joining is not found in the dataframes dict.
            ValueError: If a join column specified in the graph is missing from a DataFrame.
        """
        root_node = self.get_node(root_table_name)
        if not root_node:
            raise ValueError(f"Root node '{root_table_name}' not found in the graph.")

        if root_table_name not in dataframes:
            raise ValueError(f"DataFrame for root table '{root_table_name}' not found in the dataframes dictionary.")

        # Start with the root DataFrame, make a copy to avoid modifying original
        merged_df = dataframes[root_table_name].copy()

        # Queue for BFS traversal: stores tuples of (parent_node, child_node, join_column)
        queue: List[Tuple[Node, Node, str]] = []
        # Set to keep track of tables already joined to prevent redundant joins/cycles
        joined_tables: Set[str] = {root_table_name}

        # Initial population of the queue with the root's direct children
        for child_node, join_column in root_node.get_children_with_joins():
            if child_node.table_name not in joined_tables:
                 queue.append((root_node, child_node, join_column))
                 # Mark as potentially joinable (will be confirmed when dequeued)
                 # joined_tables.add(child_node.table_name) # Add when actually joined

        processed_edges: Set[Tuple[str, str]] = set() # Track processed parent-child pairs

        while queue:
            parent_node, child_node, join_column = queue.pop(0)

            parent_name = parent_node.table_name
            child_name = child_node.table_name

            # Avoid re-processing the same edge relationship
            if (parent_name, child_name) in processed_edges:
                continue
            processed_edges.add((parent_name, child_name))

            # Skip if child already joined (can happen in complex graphs)
            # Check joined_tables *before* attempting merge
            if child_name in joined_tables:
                 # Still need to explore children of this already-joined node
                for grandchild_node, grandchild_join_col in child_node.get_children_with_joins():
                    if grandchild_node.table_name not in joined_tables and \
                       (child_name, grandchild_node.table_name) not in processed_edges:
                         queue.append((child_node, grandchild_node, grandchild_join_col))
                continue # Skip the merge itself


            # Check if the child DataFrame exists
            if child_name not in dataframes:
                raise KeyError(f"DataFrame for table '{child_name}' not found in the dataframes dictionary.")
            child_df = dataframes[child_name]

            # --- Perform the Join ---
            # Check if join column exists in the current merged DataFrame
            if join_column not in merged_df.columns:
                 # Try adding suffix if collision happened earlier
                 suffixed_join_col_left = f"{join_column}_left"
                 suffixed_join_col_right = f"{join_column}_right"
                 if suffixed_join_col_left in merged_df.columns:
                     current_join_col = suffixed_join_col_left
                 elif suffixed_join_col_right in merged_df.columns:
                     current_join_col = suffixed_join_col_right
                 else:
                    raise ValueError(f"Join column '{join_column}' not found in the current merged DataFrame (from table '{parent_name}'). Columns: {merged_df.columns}")
            else:
                current_join_col = join_column # Join column exists directly

            # Check if join column exists in the child DataFrame
            if join_column not in child_df.columns:
                raise ValueError(f"Join column '{join_column}' not found in the child DataFrame '{child_name}'. Columns: {child_df.columns}")

            print(f"Joining: {parent_name} (on {current_join_col}) <-- {child_name} (on {join_column})") # Debug print

            # Perform the left merge
            merged_df = pd.merge(
                merged_df,
                child_df,
                on=join_column, # Assumes join column name is the same in both
                how='left',
            )
            joined_tables.add(child_name) # Mark child table as joined

            # Add children of the *current child* to the queue if not already joined/queued
            for grandchild_node, grandchild_join_col in child_node.get_children_with_joins():
                 grandchild_name = grandchild_node.table_name
                 # Add to queue only if not already joined and edge not processed
                 if grandchild_name not in joined_tables and \
                    (child_name, grandchild_name) not in processed_edges:
                     queue.append((child_node, grandchild_node, grandchild_join_col))


        return merged_df

def merge(df1: pd.DataFrame, df2: pd.DataFrame, join_keys: List[str]) -> pd.DataFrame:
    """
    Merges two DataFrames on the specified join keys.
    """
    return pd.merge(df1, df2, on=join_keys, how='inner')

def save_to_csv(df: pd.DataFrame) -> None:
    """
    Saves the DataFrame to a CSV file.
    """
    df.to_csv('temp/candidate.csv', index=False)

global_db = None
def embedding_search(datalake: DataLake, kitana_history: KitanaHistory, testcase: TestCase, query:str, exclude_list:List[str] , top_k:int = 5) -> Optional[List[str]]:
    global global_db

    datalake_files = datalake.list_files()
    if global_db is None:
        global_db = EmbeddingDB(datalake_files)
        
    if global_db is not None:
    
        result = global_db.get_vector_db().get_n_closest(query, 50)
        # Filter out files in the exclude_list from the result
        result = [file for file in result if file not in exclude_list]
        result = result[:top_k]

        return result
    return None

def check_join(core_df: pd.DataFrame, join_df: pd.DataFrame, join_keys: List[str]) -> tuple:
    """
    Checks if the join keys exist in both DataFrames and counts matching rows.
    
    Parameters:
    -----------
    core_df : pd.DataFrame
        The primary DataFrame
    join_df : pd.DataFrame
        The DataFrame to join with
    join_keys : List[str]
        List of column names to use as join keys
    
    Returns:
    --------
    tuple
        (bool, int) - First element indicates if all keys exist in both DataFrames,
        second element is the count of rows in core_df that have matches in join_df
    """
    # Check if keys exist in both DataFrames
    keys_exist = True
    for key in join_keys:
        if key not in core_df.columns or key not in join_df.columns:
            keys_exist = False
            break
    
    # Count matching rows only if all keys exist
    matching_rows = 0
    if keys_exist:
        # Create a merged DataFrame with indicator=True to show which rows matched
        merged = pd.merge(core_df, join_df, on=join_keys, how='left', indicator=True)
        
        # Count rows with matches (where _merge column is 'both')
        matching_rows = (merged['_merge'] == 'both').sum()
    
    return keys_exist, matching_rows

# modify to incorporate data intelligence as well
def propose_join(columns:List[str], table: pd.DataFrame) -> str:
    table_columns = list(table.columns)
    # this is a dumbed out one where we expect the exact match to be present. This is not the case in real world
    prompt = f"""
    I am a data engineer and I need to join two tables.
    the main table has the columns: {columns} and the table I want to join has the column {table_columns}, please suggest a join key.
    Return the join key which is present in both tables.
    Your solution should be a single column name.
    If no join key is found, return "None" as join_key in the JSON
    return a json with the following format:
    {{
        "reason": "<reasoning>"
        "join_key": "<join_key>",
    }}
    return only the json, nothing else.
    """
    response = lm.get_response(prompt, 0, True)
    cleaned_response = gemini_json_cleaner(response)
    is_valid, json_obj = check_if_valid_json(cleaned_response)
    if is_valid and json_obj is not None and "join_key" in json_obj:
        return json_obj["join_key"]
    else:
        print("Invalid JSON response from the model. Or no Join Key found.")
        print(response)
        return ""

# code smell
graph = Graph()

def create_dataframes(datalake:DataLake, root_table_name, root_pd: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    nodes = graph.get_all_nodes()
    dataframes = {}
    dataframes[root_table_name] = root_pd
    for node in nodes:
        if not node.table_name == root_table_name:
            dataframes[node.table_name] = pd.read_csv(f"{datalake.datalake_folder}/{node.table_name}")
    return dataframes


def multiHopAgent(kitana_results:KitanaHistory, datalake:DataLake, test_case, top_k:int = 5, iteration:int = 0) -> bool:
    """
    Huge assumption: starting table has only one column
    """
    full_path = Path(test_case.buyer_csv_path)

    query_location = str(full_path.parent)  
    query_table = full_path.stem + ".csv" #i am so sorry for this
    query_column = test_case.target_feature
    query_pd = pd.read_csv(test_case.buyer_csv_path)
    query_pd = query_pd.drop(columns=[query_column], axis=1)
    root_column = query_pd.columns[0] 

    if kitana_results is not None:
        # I will get to work with the results as well
        # for now lets focus on BFS
        pass
    
    if graph.get_node(root_column) is None:
        # Initialize the graph with the query table
        graph.add_node(full_path.stem, {root_column})
    
    exlcude_list = graph.get_all_nodes_str()

    query = " ".join(graph.get_all_unique_column_names())

    top_selections = embedding_search(datalake, kitana_results, test_case, query, exlcude_list, top_k = 5)

    candidate_table = graph.join_tables_to_dataframe(full_path.stem, create_dataframes(datalake, full_path.stem, query_pd))

    if top_selections is None:
        return False
    
    for table in top_selections:
        proposal_pd = pd.read_csv(f"{datalake.datalake_folder}/{table}")
        proposed_column = propose_join(graph.get_all_unique_column_names(), proposal_pd)
        if len(proposed_column) > 0:
            table_node = graph.find_bottom_most_node_by_column(proposed_column)
            if table_node is None:
                print(f"No node found for column: {proposed_column}")
                continue
            # Now proceed knowing table_node is not None
            check_join_result = check_join(candidate_table, proposal_pd, [proposed_column])
            if check_join_result[0] and check_join_result[1] > 0:
                graph.add_node(table, set(list(proposal_pd.columns)))
                # add the edge to the graph
                graph.add_edge(table_node.table_name, table, proposed_column)
                candidate_table = merge(candidate_table, proposal_pd, [proposed_column])
            else:
                print(f"Join keys do not exist in both tables or no matching rows found for {table} and {query_table}")

    
    save_to_csv(candidate_table)
    return True

