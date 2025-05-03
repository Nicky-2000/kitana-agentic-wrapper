

## We want a seleciton of "value functions" to use in the budget handler
    # These functions should return an float in the range [0, 1] based on what we think the value of the table is
    # 1. Just use cosine similarity between the table and the query
    
    
    # using the history to come up with combined value functions
    # Things we can use:
    # 1. Cosine similarity of the table and the query
    # 2. Cosine similarity of the table and the tables that were good for the query in the history
    # 3. Cosine similarity of the table and the tables that WERE BADDDD for the query in the history
    # 4. We could do some clustering of sorts and select from clusters that were good for the query. 
    
