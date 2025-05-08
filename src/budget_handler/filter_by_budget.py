

## We want a few budget functions

# 1. Greedy filter based on VALUE... Keep taking tables until we run out of budget
# 2. Random filter... Randomly select tables until we run out of budget
# 3. Maximize Value per Cost (Tokens). This is similar to a coin change or knapsack problem

from dataclasses import dataclass
from itertools import combinations
import random

@dataclass
class TableCostMetadata:
    name: str
    tokens: int
    

def filter_tables_by_budget(
    tables: list[TableCostMetadata], 
    max_tokens: int, 
    strategy: str = "greedy"
) -> list[str]:

    if strategy == "greedy":
        return greedy_filter(tables, max_tokens)
    elif strategy == "random":
        return random_filter(tables, max_tokens)
    elif strategy == "value_per_token":
        return value_per_token_filter(tables, max_tokens)
    elif strategy == "knapsack_brute_force":
        return brute_force_knapsack(tables, max_tokens)
    elif strategy == "knapsack_dp":
        return knapsack_filter(tables, max_tokens)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def greedy_filter(tableCosts: list[TableCostMetadata], max_tokens: int) -> list[str]:
    selected = []
    total = 0

    for t in tableCosts:
        # Always add at least one table
        if not selected or total + t.tokens <= max_tokens:
            selected.append(t.name)
            total += t.tokens
        else:
            break

    return selected


def random_filter(tableCosts: list[TableCostMetadata], max_tokens: int) -> list[str]:
    shuffled = tableCosts[:]
    random.shuffle(shuffled)

    selected = []
    total = 0

    for t in shuffled:
        # Always add at least one table
        if not selected or total + t.tokens <= max_tokens: 
            selected.append(t.name)
            total += t.tokens
        else:
            break

    return selected


def value_per_token_filter(tableCosts: list[TableCostMetadata], max_tokens: int) -> list[str]:
    sorted_by_ratio = sorted(tableCosts, key=lambda t: t.value / t.tokens, reverse=True)

    selected = []
    total = 0

    for t in sorted_by_ratio:
        if not selected or total + t.tokens <= max_tokens:
            selected.append(t.name)
            total += t.tokens
        else:
            continue

    return selected


# This is O(num_tables * max_tokens). Too tired to figure out when this is better than brute force
# This might be bad if max_tokens is very big (millions)
def knapsack_filter(tableCosts: list[TableCostMetadata], max_tokens: int) -> list[str]:
    n = len(tableCosts)
    W = max_tokens # This is the max token budget

    # Create DP table: rows = items, cols = token budget
    dp = [[0] * (W + 1) for _ in range(n + 1)]

    # Build tableCosts[i] = value of max set using first i tables, budget j
    for i in range(1, n + 1):
        value = tableCosts[i - 1].value
        tokens = tableCosts[i - 1].tokens
        for j in range(W + 1):
            if tokens <= j:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - tokens] + value)
            else:
                dp[i][j] = dp[i - 1][j]

    # Backtrack to find selected items
    selected = []
    j = W
    for i in range(n, 0, -1):
        if dp[i][j] != dp[i - 1][j]:
            selected.append(tableCosts[i - 1].name)
            j -= tableCosts[i - 1].tokens

    selected.reverse()  # Optional: return in input order
    return selected

# This is O(2^n) but might be better if max_tokens is huge and n is small
def brute_force_knapsack(tables: list[TableCostMetadata], max_tokens: int) -> list[str]:
    best_value = 0
    best_subset = []

    for r in range(1, len(tables)+1):
        for subset in combinations(tables, r):
            total_tokens = sum(t.tokens for t in subset)
            if total_tokens > max_tokens:
                continue
            total_value = sum(t.value for t in subset)
            if total_value > best_value:
                best_value = total_value
                best_subset = subset

    return [t.name for t in best_subset]

