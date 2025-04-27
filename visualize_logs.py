import json
import matplotlib.pyplot as plt
from src.kitana_history.query_history import KitanaHistory, KitanaResults
# from termcolor import colored  # If you can't install termcolor, we'll avoid it

# -------------------- Load the history --------------------
def load_history(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    
    history = KitanaHistory()
    for result_data in data["kitana_results"]:
        history.kitana_results.append(KitanaResults.from_dict(result_data))
    history.files_cleaned = data["files_cleaned"]
    return history

# -------------------- Plot final accuracy --------------------
def plot_kitana_history(history: KitanaHistory, title="Kitana Improvement"):
    final_accuracies = [result.accuracy[-1] for result in history.kitana_results]
    steps = list(range(len(final_accuracies)))

    plt.figure(figsize=(8,5))
    plt.plot(steps, final_accuracies, marker='o')
    plt.title(title)
    plt.xlabel("Kitana Experiment Step")
    plt.ylabel("Final Accuracy")
    plt.grid(True)

    for i, files in enumerate(history.files_cleaned):
        if files:
            plt.annotate(f"+{len(files)} files", (i, final_accuracies[i]),
                         textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.show()

# -------------------- Check cleaned files usage --------------------
def filename_to_table_name(filename: str) -> str:
    return filename.replace(".csv", "").strip()

def check_cleaned_files_used(history: KitanaHistory):
    cleaned_file_matches = []

    for i, kitana_result in enumerate(history.kitana_results):
        used_table_names = set(aug.table_name for aug in kitana_result.augplan)

        if i == 0:
            cleaned_file_matches.append(False)
            continue

        cleaned_files_before_this_run = history.files_cleaned[i-1]
        cleaned_tables = set(filename_to_table_name(f) for f in cleaned_files_before_this_run)

        match_found = any(table in used_table_names for table in cleaned_tables)
        cleaned_file_matches.append(match_found)

    return cleaned_file_matches

# -------------------- Plot cleaned file usage --------------------
def plot_cleaned_file_usage(history: KitanaHistory, title="Usage of Cleaned Files"):
    matches = check_cleaned_files_used(history)
    steps = list(range(len(matches)))

    plt.figure(figsize=(8,4))
    plt.plot(steps, matches, marker='o', linestyle="--", label="Cleaned File Used?")
    plt.ylim(-0.1, 1.1)
    plt.yticks([0, 1], labels=["No Match", "Matched"])
    plt.xticks(steps)
    plt.xlabel("Kitana Experiment Step")
    plt.ylabel("Cleaned Files Matched")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# -------------------- Print augplan with highlights --------------------
def print_augplan_with_highlights(history: KitanaHistory):
    all_cleaned_files = [file for batch in history.files_cleaned for file in batch]
    cleaned_tables = set(filename_to_table_name(f) for f in all_cleaned_files)

    for run_idx, kitana_result in enumerate(history.kitana_results):
        print(f"\n--- Kitana Run {run_idx} ---")
        print(f"Final Accuracy: {kitana_result.accuracy[-1]:.4f}")
        for aug in kitana_result.augplan:
            table_name = aug.table_name
            col_name = aug.column_name

            if table_name in cleaned_tables:
                # No colored output if you can't install termcolor, just mark it
                display_text = f"{table_name}.{col_name}  [CLEANED]"
            else:
                display_text = f"{table_name}.{col_name}"

            print(f"  {display_text}")

# -------------------- Example Usage --------------------
if __name__ == "__main__":
    history_path = "kitana_logs/test_case_3_history.json"  # <-- change path if needed
    history = load_history(history_path)
    
    plot_kitana_history(history, title="Test Case 3: Accuracy Improvement")
    plot_cleaned_file_usage(history, title="Did Cleaned Files Help Kitana?")
    print_augplan_with_highlights(history)
