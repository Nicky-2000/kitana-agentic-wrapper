import sys
from pathlib import Path

# This is needed if you don't have the vscode settings "python.autoComplete.extraPaths" set to "kitana-e2e"
sys.path.insert(0, str(Path(__file__).resolve().parent / "kitana-e2e"))

import pandas as pd
import numpy as np
from search_engine.experiment import ScaledExperiment
from search_engine.config import (
    Config,
    DataConfig,
    SearchConfig,
    ExperimentConfig,
    LoggingConfig,
)


def plot_two_experiments_results_max(
    origin_result: dict, output_fig="experiment_comparison.png"
):
    acc1 = origin_result["accuracy"]
    max_len = len(acc1)
    acc1_padded = list(acc1) + [np.nan] * (max_len - len(acc1))

    iterations = range(1, max_len + 1)
    df = pd.DataFrame(
        {
            "iteration": iterations,
            "origin_exp": acc1_padded,
        }
    )

    from search_engine.utils.plot_utils import plot_whiskers

    plot_whiskers(
        df=df,
        x_col="iteration",
        y_cols=["origin_exp"],
        labels=["Origin Exp"],
        colors=["blue"],
        linestyles=["-"],
        figsize=(8, 6),
        resultname=output_fig,
        xlabel="Iteration",
        ylabel="Accuracy",
    )

    return df


config1 = Config(
    search=SearchConfig(iterations=12),
    data=DataConfig(
        directory_path="data/original_sample_execution/country_extend_table_search/seller",
        buyer_csv="data/original_sample_execution/country_extend_table_search/buyer/master.csv",
        join_keys=[["Country"], ["year"]],
        target_feature="suicides_no",
        one_target_feature=False,
        need_to_clean_data=True,
    ),
    experiment=ExperimentConfig(plot_results=True, results_dir="results/"),
    logging=LoggingConfig(level="ERROR", file="logs/original_sample_execution.log"),
)

# Run exps
company = ScaledExperiment(config1)
company_experiment_result = company.run()
# Return as list
results_list = [company_experiment_result]
# plot
plot_two_experiments_results_max(
    origin_result=company_experiment_result,
    output_fig="results/original_sample_execution/comparison_country_extend_table_search_whiskers.png",
)
