import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def create_stratified_splits(
    data: pd.DataFrame,
    target_col: str = "resolution",
    stratify_cols_info: dict[str, int] = {"CommercialTypeName4": 1000, "brand_name": 500},
    n_splits: int = 3,
    val_size: float = 0.2,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Creates stratified splits for the given DataFrame using MultilabelStratifiedShuffleSplit.

    Always stratifies based on the target column and the specified stratification columns.
    Possible stratification columns are: {CommercialTypeName4, brand_name}.

    Params:
    - data: The input DataFrame containing the data to split.
    - target_col: The name of the target column to stratify on.
    - stratify_cols: A dictionary mapping column names to their minimum sample sizes for stratification.
    - n_splits: The number of splits to create.
    - val_size: The proportion of the dataset to include in the validation split.
    - shuffle: Whether to shuffle the data before splitting.
    - random_state: The random seed to use for reproducibility.

    Returns a list of train/validation index tuples.
    """
    stratify_cols = [target_col]

    if "CommercialTypeName4" in stratify_cols_info:
        # Create other group for rare categories
        data["CommercialTypeName4_for_strat"] = data["CommercialTypeName4"].where(
            data["CommercialTypeName4"].map(data["CommercialTypeName4"].value_counts())
            > stratify_cols_info["CommercialTypeName4"],
            "<OTHER>",
        )
        stratify_cols.append("CommercialTypeName4_for_strat")

    if "brand_name" in stratify_cols_info:
        # Create other group for rare categories
        data["brand_name_for_strat"] = data["brand_name"].where(
            data["brand_name"].map(data["brand_name"].value_counts()) > stratify_cols_info["brand_name"], "<OTHER>"
        )
        stratify_cols.append("brand_name_for_strat")

    mskf = MultilabelStratifiedShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=random_state)
    splits = []
    for train_idx, val_idx in mskf.split(data, data[stratify_cols]):
        splits.append((train_idx, val_idx))

    # Remove temporary columns
    temp_cols = [col for col in stratify_cols if col != target_col]
    if temp_cols:
        data.drop(columns=temp_cols, inplace=True)

    return splits
