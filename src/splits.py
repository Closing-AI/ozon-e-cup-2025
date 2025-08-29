import networkx as nx
import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import StratifiedGroupKFold


def create_train_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Creates train/validation splits for training.

    Splits will be stratified by target column ("resolution") and grouping by categorical columns ("SellerID" and "brand_name").

    Returns a list of train/validation index tuples.
    """
    # Build a graph: nodes are row indices, edges connect rows with same SellerID or brand_name
    G = nx.Graph()
    for col in ["SellerID", "brand_name"]:
        for _, group in data.groupby(col):
            idx = group.index.tolist()
            # Connect all indices in this group
            for i in range(len(idx) - 1):
                G.add_edge(idx[i], idx[i + 1])

    # Find connected components (each is a group)
    component_labels = {}
    for group_id, component in enumerate(nx.connected_components(G)):
        for idx in component:
            component_labels[idx] = group_id

    # Assign group labels to train
    data["union_group"] = data.index.map(component_labels)

    # Create stratified splits
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    for train_idx, val_idx in sgkf.split(X=data, y=data["resolution"], groups=data["union_group"]):
        splits.append((train_idx, val_idx))

    # Remove temporary column
    data.drop(columns=["union_group"], inplace=True)

    return splits


def create_final_splits(
    data: pd.DataFrame,
    n_splits: int = 2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Creates stratified splits for the given DataFrame using MultilabelStratifiedShuffleSplit.

    Always stratifies based on the target column ("resolution") and the categorical columns ("CommercialTypeName4", "brand_name", "SellerID").

    Params:
    - data: The input DataFrame containing the data to split.
    - n_splits: The number of splits to create.
    - val_size: The proportion of the dataset to include in the validation split.
    - random_state: The random seed to use for reproducibility.

    Returns a list of train/validation index tuples.
    """
    data["brand_name"] = data["brand_name"].fillna("<UNKNOWN_BRAND_NAME>").astype(str)

    mskf = MultilabelStratifiedShuffleSplit(n_splits=n_splits, test_size=val_size, random_state=random_state)
    splits = []
    for train_idx, val_idx in mskf.split(data, data[["resolution", "SellerID", "CommercialTypeName4", "brand_name"]]):
        splits.append((train_idx, val_idx))

    data["brand_name"] = data["brand_name"].replace("<UNKNOWN_BRAND_NAME>", np.nan)

    return splits
