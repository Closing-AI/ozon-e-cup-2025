import numpy as np
import pandas as pd
from category_encoders import BinaryEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize


class NumericEncoder(TransformerMixin, BaseEstimator):
    """
    Class for numeric column preprocessing:

    - Apply np.log1p if required
    - Apply StandardScaler
    - Fill missing values

    Args:
        col_name (str): The name of the column to encode.
        apply_log (bool): Whether to apply np.log1p to the column.
        filling_value (float): The value to use for filling missing values.
    """

    def __init__(self, col_name: str, apply_log: bool, filling_value: float = -1.0):
        self.col_name = col_name
        self.apply_log = apply_log
        self.filling_value = filling_value

        self._scaler = StandardScaler()
        self._scaler.set_output(transform="pandas")

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()

        if self.apply_log:
            X[[self.col_name]] = np.log1p(X[[self.col_name]])

        self._scaler = self._scaler.fit(X[[self.col_name]].dropna())

        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()

        if self.apply_log:
            X[[self.col_name]] = np.log1p(X[[self.col_name]])

        X[[self.col_name]] = self._scaler.transform(X[[self.col_name]])
        X[[self.col_name]] = X[[self.col_name]].fillna(self.filling_value)

        return X


class CategoryEncoder(TransformerMixin, BaseEstimator):
    """
    Class for category column preprocessing (Group rare categories and apply Binary Encoding).

    Args:
        col_name (str): The name of the column to encode.
        min_freq (int): Categories which frequency is below this threshold will be put in f"<RARE_{self.col_name}>" category.
    """

    def __init__(self, col_name: str, min_freq: int):
        self.col_name = col_name
        self.min_freq = min_freq

        self._encoder = BinaryEncoder(cols=[self.col_name], handle_missing="value", handle_unknown="value")
        self._rare_col_name = f"<RARE_{self.col_name.upper()}>"

    def _group_rare_categories(self, series: pd.Series):
        """
        Puts rare categories into a special category (check whether value is in self.frequent_categories).
        """
        mask = series.notna() & ~series.isin(self.frequent_categories_)
        series.loc[mask] = self._rare_col_name

        return series

    def fit(self, X: pd.DataFrame, y=None):
        """
        Memorize the most frequent categories and fits binary encoder.
        """
        X_one_col = X[[self.col_name]].copy()
        series = X_one_col[self.col_name]

        value_counts = series.dropna().value_counts()
        self.frequent_categories_ = value_counts.loc[lambda x: x >= self.min_freq].index.tolist()

        X_one_col[self.col_name] = self._group_rare_categories(series)
        self._encoder = self._encoder.fit(X_one_col)

        return self

    def transform(self, X: pd.DataFrame, y=None):
        """
        Applies Binary Encoding to the column.
        """
        X = X.copy()
        X[self.col_name] = self._group_rare_categories(X[self.col_name].copy())

        return pd.merge(X, self._encoder.transform(X[[self.col_name]]), left_index=True, right_index=True)


CATEGORIAL_PREP_PIPELINE = Pipeline(
    [
        ("brand_name", CategoryEncoder("brand_name", min_freq=10)),
        ("CommercialTypeName4", CategoryEncoder(col_name="CommercialTypeName4", min_freq=10)),
        ("SellerID", CategoryEncoder(col_name="SellerID", min_freq=15)),
    ],
    verbose=True,
)


NUMERIC_PREP_PIPELINE = Pipeline(
    [
        ("rating_1_count", NumericEncoder("rating_1_count", apply_log=True)),
        ("rating_2_count", NumericEncoder("rating_2_count", apply_log=True)),
        ("rating_3_count", NumericEncoder("rating_3_count", apply_log=True)),
        ("rating_4_count", NumericEncoder("rating_4_count", apply_log=True)),
        ("rating_5_count", NumericEncoder("rating_5_count", apply_log=True)),
        ("comments_published_count", NumericEncoder("comments_published_count", apply_log=True)),
        ("photos_published_count", NumericEncoder("photos_published_count", apply_log=True)),
        ("videos_published_count", NumericEncoder("videos_published_count", apply_log=True)),
        ("PriceDiscounted", NumericEncoder("PriceDiscounted", apply_log=False)),
        ("item_time_alive", NumericEncoder("item_time_alive", apply_log=True)),
        ("item_count_fake_returns7", NumericEncoder("item_count_fake_returns7", apply_log=False)),
        ("item_count_fake_returns30", NumericEncoder("item_count_fake_returns30", apply_log=False)),
        ("item_count_fake_returns90", NumericEncoder("item_count_fake_returns90", apply_log=False)),
        ("item_count_sales7", NumericEncoder("item_count_sales7", apply_log=True)),
        ("item_count_sales30", NumericEncoder("item_count_sales30", apply_log=True)),
        ("item_count_sales90", NumericEncoder("item_count_sales90", apply_log=True)),
        ("item_count_returns7", NumericEncoder("item_count_returns7", apply_log=False)),
        ("item_count_returns30", NumericEncoder("item_count_returns30", apply_log=False)),
        ("item_count_returns90", NumericEncoder("item_count_returns90", apply_log=False)),
        ("GmvTotal7", NumericEncoder("GmvTotal7", apply_log=False)),
        ("GmvTotal30", NumericEncoder("GmvTotal30", apply_log=False)),
        ("GmvTotal90", NumericEncoder("GmvTotal90", apply_log=False)),
        ("ExemplarAcceptedCountTotal7", NumericEncoder("ExemplarAcceptedCountTotal7", apply_log=True)),
        ("ExemplarAcceptedCountTotal30", NumericEncoder("ExemplarAcceptedCountTotal30", apply_log=True)),
        ("ExemplarAcceptedCountTotal90", NumericEncoder("ExemplarAcceptedCountTotal90", apply_log=True)),
        ("OrderAcceptedCountTotal7", NumericEncoder("OrderAcceptedCountTotal7", apply_log=True)),
        ("OrderAcceptedCountTotal30", NumericEncoder("OrderAcceptedCountTotal30", apply_log=True)),
        ("OrderAcceptedCountTotal90", NumericEncoder("OrderAcceptedCountTotal90", apply_log=True)),
        ("ExemplarReturnedCountTotal7", NumericEncoder("ExemplarReturnedCountTotal7", apply_log=True)),
        ("ExemplarReturnedCountTotal30", NumericEncoder("ExemplarReturnedCountTotal30", apply_log=True)),
        ("ExemplarReturnedCountTotal90", NumericEncoder("ExemplarReturnedCountTotal90", apply_log=True)),
        ("ExemplarReturnedValueTotal7", NumericEncoder("ExemplarReturnedValueTotal7", apply_log=False)),
        ("ExemplarReturnedValueTotal30", NumericEncoder("ExemplarReturnedValueTotal30", apply_log=False)),
        ("ExemplarReturnedValueTotal90", NumericEncoder("ExemplarReturnedValueTotal90", apply_log=False)),
        ("ItemVarietyCount", NumericEncoder("ItemVarietyCount", apply_log=True)),
        ("ItemAvailableCount", NumericEncoder("ItemAvailableCount", apply_log=True)),
        ("seller_time_alive", NumericEncoder("seller_time_alive", apply_log=True)),
    ],
    verbose=True,
)
