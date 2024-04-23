"""Preprocessing data class and functions."""

from typing import List
from typing import Tuple
import pandas as pd
from scipy.stats import iqr
from scipy.stats import scoreatpercentile
from sklearn.preprocessing import RobustScaler


def fillna_median(
    df: pd.DataFrame,
    grp: List[str],
    cols: List[str]
) -> pd.DataFrame:
    """
    Fill missing values in specified columns using the median value grouped
    by specified columns.

    Args:
        df (DataFrame): The DataFrame to process.
        grp (List[str]): The list of column names to group by for calculating
        medians.
        cols (List[str]): The list of column names in which to fill missing
        values with their respective grouped medians.

    Returns:
        DataFrame: The DataFrame with missing values filled in the specified
        columns.
    """

    for col in cols:
        medians = df.groupby(grp)[col].transform('median')
        df[col] = df[col].fillna(medians)

    return df


def get_outliers_limits(df: pd.DataFrame, col: str) -> Tuple[float, float]:
    """
    Calculate the lower and upper limits for outlier detection in a given
    column using the IQR method.

    Args:
        df (DataFrame): The DataFrame containing the data.
        col (str): The column in which to calculate the outlier limits.

    Returns:
        float: The lower limit for outlier detection.
        float: The upper limit for outlier detection.
    """
    values = df[col]
    iqr_ = iqr(values)
    q1 = scoreatpercentile(values, 25)
    q3 = scoreatpercentile(values, 75)
    limit_low = q1 - 1.5 * iqr_
    limit_upp = q3 + 1.5 * iqr_

    return limit_low, limit_upp


def scale(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Scale specified columns in a DataFrame using the RobustScaler, which is less sensitive to outliers.

    Args:
        df (DataFrame): The DataFrame to be scaled.
        cols (List[str]): A list of column names to be scaled.

    Returns:
        DataFrame: The DataFrame with the specified columns scaled.
    """
    scaler = RobustScaler()
    df[cols] = scaler.fit_transform(df[cols])

    return df
