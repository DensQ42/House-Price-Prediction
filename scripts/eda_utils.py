import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

def plot_missing_values(df_train: pd.DataFrame, df_test: pd.DataFrame, train_name: str = 'Train dataset', test_name: str = 'Test dataset') -> None:
    """
    Visualize the distribution of missing values for each column in the provided datasets.

    This function creates two heatmaps displaying the locations of missing values
    in the training and test datasets. Each row in the heatmap represents a feature (column),
    and each column in the heatmap represents an observation (row) in the dataset.

    Args:
        df_train (pd.DataFrame): The training dataset.
        df_test (pd.DataFrame): The test dataset.
        train_name (str, optional): Title for the training dataset heatmap. Defaults to 'Train dataset'.
        test_name (str, optional): Title for the test dataset heatmap. Defaults to 'Test dataset'.

    Returns:
        None: This function only displays plots and does not return any value.

    Example:
        >>> plot_missing_values(train_df, test_df, 'Training Set', 'Testing Set')
    """
    plt.figure(figsize=(10,6), dpi=150)

    plt.subplot(2, 1, 1)
    sns.heatmap(df_train.T.isnull(), cbar=False, cmap='magma', xticklabels=False)
    plt.title(train_name)
    plt.ylabel('Features')

    plt.subplot(2, 1, 2)
    sns.heatmap(df_test.T.isnull(), cbar=False, cmap='magma', xticklabels=False)
    plt.title(test_name)
    plt.ylabel('Features')

    plt.tight_layout()
    plt.show()


def histplot_plus_norm(data: pd.Series, title: str = '', bins: int = 30) -> None:
    """
    Plots a histogram with KDE and overlays a fitted normal distribution.

    This function visualizes the distribution of the input data using a histogram
    and kernel density estimation (KDE), and overlays the probability density
    function (PDF) of a normal distribution fitted to the data.

    The plot is styled to remove x and y axis ticks and gridlines for a cleaner visual.

    Args:
        data (pd.Series): The input data series to visualize.
        title (str, optional): The title of the plot. Defaults to an empty string.
        bins (int, optional): The number of bins to use for the histogram. Defaults to 30.

    Returns:
        None: This function only displays the plot and does not return any value.

    Example:
        >>> histplot_plus_norm(df['SalePrice'], title='Sale Price Distribution', bins=40)
    """
    mu, sigma = norm.fit(data)
    xx = np.linspace(min(data), max(data), 500)

    sns.histplot(data, kde=True, stat='density', bins=bins)
    plt.plot(xx, norm.pdf(xx, mu, sigma), 'r-', label='Normal Fit')

    plt.title(title)
    plt.gca().set_yticks([])
    plt.gca().set_xticks([])
    plt.grid(False)


def plot_distributions_with_transforms(data: pd.Series) -> None:
    """
    Plots the distribution of the original data and its transformations (square root and logarithmic).

    This function visualizes:
        1. The original distribution.
        2. The square root transformed distribution.
        3. The logarithmic transformed distribution.

    Each distribution is displayed with a histogram, KDE, and a fitted normal distribution
    using the `histplot_plus_norm` function.

    Args:
        data (pd.Series): The input data series to visualize.

    Returns:
        None: This function only displays plots and does not return any value.

    Example:
        >>> plot_distributions_with_transforms(df['SalePrice'])
    """
    plt.figure(figsize=(12, 3), dpi=200)

    plt.subplot(1, 3, 1)
    histplot_plus_norm(data=data, title='Original Distribution')

    plt.subplot(1, 3, 2)
    histplot_plus_norm(data=np.sqrt(data), title='Square Root Transform')

    plt.subplot(1, 3, 3)
    histplot_plus_norm(data=np.log(data), title='Log Transform')

    plt.tight_layout()
    plt.show()
