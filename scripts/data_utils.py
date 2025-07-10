import pandas as pd
import shutil
from IPython.display import display

def describe_dataframe(df: pd.DataFrame, head: int = 5, delete_duplicates: bool = False) -> pd.DataFrame:
    """
    Prints a quick overview of the provided DataFrame for exploratory data analysis (EDA).

    This function displays the first rows, shape, basic statistics, missing values,
    number of duplicates, and unique values per column. Optionally, it can remove duplicate rows
    and return the cleaned DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to analyze.
        head (int, optional): The number of top rows to display. Defaults to 5.
        delete_duplicates (bool, optional): Whether to remove duplicate rows from the DataFrame. Defaults to False.

    Returns:
        pd.DataFrame: A copy of the input DataFrame, possibly with duplicates removed.

    Example:
        >>> cleaned_df = describe_dataframe(df, head=10, delete_duplicates=True)
    """

    def rows_separator():
        try:
            w = shutil.get_terminal_size().columns
        except:
            w = 100

        print('_' * w, end='\n\n')

    print(f'First {head} rows of the DataFrame:')
    display(df.head(head))
    rows_separator()

    print('Shape of the DataFrame:', df.shape)
    rows_separator()

    print('DataFrame info:')
    df.info()
    rows_separator()

    print('Descriptive statistics of the dataframe:')
    display(df.describe().T)
    rows_separator()

    print('Missing values per column:')
    display(df.isna().sum())
    print('Total missing values:', df.isna().sum().sum())
    rows_separator()

    p = round(df.duplicated().sum()/df.shape[0]*100, 3)
    print(f'Duplicate rows: {p}%', )
    rows_separator()

    if delete_duplicates:
        if p:
            df = df.drop_duplicates(inplace=False, ignore_index=False)
            print('Duplicate rows removed.')
        else:
            print('No duplicate rows to remove.')
        rows_separator()

    print('Unique values per column:')
    display(df.nunique())

    return df