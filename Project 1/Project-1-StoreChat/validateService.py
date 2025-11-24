import pandas as pd

def clean_dataframe(df: pd.DataFrame,  enforce_positive_price: bool):
    original_df = df.copy()
    # Convert all empty-string cells to NaN
    df = df.replace('', pd.NA)
    required_fields = list(df.columns)
    df = df.dropna(subset=required_fields, how='any')
    # If needed, drop negative prices
    if enforce_positive_price and "price" in df.columns:
        df = df[df["price"] >= 0]
    rejected_df = pd.concat([original_df, df]).drop_duplicates(keep=False)

    return df, rejected_df