import pandas as pd
import logger
def clean_dataframe(df: pd.DataFrame,  enforce_positive_price: bool):
    original_df = df.copy()
    df = df.replace('', pd.NA)
    required_fields = list(df.columns)
    
    missing_count = df[required_fields].isna().any(axis=1).sum()
    if missing_count > 0:
        logger.log_warning(f"Rejected {missing_count} rows: Missing required values")
    df = df.dropna(subset=required_fields, how='any')
    # If needed, drop negative prices
    if enforce_positive_price and "price" in df.columns:
        negative_count = (df["price"] < 0).sum()
        if negative_count > 0:
            logger.log_warning(f"Rejected {negative_count} rows: Negative prices")
        df = df[df["price"] >= 0]


    rejected_df = pd.concat([original_df, df]).drop_duplicates(keep=False)
    return df, rejected_df