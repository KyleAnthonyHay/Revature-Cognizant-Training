import pandas as pd
import logger

def clean_dataframe(df: pd.DataFrame, enforce_positive_price: bool):
    original_df = df.copy()
    df = df.replace('', pd.NA)
    required_fields = list(df.columns)
    
    rejected_rows = []
    rejection_reasons = []
    
    for idx, row in df.iterrows():
        reason_parts = []
        
        missing_fields = [field for field in required_fields if pd.isna(row[field])]
        if missing_fields:
            reason_parts.append(f"Missing fields: {', '.join(missing_fields)}")
        
        if enforce_positive_price and "price" in row and pd.notna(row["price"]) and row["price"] < 0:
            reason_parts.append("Negative price")
        
        if reason_parts:
            rejected_rows.append(idx)
            rejection_reasons.append("; ".join(reason_parts))
    
    rejected_df = df.loc[rejected_rows].copy() if rejected_rows else pd.DataFrame()
    if len(rejected_df) > 0:
        rejected_df['rejection_reason'] = rejection_reasons
    
    df = df.drop(rejected_rows)
    
    missing_count = len(rejected_df[rejected_df['rejection_reason'].str.contains('Missing fields', na=False)]) if len(rejected_df) > 0 else 0
    if missing_count > 0:
        logger.log_warning(f"Rejected {missing_count} rows: Missing required values")
    
    negative_count = len(rejected_df[rejected_df['rejection_reason'].str.contains('Negative price', na=False)]) if len(rejected_df) > 0 else 0
    if enforce_positive_price and negative_count > 0:
        logger.log_warning(f"Rejected {negative_count} rows: Negative prices")
    
    return df, rejected_df