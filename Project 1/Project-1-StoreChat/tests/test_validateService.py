import pytest
import pandas as pd
from src.validateService import clean_dataframe

#--------------------------------Fixtures----------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'product_id': ['P1', 'P2', 'P3'],
        'product_name': ['Product 1', 'Product 2', 'Product 3'],
        'price': [100.50, 200.75, 50.00]
    })

@pytest.fixture
def df_with_missing():
    return pd.DataFrame({
        'product_id': ['P1', 'P2', None, 'P4'],
        'product_name': ['Product 1', None, 'Product 3', 'Product 4'],
        'price': [100.50, 200.75, 50.00, 75.00]
    })

@pytest.fixture
def df_with_negative_price():
    return pd.DataFrame({
        'product_id': ['P1', 'P2', 'P3'],
        'product_name': ['Product 1', 'Product 2', 'Product 3'],
        'price': [100.50, -50.00, 200.75]
    })

@pytest.fixture
def df_with_empty_strings():
    return pd.DataFrame({
        'product_id': ['P1', '', 'P3'],
        'product_name': ['Product 1', 'Product 2', ''],
        'price': [100.50, 200.75, 50.00]
    })

#--------------------------------Tests----------------------------
def test_clean_dataframe_no_rejections(sample_df):
    cleaned_df, rejected_df = clean_dataframe(sample_df.copy(), enforce_positive_price=True)
    
    assert len(cleaned_df) == 3
    assert len(rejected_df) == 0
    assert list(cleaned_df.index) == [0, 1, 2]

def test_clean_dataframe_multiple_rejection_reasons():
    df = pd.DataFrame({
        'product_id': ['P1', None],
        'product_name': ['Product 1', 'Product 2'],
        'price': [-50.00, 100.00]
    })
    
    cleaned_df, rejected_df = clean_dataframe(df.copy(), enforce_positive_price=True)
    
    assert len(cleaned_df) == 0
    assert len(rejected_df) == 2
    
    first_reason = rejected_df.iloc[0]['rejection_reason']
    assert 'Negative price' in first_reason
    
    second_reason = rejected_df.iloc[1]['rejection_reason']
    assert 'Missing fields' in second_reason

def test_clean_dataframe_partial_rejections():
    df = pd.DataFrame({
        'product_id': ['P1', None, 'P3', 'P4'],
        'product_name': ['Product 1', 'Product 2', 'Product 3', None],
        'price': [100.50, 200.75, -50.00, 75.00]
    })
    
    cleaned_df, rejected_df = clean_dataframe(df.copy(), enforce_positive_price=True)
    
    assert len(cleaned_df) == 1
    assert len(rejected_df) == 3
    assert cleaned_df.iloc[0]['product_id'] == 'P1'

def test_clean_dataframe_no_price_column():
    df = pd.DataFrame({
        'product_id': ['P1', 'P2'],
        'product_name': ['Product 1', 'Product 2']
    })
    
    cleaned_df, rejected_df = clean_dataframe(df.copy(), enforce_positive_price=True)
    
    assert len(cleaned_df) == 2
    assert len(rejected_df) == 0

def test_clean_dataframe_zero_price():
    df = pd.DataFrame({
        'product_id': ['P1', 'P2'],
        'product_name': ['Product 1', 'Product 2'],
        'price': [0.00, 100.00]
    })
    
    cleaned_df, rejected_df = clean_dataframe(df.copy(), enforce_positive_price=True)
    
    assert len(cleaned_df) == 2
    assert len(rejected_df) == 0