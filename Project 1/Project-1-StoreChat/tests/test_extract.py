import pytest
import pandas as pd
import os
from extract import (
    clean_column_names,
    convert_data_types,
    validate_data,
    add_category_images,
    process_products
)

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Product ID': ['P1', 'P2'],
        'Product Name': ['Test Product', 'Another Product'],
        'Category ID': ['CAT-1', 'CAT-2'],
        'Launch Date': ['2023-01-01', '2023-02-01'],
        'Price': ['100.50', '200.75']
    })

def test_clean_column_names(sample_df):
    result = clean_column_names(sample_df.copy())
    assert 'product_id' in result.columns
    assert 'product_name' in result.columns
    assert 'category_id' in result.columns
    assert 'launch_date' in result.columns
    assert 'price' in result.columns

def test_convert_data_types(sample_df):
    df = clean_column_names(sample_df.copy())
    result = convert_data_types(df)
    assert result['price'].dtype == float
    assert pd.api.types.is_datetime64_any_dtype(result['launch_date'])
    assert result['product_name'].dtype == object

def test_validate_data(sample_df):
    df = clean_column_names(sample_df.copy())
    df = convert_data_types(df)
    results = validate_data(df)
    assert isinstance(results, dict)
    assert 'missing_product_id' in results
    assert results['missing_product_id'] == 0

def test_add_category_images(sample_df):
    df = clean_column_names(sample_df.copy())
    result = add_category_images(df)
    assert 'image_url' in result.columns
    assert result.loc[0, 'image_url'] == 'images/Accessories.jpg'
    assert result.loc[1, 'image_url'] == 'images/Audio.jpg'
    
def test_process_products(tmp_path, sample_df):
    input_file = tmp_path / "test_input.csv"
    output_file = tmp_path / "test_output.csv"
    
    sample_df.to_csv(input_file, index=False)
    df, validation_results = process_products(input_file, output_file)
    
    assert os.path.exists(output_file)
    assert 'image_url' in df.columns
    assert isinstance(validation_results, dict)