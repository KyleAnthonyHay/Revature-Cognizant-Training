import pytest
import pandas as pd
import os
from src.extract import (
    clean_column_names,
    convert_data_types,
    validate_data,
    add_category_images,
    process_products,
    convert_sales_data_types,
    add_time_dimensions,
    validate_sales_data,
    deduplicate_sales,
    process_sales
)
#--------------------------------Fixtures--------------------------------
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'Product ID': ['P1', 'P2'],
        'Product Name': ['Test Product', 'Another Product'],
        'Category ID': ['CAT-1', 'CAT-2'],
        'Launch Date': ['2023-01-01', '2023-02-01'],
        'Price': ['100.50', '200.75']
    })
@pytest.fixture
def sample_sales_df():
    return pd.DataFrame({
        'Sale ID': ['S1', 'S2', 'S3'],
        'Sale Date': ['16-06-2023', '13-04-2022', '05-07-2021'],
        'Store ID': ['ST-10', 'ST-63', 'ST-26'],
        'Product ID': ['P-38', 'P-48', 'P-79'],
        'Quantity': ['10', '5', '7']
    })
#--------------------------------Tests for Products.csv--------------------------------
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
    assert result.loc[0, 'image_url'] == '../dataset/images/Accessories.jpg'
    assert result.loc[1, 'image_url'] == '../dataset/images/Audio.jpg'
    
def test_process_products(tmp_path, sample_df):
    input_file = tmp_path / "test_input.csv"
    output_file = tmp_path / "test_output.csv"
    
    sample_df.to_csv(input_file, index=False)
    df, validation_results = process_products(input_file, output_file)
    
    assert os.path.exists(output_file)
    assert 'image_url' in df.columns
    assert isinstance(validation_results, dict)

#--------------------------------Tests for Sales.csv--------------------------------
def test_convert_sales_data_types(sample_sales_df):
    df = clean_column_names(sample_sales_df.copy())
    result = convert_sales_data_types(df)
    assert result['sale_id'].dtype == object
    assert pd.api.types.is_datetime64_any_dtype(result['sale_date'])
    assert result['store_id'].dtype == object
    assert result['product_id'].dtype == object
    assert pd.api.types.is_integer_dtype(result['quantity'])

def test_add_time_dimensions(sample_sales_df):
    df = clean_column_names(sample_sales_df.copy())
    df = convert_sales_data_types(df)
    result = add_time_dimensions(df)
    assert 'sale_year' in result.columns
    assert 'sale_month' in result.columns
    assert 'sale_quarter' in result.columns
    assert 'sale_day_of_week' in result.columns
    assert 'sale_week' in result.columns
    assert result.loc[0, 'sale_year'] == 2023

def test_validate_sales_data(sample_sales_df):
    df = clean_column_names(sample_sales_df.copy())
    df = convert_sales_data_types(df)
    results = validate_sales_data(df)
    assert isinstance(results, dict)
    assert 'missing_sale_id' in results
    assert 'zero_or_negative_quantity' in results
    assert 'future_dates' in results
    assert results['missing_sale_id'] == 0

def test_deduplicate_sales(sample_sales_df):
    df = clean_column_names(sample_sales_df.copy())
    duplicate_df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    result = deduplicate_sales(duplicate_df)
    assert len(result) == len(df)
    assert result['sale_id'].nunique() == len(result)

def test_process_sales(tmp_path, sample_sales_df):
    input_file = tmp_path / "test_sales_input.csv"
    output_file = tmp_path / "test_sales_output.csv"
    
    sample_sales_df.to_csv(input_file, index=False)
    df, validation_results = process_sales(input_file, output_file)
    
    assert os.path.exists(output_file)
    assert 'sale_year' in df.columns
    assert 'sale_month' in df.columns
    assert isinstance(validation_results, dict)