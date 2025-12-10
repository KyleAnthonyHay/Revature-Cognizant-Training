# StoreChat ETL Project

This project implements an ETL (Extract, Transform, Load) pipeline to process sales, product, store, and category data for the StoreChat application.

## ETL Flow

The pipeline processes data in the following stages:

1.  **Extract**: Raw data is read from CSV files (`products.csv`, `sales.csv`, `stores.csv`, `category.csv`).
2.  **Transform**: Data undergoes cleaning, validation, and enrichment.
    *   `src/extract.py`: Handles initial processing for products and sales (type conversion, adding time dimensions, image mapping).
    *   `src/validateService.py`: Validates data integrity (missing fields, negative prices) before loading.
3.  **Load**: Valid records are inserted into a PostgreSQL database, while invalid records are logged to a rejection table.
    *   `src/repo.py`: Manages database connections and insertion logic.

### Database Tables

*   **categories**: Product categories.
*   **products**: Product details including pricing and image URLs.
*   **stores**: Store locations and metadata.
*   **sales**: Transactional sales data with derived time dimensions.
*   **rejected_fields**: Stores records that failed validation with reasons.

## Cleaning Process

The cleaning logic ensures data quality by:
*   Standardizing column names (snake_case).
*   Converting data types (strings to dates/numerics).
*   Deduplicating records (specifically for sales).
*   Validating required fields and business rules (e.g., non-negative prices).
*   Enriching data (e.g., adding `sale_quarter`, `sale_day_of_week`).

## Testing & Logging

*   **Logging**: Operations are logged to `src/logs/app.log` (Info, Warning, Error levels).
*   **Testing**: Unit tests are located in the `tests/` directory (e.g., `test_extract.py`).


