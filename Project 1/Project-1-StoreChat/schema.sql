CREATE TABLE IF NOT EXISTS categories (
    category_id VARCHAR(255) PRIMARY KEY,
    category_name VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    product_id VARCHAR(255) PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    category_id VARCHAR(255) NOT NULL,
    launch_date DATE NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    image_url VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS rejected_fields ( 
    rejected_id SERIAL PRIMARY KEY,
    source_table VARCHAR(255) NOT NULL,
    rejection_reason TEXT NOT NULL,
    rejected_data JSONB NOT NULL,
    rejected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

--------------------------------STORES ANDSALES TABLES------------------------------
CREATE TABLE IF NOT EXISTS stores (
    store_id VARCHAR(255) PRIMARY KEY,
    store_name VARCHAR(255) NOT NULL,
    city VARCHAR(255) NOT NULL,
    country VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS sales (
    sale_id VARCHAR(255) PRIMARY KEY,
    sale_date DATE NOT NULL,
    store_id VARCHAR(255) NOT NULL,
    product_id VARCHAR(255) NOT NULL,
    quantity INT NOT NULL CHECK (quantity > 0),
    sale_year INT NOT NULL,
    sale_month INT NOT NULL CHECK (sale_month BETWEEN 1 AND 12),
    sale_quarter INT NOT NULL CHECK (sale_quarter BETWEEN 1 AND 4),
    sale_day_of_week VARCHAR(20) NOT NULL,
    sale_week INT NOT NULL CHECK (sale_week BETWEEN 1 AND 53),
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    FOREIGN KEY (product_id) REFERENCES products(product_id)
);

--------------------------------STORE SALES SUMMARY TABLE------------------------------
CREATE TABLE IF NOT EXISTS store_sales_summary (
    summary_id SERIAL PRIMARY KEY,
    store_id VARCHAR(255) NOT NULL,
    sale_year INT NOT NULL,
    sale_month INT NOT NULL CHECK (sale_month BETWEEN 1 AND 12),
    total_quantity INT NOT NULL,
    total_transactions INT NOT NULL,
    avg_quantity_per_transaction DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (store_id) REFERENCES stores(store_id),
    UNIQUE (store_id, sale_year, sale_month)
);

--------------------------------INDECES------------------------------
CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(sale_date);
CREATE INDEX IF NOT EXISTS idx_sales_store ON sales(store_id);
CREATE INDEX IF NOT EXISTS idx_sales_product ON sales(product_id);
CREATE INDEX IF NOT EXISTS idx_sales_year_month ON sales(sale_year, sale_month);
CREATE INDEX IF NOT EXISTS idx_store_sales_store ON store_sales_summary(store_id);
CREATE INDEX IF NOT EXISTS idx_store_sales_year_month ON store_sales_summary(sale_year, sale_month);
-- CREATE INDEX IF NOT EXISTS idx_products_category ON products (category_id);
-- CREATE INDEX IF NOT EXISTS idx_products_launch_date ON products (launch_date);