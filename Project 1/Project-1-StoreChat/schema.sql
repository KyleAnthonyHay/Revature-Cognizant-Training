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

-- CREATE INDEX IF NOT EXISTS idx_products_category ON products (category_id);
-- CREATE INDEX IF NOT EXISTS idx_products_launch_date ON products (launch_date);