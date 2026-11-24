-- database/schema.sql
PRAGMA foreign_keys = ON;

CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    first_name   TEXT NOT NULL,
    last_name    TEXT NOT NULL,
    email        TEXT NOT NULL UNIQUE,
    segment      TEXT NOT NULL CHECK (segment IN ('Consumer','Corporate','Small Business')),
    region       TEXT NOT NULL,
    city         TEXT,
    signup_date  DATE NOT NULL
);

CREATE TABLE products (
    product_id    INTEGER PRIMARY KEY,
    product_name  TEXT NOT NULL,
    category      TEXT NOT NULL,
    sub_category  TEXT,
    unit_price    REAL NOT NULL CHECK (unit_price > 0),
    unit_cost     REAL NOT NULL CHECK (unit_cost > 0)
);

CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date   DATE NOT NULL,
    status       TEXT NOT NULL CHECK (status IN ('Completed','Cancelled','Returned')),
    region       TEXT NOT NULL
);

CREATE TABLE order_items (
    order_item_id INTEGER PRIMARY KEY,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    product_id    INTEGER NOT NULL REFERENCES products(product_id),
    quantity      INTEGER NOT NULL CHECK (quantity > 0),
    unit_price    REAL NOT NULL,
    unit_cost     REAL NOT NULL,
    discount_pct  REAL NOT NULL DEFAULT 0 CHECK (discount_pct BETWEEN 0 AND 1)
);

CREATE TABLE payments (
    payment_id      INTEGER PRIMARY KEY,
    order_id        INTEGER NOT NULL REFERENCES orders(order_id),
    payment_method  TEXT NOT NULL,
    amount          REAL NOT NULL,
    payment_date    DATE NOT NULL,
    payment_status  TEXT NOT NULL
);

CREATE INDEX idx_orders_date        ON orders(order_date);
CREATE INDEX idx_orders_customer    ON orders(customer_id);
CREATE INDEX idx_orders_status_date ON orders(status, order_date);
CREATE INDEX idx_items_order        ON order_items(order_id);
CREATE INDEX idx_items_product      ON order_items(product_id);
CREATE INDEX idx_payments_order     ON payments(order_id);
CREATE INDEX idx_products_category  ON products(category);
CREATE INDEX idx_customers_region   ON customers(region);