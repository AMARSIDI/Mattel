CREATE DATABASE IF NOT EXISTS mattel;
USE mattel;

-- Dimension tables

-- Customers: one row per subscriber
CREATE TABLE IF NOT EXISTS customers (
  cust_num_bigint BIGINT PRIMARY KEY,         -- The full MSISDN (matches CUST_NUM in CDR)
  cust_type ENUM('S','A') NOT NULL            -- Subscriber Type ('S': Subscriber, 'A': Account)
);

-- Object types: Lookup table for balance objects (F=Free Resource, B=Balance, C=Credit)

CREATE TABLE IF NOT EXISTS object_types (
  object_type_id INT PRIMARY KEY AUTO_INCREMENT,
  object_type_code CHAR(1) NOT NULL UNIQUE,     -- 'F', 'B', 'C'
  description VARCHAR(100) NOT NULL             -- e.g., 'Free Resource', 'Main Balance', 'Credit Balance'
);

-- Insert known object types if they don't exist
INSERT IGNORE INTO object_types (object_type_code, description) VALUES
('F', 'Free Resource'),
('B', 'Main Balance'),
('c', 'Credit Balance');

-- Objects: Specific instances of balances/resources
CREATE TABLE IF NOT EXISTS objects (
  object_id_bigint BIGINT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each balance/resource
  cust_num_bigint BIGINT NOT NULL,                -- Links to customer
  object_type_id INT NOT NULL,                    -- Links to object_types
  expiry_date DATETIME NULL,                      -- Expiry date of this resource
  FOREIGN KEY (cust_num_bigint) REFERENCES customers(cust_num_bigint),
  FOREIGN KEY (object_type_id) REFERENCES object_types(object_type_id)
);

-- Offers: Details about plans, bundles, loans
CREATE TABLE IF NOT EXISTS offers (
  offer_id INT PRIMARY KEY,                       -- Matches OFFER_ID in CDR
  offer_name VARCHAR(255) NOT NULL                -- Offer name from CDR mapping
);

-- Insert offer names from CDR documentation if they don't exist
INSERT IGNORE INTO offers (offer_id, offer_name) VALUES
(1482110, '10 min MaxPlus'),
(1050552, 'Pass NCD 200Mo'),
(3771305, 'Khamssa Data'),
(1477069, 'Khamssa Voix'),
(9430556, '1H_Internet_Illimite'),
(1853344, '3H_Data_Illimite'),
(1853351, '5H_Data_Illimite'),
(1482815, 'Loan_35Mo'),
(1582989, 'Loan_3Min_6SMS_5Mo'),
(1382991, 'Loan_70Mo'),
(5829986, 'Loan_6Min_12SMS_15Mo'),
(5830005, 'Loan_150Mo'),
(1883001, 'Loan_12Min_24SMS_35Mo'),
(1483001, 'Loan_400Mo'),
(1883002, 'Loan_30Min_60SMS_120Mo');

-- Operation types: CDR event operation codes
CREATE TABLE IF NOT EXISTS operation_types (
  operation_type_id SMALLINT PRIMARY KEY,
  description VARCHAR(255) NOT NULL
);

-- Insert confirmed operation types from CDR documentation if they don't exist
INSERT IGNORE INTO operation_types (operation_type_id, description) VALUES
(1, 'Usage - Type 1'),
(2, 'Usage - Type 2'),
(3, 'Usage - Type 3'),
(4, 'Add new Free Resource'),
(5, 'Add to existing Free Resource');

-- CDR Events: Transactional data
CREATE TABLE IF NOT EXISTS cdr_events (
  cdr_event_id BIGINT AUTO_INCREMENT PRIMARY KEY,
  cdr_id VARCHAR(255) NOT NULL,                 -- Matches CDR_ID in documentation
  creation_time DATETIME NOT NULL,                -- Event timestamp
  
  -- Dimension links
  cust_num BIGINT NOT NULL,                       -- Customer
  object_id BIGINT NOT NULL,                      -- Object/resource
  offer_id INT NULL,                              -- Associated offer
  operation_type_id SMALLINT NOT NULL,            -- Operation type
  
  -- Amount fields
  deduct_amount DECIMAL(15,2) DEFAULT 0,          -- Amount deducted
  charge_amount DECIMAL(15,2) DEFAULT 0,          -- Amount charged/added
  
  -- Balance snapshots
  current_amount_pre DECIMAL(15,2),               -- Balance before
  current_amount_post DECIMAL(15,2),              -- Balance after
  
  -- Expiry times
  charge_expire_time DATETIME NULL,               -- When charged amount expires
  current_amount_expire_time DATETIME NULL,       -- When current amount expires
  
  -- Foreign keys
  FOREIGN KEY (cust_num) REFERENCES customers(cust_num_bigint),
  FOREIGN KEY (object_id) REFERENCES objects(object_id_bigint),
  FOREIGN KEY (offer_id) REFERENCES offers(offer_id),
  FOREIGN KEY (operation_type_id) REFERENCES operation_types(operation_type_id)
);

-- --------------------------------------------------------------------------
-- TABLES FOR EMPLOYEE AUTHENTICATION AND AUTHORIZATION
-- --------------------------------------------------------------------------

-- Roles: Defines different user roles within the system
CREATE TABLE IF NOT EXISTS roles (
  role_id INT PRIMARY KEY AUTO_INCREMENT,
  role_name VARCHAR(50) NOT NULL UNIQUE       -- e.g., 'admin', 'employee', 'viewer'
);

-- Insert basic roles if they don't exist
INSERT IGNORE INTO roles (role_name) VALUES
('admin'),
('employee');

-- Employees: Stores information about employees who can log into the web app
CREATE TABLE IF NOT EXISTS employees (
  employee_id INT PRIMARY KEY AUTO_INCREMENT,
  employee_gmail VARCHAR(255) NOT NULL UNIQUE,      -- Employee's Gmail address for login
  employee_phone VARCHAR(20) NULL UNIQUE,           -- Employee's phone number (optional, consider security)
  password_hash VARCHAR(255) NOT NULL,              -- Hashed password for security
  role_id INT NOT NULL,                             -- Foreign key to the roles table
  is_active BOOLEAN DEFAULT TRUE,                   -- To activate/deactivate accounts
  
  -- New fields for email verification
  email_verification_token VARCHAR(100) NULL UNIQUE, -- Token sent to user for email verification
  email_verification_token_expires_at DATETIME NULL, -- Expiry time for the verification token
  is_email_verified BOOLEAN DEFAULT FALSE,          -- Flag to check if email has been verified
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  FOREIGN KEY (role_id) REFERENCES roles(role_id)
);

-- Example of how to insert an admin user (replace with actual secure password hashing in your app)
-- Remember to set is_email_verified = TRUE for pre-existing admin accounts if they don't need to verify.
-- For demonstration, 'hashed_password_for_admin' is a placeholder.
-- In a real application, you would hash the password before inserting it.
-- INSERT IGNORE INTO employees (employee_gmail, employee_phone, password_hash, role_id, is_email_verified) 
-- VALUES ('admin@example.com', '1234567890', 'hashed_password_for_admin', (SELECT role_id FROM roles WHERE role_name = 'admin'), TRUE);


-- --------------------------------------------------------------------------
-- INDEXES FOR PERFORMANCE
-- --------------------------------------------------------------------------

-- Existing indexes (ensure they are created if not already)
CREATE INDEX  idx_cdr_creation_time ON cdr_events(creation_time);
CREATE INDEX  idx_cdr_cust_num ON cdr_events(cust_num);
CREATE INDEX  idx_cdr_object_id ON cdr_events(object_id);
CREATE INDEX  idx_cdr_offer_id ON cdr_events(offer_id);
CREATE INDEX  idx_cdr_operation_type_id ON cdr_events(operation_type_id);

CREATE INDEX  idx_objects_cust_num ON objects(cust_num_bigint);
CREATE INDEX  idx_objects_object_type_id ON objects(object_type_id);

-- New indexes for auth tables
CREATE INDEX  idx_employees_email ON employees(employee_gmail);
CREATE INDEX  idx_employees_role_id ON employees(role_id);
CREATE INDEX  idx_employees_verification_token ON employees(email_verification_token);

-- --------------------------------------------------------------------------
-- QUERIES (from your original script for reference)
-- --------------------------------------------------------------------------
-- USE mattel;
-- SELECT COUNT(*) FROM cdr_events;
-- SELECT MIN(creation_time), MAX(creation_time) FROM cdr_events;
-- SELECT offer_id, COUNT(*) FROM cdr_events GROUP BY offer_id ORDER BY COUNT(*) DESC LIMIT 5;
-- SELECT operation_type_id, COUNT(*) FROM cdr_events GROUP BY operation_type_id ORDER BY COUNT(*) DESC LIMIT 5;

-- To see the new tables:
-- SELECT * FROM roles;
-- SELECT * FROM employees;

