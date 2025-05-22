
CREATE DATABASE mattel;
USE mattel;

-- Dimension tables

-- Customers: one row per subscriber
select * from customers;
CREATE TABLE customers (
  cust_num_bigint BIGINT PRIMARY KEY,          -- The full MSISDN (matches CUST_NUM in CDR)
  cust_type ENUM('S','A') NOT NULL            -- Subscriber Type ('S': Subscriber, 'A': Account)
);

-- Object types: Lookup table for balance objects (F=Free Resource, B=Balance, C=Credit)
select * from object_types;

CREATE TABLE object_types (
  object_type_id INT PRIMARY KEY AUTO_INCREMENT,
  object_type_code CHAR(1) NOT NULL UNIQUE,     -- 'F', 'B', 'C'
  description VARCHAR(100) NOT NULL             -- e.g., 'Free Resource', 'Main Balance', 'Credit Balance'
);

-- Insert known object types
INSERT INTO object_types (object_type_code, description) VALUES
('F', 'Free Resource'),
('B', 'Main Balance'),
('C', 'Credit Balance');

-- Objects: Specific instances of balances/resources
select * from objects;
CREATE TABLE objects (
  object_id_bigint BIGINT PRIMARY KEY AUTO_INCREMENT, -- Unique ID for each balance/resource
  cust_num_bigint BIGINT NOT NULL,               -- Links to customer
  object_type_id INT NOT NULL,                   -- Links to object_types
  expiry_date DATETIME NULL,                     -- Expiry date of this resource
  FOREIGN KEY (cust_num_bigint) REFERENCES customers(cust_num_bigint),
  FOREIGN KEY (object_type_id) REFERENCES object_types(object_type_id)
);

-- Offers: Details about plans, bundles, loans
select * from offers;
CREATE TABLE offers (
  offer_id INT PRIMARY KEY,                     -- Matches OFFER_ID in CDR
  offer_name VARCHAR(255) NOT NULL              -- Offer name from CDR mapping
);

-- Insert offer names from CDR documentation
INSERT INTO offers (offer_id, offer_name) VALUES
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
select * from operation_types;
CREATE TABLE operation_types (
  operation_type_id SMALLINT PRIMARY KEY,
  description VARCHAR(255) NOT NULL
);

-- Insert confirmed operation types from CDR documentation
INSERT INTO operation_types (operation_type_id, description) VALUES
(1, 'Usage - Type 1'),
(2, 'Usage - Type 2'),
(3, 'Usage - Type 3'),
(4, 'Add new Free Resource'),
(5, 'Add to existing Free Resource');

-- CDR Events: Transactional data
select * from cdr_events;
CREATE TABLE cdr_events (
  cdr_event_id BIGINT AUTO_INCREMENT PRIMARY KEY,
  cdr_id VARCHAR(255) NOT NULL,                 -- Matches CDR_ID in documentation
  creation_time DATETIME NOT NULL,               -- Event timestamp
  
  -- Dimension links
  cust_num BIGINT NOT NULL,                      -- Customer
  object_id BIGINT NOT NULL,                     -- Object/resource
  offer_id INT NULL,                             -- Associated offer
  operation_type_id SMALLINT NOT NULL,           -- Operation type
  
  -- Amount fields
  deduct_amount DECIMAL(15,2) DEFAULT 0,         -- Amount deducted
  charge_amount DECIMAL(15,2) DEFAULT 0,         -- Amount charged/added
  
  -- Balance snapshots
  current_amount_pre DECIMAL(15,2),              -- Balance before
  current_amount_post DECIMAL(15,2),             -- Balance after
  
  -- Expiry times
  charge_expire_time DATETIME NULL,              -- When charged amount expires
  current_amount_expire_time DATETIME NULL,      -- When current amount expires
  
  -- Foreign keys
  FOREIGN KEY (cust_num) REFERENCES customers(cust_num_bigint),
  FOREIGN KEY (object_id) REFERENCES objects(object_id_bigint),
  FOREIGN KEY (offer_id) REFERENCES offers(offer_id),
  FOREIGN KEY (operation_type_id) REFERENCES operation_types(operation_type_id)
);

-- Create indexes for performance
CREATE INDEX idx_cdr_creation_time ON cdr_events(creation_time);
CREATE INDEX idx_cdr_cust_num ON cdr_events(cust_num);
CREATE INDEX idx_cdr_object_id ON cdr_events(object_id);
CREATE INDEX idx_cdr_offer_id ON cdr_events(offer_id);
CREATE INDEX idx_cdr_operation_type_id ON cdr_events(operation_type_id);

CREATE INDEX idx_objects_cust_num ON objects(cust_num_bigint);
CREATE INDEX idx_objects_object_type_id ON objects(object_type_id);

USE mattel;
SELECT COUNT(*) FROM cdr_events;
SELECT MIN(creation_time), MAX(creation_time) FROM cdr_events;
SELECT offer_id, COUNT(*) FROM cdr_events GROUP BY offer_id ORDER BY COUNT(*) DESC LIMIT 5;
SELECT operation_type_id, COUNT(*) FROM cdr_events GROUP BY operation_type_id ORDER BY COUNT(*) DESC LIMIT 5;