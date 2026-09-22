create database cwpc01;
use cwpc01;
create table customers(
customerID int primary key,
name varchar(100),
email varchar(100),
city varchar(100)
);
create table orders(
orderID int primary key,
customerID int ,
orderDate date,
totalamount decimal(10,2)
);
INSERT INTO customers (customerID, name, email, city)
VALUES
(101, 'Rahul Sharma', 'rahul@gmail.com', 'Mumbai'),
(102, 'Priya Patil', 'priya@gmail.com', 'Pune'),
(103, 'Amit Shah', 'amit@gmail.com', 'Mumbai'),
(104, 'Sneha Joshi', 'sneha@gmail.com', 'Delhi'),
(105, 'Rohan Mehta', 'rohan@gmail.com', 'Pune');


INSERT INTO orders (orderID, customerID, orderDate, totalamount)
VALUES
(1, 101, '2026-08-01', 2500.50),
(2, 102, '2026-08-02', 1200.00),
(3, 103, '2026-08-03', 3500.75),
(4, 104, '2026-08-04', 1800.25),
(5, 105, '2026-08-05', 4200.00);

select * from orders;
select * from customers;

SELECT c.customerID, c.name, c.city,
       o.orderID, o.orderDate, o.totalamount
FROM customers c
INNER JOIN orders o
ON c.customerID = o.customerID;