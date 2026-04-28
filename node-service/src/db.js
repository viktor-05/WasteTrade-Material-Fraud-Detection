// Database connection pool. Single shared pool for the whole app.
const mysql = require('mysql2/promise');

const pool = mysql.createPool({
  host: process.env.DB_HOST || 'localhost',
  port: parseInt(process.env.DB_PORT || '3306', 10),
  user: process.env.DB_USER || 'root',
  password: process.env.DB_PASSWORD || '',
  database: process.env.DB_NAME || 'wastetrade',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
  // JSON columns: mysql2 parses these automatically by default. Good.
  // dateStrings: false -> get JS Date objects. Good.
});

module.exports = { pool };
