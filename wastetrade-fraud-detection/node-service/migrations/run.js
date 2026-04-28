/**
 * Migration runner. Reads every .sql file in this directory in alphabetical
 * order and runs it. Idempotent because all our DDL uses IF NOT EXISTS.
 */
require('dotenv').config();
const fs = require('fs');
const path = require('path');
const mysql = require('mysql2/promise');

async function main() {
  const conn = await mysql.createConnection({
    host: process.env.DB_HOST || 'localhost',
    port: parseInt(process.env.DB_PORT || '3306', 10),
    user: process.env.DB_USER || 'root',
    password: process.env.DB_PASSWORD || '',
    database: process.env.DB_NAME || 'wastetrade',
    multipleStatements: true,
  });

  const dir = __dirname;
  const files = fs.readdirSync(dir)
    .filter(f => f.endsWith('.sql'))
    .sort();

  for (const file of files) {
    const sql = fs.readFileSync(path.join(dir, file), 'utf8');
    console.log(`-> running ${file}`);
    await conn.query(sql);
  }

  await conn.end();
  console.log('migrations complete');
}

main().catch(err => {
  console.error('migration failed:', err);
  process.exit(1);
});
