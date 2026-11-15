# PostgreSQL Setup Guide for m1.local

**Quick reference for the developer setting up the database on m1.local**

---

## Prerequisites

- PostgreSQL 12+ installed on m1.local
- Access to PostgreSQL superuser account (usually `postgres`)

---

## Step 1: Create Database

```bash
# Connect as postgres superuser
psql -U postgres

# Or on macOS with Homebrew:
psql postgres
```

```sql
-- Create the database
CREATE DATABASE worker_management;

-- Verify creation
\l worker_management
```

---

## Step 2: Create User

```sql
-- Create user with password
CREATE USER worker_mgmt_user WITH PASSWORD 'your_secure_password_here';

-- Grant database access
GRANT ALL PRIVILEGES ON DATABASE worker_management TO worker_mgmt_user;

-- Connect to the new database
\c worker_management

-- Grant schema permissions
GRANT ALL ON SCHEMA public TO worker_mgmt_user;

-- Grant permissions on future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO worker_mgmt_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO worker_mgmt_user;

-- Exit
\q
```

---

## Step 3: Run Schema Script

```bash
# From the directory containing postgres_schema.sql
psql -U worker_mgmt_user -d worker_management -f postgres_schema.sql

# If prompted for password, enter the password you set above
```

**Expected output:**
```
CREATE TYPE
CREATE TABLE
CREATE TABLE
... (multiple CREATE statements)
CREATE INDEX
... (multiple index creations)
CREATE FUNCTION
CREATE TRIGGER
... (multiple trigger creations)
INSERT 0 1
```

---

## Step 4: Grant Permissions (Post-Schema)

Since tables now exist, grant explicit permissions:

```bash
psql -U postgres -d worker_management
```

```sql
-- Grant permissions on all existing tables
GRANT ALL ON ALL TABLES IN SCHEMA public TO worker_mgmt_user;

-- Grant permissions on all sequences (for BIGSERIAL columns)
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO worker_mgmt_user;

-- Verify grants
\dp workers
\dp worker_states

-- Exit
\q
```

---

## Step 5: Verify Setup

### Test 1: Connection
```bash
psql -U worker_mgmt_user -d worker_management -c "SELECT version();"
```

**Expected:** PostgreSQL version string

### Test 2: List Tables
```bash
psql -U worker_mgmt_user -d worker_management -c "\dt"
```

**Expected:** List of 11 tables

### Test 3: Check Schema Migration
```bash
psql -U worker_mgmt_user -d worker_management -c "SELECT * FROM schema_migrations;"
```

**Expected:**
```
 id | version |                    description                     |        applied_at
----+---------+----------------------------------------------------+---------------------------
  1 | 1.0.0   | Initial PostgreSQL schema for worker management... | 2025-11-07 12:00:00+00
```

### Test 4: Insert Test Worker
```bash
psql -U worker_mgmt_user -d worker_management
```

```sql
-- Insert test worker
INSERT INTO workers (worker_id, worker_name, is_agent, config)
VALUES (
    gen_random_uuid(),
    'test_worker',
    true,
    '{"model": "mlx", "temperature": 0.1}'::jsonb
);

-- Verify insertion
SELECT worker_id, worker_name, config FROM workers WHERE worker_name = 'test_worker';

-- Clean up test data
DELETE FROM workers WHERE worker_name = 'test_worker';

-- Exit
\q
```

---

## Step 6: Configure PostgreSQL for Remote Access

### Edit postgresql.conf
```bash
# Find config file location
psql -U postgres -c "SHOW config_file;"

# Edit the file (path may vary)
sudo nano /var/lib/postgresql/data/postgresql.conf
# Or on macOS Homebrew:
nano /opt/homebrew/var/postgresql@14/postgresql.conf
```

**Add/modify:**
```conf
listen_addresses = '*'  # or 'localhost,m1.local'
max_connections = 100
```

### Edit pg_hba.conf
```bash
# Find HBA file location
psql -U postgres -c "SHOW hba_file;"

# Edit the file
sudo nano /var/lib/postgresql/data/pg_hba.conf
# Or on macOS:
nano /opt/homebrew/var/postgresql@14/pg_hba.conf
```

**Add at the end:**
```conf
# Allow connections from the API server
# Replace <api_server_ip> with actual IP or use 0.0.0.0/0 for any IP (less secure)
host    worker_management    worker_mgmt_user    <api_server_ip>/32    md5

# For testing, you can temporarily allow all IPs:
# host    worker_management    worker_mgmt_user    0.0.0.0/0    md5
```

### Restart PostgreSQL
```bash
# Ubuntu/Debian
sudo systemctl restart postgresql

# macOS Homebrew
brew services restart postgresql@14

# Or generic:
pg_ctl restart -D /path/to/data
```

---

## Step 7: Test Remote Connection

**From the API server machine** (not m1.local):

```bash
psql -h m1.local -U worker_mgmt_user -d worker_management -c "SELECT COUNT(*) FROM workers;"
```

**Expected:** Connection succeeds and returns `0` (or number of workers)

If connection fails, check:
1. Firewall on m1.local allows port 5432
2. PostgreSQL is listening on the correct interface
3. pg_hba.conf allows the API server's IP
4. Network connectivity: `ping m1.local`

---

## Step 8: Security Recommendations

1. **Strong Password**: Use a complex password for `worker_mgmt_user`
2. **Restrict IP Access**: In `pg_hba.conf`, specify exact API server IP instead of `0.0.0.0/0`
3. **SSL/TLS**: Consider enabling SSL for encrypted connections
4. **Regular Backups**: Set up automated backups
5. **Monitoring**: Enable PostgreSQL logging and monitoring

### Enable Logging (Optional)
```bash
# Edit postgresql.conf
nano /path/to/postgresql.conf
```

```conf
logging_collector = on
log_directory = 'pg_log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_statement = 'all'  # Or 'ddl' for less verbose logging
log_connections = on
log_disconnections = on
```

Restart PostgreSQL after changes.

---

## Step 9: Create Backup Script (Optional)

```bash
# Create backup script
cat > /usr/local/bin/backup-worker-db.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/var/backups/postgresql"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p $BACKUP_DIR

pg_dump -U worker_mgmt_user -d worker_management \
    -F c -b -v -f $BACKUP_DIR/worker_management_$TIMESTAMP.backup

# Keep only last 7 days of backups
find $BACKUP_DIR -name "worker_management_*.backup" -mtime +7 -delete

echo "Backup completed: worker_management_$TIMESTAMP.backup"
EOF

# Make executable
chmod +x /usr/local/bin/backup-worker-db.sh

# Test backup
/usr/local/bin/backup-worker-db.sh

# Add to cron for daily backups at 2 AM
(crontab -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/backup-worker-db.sh") | crontab -
```

---

## Troubleshooting

### Issue: "FATAL: database 'worker_management' does not exist"
**Solution:** Run Step 1 to create the database

### Issue: "FATAL: role 'worker_mgmt_user' does not exist"
**Solution:** Run Step 2 to create the user

### Issue: "ERROR: permission denied for schema public"
**Solution:** Run the GRANT commands in Step 2 and Step 4

### Issue: "could not connect to server: Connection refused"
**Solution:**
- Check if PostgreSQL is running: `systemctl status postgresql` or `brew services list`
- Check `postgresql.conf` for `listen_addresses`
- Check firewall settings

### Issue: "no pg_hba.conf entry for host"
**Solution:** Add appropriate entry to `pg_hba.conf` (Step 6)

---

## Quick Reference

### Database Details
- **Host**: m1.local
- **Port**: 5432 (default)
- **Database**: worker_management
- **User**: worker_mgmt_user
- **Password**: [Set in Step 2]

### Connection String
```
postgresql://worker_mgmt_user:password@m1.local:5432/worker_management
```

### Useful Commands
```bash
# Connect to database
psql -h m1.local -U worker_mgmt_user -d worker_management

# List databases
psql -U postgres -l

# List tables in worker_management
psql -U worker_mgmt_user -d worker_management -c "\dt"

# Check table sizes
psql -U worker_mgmt_user -d worker_management -c "
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
"

# Check active connections
psql -U postgres -c "SELECT * FROM pg_stat_activity WHERE datname = 'worker_management';"
```

---

## Next Steps

Once setup is complete:
1. **Share credentials** with the API developer (use secure method)
2. **Test remote connection** from the API server
3. **Monitor logs** for any connection issues
4. **Set up monitoring** (optional but recommended)

---

**Setup complete! The database is ready for the application to connect.**
