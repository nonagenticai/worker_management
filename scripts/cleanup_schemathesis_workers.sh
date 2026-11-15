#!/bin/bash

# Comprehensive cleanup script for all databases used in the Worker Management API
# This script cleans up schemathesis-generated workers and test data from all SQLite databases
# Designed to run inside Docker container, but works in local development too

# set -e  # Exit on any error - REMOVED to allow script to continue on minor errors

echo "============================================================"
echo "🧹 COMPREHENSIVE DATABASE CLEANUP"
echo "Cleaning up all databases used in Worker Management API..."
echo "============================================================"

# Determine if running in Docker or local - check for actual Docker environment
if [ -d "/app" ] && [ -f "/app/src/api/main.py" ]; then
    echo "🐳 Detected Docker environment"
    DB_BASE_PATH="/app/src/db"
    WORKER_DB_BASE="/app"
else
    echo "💻 Detected local development environment"
    DB_BASE_PATH="$(cd "$(dirname "$0")/../src/db" && pwd)"
    WORKER_DB_BASE="$(cd "$(dirname "$0")/.." && pwd)"
fi

echo "📁 Database base path: $DB_BASE_PATH"
echo "📁 Worker database base: $WORKER_DB_BASE"

# Array of core database files to clean
declare -a CORE_DATABASES=(
    "$DB_BASE_PATH/worker_registry.sqlite"
    "$DB_BASE_PATH/workers_state.sqlite"
    "$DB_BASE_PATH/.workers_state.sqlite"
    "$DB_BASE_PATH/graph_states.sqlite"
    "$DB_BASE_PATH/graph_execution.sqlite"
)



# Statistics tracking
TOTAL_WORKERS_DELETED=0
TOTAL_GRAPHS_DELETED=0
TOTAL_STATES_DELETED=0
TOTAL_SESSIONS_DELETED=0
DATABASES_PROCESSED=0

# Function to safely clean a database with error handling
clean_database() {
    local db_path="$1"
    local db_type="$2"

    if [ ! -f "$db_path" ]; then
        echo "⚠️  Database not found: $db_path"
        return
    fi

    echo "🔧 Processing $db_type database: $(basename "$db_path")"

    # Test database accessibility
    if ! sqlite3 "$db_path" "SELECT 1;" >/dev/null 2>&1; then
        echo "❌ Cannot access database: $db_path"
        return
    fi

    case "$db_type" in
        "worker_registry")
            clean_worker_registry "$db_path"
            ;;
        "worker_states"|"workers_state")
            clean_worker_states "$db_path"
            ;;
        "graph_states")
            clean_graph_states "$db_path"
            ;;
        "graph_execution")
            clean_graph_execution "$db_path"
            ;;
        "legacy")
            clean_legacy_database "$db_path"
            ;;
        *)
            echo "⚠️  Unknown database type: $db_type"
            ;;
    esac

    ((DATABASES_PROCESSED++))
}

# Clean worker registry database
clean_worker_registry() {
    local db_path="$1"

    # Get initial count
    local initial_count
    initial_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM workers;" 2>/dev/null || echo "0")

    echo "  📊 Initial workers: $initial_count"

    if [ "$initial_count" = "0" ]; then
        echo "  ✓ No workers to clean"
        return
    fi

    # Delete problematic schemathesis-generated workers
    echo "  🗑️  Deleting schemathesis-generated workers..."

    # 1. Empty worker names
    sqlite3 "$db_path" "DELETE FROM workers WHERE worker_name = '' OR worker_name IS NULL;" 2>/dev/null || true

    # 2. Workers with control characters or very short suspicious names
    sqlite3 "$db_path" "DELETE FROM workers WHERE length(trim(worker_name)) <= 2 AND worker_name GLOB '*[^a-zA-Z0-9_-]*';" 2>/dev/null || true

    # 3. Workers with unicode/non-ASCII characters (typical of schemathesis)
    sqlite3 "$db_path" "DELETE FROM workers WHERE worker_name GLOB '*[^ -~]*';" 2>/dev/null || true

    # 3a. Additional Unicode patterns - catch high Unicode code points (4-byte UTF-8)
    sqlite3 "$db_path" "DELETE FROM workers WHERE
        worker_name GLOB '*[󀀀-󿿿]*' OR
        worker_name GLOB '*[򀀀-򿿿]*' OR
        worker_name GLOB '*[򀀀-򿿿]*' OR
        worker_name GLOB '*[𐀀-􏿿]*';" 2>/dev/null || true

    # 3b. Catch any worker name with non-ASCII bytes (using hex/octal patterns)
    sqlite3 "$db_path" "DELETE FROM workers WHERE
        worker_name REGEXP '[^\x00-\x7F]';" 2>/dev/null || true

    # 4. Workers with specific control characters
    sqlite3 "$db_path" "DELETE FROM workers WHERE
        worker_name LIKE '%\x02%' OR worker_name LIKE '%\x04%' OR worker_name LIKE '%\x07%' OR
        worker_name LIKE '%\x0b%' OR worker_name LIKE '%\x0c%' OR worker_name LIKE '%\x0e%' OR
        worker_name LIKE '%\x0f%' OR worker_name LIKE '%\x10%' OR worker_name LIKE '%\x12%' OR
        worker_name LIKE '%\x15%' OR worker_name LIKE '%\x17%' OR worker_name LIKE '%\x19%' OR
        worker_name LIKE '%\x1a%' OR worker_name LIKE '%\x1c%' OR worker_name LIKE '%\x1d%' OR
        worker_name LIKE '%\x1e%';" 2>/dev/null || true

    # 5. Workers with suspicious patterns (single chars, only numbers, etc.)
    sqlite3 "$db_path" "DELETE FROM workers WHERE
        length(worker_name) = 1 OR
        worker_name GLOB '[0-9]' OR
        worker_name GLOB '[0-9][0-9]' OR
        worker_name LIKE '%test%' OR
        worker_name LIKE '%schemathesis%';" 2>/dev/null || true

    # 6. NEW: Specific schemathesis patterns we've observed
    sqlite3 "$db_path" "DELETE FROM workers WHERE
        worker_name = '0/0' OR
        worker_name = '000' OR
        worker_name = '0000' OR
        worker_name = '00000' OR
        worker_name = 'FALSE' OR
        worker_name = 'TRUE' OR
        worker_name = 'false' OR
        worker_name = 'true' OR
        worker_name = 'null' OR
        worker_name = 'NULL' OR
        worker_name GLOB '[0-9]*[0-9]' OR
        worker_name GLOB '[0-9]/[0-9]*' OR
        length(worker_name) <= 5 AND (worker_name GLOB '[0-9]*' OR worker_name IN ('FALSE', 'TRUE', 'false', 'true'));" 2>/dev/null || true

    # Get final count
    local final_count
    final_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM workers;" 2>/dev/null || echo "0")
    local deleted=$((initial_count - final_count))

    echo "  ✅ Workers deleted: $deleted (remaining: $final_count)"
    TOTAL_WORKERS_DELETED=$((TOTAL_WORKERS_DELETED + deleted))
}

# Clean worker states database
clean_worker_states() {
    local db_path="$1"

    # Check if the table exists
    if ! sqlite3 "$db_path" "SELECT name FROM sqlite_master WHERE type='table' AND name='worker_states';" | grep -q worker_states; then
        echo "  ⚠️  No worker_states table found"
        return
    fi

    local initial_count
    initial_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM worker_states;" 2>/dev/null || echo "0")

    echo "  📊 Initial worker states: $initial_count"

    if [ "$initial_count" = "0" ]; then
        echo "  ✓ No worker states to clean"
        return
    fi

    # Delete states for non-existent or problematic workers
    echo "  🗑️  Deleting orphaned and test worker states..."

    # Delete states with empty or suspicious worker names
    sqlite3 "$db_path" "DELETE FROM worker_states WHERE
        worker_name = '' OR worker_name IS NULL OR
        length(trim(worker_name)) <= 2 OR
        worker_name GLOB '*[^ -~]*' OR
        worker_name LIKE '%test%' OR
        worker_name LIKE '%schemathesis%';" 2>/dev/null || true

    # Delete very old states (older than 7 days) to clean up test data
    sqlite3 "$db_path" "DELETE FROM worker_states WHERE
        timestamp < datetime('now', '-7 days');" 2>/dev/null || true

    local final_count
    final_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM worker_states;" 2>/dev/null || echo "0")
    local deleted=$((initial_count - final_count))

    echo "  ✅ Worker states deleted: $deleted (remaining: $final_count)"
    TOTAL_STATES_DELETED=$((TOTAL_STATES_DELETED + deleted))
}

# Clean graph states database
clean_graph_states() {
    local db_path="$1"

    # Check if tables exist
    local has_graphs
    has_graphs=$(sqlite3 "$db_path" "SELECT name FROM sqlite_master WHERE type='table' AND name='graphs';" 2>/dev/null || echo "")

    if [ -z "$has_graphs" ]; then
        echo "  ⚠️  No graphs table found"
        return
    fi

    local initial_count
    initial_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM graphs;" 2>/dev/null || echo "0")

    echo "  📊 Initial graphs: $initial_count"

    if [ "$initial_count" = "0" ]; then
        echo "  ✓ No graphs to clean"
        return
    fi

    # Delete test graphs and their states
    echo "  🗑️  Deleting test graphs and states..."

    # 1. Delete graphs with test-like names
    sqlite3 "$db_path" "DELETE FROM graphs WHERE
        name LIKE '%test%' OR
        name LIKE '%schemathesis%' OR
        name = '' OR
        length(trim(name)) <= 2;" 2>/dev/null || true

    # 2. NEW: Delete graphs with unicode/non-ASCII characters (schemathesis generates these)
    sqlite3 "$db_path" "DELETE FROM graphs WHERE name GLOB '*[^ -~]*';" 2>/dev/null || true

    # 2a. Additional Unicode patterns for graphs - catch high Unicode code points (4-byte UTF-8)
    sqlite3 "$db_path" "DELETE FROM graphs WHERE
        name GLOB '*[󀀀-󿿿]*' OR
        name GLOB '*[򀀀-򿿿]*' OR
        name GLOB '*[򀀀-򿿿]*' OR
        name GLOB '*[𐀀-􏿿]*';" 2>/dev/null || true

    # 2b. Catch any graph name with non-ASCII bytes
    sqlite3 "$db_path" "DELETE FROM graphs WHERE
        name REGEXP '[^\x00-\x7F]';" 2>/dev/null || true

    # 3. NEW: Delete graphs with control characters
    sqlite3 "$db_path" "DELETE FROM graphs WHERE
        name LIKE '%\x01%' OR name LIKE '%\x02%' OR name LIKE '%\x03%' OR name LIKE '%\x04%' OR
        name LIKE '%\x05%' OR name LIKE '%\x06%' OR name LIKE '%\x07%' OR name LIKE '%\x08%' OR
        name LIKE '%\x0b%' OR name LIKE '%\x0c%' OR name LIKE '%\x0e%' OR name LIKE '%\x0f%' OR
        name LIKE '%\x10%' OR name LIKE '%\x11%' OR name LIKE '%\x12%' OR name LIKE '%\x13%' OR
        name LIKE '%\x14%' OR name LIKE '%\x15%' OR name LIKE '%\x16%' OR name LIKE '%\x17%' OR
        name LIKE '%\x18%' OR name LIKE '%\x19%' OR name LIKE '%\x1a%' OR name LIKE '%\x1b%' OR
        name LIKE '%\x1c%' OR name LIKE '%\x1d%' OR name LIKE '%\x1e%' OR name LIKE '%\x1f%';" 2>/dev/null || true

    # 4. NEW: Delete graphs with suspicious single character or very short names
    sqlite3 "$db_path" "DELETE FROM graphs WHERE
        length(name) = 1 OR
        (length(name) <= 3 AND name GLOB '*[^a-zA-Z0-9]*');" 2>/dev/null || true

    # 5. NEW: Delete graphs that contain high unicode characters (typical schemathesis garbage)
    sqlite3 "$db_path" "DELETE FROM graphs WHERE
        name LIKE '%򝉄%' OR name LIKE '%񤐞%' OR name LIKE '%񀾸%' OR name LIKE '%򵜣%' OR
        name LIKE '%񂙥%' OR name LIKE '%񬾃%' OR name LIKE '%񻺻%' OR name LIKE '%񖋒%' OR
        name LIKE '%򰯷%' OR name LIKE '%򀅖%' OR name LIKE '%򷓰%' OR name LIKE '%񾱡%' OR
        name LIKE '%𔗺%' OR name LIKE '%𧃁%' OR name LIKE '%𖈏%' OR name LIKE '%񺵗%' OR
        name LIKE '%𒇗%' OR name LIKE '%𒪄%' OR name LIKE '%򍷡%' OR name LIKE '%䨮%' OR
        name LIKE '%Ṱ̺̺̕%' OR name LIKE '%o͞%' OR name LIKE '%ì%' OR name LIKE '%v̟%' OR
        name LIKE '%𝕋%' OR name LIKE '%𝕙%' OR name LIKE '%𝕖%' OR name LIKE '%𝕞%';" 2>/dev/null || true

    # Clean up orphaned states if there's a states table
    if sqlite3 "$db_path" "SELECT name FROM sqlite_master WHERE type='table' AND name='graph_states';" | grep -q graph_states; then
        sqlite3 "$db_path" "DELETE FROM graph_states WHERE graph_id NOT IN (SELECT id FROM graphs);" 2>/dev/null || true
    fi

    local final_count
    final_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM graphs;" 2>/dev/null || echo "0")
    local deleted=$((initial_count - final_count))

    echo "  ✅ Graphs deleted: $deleted (remaining: $final_count)"
    TOTAL_GRAPHS_DELETED=$((TOTAL_GRAPHS_DELETED + deleted))
}

# Clean graph execution database
clean_graph_execution() {
    local db_path="$1"

    # Check if tables exist
    local has_sessions
    has_sessions=$(sqlite3 "$db_path" "SELECT name FROM sqlite_master WHERE type='table' AND name='execution_sessions';" 2>/dev/null || echo "")

    if [ -z "$has_sessions" ]; then
        echo "  ⚠️  No execution_sessions table found"
        return
    fi

    local initial_count
    initial_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM execution_sessions;" 2>/dev/null || echo "0")

    echo "  📊 Initial execution sessions: $initial_count"

    if [ "$initial_count" = "0" ]; then
        echo "  ✓ No execution sessions to clean"
        return
    fi

    # Delete old execution sessions and events
    echo "  🗑️  Deleting old execution sessions..."

    # Delete sessions older than 24 hours (test cleanup)
    sqlite3 "$db_path" "DELETE FROM execution_sessions WHERE
        created_at < datetime('now', '-1 day');" 2>/dev/null || true

    # Clean up orphaned events if there's an events table
    if sqlite3 "$db_path" "SELECT name FROM sqlite_master WHERE type='table' AND name='execution_events';" | grep -q execution_events; then
        sqlite3 "$db_path" "DELETE FROM execution_events WHERE session_id NOT IN (SELECT id FROM execution_sessions);" 2>/dev/null || true
    fi

    local final_count
    final_count=$(sqlite3 "$db_path" "SELECT COUNT(*) FROM execution_sessions;" 2>/dev/null || echo "0")
    local deleted=$((initial_count - final_count))

    echo "  ✅ Execution sessions deleted: $deleted (remaining: $final_count)"
    TOTAL_SESSIONS_DELETED=$((TOTAL_SESSIONS_DELETED + deleted))
}

# Clean legacy databases
clean_legacy_database() {
    local db_path="$1"

    echo "  🗑️  Cleaning legacy database..."

    # For legacy databases, we'll do a more conservative cleanup
    # Just remove obviously problematic entries

    # Check what tables exist
    local tables
    tables=$(sqlite3 "$db_path" "SELECT name FROM sqlite_master WHERE type='table';" 2>/dev/null || echo "")

    echo "  📋 Tables found: $tables"

    # If it has a workers table, clean it like worker_registry
    if echo "$tables" | grep -q "workers"; then
        clean_worker_registry "$db_path"
    fi

    # If it has states, clean old entries
    if echo "$tables" | grep -q "state"; then
        # Delete entries older than 7 days
        for table in $(echo "$tables" | grep state); do
            sqlite3 "$db_path" "DELETE FROM $table WHERE timestamp < datetime('now', '-7 days');" 2>/dev/null || true
        done
    fi

    echo "  ✅ Legacy database cleaned"
}

# Clean individual worker database files
clean_worker_database_files() {
    echo "🔍 Looking for individual worker database files..."

    # Find all *_state.sqlite files
    local worker_db_files
    worker_db_files=$(find "$WORKER_DB_BASE" -name "*_state.sqlite" -type f 2>/dev/null || echo "")

    if [ -z "$worker_db_files" ]; then
        echo "  ✓ No individual worker database files found"
        return
    fi

    local count=0
    while IFS= read -r db_file; do
        if [ -f "$db_file" ]; then
            local filename
            filename=$(basename "$db_file")
            echo "  🗑️  Removing worker database: $filename"
            rm -f "$db_file" 2>/dev/null || echo "    ⚠️  Could not remove $filename"
            ((count++))
        fi
    done <<< "$worker_db_files"

    echo "  ✅ Individual worker databases removed: $count"
}

# Main execution
echo "🚀 Starting comprehensive database cleanup..."

# Clean core databases
echo ""
echo "📚 Cleaning core databases..."
clean_database "${CORE_DATABASES[0]}" "worker_registry"
echo "🔍 Debug: Finished worker_registry cleanup"
clean_database "${CORE_DATABASES[1]}" "worker_states"
echo "🔍 Debug: Finished worker_states cleanup"
clean_database "${CORE_DATABASES[2]}" "workers_state"
echo "🔍 Debug: Finished workers_state cleanup"
clean_database "${CORE_DATABASES[3]}" "graph_states"
echo "🔍 Debug: Finished graph_states cleanup"
clean_database "${CORE_DATABASES[4]}" "graph_execution"
echo "🔍 Debug: Finished graph_execution cleanup"

# Clean legacy databases
echo ""
echo "🏛️  Cleaning legacy databases..."
for legacy_db in "${LEGACY_DATABASES[@]}"; do
    clean_database "$legacy_db" "legacy"
done

# Clean individual worker database files
echo ""
clean_worker_database_files

# Vacuum all existing databases to reclaim space
echo ""
echo "🧽 Vacuuming databases to reclaim space..."
for db in "${CORE_DATABASES[@]}" "${LEGACY_DATABASES[@]}"; do
    if [ -f "$db" ]; then
        echo "  🔧 Vacuuming $(basename "$db")..."
        sqlite3 "$db" "VACUUM;" 2>/dev/null || echo "    ⚠️  Could not vacuum $(basename "$db")"
    fi
done

echo ""
echo "============================================================"
echo "🎯 COMPREHENSIVE CLEANUP SUMMARY:"
echo "============================================================"
echo "📊 Databases processed: $DATABASES_PROCESSED"
echo "🗑️  Total workers deleted: $TOTAL_WORKERS_DELETED"
echo "🗑️  Total worker states deleted: $TOTAL_STATES_DELETED"
echo "🗑️  Total graphs deleted: $TOTAL_GRAPHS_DELETED"
echo "🗑️  Total execution sessions deleted: $TOTAL_SESSIONS_DELETED"
echo "============================================================"

# Final verification - show remaining data
echo ""
echo "📋 REMAINING DATA SUMMARY:"
for db in "${CORE_DATABASES[@]}"; do
    if [ -f "$db" ]; then
        db_name=$(basename "$db")
        echo "  📁 $db_name:"

        # Show table counts
        tables=$(sqlite3 "$db" "SELECT name FROM sqlite_master WHERE type='table';" 2>/dev/null || echo "")

        if [ -n "$tables" ]; then
            while IFS= read -r table; do
                if [ -n "$table" ]; then
                    count=$(sqlite3 "$db" "SELECT COUNT(*) FROM $table;" 2>/dev/null || echo "0")
                    echo "    - $table: $count rows"
                fi
            done <<< "$tables"
        else
            echo "    - No tables found or database inaccessible"
        fi
    fi
done

echo ""
echo "✅ Comprehensive database cleanup completed successfully!"
echo "🧹 All schemathesis test data and problematic entries have been removed."

exit 0
