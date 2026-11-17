# Worker to Agent Registry Migration Script

## Overview

The migration script `/Users/mini2/Movies/worker_management/scripts/migrate_to_agent_registry.py` was created to populate skills and workload information for existing workers in the Worker Registry database.

## What It Does

The migration script performs the following operations for each worker:

1. **Reads Workers**: Fetches all existing workers from the `WorkerRegistryDB`
2. **Extracts Skills**:
   - Parses the `config` JSON to extract `tool_names` and `agent_type`
   - Maps each `tool_name` to a skill with "intermediate" level
   - Maps `agent_type` capabilities to additional skills based on predefined capability mappings
3. **Initializes Workload**:
   - Sets `capacity` to 5 (max concurrent tasks)
   - Sets `current_tasks` to 0
4. **Sets Status**: Workers get `operational_status='offline'` (will be updated on first heartbeat)

## Database Tables Created

The script creates two new tables in the Worker Registry database:

### worker_skills
```sql
CREATE TABLE worker_skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT NOT NULL,
    skill_name TEXT NOT NULL,
    level TEXT NOT NULL DEFAULT 'intermediate',
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    UNIQUE(worker_id, skill_name)
)
```

### worker_workload
```sql
CREATE TABLE worker_workload (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    worker_id TEXT NOT NULL UNIQUE,
    capacity INTEGER NOT NULL DEFAULT 5,
    current_tasks INTEGER NOT NULL DEFAULT 0,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL
)
```

## Agent Type Capability Mappings

The script includes intelligent capability mappings for different agent types:

| Agent Type | Capabilities |
|------------|--------------|
| `pydantic_ai` | data_validation, schema_design, type_checking, python_programming |
| `claude_code` | code_generation, code_analysis, debugging, refactoring, documentation |
| `codex` | code_completion, code_translation, algorithm_design, code_optimization |
| `langchain` | chain_building, prompt_engineering, llm_integration, data_processing |
| `openai` | text_generation, conversation, summarization, question_answering |
| `generic` | task_execution, data_processing |

## Usage

### Basic Migration
```bash
python scripts/migrate_to_agent_registry.py
```

### Dry Run (Preview Changes)
```bash
python scripts/migrate_to_agent_registry.py --dry-run
```

### Force Re-migration
```bash
python scripts/migrate_to_agent_registry.py --force
```

### Verbose Output
```bash
python scripts/migrate_to_agent_registry.py --verbose
```

### Combined Options
```bash
python scripts/migrate_to_agent_registry.py --dry-run --verbose
python scripts/migrate_to_agent_registry.py --force --verbose
```

## Features

### Idempotent
- By default, skips workers that have already been migrated
- Checks for existing skills and workload before migration
- Use `--force` flag to re-migrate already migrated workers

### Dry Run Mode
- Preview what would be migrated without making changes
- Shows exactly which workers would be affected
- Displays skill counts and workload initialization status

### Error Handling
- Gracefully handles malformed config JSON
- Continues migration even if individual workers fail
- Provides detailed error reporting in the summary

### Comprehensive Logging
- INFO level by default
- DEBUG level with `--verbose` flag
- Summary report at the end showing:
  - Total workers processed
  - Workers successfully migrated
  - Workers skipped
  - Workers that failed
  - Total skills added

## Implementation Details

### Database Classes (Inline)

The script includes inline implementations of `SkillsDB` and `WorkloadDB` classes:

#### SkillsDB
- `init_table()`: Creates the worker_skills table
- `add_skill(worker_id, skill_name, level)`: Adds/updates a skill
- `get_skills(worker_id)`: Retrieves all skills for a worker
- `delete_worker_skills(worker_id)`: Removes all skills for a worker

#### WorkloadDB
- `init_table()`: Creates the worker_workload table
- `init_workload(worker_id, capacity, current_tasks)`: Initializes/updates workload
- `get_workload(worker_id)`: Retrieves workload information
- `delete_worker_workload(worker_id)`: Removes workload for a worker

### Pydantic Models

The script uses Pydantic models for type safety:

```python
class SkillLevel(BaseModel):
    skill_name: str
    level: str = "intermediate"  # beginner, intermediate, expert


class WorkloadInfo(BaseModel):
    capacity: int = 5
    current_tasks: int = 0
```

## Example Output

### Dry Run
```
======================================================================
Starting worker migration to agent registry
======================================================================
DRY RUN MODE: No changes will be made
Found 7 worker(s) to migrate
[DRY RUN] Migrated data_processor: 7 skills, workload initialized
[DRY RUN] Migrated kanban_management_agent: 4 skills, workload initialized
[DRY RUN] Migrated glm_mcp_agent: 4 skills, workload initialized
[DRY RUN] Migrated my_agent: 5 skills, workload initialized
[DRY RUN] Migrated github_only_agent: 4 skills, workload initialized
[DRY RUN] Migrated usage_metrics_test: 4 skills, workload initialized
[DRY RUN] Migrated claude_code_agent: 5 skills, workload initialized
======================================================================
Migration Summary
======================================================================
Total workers: 7
Migrated: 7
Skipped: 0
Failed: 0
Total skills added: 33
======================================================================
DRY RUN completed. Run without --dry-run to apply changes.
```

### Actual Migration
```
======================================================================
Starting worker migration to agent registry
======================================================================
Found 7 worker(s) to migrate
Migrated data_processor: 7 skills, workload initialized
Migrated kanban_management_agent: 4 skills, workload initialized
...
======================================================================
Migration Summary
======================================================================
Total workers: 7
Migrated: 7
Skipped: 0
Failed: 0
Total skills added: 33
======================================================================
Migration completed successfully!
```

## Exit Codes

- `0`: Success (all workers migrated successfully)
- `1`: Failure (one or more workers failed to migrate)

## Dependencies

- Python 3.12+
- aiosqlite
- pydantic >= 2.0
- asyncio

## Notes

- The script is safe to run multiple times (idempotent by default)
- Database tables are created automatically if they don't exist
- The `operational_status` column is added to the workers table if missing
- Skill deduplication is handled automatically (unique constraint on worker_id + skill_name)
- All timestamps are stored in UTC ISO format

## Testing

The script has been tested with:
- 7 existing workers in the database
- Various agent types (pydantic_ai, claude_code, generic)
- Multiple tool configurations
- Dry run mode verification

## Future Enhancements

Potential improvements for future versions:
- Support for custom skill levels per tool
- Batch processing for large worker counts
- Rollback functionality
- Export/import of skill mappings
- Integration with worker heartbeat system
- Real-time operational status updates
