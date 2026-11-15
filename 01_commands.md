uv run pytest . --json-report --json-report-file=report.json

find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

docker compose down --remove-orphans && docker compose up --build -d --force-recreate

ruff check --select PLR1714 --fix --unsafe-fixes

docker compose down --remove-orphans && docker compose up --build -d --force-recreate

uv run ty check src/ > zz_ty_results.txt


schemathesis run --checks=all --max-response-time=30000 http://localhost:8887/openapi.json > ./schemathesis_results.txt


worker
schemathesis run --checks=all --max-response-time=30000 http://localhost:8887/openapi.json > schemathesis_results.txt

use 45 subagents to work on the final items \                              │
│      The remaining 18% is primarily:                                         │
│       1. Security middleware rate limiting (still active despite disabling)  │
│       2. Schema documentation gaps (missing 405 responses)                   │
│       3. Edge case validation strictness (graph validation)

isort -c -df -vb .

get to 100% examples 100% coverage and 100% fuzzing pass @schemathesis_results.txt and fix all failures

starting with source files and only after that test files, scan the codebase for "(todo|fixme|hack|temporary|placeholder|for now|workaround|stub|mock|dummy)" statements and and NotImplemented, pass statements, and ellipsis in comments, and look for "# For now" for example, take the first file you find this in, analyze what the statement is about, and if it can and should be implemented. Stop when you found the first item, inspect it to see if you could implement it right now, if not, continue your search to find the next file. Implement it if it makes sense and ensure the code is type safe (pydantic v2, v3-ready), would pass mypy and pylint inspection, has meaningful log statements, docstrings and comments, has no stub implementations, has no todo items, has no future tasks. Only work on the first file you find and the implementation in other files that might be needed on the item you are working on! Validate that all code you write is type safe (pydantic v2, v3-ready), would pass mypy and pylint inspection, has meaningful log statements, docstrings and comments, has no stub implementations, has no todo items, has no future tasks.
  - Type-safe: Uses proper type annotations with Literal types for model providers
  - Comprehensive: Validates model provider, creates PydanticAI agent, executes tasks, and handles errors
  - Well-documented: Includes detailed docstring and meaningful log statements
  - No placeholders: Complete implementation with proper error handling and result formatting
  - Pydantic v2 compliant: Uses modern Pydantic patterns and ConfigDict
  - Passes validation: MyPy and PyLint compliant code

python code_quality_scanner.py --save zz_quality.txt


  "Analyze the quality issues in @zz_quality.txt and systematically address them in priority order:

  1. **Critical Issues First**: Start with errors, then warnings, then info
  2. **File-by-File Approach**: Process one complete file at a time to avoid context switching
  3. **Batch Similar Issues**: Group similar issues (like trailing whitespace) for efficient resolution
  4. **Vulture Caution**: Vulture often flags false positives for unused code. ALWAYS verify before removing:
     - Check if classes/functions are used via dynamic imports, metaclasses, or reflection
     - Search the entire codebase for string references to the name
     - Look for decorator usage, plugin systems, or configuration-driven instantiation
     - When uncertain, mark as potentially unused but DO NOT remove
  5. **Progress Tracking**: Use TodoWrite to track which files/issues you've completed
  6. **Verification**: After each file, run relevant linters to confirm fixes work
  7. **Documentation**: For each file processed, briefly note what types of issues were resolved

  Focus on the first file with the worst health score and work systematically. Aim to completely resolve 3-5
  files per session rather than partially addressing many files. This will show measurable progress in the
  overall codebase health."

  Key addition:
  - Vulture verification protocol to prevent accidentally removing code that's used dynamically or through
  non-obvious patterns
  - Conservative approach when dealing with "unused" code warnings
  - Search methodology to verify actual usage before removal


Quick scan @zz_quality.txt to identify files with issues, then immediately launch parallel subagents:

  **Code Improvement Agents**
  1. FAST SCAN (DO NOT read entire file):
    - Read just enough to identify unique file paths with health scores
    - Skip node_modules/* and code_quality_scanner.py
    - Prioritize non-test files first (tests assumed outdated)
    - Stop reading once you have 10-15 files to process
  2. IMMEDIATE SUBAGENT LAUNCH:
    - Launch Task subagents in parallel for identified files
    - Each subagent gets this focused prompt:
    Fix all quality issues in [filename]:
  - Read the specific file section from zz_quality.txt
  - Fix issues by severity: errors > warnings > info
  - For vulture 'unused' warnings: search codebase for usage before removing
  - Run relevant linter on file when done
  - Report: issues fixed vs issues kept with reasons
  3. NO ANALYSIS PARALYSIS:
    - Don't categorize or analyze patterns
    - Don't create elaborate plans
    - Just identify files → launch subagents → move on

  **Codebase Integrity and Completeness**
  You are responsible for checking the entire codebase for architectural integrity and code completeness. Work with the 8 code improvement subagents and add a nineth subagent that has the task to ensure you regularly (every 5 mins) run this command to ensure that the docker containers compose "docker compose down --remove-orphans && docker compose up --build -d --force-recreate" and then check the containers that are started from the @docker-compose.yml


  **Code Quality Improvement Protocol**
  - Type-safe: Uses proper type annotations with Literal types for model providers
  - Comprehensive: Validates model provider, creates PydanticAI agent, executes tasks, and handles errors
  - Well-documented: Includes detailed docstring and meaningful log statements
  - No placeholders: Complete implementation with proper error handling and result formatting
  - Pydantic v2 compliant: Uses modern Pydantic patterns and ConfigDict
  - Passes validation: MyPy and PyLint compliant code

Start NOW by reading just enough of zz_quality.txt to get 8 file paths, then immediately launch 8 parallel
  subagents. keep test files for last, ignore node_modules/* and code_quality_scanner.py
  Launch the 9th subagent to ensure the docker containers are running and the codebase is complete.
  Launch the 10th subagent to ensure the architecture of the codebase is clean and complete and have this subagent inform the 8 subagents to ensure they are not working on the same files and that they are working with the correct architecture and software design and data models.




Usage:
  # Scan current directory
  ./code_quality_scanner.py

  # Scan specific path
  ./code_quality_scanner.py /path/to/codebase

  # Exclude specific tools
  ./code_quality_scanner.py --exclude bandit safety

  # Save report to file
  ./code_quality_scanner.py --save report.txt

  # JSON output
  ./code_quality_scanner.py --output json

python code_quality_scanner.py --save zz_quality.txt
inspect and improve the codebase as inspected by @zz_quality.txt

# Prune old images
docker system prune -f

# Stop all services
docker-compose down --volumes --remove-orphans

# Remove the problematic dynamic container if it still exists
docker rm -f code_graph_helper_container 2>/dev/null || true

# Rebuild with the fixed docker_service.py
docker-compose build --no-cache api

# Start services
docker-compose up -d

# Monitor for success
docker-compose logs -f api | grep -E "(Docker|ERROR|WARNING)"
