# Worker Management API Endpoint Testing

This directory contains a comprehensive test script for all critical worker management endpoints.

## Test Script: `test_worker_endpoints.py`

### What it tests:

1. **Create a worker** - `POST /workers`
2. **List workers** - `GET /workers`
3. **Get worker status** - `GET /workers/{worker_name}`
4. **Update worker** - `PUT /workers/{worker_name}`
5. **Delete individual worker** - `DELETE /workers/{worker_name}`
6. **Delete all workers** - Custom implementation (lists all workers and deletes each one)
7. **Get worker states** - `GET /workers/{worker_name}/states`
8. **Delete worker states** - `DELETE /workers/{worker_name}/states`
9. **Delete all worker states** - `DELETE /workers/states`

### Requirements:

```bash
pip install aiohttp
```

Or install from the requirements file:

```bash
pip install -r test_requirements.txt
```

### Usage:

1. **Start your Worker Management API server** (default: http://localhost:8000)

2. **Run the test script:**

```bash
python test_worker_endpoints.py
```

Or if you want to make it executable:

```bash
chmod +x test_worker_endpoints.py
./test_worker_endpoints.py
```

### Configuration:

You can modify the `BASE_URL` variable at the top of the script if your API is running on a different host/port:

```python
BASE_URL = "http://localhost:8000"  # Change this if needed
```

### Output:

The script provides:
- **Real-time feedback** during test execution
- **Detailed error messages** for failed tests
- **Response data** for successful tests
- **Summary report** at the end showing pass/fail counts

### Example Output:

```
🚀 Starting Worker Management API Endpoint Tests
============================================================

✅ PASS: Health Check
    Details: {
      "status": "healthy",
      "service": "worker-management-api"
    }

✅ PASS: Create Worker
    Details: {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "name": "test_worker_1703123456",
      "status": "created",
      "is_agent": true
    }

...

============================================================
📊 TEST SUMMARY
============================================================
Total Tests: 10
✅ Passed: 8
❌ Failed: 2
Success Rate: 80.0%

❌ FAILED TESTS:
  • Create Worker: Status 422: API key required for the specified model provider

✅ PASSED TESTS:
  • Health Check
  • List Workers
  • Get Worker Status
  • Update Worker
  • Get Worker States
  • Delete Worker States
  • Delete All Worker States
  • Delete Individual Worker

============================================================
🏁 Test run completed!
```

### Notes:

- The script creates a unique test worker name using a timestamp to avoid conflicts
- All tests are run in a logical order (create → read → update → delete)
- The "Delete All Workers" test is implemented by listing all workers and deleting each individually since there's no dedicated endpoint
- Failed tests don't stop execution - all tests will run regardless
- The script is designed to be safe and will clean up test data

### Troubleshooting:

- **Connection refused**: Make sure your API server is running on the expected port
- **422 Validation errors**: Check if you have required API keys set (like OPENAI_API_KEY)
- **Timeout errors**: The script has a 30-second timeout per request - increase it if needed
