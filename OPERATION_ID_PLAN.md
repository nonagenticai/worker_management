# Operation ID Plan: Worker Management — VERIFY ONLY

## Status: CLEAN (has operation_ids)

## Files
- `src/api/workers/crud_endpoints.py`
- `src/api/workers/state_endpoints.py`
- `src/api/workers/message_endpoints.py`
- `src/api/workers/model_registry_endpoints.py`
- `src/api/workers/convenience_endpoints.py`
- `src/api/graph_management_api.py`
- `src/api/graph_execution_api.py`
- `src/api/worker_states.py`

## Current operation_ids (~40 total)

### Worker CRUD
| operation_id | Method | Path |
|--------------|--------|------|
| `create_worker` | POST | /api/workers |
| `list_workers` | GET | /api/workers |
| `list_worker_types` | GET | /api/workers/types |
| `get_worker` | GET | /api/workers/{id} |
| `update_worker` | PUT | /api/workers/{id} |
| `delete_worker` | DELETE | /api/workers/{id} |
| `delete_all_workers` | DELETE | /api/workers |

### Worker States
| operation_id | Method | Path |
|--------------|--------|------|
| `list_all_worker_states` | GET | /api/workers/states |
| `delete_all_worker_states` | DELETE | /api/workers/states |
| `get_worker_states` | GET | /api/workers/{id}/states |
| `get_latest_worker_state` | GET | /api/workers/{id}/states/latest |
| `delete_worker_states` | DELETE | /api/workers/{id}/states |
| `cleanup_worker_instance` | POST | /api/workers/{id}/cleanup |
| `list_worker_states` | GET | /api/worker-states |
| `create_worker_states` | POST | /api/worker-states |

### Worker Messages
| operation_id | Method | Path |
|--------------|--------|------|
| `send_worker_message` | POST | /api/workers/{id}/message |

### Model Registry
| operation_id | Method | Path |
|--------------|--------|------|
| `list_mlx_models` | GET | /api/models/mlx |
| `refresh_mlx_models` | POST | /api/models/mlx/refresh |
| `list_inference_router_models` | GET | /api/models/inference-router |
| `refresh_inference_router_models` | POST | /api/models/inference-router/refresh |

### Convenience
| operation_id | Method | Path |
|--------------|--------|------|
| `create_pydantic_ai_worker` | POST | /api/workers/pydantic-ai |
| `create_claude_code_worker` | POST | /api/workers/claude-code |
| `create_codex_worker` | POST | /api/workers/codex |
| `create_remote_claude_worker` | POST | /api/workers/remote/claude-code |
| `create_remote_codex_worker` | POST | /api/workers/remote/codex |

### Graph Management
| operation_id | Method | Path |
|--------------|--------|------|
| `create_graph` | POST | /api/graphs |
| `list_graphs` | GET | /api/graphs |
| `get_graph` | GET | /api/graphs/{id} |
| `update_graph` | PUT | /api/graphs/{id} |
| `delete_graph` | DELETE | /api/graphs/{id} |
| `delete_all_graphs` | DELETE | /api/graphs |
| `get_graph_states` | GET | /api/graphs/{id}/states |
| `delete_graph_states` | DELETE | /api/graphs/{id}/states |
| `delete_all_graph_states` | DELETE | /api/graphs/states |
| `get_graph_structure` | GET | /api/graphs/{id}/structure |
| `validate_graph` | POST | /api/graphs/{id}/validate |

### Graph Execution
| operation_id | Method | Path |
|--------------|--------|------|
| `execute_graph` | POST | /api/graphs/{id}/execute |
| `execute_graph_stepwise` | POST | /api/graphs/{id}/execute/stepwise |
| `get_session_status` | GET | /api/sessions/{id} |
| `execution_health` | GET | /api/execution/health |

### Workflow Routing (`src/api/workflow_routing/routes.py`)
| operation_id | Method | Path |
|--------------|--------|------|
| `workflow_routing_active` | GET | /api/workflow-routing/active |
| `workflow_routing_observed` | GET | /api/workflow-routing/observed |
| `workflow_routing_promote` | POST | /api/workflow-routing/promote |
| `workflow_routing_rollback` | POST | /api/workflow-routing/rollback |
| `workflow_routing_extend` | POST | /api/workflow-routing/extend |
| `workflow_routing_clear_prior` | POST | /api/workflow-routing/clear-prior |

## Verification Steps
```bash
cd /Users/mini1/Movies/worker_management
# Count all routes vs operation_ids
grep -rc '@router\.\(get\|post\|put\|delete\)(' src/api/ --include="*.py" | grep -v ':0'
grep -rc 'operation_id=' src/api/ --include="*.py" | grep -v ':0'
# Ensure counts match for each file
# Check health routes in main app file too
```
