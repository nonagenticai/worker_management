"""Pydantic Graph State Model - Core state management for pydantic_graph integration.

This module provides the WorkflowState dataclass that serves as the central state management
system for graph execution within the pydantic_graph framework. It tracks execution progress,
node outputs, error states, and performance metrics throughout the workflow lifecycle.

The WorkflowState class is designed to be framework-agnostic while providing comprehensive
tracking capabilities for distributed workflow execution across multiple workers.

Classes:
    WorkflowState: Core dataclass for managing workflow execution state
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class WorkflowState:
    """State shared across all nodes during graph execution.

    This is the core state model that pydantic_graph uses to track
    execution progress, node outputs, and overall workflow state.

    The WorkflowState maintains comprehensive information about:
    - Node execution status and outputs
    - Error tracking and retry counts
    - Performance metrics and timing
    - Worker assignments for distributed execution
    - Framework-specific state data

    Attributes:
        session_id (str): Unique identifier for the execution session
        graph_id (str): Identifier for the graph being executed
        variables (Dict[str, Any]): Global variables available to all nodes
        node_outputs (Dict[str, Any]): Mapping of node_id to execution output
        completed_nodes (List[str]): List of successfully completed node IDs
        failed_nodes (List[str]): List of failed node IDs
        start_time (Optional[datetime]): Workflow execution start timestamp
        end_time (Optional[datetime]): Workflow execution end timestamp
        error_message (Optional[str]): Global error message if workflow fails
        node_status (Dict[str, str]): Mapping of node_id to current status
        worker_assignments (Dict[str, str]): Mapping of node_id to assigned worker
        node_errors (Dict[str, str]): Mapping of node_id to error messages
        current_node (Optional[str]): ID of currently executing node
        retry_counts (Dict[str, int]): Retry attempt counts per node
        node_durations (Dict[str, float]): Execution duration in seconds per node
        framework_state (Dict[str, Any]): Framework-specific state storage
    """

    # Reserved key inside ``framework_state`` for inference telemetry data.
    # See ``record_inference_attempt`` / ``mark_inference_success`` /
    # ``set_inference_request`` / ``set_validation_result`` for the accessors
    # that read and write under this key.
    INFERENCE_KEY = "inference"

    session_id: str
    graph_id: str
    variables: dict[str, Any]
    node_outputs: dict[str, Any]  # node_id -> output data
    completed_nodes: list[str]
    failed_nodes: list[str]
    start_time: datetime | None = None
    end_time: datetime | None = None
    error_message: str | None = None

    # Extended state for enhanced functionality
    node_status: dict[str, str] = field(
        default_factory=dict
    )  # node_id -> status (pending, running, completed, failed)
    worker_assignments: dict[str, str] = field(
        default_factory=dict
    )  # node_id -> worker_name mapping
    node_errors: dict[str, str] = field(default_factory=dict)  # node_id -> error_message mapping
    current_node: str | None = None
    retry_counts: dict[str, int] = field(default_factory=dict)  # node_id -> retry count
    node_durations: dict[str, float] = field(
        default_factory=dict
    )  # node_id -> execution time in seconds
    framework_state: dict[str, Any] = field(default_factory=dict)  # For framework-specific data

    # Run identity threaded from the API layer (execute_graph) so graph-executed
    # LLM workers enforce the SAME BYO posture as the HTTP message path. The
    # OFFICIAL WorkerNode (graph_factory) reads these and forwards them as a raw
    # X-Tenant-Id / X-Subject-Id on its loopback call to /workers/{name}/message;
    # the auth middleware's trusted-network branch then re-resolves byo_mode from
    # licenses by tenant_id and enforces byo_required. Absent (None) ⇒ the loopback
    # sends no tenant header and falls back to DEFAULT_TENANT_ID — byte-identical
    # to the pre-fix behavior. byo_mode is deliberately NOT carried here: the
    # middleware re-resolves the authoritative posture, which beats a stale snapshot.
    tenant_id: str | None = None
    subject_id: str | None = None

    def __post_init__(self) -> None:
        """Initialize default values for mutable fields.

        This method is called automatically after dataclass initialization
        to set up any additional state that cannot be handled by field factories.
        Currently, all initialization is handled by field default factories.
        """
        # Fields are now non-optional with factory defaults
        # Initialization handled by field factories

    def get_node_output(self, node_id: str) -> Any:
        """Get output from a previously executed node.

        Args:
            node_id (str): The unique identifier of the node

        Returns:
            Any: The output data from the node, or None if node has not executed

        Examples:
            >>> state.get_node_output("data_processor")
            {"processed_count": 100, "status": "success"}
        """
        return self.node_outputs.get(node_id)

    def set_node_output(self, node_id: str, output: Any) -> None:
        """Set output for a node and mark as completed.

        This method stores the node's execution result and automatically
        updates the node's status to completed if not already in the
        completed_nodes list.

        Args:
            node_id (str): The unique identifier of the node
            output (Any): The output data from the node execution

        Examples:
            >>> state.set_node_output("processor", {"count": 42})
            >>> "processor" in state.completed_nodes
            True
        """
        self.node_outputs[node_id] = output
        if node_id not in self.completed_nodes:
            self.completed_nodes.append(node_id)
        self.node_status[node_id] = "completed"

    def mark_node_failed(self, node_id: str, error_message: str | None = None) -> None:
        """Mark a node as failed and record error information.

        This method updates the node's status to failed, adds it to the
        failed_nodes list, and optionally stores an error message. If this
        is the first workflow-level error, it sets the global error_message.

        Args:
            node_id (str): The unique identifier of the failed node
            error_message (Optional[str]): Description of the failure reason

        Examples:
            >>> state.mark_node_failed("validator", "Invalid input format")
            >>> state.node_status["validator"]
            "failed"
        """
        if node_id not in self.failed_nodes:
            self.failed_nodes.append(node_id)
        self.node_status[node_id] = "failed"
        if error_message:
            self.node_errors[node_id] = error_message
            # Set workflow-level error message if this is the first error
            if not self.error_message:
                self.error_message = error_message

    def mark_node_running(self, node_id: str) -> None:
        """Mark a node as currently running.

        Updates the node status to running and sets it as the current
        node being executed. This is used for tracking execution progress
        and determining the active node in the workflow.

        Args:
            node_id (str): The unique identifier of the node to mark as running

        Examples:
            >>> state.mark_node_running("data_processor")
            >>> state.current_node
            "data_processor"
        """
        self.current_node = node_id
        self.node_status[node_id] = "running"

    def increment_retry_count(self, node_id: str) -> int:
        """Increment retry count for a node and return new count.

        This method tracks how many times a node has been retried,
        which is useful for implementing retry policies and debugging
        persistent failures.

        Args:
            node_id (str): The unique identifier of the node

        Returns:
            int: The new retry count after incrementing

        Examples:
            >>> count = state.increment_retry_count("flaky_node")
            >>> print(f"Retry attempt {count}")
            "Retry attempt 1"
        """
        current_count = self.retry_counts.get(node_id, 0)
        new_count = current_count + 1
        self.retry_counts[node_id] = new_count
        return new_count

    def set_node_duration(self, node_id: str, duration: float) -> None:
        """Set execution duration for a node.

        Records the time taken for a node to execute, which is useful
        for performance monitoring and optimization.

        Args:
            node_id (str): The unique identifier of the node
            duration (float): Execution time in seconds

        Examples:
            >>> state.set_node_duration("heavy_computation", 45.7)
            >>> state.node_durations["heavy_computation"]
            45.7
        """
        self.node_durations[node_id] = duration

    def get_progress_percentage(self, total_nodes: int) -> float:
        """Calculate execution progress as percentage.

        Computes the percentage of nodes that have completed execution
        (either successfully or with failure) relative to the total number
        of nodes in the graph.

        Args:
            total_nodes (int): The total number of nodes in the graph

        Returns:
            float: Progress percentage between 0.0 and 100.0

        Examples:
            >>> state.get_progress_percentage(10)  # 5 completed, 5 remaining
            50.0
        """
        if total_nodes == 0:
            return 100.0
        completed_count = len(self.completed_nodes) + len(self.failed_nodes)
        return min(100.0, (completed_count / total_nodes) * 100.0)

    def is_execution_complete(self, total_nodes: int) -> bool:
        """Check if execution is complete (all nodes processed).

        Determines whether all nodes in the graph have been processed,
        either successfully completed or failed. This indicates that
        no more nodes are pending execution.

        Args:
            total_nodes (int): The total number of nodes in the graph

        Returns:
            bool: True if all nodes have been processed, False otherwise

        Examples:
            >>> state.is_execution_complete(5)  # All 5 nodes done
            True
        """
        return len(self.completed_nodes) + len(self.failed_nodes) == total_nodes

    def has_failed_nodes(self) -> bool:
        """Check if any nodes have failed.

        Returns:
            bool: True if there are any failed nodes, False otherwise

        Examples:
            >>> state.has_failed_nodes()
            False
            >>> state.mark_node_failed("bad_node")
            >>> state.has_failed_nodes()
            True
        """
        return len(self.failed_nodes) > 0

    def get_execution_summary(self) -> dict[str, Any]:
        """Get a summary of the execution state.

        Provides a comprehensive overview of the workflow execution
        including status, progress, timing, and error information.

        Returns:
            Dict[str, Any]: Summary containing:
                - session_id: Execution session identifier
                - graph_id: Graph identifier
                - status: Overall execution status
                - completed_nodes: Count of completed nodes
                - failed_nodes: Count of failed nodes
                - total_processed: Total nodes processed
                - current_node: Currently executing node
                - execution_time: Total execution time if complete
                - has_errors: Whether any errors occurred
                - error_message: Global error message if any

        Examples:
            >>> summary = state.get_execution_summary()
            >>> print(summary["status"])
            "running"
        """
        total_processed = len(self.completed_nodes) + len(self.failed_nodes)
        total_time = None
        if self.start_time and self.end_time:
            total_time = (self.end_time - self.start_time).total_seconds()

        return {
            "session_id": self.session_id,
            "graph_id": self.graph_id,
            "status": self._get_overall_status(),
            "completed_nodes": len(self.completed_nodes),
            "failed_nodes": len(self.failed_nodes),
            "total_processed": total_processed,
            "current_node": self.current_node,
            "execution_time": total_time,
            "has_errors": self.has_failed_nodes(),
            "error_message": self.error_message,
        }

    def _get_overall_status(self) -> str:
        """Get overall execution status.

        Determines the current state of the workflow execution based on
        error conditions, completion status, and current activity.

        Returns:
            str: One of "failed", "completed", "completed_with_errors",
                 "running", or "pending"

        Examples:
            >>> state._get_overall_status()
            "running"
        """
        if self.error_message:
            return "failed"
        elif self.end_time:
            return "completed" if not self.has_failed_nodes() else "completed_with_errors"
        elif self.current_node:
            return "running"
        else:
            return "pending"

    # ------------------------------------------------------------------
    # Inference telemetry accessors
    #
    # These helpers read and write a small, well-defined bag under
    # ``framework_state[INFERENCE_KEY]``. The bag is lazily initialised on
    # first access so that existing ``WorkflowState`` instances (and
    # serialised state that predates this feature) keep working without any
    # migration. The shape is:
    #
    #     {
    #         "attempts": [],              # list[dict] of per-provider attempts
    #         "chosen_provider": None,     # str | None — winning provider
    #         "served_model": None,        # str | None — read-back model that served it
    #         "final_response": None,      # Any — winning response payload
    #         "validation_passed": None,   # bool | None — set by EvalSubgraphNode
    #         "quality_score": None,       # float | None — set by validator-LLM-worker
    #         "request_payload": None,     # dict | None — original request
    #     }
    # ------------------------------------------------------------------

    def _get_or_create_inference_bag(self) -> dict[str, Any]:
        """Return the inference bag, creating it with defaults if missing.

        Centralising the initialisation in one helper keeps the bag shape
        consistent for every call site. Callers receive the live dict; any
        mutation persists in ``framework_state``.

        Returns:
            Dict[str, Any]: The inference bag stored under
            ``framework_state[INFERENCE_KEY]``.
        """
        bag = self.framework_state.get(self.INFERENCE_KEY)
        if not isinstance(bag, dict):
            bag = {
                "attempts": [],
                "chosen_provider": None,
                "served_model": None,
                "final_response": None,
                "validation_passed": None,
                "quality_score": None,
                "request_payload": None,
            }
            self.framework_state[self.INFERENCE_KEY] = bag
        return bag

    def record_inference_attempt(
        self,
        provider: str,
        status: Literal["ok", "error"],
        latency_ms: int,
        tokens_prompt: int | None = None,
        tokens_completion: int | None = None,
        error: str | None = None,
        error_classification: str | None = None,
        served_model: str | None = None,
    ) -> None:
        """Append an attempt dict to ``framework_state[INFERENCE_KEY]['attempts']``.

        Called by provider nodes (Task 3) once per provider try, whether the
        attempt succeeded or failed. The appended record is a flat dict of
        JSON-serialisable fields so it round-trips through ``to_dict`` /
        ``from_dict`` without any custom encoding.

        Args:
            provider: Provider name (e.g. ``"pgx-local"``, ``"claude"``).
            status: ``"ok"`` if the attempt returned a response,
                ``"error"`` otherwise.
            latency_ms: Wall-clock latency of the attempt, in milliseconds.
            tokens_prompt: Prompt token count, if known.
            tokens_completion: Completion token count, if known.
            error: Short error description when ``status == "error"``.
            error_classification: Stable label aligned with IR's
                ``X-IR-Error-Classification`` header (transient_network,
                upstream_timeout, upstream_5xx, …); only meaningful when
                ``status == "error"``.
            served_model: The model the upstream REPORTED serving (read back
                from the response), or ``None`` when it did not say. Never
                the configured/requested model name.
        """
        bag = self._get_or_create_inference_bag()
        attempts: list[dict[str, Any]] = bag["attempts"]
        attempts.append(
            {
                "provider": provider,
                "status": status,
                "latency_ms": latency_ms,
                "tokens_prompt": tokens_prompt,
                "tokens_completion": tokens_completion,
                "error": error,
                "error_classification": error_classification,
                "served_model": served_model,
            }
        )

    def mark_inference_success(
        self,
        provider: str,
        response: Any,
        tokens_prompt: int,
        tokens_completion: int,
        served_model: str | None = None,
    ) -> None:
        """Record the winning provider and response in the inference bag.

        The ``tokens_prompt`` / ``tokens_completion`` arguments are kept in
        the method signature so callers can't forget them; the canonical
        per-attempt token record lives in the ``attempts`` list populated
        by :meth:`record_inference_attempt`. This method only flags the
        overall winner.

        Args:
            provider: Provider name that produced the winning response.
            response: Final response payload to surface to the caller.
            tokens_prompt: Prompt token count for the winning attempt.
            tokens_completion: Completion token count for the winning attempt.
            served_model: Read-back served model of the winning response
                (``None`` when the upstream did not report one).
        """
        # Token totals are deliberately not stored as top-level bag fields —
        # they live on the matching attempt entry populated by
        # ``record_inference_attempt``. Accepting them here keeps the call
        # site signature explicit and forward-compatible.
        _ = tokens_prompt, tokens_completion
        bag = self._get_or_create_inference_bag()
        bag["chosen_provider"] = provider
        bag["served_model"] = served_model
        bag["final_response"] = response
        # The chain recovered: a winning provider response means the workflow
        # did NOT fail overall, even if an earlier provider in the fallback
        # chain raised and set the global error_message (see mark_node_failed).
        # Clear it so the run is reported "completed" — the per-attempt failure
        # audit survives in failed_nodes / node_errors / the inference attempts.
        self.error_message = None

    def set_inference_request(self, payload: dict[str, Any]) -> None:
        """Set the request payload in the inference bag.

        Called once at the start of an inference chain so that downstream
        nodes, telemetry, and trainer queries can see exactly what request
        the chain ran against.

        Args:
            payload: The original request payload.
        """
        bag = self._get_or_create_inference_bag()
        bag["request_payload"] = payload

    def set_validation_result(self, passed: bool) -> None:  # noqa: FBT001, RUF100
        """Set ``validation_passed`` in the inference bag.

        Args:
            passed: True if the response passed validation (the
                ``EvalSubgraphNode`` in Task 4 calls this).
        """
        bag = self._get_or_create_inference_bag()
        bag["validation_passed"] = passed

    def set_quality_score(self, score: float) -> None:
        """Set ``quality_score`` (0.0–1.0) in the inference bag.

        Populated by a validator-LLM-worker (e.g. ``qa_validator`` seeded in
        ``scripts/seed_provisioning_workers.py``) when it returns a numeric
        score alongside the boolean ``validation_passed``. The orchestrator
        finalize path pulls it from the bag and persists it onto
        ``graph_execution_sessions.quality_score`` (NUMERIC(5,4), added in
        migration 018) so the metering ETL can enrich ``exec_quality.quality_score``
        via the by-correlation endpoint.

        Args:
            score: Quality score in [0.0, 1.0]. Values outside the range are
                accepted as-is (DB precision is 5,4 → up to 9.9999); validation
                clamping belongs at the validator-worker boundary, not here.
        """
        bag = self._get_or_create_inference_bag()
        bag["quality_score"] = score

    def to_dict(self) -> dict[str, Any]:
        """Convert state to dictionary for serialization.

        Transforms the WorkflowState instance into a dictionary format
        suitable for JSON serialization, database storage, or network
        transmission. Datetime objects are converted to ISO format strings.

        Returns:
            Dict[str, Any]: Dictionary representation of the workflow state

        Examples:
            >>> state_dict = state.to_dict()
            >>> "session_id" in state_dict
            True
        """
        return {
            "session_id": self.session_id,
            "graph_id": self.graph_id,
            "variables": self.variables,
            "node_outputs": self.node_outputs,
            "completed_nodes": self.completed_nodes,
            "failed_nodes": self.failed_nodes,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "error_message": self.error_message,
            "node_status": self.node_status,
            "worker_assignments": self.worker_assignments,
            "node_errors": self.node_errors,
            "current_node": self.current_node,
            "retry_counts": self.retry_counts,
            "node_durations": self.node_durations,
            "framework_state": self.framework_state,
            "tenant_id": self.tenant_id,
            "subject_id": self.subject_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowState:
        """Create state from dictionary.

        Reconstructs a WorkflowState instance from a dictionary representation,
        typically used for deserialization from JSON, database records, or
        network messages. Handles parsing of ISO format datetime strings.

        Args:
            data (Dict[str, Any]): Dictionary containing workflow state data

        Returns:
            WorkflowState: Reconstructed workflow state instance

        Raises:
            ValueError: If datetime strings cannot be parsed
            KeyError: If required fields are missing from data

        Examples:
            >>> data = {"session_id": "123", "graph_id": "my_graph", ...}
            >>> state = WorkflowState.from_dict(data)
            >>> state.session_id
            "123"
        """
        # Handle datetime parsing
        start_time = None
        if data.get("start_time"):
            start_time = datetime.fromisoformat(data["start_time"])

        end_time = None
        if data.get("end_time"):
            end_time = datetime.fromisoformat(data["end_time"])

        return cls(
            session_id=data["session_id"],
            graph_id=data["graph_id"],
            variables=data.get("variables", {}),
            node_outputs=data.get("node_outputs", {}),
            completed_nodes=data.get("completed_nodes", []),
            failed_nodes=data.get("failed_nodes", []),
            start_time=start_time,
            end_time=end_time,
            error_message=data.get("error_message"),
            node_status=data.get("node_status", {}),
            worker_assignments=data.get("worker_assignments", {}),
            node_errors=data.get("node_errors", {}),
            current_node=data.get("current_node"),
            retry_counts=data.get("retry_counts", {}),
            node_durations=data.get("node_durations", {}),
            framework_state=data.get("framework_state", {}),
            tenant_id=data.get("tenant_id"),
            subject_id=data.get("subject_id"),
        )
