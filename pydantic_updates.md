## Plan for `pydantic-ai` v0.2.17 Update

**Objective:** Update the `worker_management` system to leverage new features in `pydantic-ai` v0.2.17, specifically implementing token usage metrics and verifying MCP compatibility.

**Release Highlights from `pydantic-ai` v0.2.17:**
*   **Token Usage Metrics:** Added to `InstrumentedModel` (PR #1898).
*   **MCP Custom HTTP Client:** `MCPServerHTTP` now allows users to supply their own `httpx.AsyncClient` (PR #1925).
*   Other changes include `service_tier` in `OpenAIModelSettings`, adjustments to sampling settings for OpenAI reasoning models, and `fileData` support for Gemini/Google models.

### 1. Implement Token Usage Metrics

*   **Goal:** Track token usage for LLM interactions and include this information in API responses when messages are sent.
*   **Approach:** Utilize the new token usage metrics available in `pydantic-ai`'s `InstrumentedModel`.

*   **Steps:**
    1.  **Identify Models:** Locate the Pydantic models in the `worker_management` codebase that are used for LLM interactions (likely those interacting with `pydantic-ai`'s core functionalities).
    2.  **Integrate `InstrumentedModel`:**
        *   If these models don't already inherit from or utilize a base that incorporates `InstrumentedModel`'s features, modify them. This might involve changing the base class or wrapping existing models.
        *   Consult the `pydantic-ai` documentation (or PR #1898 if necessary) to understand how `InstrumentedModel` exposes token metrics (e.g., attributes like `prompt_tokens`, `completion_tokens`).
    3.  **Update API Responses:**
        *   Modify the API endpoint(s) responsible for sending messages and receiving LLM responses.
        *   After an LLM call, retrieve the token usage data from the `InstrumentedModel` instance.
        *   Add a new section/field to the API response JSON (e.g., `usage_metrics: {"prompt_tokens": <value>, "completion_tokens": <value>, "total_tokens": <value>}`) to convey this information.
    4.  **Testing:**
        *   Send test requests to the updated API.
        *   Verify that the token usage metrics are present in the response and accurately reflect the interaction.

### 2. Check and Update MCP Implementation

*   **Goal:** Ensure the existing MCP implementation is compatible with `pydantic-ai` v0.2.17 and leverage new features if beneficial.
*   **Approach:** Review the usage of `MCPServerHTTP` in light of the new capability to supply a custom `httpx.AsyncClient`.

*   **Steps:**
    1.  **Locate MCP Server Code:** Find where `MCPServerHTTP` (or other `pydantic-ai` MCP components) is instantiated and configured within the `worker_management` project.
    2.  **Evaluate `httpx.AsyncClient` Customization:**
        *   Determine if the current implementation uses the default `httpx.AsyncClient` provided by `pydantic-ai` or if there's a need for a custom client (e.g., for specific timeouts, proxy configurations, retry strategies, or custom headers).
        *   If a custom client is desirable, update the instantiation of `MCPServerHTTP` to pass your configured `httpx.AsyncClient` instance.
        *   If the default client is sufficient, no immediate code change might be required for this specific feature, but it's good to be aware of this new flexibility.
    3.  **General Compatibility Review:**
        *   Briefly review other MCP-related parts of your code to ensure no deprecated methods are being used or if new best practices from v0.2.17 are applicable. (The release notes primarily highlight the `httpx.AsyncClient` change for MCP).
    4.  **Testing:**
        *   Test MCP functionalities to ensure they operate as expected after any changes or verifications.

### 3. Codebase Investigation (Next Steps for Cascade)

*   To proceed with the implementation, I will need to:
    *   Search the codebase for usages of `pydantic-ai` models to identify candidates for `InstrumentedModel`.
    *   Search the codebase for `MCPServerHTTP` to check the current MCP client setup.

---
