"""The one place SAM talks to n8n.

Every call goes through the configured `n8n_base_url`. A workflow's own
contents can never choose the target: a downloaded file naming some other host
would otherwise turn SAM's control client into a request forwarder pointed
wherever the file said.

Only the documented public API (v1) is used. That API has no endpoint for
manually running a workflow, so SAM does not run one -- rather than reaching
for the internal routes the editor UI uses, which are unsupported and change
without notice. What it does have is create, get, list, update, delete,
activate, deactivate, execution history and execution stop, and that is the
whole surface offered here.
"""

from __future__ import annotations

from typing import Any

import httpx

from .models import N8nExecutionStatus, WorkflowError, WorkflowErrorCode

TIMEOUT_SECONDS = 25.0
MAX_RESPONSE_BYTES = 2_000_000

# Exactly the n8n API key scopes the calls below need, and nothing else.
# Declared here rather than in a settings file because this class is the only
# thing that talks to n8n: if a method is added, the scope it needs is added
# on the same screen, and the health panel can tell an operator whether their
# key is broader than the product has any use for.
REQUIRED_SCOPES = frozenset({
    "workflow:list",        # status(), list_workflows()
    "workflow:read",        # get_workflow()
    "workflow:create",      # create_workflow()
    "workflow:activate",    # set_active(True)
    "workflow:deactivate",  # set_active(False)
    "credential:list",      # list_credentials() -- names and types only
    "execution:list",       # executions()
})
# Execution output goes in front of a model, so it is trimmed here rather than
# wherever it happens to be rendered.
MAX_OUTPUT_ITEMS = 5
MAX_OUTPUT_CHARS = 1500


class N8nClient:
    """A thin, honest wrapper over n8n's public REST API."""

    def __init__(self, base_url: str, api_key: str | None, *, client_factory: Any = None) -> None:
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self._client_factory = client_factory or (
            lambda: httpx.Client(timeout=TIMEOUT_SECONDS, trust_env=False, follow_redirects=False)
        )

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def _require(self) -> None:
        if not self.base_url:
            raise WorkflowError("No n8n instance is configured.", WorkflowErrorCode.NOT_CONFIGURED)
        if not self.api_key:
            raise WorkflowError("No n8n API key is configured.", WorkflowErrorCode.NOT_CONFIGURED)

    def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        """One request to the configured instance. `path` is never caller-supplied."""
        self._require()
        url = f"{self.base_url}/api/v1{path}"
        try:
            with self._client_factory() as client:
                response = client.request(
                    method, url,
                    headers={"X-N8N-API-KEY": self.api_key or "", "Accept": "application/json"},
                    **kwargs,
                )
        except httpx.TimeoutException as exc:
            raise WorkflowError("n8n did not answer in time.", WorkflowErrorCode.TIMEOUT) from exc
        except httpx.HTTPError as exc:
            raise WorkflowError(f"Could not reach n8n: {type(exc).__name__}", WorkflowErrorCode.NETWORK) from exc
        if response.status_code in (401, 403):
            raise WorkflowError("n8n rejected the API key.", WorkflowErrorCode.AUTH)
        if response.status_code == 404:
            raise WorkflowError("n8n has no such workflow or execution.", WorkflowErrorCode.NOT_FOUND)
        if response.status_code == 429:
            raise WorkflowError("n8n is rate limiting this key.", WorkflowErrorCode.RATE_LIMIT)
        if response.status_code >= 400:
            # The body may echo the request; only the status is reported on.
            raise WorkflowError(f"n8n returned HTTP {response.status_code}.", WorkflowErrorCode.EXECUTION_FAILED)
        if not response.content:
            return {}
        if len(response.content) > MAX_RESPONSE_BYTES:
            raise WorkflowError("n8n response was too large to parse.", WorkflowErrorCode.TOO_LARGE)
        try:
            return response.json()
        except ValueError as exc:
            raise WorkflowError("n8n returned a response that is not JSON.",
                                WorkflowErrorCode.EXECUTION_FAILED) from exc

    # -- status ------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Cheap reachability: list one workflow rather than run anything."""
        if not self.base_url or not self.api_key:
            return {"status": "NOT_CONFIGURED", "configured": False,
                    "detail": "Set the n8n URL and API key in Settings to enable import."}
        try:
            payload = self._call("GET", "/workflows", params={"limit": 1})
        except WorkflowError as exc:
            mapping = {
                WorkflowErrorCode.AUTH: "AUTH_ERROR",
                WorkflowErrorCode.TIMEOUT: "UNREACHABLE",
                WorkflowErrorCode.NETWORK: "UNREACHABLE",
                WorkflowErrorCode.RATE_LIMIT: "RATE_LIMITED",
            }
            return {"status": mapping.get(exc.code, "INCOMPATIBLE"), "configured": True,
                    "base_url": self.base_url, "detail": str(exc)}
        return {"status": "CONNECTED", "configured": True, "base_url": self.base_url,
                "workflows_visible": len(payload.get("data") or []),
                "detail": "Authenticated against the n8n public API."}

    # -- workflows ---------------------------------------------------------
    def list_workflows(self, limit: int = 20) -> list[dict[str, Any]]:
        payload = self._call("GET", "/workflows", params={"limit": max(1, min(int(limit), 100))})
        return [
            {"id": str(item.get("id") or ""), "name": str(item.get("name") or ""),
             "active": bool(item.get("active")), "updatedAt": str(item.get("updatedAt") or ""),
             "tags": [str(t.get("name") or "") for t in (item.get("tags") or []) if isinstance(t, dict)]}
            for item in (payload.get("data") or []) if isinstance(item, dict)
        ]

    def get_workflow(self, workflow_id: str) -> dict[str, Any]:
        return self._call("GET", f"/workflows/{workflow_id}")

    def create_workflow(self, workflow: dict[str, Any]) -> dict[str, Any]:
        """Import as inactive, always.

        n8n ignores `active` on create, but the field is stripped anyway so the
        intent is unambiguous in the payload SAM sends: importing a workflow
        must never be what starts it running.
        """
        body = {
            "name": str(workflow.get("name") or "Imported by SAM"),
            "nodes": workflow.get("nodes") or [],
            "connections": workflow.get("connections") or {},
            "settings": workflow.get("settings") or {},
        }
        created = self._call("POST", "/workflows", json=body)
        return {"id": str(created.get("id") or ""), "name": str(created.get("name") or ""),
                "active": bool(created.get("active"))}

    def set_active(self, workflow_id: str, active: bool) -> dict[str, Any]:
        verb = "activate" if active else "deactivate"
        result = self._call("POST", f"/workflows/{workflow_id}/{verb}")
        return {"id": str(result.get("id") or workflow_id), "active": bool(result.get("active", active))}

    def delete_workflow(self, workflow_id: str) -> dict[str, Any]:
        self._call("DELETE", f"/workflows/{workflow_id}")
        return {"id": workflow_id, "deleted": True}

    # -- credentials -------------------------------------------------------
    def list_credentials(self) -> list[dict[str, str]]:
        """Names and types only. n8n keeps the values; SAM never asks for them."""
        payload = self._call("GET", "/credentials")
        return [
            {"id": str(item.get("id") or ""), "name": str(item.get("name") or ""),
             "type": str(item.get("type") or "")}
            for item in (payload.get("data") or []) if isinstance(item, dict)
        ]

    # -- executions --------------------------------------------------------
    def execute(self, workflow_id: str, payload: dict[str, Any] | None = None) -> N8nExecutionStatus:
        """Not available: the public API exposes no manual-run endpoint.

        Said plainly rather than reached for through the editor's internal
        routes, which are undocumented and would break without warning.
        """
        raise WorkflowError(
            "n8n's public API has no endpoint for running a workflow on demand. SAM can import and "
            "activate it, or you can run it from the n8n editor; SAM will read the execution either way.",
            WorkflowErrorCode.UNSUPPORTED_OPERATION,
        )

    def executions(self, workflow_id: str = "", limit: int = 10) -> list[N8nExecutionStatus]:
        params: dict[str, Any] = {"limit": max(1, min(int(limit), 50))}
        if workflow_id:
            params["workflowId"] = workflow_id
        payload = self._call("GET", "/executions", params=params)
        return [self._execution(item) for item in (payload.get("data") or []) if isinstance(item, dict)]

    def execution(self, execution_id: str, *, include_data: bool = False) -> N8nExecutionStatus:
        return self._execution(self._call("GET", f"/executions/{execution_id}",
                                          params={"includeData": "true"} if include_data else None))

    def stop_execution(self, execution_id: str) -> dict[str, Any]:
        result = self._call("POST", f"/executions/{execution_id}/stop")
        return {"execution_id": execution_id, "stopped": True, "status": str(result.get("status") or "canceled")}

    @staticmethod
    def _execution(item: dict[str, Any]) -> N8nExecutionStatus:
        started, finished = str(item.get("startedAt") or ""), str(item.get("stoppedAt") or "")
        status = str(item.get("status") or ("error" if item.get("finished") is False else "unknown"))
        failed_node, error, outputs = "", "", []
        data = item.get("data") if isinstance(item.get("data"), dict) else {}
        result = data.get("resultData") if isinstance(data.get("resultData"), dict) else {}
        problem = result.get("error") if isinstance(result.get("error"), dict) else {}
        if problem:
            failed_node = str((problem.get("node") or {}).get("name") or "") if isinstance(problem.get("node"), dict) else ""
            error = str(problem.get("message") or "")[:400]
        run_data = result.get("runData") if isinstance(result.get("runData"), dict) else {}
        for node_name, runs in list(run_data.items())[:MAX_OUTPUT_ITEMS]:
            text = str(runs)[:MAX_OUTPUT_CHARS]
            outputs.append({"node": str(node_name), "preview": text})
        return N8nExecutionStatus(
            execution_id=str(item.get("id") or ""), workflow_id=str(item.get("workflowId") or ""),
            status=status, started_at=started, finished_at=finished,
            failed_node=failed_node, error=error, outputs=tuple(outputs),
        )
