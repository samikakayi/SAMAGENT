"""Publishing a workflow, on whichever n8n the operator happens to run.

Current n8n calls this publish/unpublish; `activate`/`deactivate` still answer
but are marked deprecated, and in 2.40.5 they are literally aliases onto the
publish handler. SAM uses the supported name and keeps the old one strictly as
a fallback for an older instance -- which makes *when* it falls back the thing
worth testing. A missing endpoint is evidence about a version. A rejected key
is not, and quietly retrying a different URL after one would turn a clear
refusal into a puzzle.
"""

from __future__ import annotations

import httpx
import pytest

from sam_backend.workflows import WorkflowError, WorkflowErrorCode
from sam_backend.workflows.n8n import REQUIRED_SCOPES, N8nClient


def client(handler) -> N8nClient:
    return N8nClient("http://localhost:5678", "key",
                     client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)))


def modern(request: httpx.Request) -> httpx.Response:
    """An n8n that has publish/unpublish and nothing older."""
    path = request.url.path
    if path.endswith(("/publish", "/unpublish")):
        return httpx.Response(200, json={"id": "wf-1", "active": path.endswith("/publish")},
                              request=request)
    if path.endswith(("/activate", "/deactivate")):
        return httpx.Response(404, json={"message": "not found"}, request=request)
    return httpx.Response(404, request=request)


def legacy(request: httpx.Request) -> httpx.Response:
    """An n8n old enough that publish does not exist yet."""
    path = request.url.path
    if path.endswith(("/activate", "/deactivate")):
        return httpx.Response(200, json={"id": "wf-1", "active": path.endswith("/activate")},
                              request=request)
    return httpx.Response(404, json={"message": "unknown endpoint"}, request=request)


# --- the supported endpoint comes first ---------------------------------------


@pytest.mark.parametrize("active, expected_path", [(True, "/publish"), (False, "/unpublish")])
def test_current_n8n_is_asked_to_publish_not_to_activate(active, expected_path):
    seen: list[str] = []

    def handler(request):
        seen.append(request.url.path)
        return modern(request)

    result = client(handler).set_active("wf-1", active)

    assert seen == [f"/api/v1/workflows/wf-1{expected_path}"]
    assert result == {"id": "wf-1", "active": active}
    assert not any("activate" in path for path in seen), "the deprecated alias was never tried"


def test_the_supported_endpoint_is_not_re_probed_every_call():
    seen: list[str] = []

    def handler(request):
        seen.append(request.url.path)
        return modern(request)

    engine = client(handler)
    engine.set_active("wf-1", True)
    engine.set_active("wf-1", False)

    assert seen == ["/api/v1/workflows/wf-1/publish", "/api/v1/workflows/wf-1/unpublish"]


# --- and the old one only when the new one genuinely is not there -------------


def test_an_older_n8n_falls_back_to_the_deprecated_alias():
    seen: list[str] = []

    def handler(request):
        seen.append(request.url.path)
        return legacy(request)

    result = client(handler).set_active("wf-1", True)

    assert seen == ["/api/v1/workflows/wf-1/publish", "/api/v1/workflows/wf-1/activate"]
    assert result == {"id": "wf-1", "active": True}


def test_the_fallback_is_remembered_so_it_costs_one_extra_call_once():
    seen: list[str] = []

    def handler(request):
        seen.append(request.url.path)
        return legacy(request)

    engine = client(handler)
    engine.set_active("wf-1", True)
    engine.set_active("wf-1", False)

    assert seen == [
        "/api/v1/workflows/wf-1/publish",
        "/api/v1/workflows/wf-1/activate",
        "/api/v1/workflows/wf-1/deactivate",
    ], "after one proof, the old endpoint is used directly"


@pytest.mark.parametrize("code, expected", [
    (401, WorkflowErrorCode.AUTH),
    (403, WorkflowErrorCode.AUTH),
    (400, WorkflowErrorCode.EXECUTION_FAILED),
    (409, WorkflowErrorCode.EXECUTION_FAILED),
    (429, WorkflowErrorCode.RATE_LIMIT),
    (500, WorkflowErrorCode.EXECUTION_FAILED),
])
def test_only_a_missing_endpoint_may_change_which_url_sam_calls(code, expected):
    """A rejected key or a refused request is an answer, not a version hint."""
    seen: list[str] = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(code, json={"message": "no"}, request=request)

    with pytest.raises(WorkflowError) as raised:
        client(handler).set_active("wf-1", True)

    assert raised.value.code is expected
    assert seen == ["/api/v1/workflows/wf-1/publish"], "no hidden retry on a different URL"


def test_an_insufficient_scope_stays_a_permission_failure():
    """The narrow key is the point; a 403 must not look like an old n8n."""
    def handler(request):
        return httpx.Response(403, json={"message": "missing scope workflow:activate"},
                              request=request)

    with pytest.raises(WorkflowError) as raised:
        client(handler).set_active("wf-1", True)

    assert raised.value.code is WorkflowErrorCode.AUTH
    assert "rejected the API key" in str(raised.value)


def test_a_workflow_that_does_not_exist_still_reports_not_found():
    """Both endpoints 404, so the fallback runs and the honest answer survives."""
    def handler(request):
        return httpx.Response(404, json={"message": "workflow not found"}, request=request)

    with pytest.raises(WorkflowError) as raised:
        client(handler).set_active("missing", True)

    assert raised.value.code is WorkflowErrorCode.NOT_FOUND


def test_a_404_for_a_missing_workflow_does_not_pin_the_client_to_the_old_api():
    """Otherwise one typo'd id would downgrade every later call."""
    calls: list[str] = []

    def handler(request):
        calls.append(request.url.path)
        if "missing" in request.url.path:
            return httpx.Response(404, json={"message": "workflow not found"}, request=request)
        return modern(request)

    engine = client(handler)
    with pytest.raises(WorkflowError):
        engine.set_active("missing", True)
    calls.clear()
    engine.set_active("wf-1", True)

    assert calls == ["/api/v1/workflows/wf-1/publish"]


# --- the key does not need to change ------------------------------------------


def test_publishing_needs_no_scope_the_narrow_key_lacks():
    """n8n gives publish the same two scopes the deprecated aliases carried."""
    assert {"workflow:activate", "workflow:deactivate"} <= REQUIRED_SCOPES
    assert len(REQUIRED_SCOPES) == 7, "migrating endpoints must not widen the key"
    assert "workflow:update" not in REQUIRED_SCOPES, "publishing is not updating"
    assert "workflow:delete" not in REQUIRED_SCOPES


def test_an_empty_publish_response_still_answers_honestly():
    """n8n may answer 200 with no body; the requested state is then the answer."""
    def handler(request):
        return httpx.Response(200, content=b"", request=request)

    assert client(handler).set_active("wf-1", True) == {"id": "wf-1", "active": True}
