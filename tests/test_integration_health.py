"""Integration health: can SAM say what is wrong before it goes wrong.

The panel exists because credentials fail quietly -- they expire, get revoked
somewhere else, or carry authority nobody intended. So these tests care about
two things above all: that what is reported is *known* rather than guessed,
and that reporting it never becomes a way to read a secret.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sam_backend.config import Settings
from sam_backend.db import Database
from sam_backend.integrations import (
    ExpiryState,
    IntegrationHealth,
    IntegrationReport,
    IntegrationState,
    PrivilegeState,
    clear_metadata,
    describe_expiry,
    describe_privilege,
    read_metadata,
    record_metadata,
)
from sam_backend.secrets import SecretStore
from sam_backend.workflows.n8n import REQUIRED_SCOPES

# Shaped like a real key so the store accepts it; never a real one.
N8N_SENTINEL = "eyJhbGciHEALTHTESTSENTINELabcdefghij0123456789"


def database(tmp_path):
    return Database(tmp_path / "health.db")


def store(tmp_path):
    return SecretStore(tmp_path)


# --- expiry is reported only when it is known ---------------------------------


def test_no_recorded_expiry_is_said_plainly_not_guessed():
    state, date, days = describe_expiry(None)

    assert state is ExpiryState.UNKNOWN_EXPIRY
    assert date == "" and days is None


@pytest.mark.parametrize("offset_days, expected", [
    (90, ExpiryState.OK),
    (13, ExpiryState.EXPIRING_SOON),
    (3, ExpiryState.EXPIRING_SOON),
    (-1, ExpiryState.EXPIRED),
])
def test_a_recorded_expiry_becomes_a_countdown(offset_days, expected):
    now = datetime(2026, 9, 23, tzinfo=UTC)
    state, date, days = describe_expiry(
        (now + timedelta(days=offset_days, hours=1)).isoformat(), now=now)

    assert state is expected
    assert date and days == offset_days


def test_expiry_is_accepted_in_the_shapes_providers_actually_send():
    now = datetime(2026, 9, 23, tzinfo=UTC)
    unix = (now + timedelta(days=30)).timestamp()

    from_unix, date_a, _ = describe_expiry(unix, now=now)
    from_iso, date_b, _ = describe_expiry("2026-12-22T05:00:00Z", now=now)

    assert from_unix is ExpiryState.OK and date_a == "2026-10-23"
    assert from_iso is ExpiryState.OK and date_b == "2026-12-22"


def test_a_unix_expiry_that_arrives_as_text_is_still_a_date():
    """n8n publishes epoch seconds, and a text-typed field hands them over as text.

    Reading only the numeric type made a real n8n key with a real 90-day
    expiry report UNKNOWN_EXPIRY -- the exact answer this module exists to
    avoid giving.
    """
    now = datetime(2026, 9, 23, tzinfo=UTC)
    epoch = str(int((now + timedelta(days=90)).timestamp()))

    state, date, days = describe_expiry(epoch, now=now)

    assert state is ExpiryState.OK
    assert date == "2026-12-22" and days == 90


def test_an_unreadable_expiry_is_unknown_rather_than_an_error():
    assert describe_expiry("next tuesday")[0] is ExpiryState.UNKNOWN_EXPIRY
    assert describe_expiry({"nope": 1})[0] is ExpiryState.UNKNOWN_EXPIRY


# --- privilege ----------------------------------------------------------------


def test_a_key_with_exactly_what_sam_uses_is_least_privilege():
    state, extra, note = describe_privilege(sorted(REQUIRED_SCOPES), REQUIRED_SCOPES)

    assert state is PrivilegeState.LEAST_PRIVILEGE
    assert extra == ()
    assert str(len(REQUIRED_SCOPES)) in note


def test_a_key_with_anything_extra_is_overprivileged_and_says_what():
    state, extra, note = describe_privilege(
        [*REQUIRED_SCOPES, "workflow:delete", "user:list"], REQUIRED_SCOPES)

    assert state is PrivilegeState.OVERPRIVILEGED
    assert extra == ("user:list", "workflow:delete")
    assert "beyond" in note


def test_a_narrower_key_is_not_called_overprivileged():
    """Too few scopes is a different problem, and not this one."""
    state, extra, _ = describe_privilege(["workflow:list"], REQUIRED_SCOPES)

    assert state is PrivilegeState.LEAST_PRIVILEGE
    assert extra == ()


def test_no_recorded_scopes_means_unknown_not_assumed_safe():
    state, _, note = describe_privilege([], REQUIRED_SCOPES)

    assert state is PrivilegeState.UNKNOWN
    assert "No scope list" in note


# --- metadata is non-secret, and only the fields that are ---------------------


def test_only_the_three_non_secret_fields_are_ever_recorded(tmp_path):
    db = database(tmp_path)

    record_metadata(db, "n8n_api_key", {
        "scopes": ["workflow:list"], "created_at": "2026-09-23T05:11:15Z",
        "expires_at": "2026-12-22T05:00:00Z",
        # Anything else offered is dropped rather than trusted.
        "value": N8N_SENTINEL, "apiKey": N8N_SENTINEL, "password": "hunter2",
    })

    stored = read_metadata(db)["n8n_api_key"]
    assert set(stored) == {"scopes", "created_at", "expires_at"}
    assert N8N_SENTINEL not in str(stored)


def test_clearing_a_credential_forgets_what_was_recorded_about_it(tmp_path):
    db = database(tmp_path)
    record_metadata(db, "n8n_api_key", {"scopes": ["workflow:list"]})

    assert clear_metadata(db, "n8n_api_key") is True
    assert "n8n_api_key" not in read_metadata(db)
    assert clear_metadata(db, "n8n_api_key") is False, "clearing twice is not an error"


def test_metadata_for_one_credential_does_not_disturb_another(tmp_path):
    db = database(tmp_path)
    record_metadata(db, "n8n_api_key", {"scopes": ["workflow:list"]})
    record_metadata(db, "groq_api_key", {"expires_at": "2027-01-01"})

    assert set(read_metadata(db)) == {"n8n_api_key", "groq_api_key"}


# --- what a report says -------------------------------------------------------


def health(tmp_path, **kwargs):
    return IntegrationHealth(Settings(data_dir=tmp_path), store(tmp_path), database(tmp_path), **kwargs)


def test_an_unconfigured_integration_is_not_an_error(tmp_path):
    report = IntegrationHealth._advise(
        IntegrationReport("groq", IntegrationState.NOT_CONFIGURED))

    assert report.needs_attention is False, "nothing is wrong with a feature you do not use"
    assert report.recommended_action == ""


def test_a_rejected_credential_says_what_to_do_about_it(tmp_path):
    report = IntegrationHealth._advise(IntegrationReport("groq", IntegrationState.AUTH_ERROR))

    assert report.needs_attention is True
    assert "Re-issue" in report.recommended_action


def test_an_expiring_credential_is_flagged_while_still_working(tmp_path):
    engine = health(tmp_path)
    facts = engine._credential_facts("n8n_api_key")
    facts.configured = True
    facts.expiry, facts.expires_at, facts.days_until_expiry = ExpiryState.EXPIRING_SOON, "2026-10-01", 5

    report = IntegrationHealth._advise(
        IntegrationReport("n8n", IntegrationState.HEALTHY, credential=facts))

    assert report.state is IntegrationState.HEALTHY, "healthy and expiring are both true"
    assert report.needs_attention is True
    assert "less than a week" in report.warnings[0]


def test_an_overprivileged_key_is_flagged_even_when_everything_works(tmp_path):
    engine = health(tmp_path)
    facts = engine._credential_facts("n8n_api_key")
    facts.configured = True
    facts.privilege = PrivilegeState.OVERPRIVILEGED
    facts.extra_scopes = ("user:list", "workflow:delete")

    report = IntegrationHealth._advise(
        IntegrationReport("n8n", IntegrationState.HEALTHY, credential=facts))

    assert report.needs_attention is True
    assert "user:list" in report.warnings[0]
    assert "only the scopes SAM needs" in report.recommended_action


def test_an_n8n_key_with_no_recorded_scopes_is_not_presumed_safe(tmp_path):
    engine = health(tmp_path)
    facts = engine._credential_facts("n8n_api_key")
    facts.configured = True

    report = IntegrationHealth._advise(
        IntegrationReport("n8n", IntegrationState.HEALTHY, credential=facts))

    assert any("cannot be ruled out" in warning for warning in report.warnings)


# --- the whole summary --------------------------------------------------------


class OfflineN8n:
    def status(self):
        return {"status": "NOT_CONFIGURED", "detail": "no instance"}


class Workflows:
    n8n = OfflineN8n()


@pytest.mark.asyncio
async def test_the_summary_covers_every_integration_and_leaks_nothing(tmp_path):
    secrets = store(tmp_path)
    secrets.set("n8n_api_key", N8N_SENTINEL)
    db = database(tmp_path)
    record_metadata(db, "n8n_api_key", {"scopes": sorted(REQUIRED_SCOPES),
                                        "expires_at": "2099-01-01T00:00:00Z"})
    engine = IntegrationHealth(Settings(data_dir=tmp_path), secrets, db, Workflows())

    summary = await engine.summary()

    names = [item["integration"] for item in summary["integrations"]]
    assert names == ["openrouter", "groq", "gemini", "n8n"]
    assert N8N_SENTINEL not in str(summary), "a health panel is not a way to read a key"

    n8n = next(item for item in summary["integrations"] if item["integration"] == "n8n")
    assert n8n["credential"]["configured"] is True
    assert n8n["credential"]["privilege"] == "LEAST_PRIVILEGE"
    assert n8n["credential"]["fingerprint"], "presence is shown as a fingerprint, not a value"
    assert n8n["credential"]["fingerprint"] not in N8N_SENTINEL


@pytest.mark.asyncio
async def test_the_panel_does_not_re_probe_on_every_render(tmp_path):
    """A dashboard must not spend a free tier to look current."""
    engine = IntegrationHealth(Settings(data_dir=tmp_path), store(tmp_path),
                               database(tmp_path), Workflows())
    calls: list[int] = []

    async def counted(name, base_url, key):
        calls.append(1)
        return engine._from_provider(name, {"status": "UNCONFIGURED"})

    engine._openai_compatible = counted
    await engine.summary()
    await engine.summary()

    assert len(calls) == 2, "the second render reused the cache rather than re-probing"


@pytest.mark.asyncio
async def test_refresh_is_what_re_checks_and_nothing_else(tmp_path):
    engine = IntegrationHealth(Settings(data_dir=tmp_path), store(tmp_path),
                               database(tmp_path), Workflows())
    calls: list[int] = []

    async def counted(name, base_url, key):
        calls.append(1)
        return engine._from_provider(name, {"status": "UNCONFIGURED"})

    engine._openai_compatible = counted
    await engine.summary()
    await engine.summary(refresh=True)

    assert len(calls) == 4, "explicitly refreshing re-probes both keyed providers"


@pytest.mark.asyncio
async def test_storing_a_new_credential_invalidates_the_cached_verdict(tmp_path):
    engine = IntegrationHealth(Settings(data_dir=tmp_path), store(tmp_path),
                               database(tmp_path), Workflows())
    await engine.summary()
    assert engine._cache is not None

    engine.invalidate()

    assert engine._cache is None, "a new key must not be judged by the old key's verdict"


# --- the metadata cannot be forged ---------------------------------------------


def test_scope_metadata_cannot_be_written_through_the_settings_route():
    """Otherwise 'least privilege' would be a claim anyone could make.

    The settings route writes to the same key-value table the metadata lives
    in, so the only thing keeping the two apart is that SettingsUpdate does not
    declare the field and Pydantic drops what it does not declare.
    """
    from sam_backend.integrations import METADATA_SETTING_KEY
    from sam_backend.schemas import SettingsUpdate

    smuggled = SettingsUpdate.model_validate({
        METADATA_SETTING_KEY: {"n8n_api_key": {"scopes": ["workflow:list"]}},
        "fallback_enabled": True,
    })

    assert METADATA_SETTING_KEY not in smuggled.provided()
    assert smuggled.provided() == {"fallback_enabled": True}


def test_storing_a_credential_is_what_records_its_scopes(client):
    """And the value never comes back, metadata or not."""
    response = client.post("/api/providers/credentials", json={
        "name": "n8n_api_key", "value": N8N_SENTINEL,
        "metadata": {"scopes": sorted(REQUIRED_SCOPES), "expires_at": "2099-01-01T00:00:00Z"},
    })

    assert response.status_code == 200
    assert N8N_SENTINEL not in response.text

    health = client.get("/api/integrations?refresh=true")
    n8n = next(i for i in health.json()["integrations"] if i["integration"] == "n8n")
    assert n8n["credential"]["privilege"] == "LEAST_PRIVILEGE"
    assert N8N_SENTINEL not in health.text
