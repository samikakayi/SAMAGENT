from __future__ import annotations

import base64
import json
import logging
import os

import pytest
from conftest import FAKE_GEMINI, FAKE_GEMINI_AQ, FAKE_GROQ, FAKE_OPENROUTER

from sam import secrets as S
from sam.config import Config


def test_round_trip_is_dpapi_v1_and_never_plaintext(tmp_path):
    store = S.SecretStore(tmp_path)
    result = store.set("groq_api_key", FAKE_GROQ)
    assert result == {"name": "groq_api_key", "stored": True, "fingerprint": S.SecretStore.fingerprint(FAKE_GROQ)}
    raw = (tmp_path / "secrets.json").read_text(encoding="utf-8")
    assert FAKE_GROQ not in raw
    if os.name == "nt":
        assert json.loads(raw)["_format"] == "dpapi-v1"
        assert store.encrypted_at_rest()
    # A fresh instance (as after a restart) reads the same value.
    assert S.SecretStore(tmp_path).get("groq_api_key") == FAKE_GROQ


@pytest.mark.skipif(os.name != "nt", reason="DPAPI is Windows-only")
def test_reads_a_file_written_in_v1_format(tmp_path):
    # Exactly what SAM v1's SecretStore._write produced.
    blob = S._dpapi("protect", json.dumps({"openrouter_api_key": FAKE_OPENROUTER, "n8n_api_key": "x" * 30}).encode())
    (tmp_path / "secrets.json").write_text(json.dumps(
        {"_format": "dpapi-v1", "data": base64.b64encode(blob).decode("ascii")}, indent=2), encoding="utf-8")
    store = S.SecretStore(tmp_path)
    assert store.get("openrouter_api_key") == FAKE_OPENROUTER
    # Saving a new key keeps every existing entry (even ones SAM 2 does not use).
    store.set("gemini_api_key", FAKE_GEMINI_AQ)
    again = S.SecretStore(tmp_path)
    assert again.get("n8n_api_key") == "x" * 30
    assert again.get("gemini_api_key") == FAKE_GEMINI_AQ


def test_opening_never_rewrites_a_legacy_plaintext_file(tmp_path):
    path = tmp_path / "secrets.json"
    path.write_text(json.dumps({"groq_api_key": FAKE_GROQ}), encoding="utf-8")
    before = path.stat().st_mtime_ns
    store = S.SecretStore(tmp_path)
    assert store.get("groq_api_key") == FAKE_GROQ
    assert path.stat().st_mtime_ns == before


@pytest.mark.parametrize("value", [FAKE_GEMINI, FAKE_GEMINI_AQ])
def test_gemini_accepts_aiza_and_aq_shapes(tmp_path, value):
    store = S.SecretStore(tmp_path)
    store.set("gemini_api_key", "  " + value + "\n")
    assert store.get("gemini_api_key") == value


@pytest.mark.parametrize("name,value", [("gemini_api_key", "AQ.short"), ("gemini_api_key", "sk-" + "a" * 30),
                                        ("groq_api_key", "AIza" + "b" * 30), ("nope", "whatever")])
def test_rejects_wrong_shapes_without_echoing_the_value(tmp_path, name, value):
    with pytest.raises(ValueError) as info:
        S.SecretStore(tmp_path).set(name, value)
    assert value not in str(info.value)


def test_resolution_order_env_then_dotenv_then_store(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / ".env").write_text("LITELLM_API_KEY=local-gateway-key-123\nGROQ_API_KEY=\n", encoding="utf-8")
    config = Config(tmp_path, environ={})
    secrets = S.Secrets(S.SecretStore(config.data_dir), config.env_value)
    assert secrets.get("litellm_api_key") == "local-gateway-key-123"
    assert secrets.source("litellm_api_key") == "environment"
    secrets.set("groq_api_key", FAKE_GROQ)          # empty .env value does not shadow the store
    assert secrets.get("groq_api_key") == FAKE_GROQ and secrets.source("groq_api_key") == "secret_store"
    process = Config(tmp_path, environ={"GROQ_API_KEY": "gsk_" + "p" * 30})
    assert S.Secrets(S.SecretStore(process.data_dir), process.env_value).get("groq_api_key") == "gsk_" + "p" * 30
    # .env values are never copied into the real process environment.
    assert os.environ.get("LITELLM_API_KEY") != "local-gateway-key-123"


def test_status_never_contains_values(tmp_path):
    secrets = S.Secrets(S.SecretStore(tmp_path), lambda name: None)
    secrets.set("groq_api_key", FAKE_GROQ)
    dumped = json.dumps(secrets.status())
    assert FAKE_GROQ not in dumped
    assert secrets.status()["groq_api_key"]["configured"] is True


@pytest.mark.parametrize("secret", [
    FAKE_GROQ, FAKE_GEMINI, FAKE_GEMINI_AQ, FAKE_OPENROUTER, "sk-" + "Ab1" * 12, "sk-proj-" + "Zz9" * 8,
    "0123456789abcdef0123456789abcdef", "QmFzZTY0U2VjcmV0VmFsdWUxMjM0NTY3ODkwQUJDREVGRw",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
])
def test_redact_masks_every_key_shape(secret):
    text = f"calling with {secret} now"
    out = S.redact(text)
    assert secret not in out and S.MASK in out
    assert out.startswith("calling with") and out.endswith("now")


def test_redact_masks_assignments_bearer_and_exact_values():
    assert "hunter2secret" not in S.redact('password = "hunter2secret"')
    assert "abc.def-ghi_jkl" not in S.redact("Authorization: Bearer abc.def-ghi_jkl")
    assert "k" * 12 not in S.redact("custom " + "k" * 12, extra_values=["k" * 12])
    out, count = S.redact_count(f"a {FAKE_GROQ} b {FAKE_GEMINI_AQ}")
    assert count == 2 and FAKE_GROQ not in out


def test_redact_keeps_normal_text():
    samples = ["ترەیدینگ ڤیو بکەرەوە", "C:/Users/samit/SAM Projects/site/index.html",
               "https://www.tradingview.com/chart/abc/", "tokens_in: 1234567 tokens_out: 42", "price 2650.35"]
    for sample in samples:
        assert S.redact(sample) == sample


def test_redact_obj_masks_secret_named_fields():
    obj = {"api_key": "short-ish", "nested": [{"token": "abcdef123"}, f"x {FAKE_GROQ}"], "count": 3}
    out = S.redact_obj(obj)
    assert out["api_key"] == S.MASK and out["nested"][0]["token"] == S.MASK
    assert FAKE_GROQ not in out["nested"][1] and out["count"] == 3


def test_log_filter_redacts_messages_and_exceptions(tmp_path, caplog):
    logger = logging.getLogger("sam.test.redaction")
    handler = logging.FileHandler(tmp_path / "log.txt", encoding="utf-8")
    logging.getLogger().addHandler(handler)
    try:
        S.install_log_redaction(None)
        logger.error("key is %s", FAKE_GROQ)
        try:
            raise RuntimeError(f"bad key {FAKE_GEMINI}")
        except RuntimeError:
            logger.exception("failed")
        handler.flush()
        content = (tmp_path / "log.txt").read_text(encoding="utf-8")
        assert FAKE_GROQ not in content and FAKE_GEMINI not in content
        assert "[REDACTED]" in content
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()
