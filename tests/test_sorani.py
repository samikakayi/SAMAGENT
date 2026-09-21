"""Sorani speech and command tests.

The user speaks Kurdish Sorani, so these cover the two things that would be
worst to get wrong: claiming Sorani works when it does not, and quietly turning
Sorani speech into some other language that merely looks like a transcript.
"""

from __future__ import annotations

import json

import numpy
import pytest

from sam_backend import sorani
from sam_backend import sorani_intent
from sam_backend.contracts import CapabilityState

# --- Normalisation ------------------------------------------------------------


def test_arabic_indic_digits_are_folded():
    assert sorani_intent.normalize("بچۆ ٥ خولەکی") == "بچۆ 5 خولەکی"


def test_extended_arabic_indic_digits_are_folded():
    assert sorani_intent.normalize("۱۵") == "15"


def test_arabic_letter_variants_are_folded_to_kurdish_ones():
    # Arabic yeh/kaf are what most keyboards emit; Kurdish uses its own forms.
    assert sorani_intent.normalize("كوردي") == "کوردی"


def test_punctuation_and_bidi_marks_do_not_block_matching():
    assert sorani_intent.normalize("سام،‏ بچۆ!") == "سام بچۆ"


# --- Timeframes ---------------------------------------------------------------


@pytest.mark.parametrize("phrase,expected", [
    ("بچۆ بۆ پێنج خولەکی", "M5"),
    ("بچۆ ٥ خولەکی", "M5"),
    ("بچۆ 5m", "M5"),
    ("پازدە خولەک", "M15"),
    ("دە خولەک", "M10"),
    ("بچۆ بۆ کاتژمێرێک", "H1"),
    ("چوار کاتژمێر", "H4"),
    ("ڕۆژانە", "D1"),
    ("هەفتانە", "W1"),
])
def test_timeframes_are_understood_however_they_are_said(phrase, expected):
    assert sorani_intent.parse_timeframe(phrase) == expected


def test_a_longer_number_word_wins_over_one_contained_in_it():
    """"پازدە" (15) contains "دە" (10) and must not be read as ten."""
    assert sorani_intent.parse_timeframe("پازدە خولەک") == "M15"


def test_a_phrase_with_no_timeframe_returns_nothing():
    assert sorani_intent.parse_timeframe("شیکاری بکە") is None


# --- Symbols and theories -----------------------------------------------------


@pytest.mark.parametrize("phrase,symbol", [
    ("زێڕ بپشکنە", "XAUUSD"),
    ("شیکاری زیو", "XAGUSD"),
    ("بیتکۆین", "BTCUSD"),
    ("XAUUSD", "XAUUSD"),
])
def test_symbols_are_recognised_from_kurdish_names(phrase, symbol):
    assert sorani_intent.parse_symbol(phrase) == symbol


def test_an_unknown_instrument_is_not_guessed():
    assert sorani_intent.parse_symbol("شتێکی تر") is None


def test_theories_are_recognised_by_kurdish_and_english_names():
    found = sorani_intent.parse_theories("وایکۆف و ساپۆرت و ict")
    assert set(found) == {"wyckoff", "snr", "ict"}


# --- Stop and wake ------------------------------------------------------------


@pytest.mark.parametrize("phrase", ["وەستە", "بەسە", "ڕابگرە", "stop"])
def test_stop_is_understood_in_sorani(phrase):
    assert sorani_intent.is_stop_command(phrase) is True


def test_an_ordinary_sentence_is_not_a_stop_command():
    assert sorani_intent.is_stop_command("شیکاری زێڕ بکە") is False


def test_the_wake_word_is_heard_in_sorani():
    assert sorani_intent.has_wake_word("سام، شیکاری بکە") is True
    assert sorani_intent.has_wake_word("شیکاری بکە") is False


# --- Intents ------------------------------------------------------------------


@pytest.mark.parametrize("phrase,action", [
    ("وەستە", "stop"),
    ("ترەیدینگ ڤیو بکەرەوە", "open_tradingview"),
    ("هێڵەکان بسڕەوە", "clear_drawings"),
    ("ئاستەکان بکێشە", "draw_levels"),
    ("چاودێری بکە", "monitor_setup"),
    ("بەراورد بکە", "compare_theories"),
    ("بچۆ بۆ پێنج خولەکی", "set_timeframe"),
    ("بچۆ بۆ زێڕ", "set_symbol"),
    ("شیکاری بکە", "analyze"),
])
def test_spoken_commands_map_to_actions(phrase, action):
    parsed = sorani_intent.parse(phrase)
    assert parsed is not None, phrase
    assert parsed.action == action


def test_a_command_carries_its_arguments():
    parsed = sorani_intent.parse("شیکاری زێڕ بکە بە وایکۆف")
    assert parsed.action == "analyze"
    assert parsed.arguments["symbol"] == "XAUUSD"
    assert parsed.arguments["theories"] == ["wyckoff"]


def test_a_symbol_and_a_timeframe_together_switch_the_symbol():
    parsed = sorani_intent.parse("بچۆ بۆ زێڕ")
    assert parsed.action == "set_symbol"
    assert parsed.arguments["symbol"] == "XAUUSD"


def test_an_unrecognised_phrase_returns_nothing_rather_than_a_guess():
    """Inventing an action from an unclear phrase could move a real chart."""
    assert sorani_intent.parse("ئەمڕۆ کەشوهەوا چۆنە") is None
    assert sorani_intent.parse("") is None


# --- Language and script detection --------------------------------------------


@pytest.mark.parametrize("code", ["ckb", "ckb-IQ", "ku", "sorani"])
def test_sorani_language_codes_are_recognised(code):
    assert sorani.is_sorani(code) is True


@pytest.mark.parametrize("code", ["en", "ar", "fa", "tr", None, ""])
def test_other_languages_are_not_treated_as_sorani(code):
    assert sorani.is_sorani(code) is False


def test_kurdish_script_is_detected_in_a_reply():
    assert sorani.looks_sorani("زێڕ لە ئاستی ٤٠٨ دایە") is True


def test_latin_text_is_not_detected_as_kurdish():
    assert sorani.looks_sorani("Gold is at 408") is False
    assert sorani.looks_sorani("") is False
    assert sorani.looks_sorani("408.50") is False


# --- Audio conversion ---------------------------------------------------------


def test_pcm_survives_a_round_trip_through_wav():
    original = numpy.linspace(-0.5, 0.5, 1600, dtype=numpy.float32)
    restored, rate = sorani.wav_to_float32(sorani.pcm_to_wav(original, 16_000))
    assert rate == 16_000
    assert len(restored) == len(original)
    # 16-bit quantisation is the only loss allowed here.
    assert float(numpy.max(numpy.abs(restored - original))) < 1e-3


# --- Provider status honesty --------------------------------------------------


def test_an_unconfigured_provider_reports_unconfigured_not_broken():
    stt, tts = sorani.build_routers(stt_key=None, tts_key=None)
    assert stt.status()["primary"]["status"] == "UNCONFIGURED"
    assert tts.status()["primary"]["status"] == "UNCONFIGURED"


def test_stt_health_does_not_spend_a_recognition_round_trip():
    """Status must not transcribe silence; that delay used to sit in front of every spoken turn."""
    client = sorani.KurdishTTSClient(stt_key="present-key", tts_key=None)

    def boom(*_args, **_kwargs):
        raise AssertionError("health_check must not call transcribe")

    client.transcribe = boom
    status = sorani.KurdishTTSSTTProvider(client).health_check()
    assert status["status"] == "CONNECTED"
    assert status["provider"] == "kurdishtts"


def test_provider_messages_are_written_in_sorani():
    """The user reads Kurdish, so a provider failure has to be readable to them."""
    for message in (sorani.MESSAGE_NO_PROVIDER, sorani.MESSAGE_AUTH_FAILED,
                    sorani.MESSAGE_RATE_LIMITED, sorani.MESSAGE_UNREACHABLE):
        assert sorani.looks_sorani(message), message


@pytest.mark.parametrize("code,expected", [
    (401, "AUTH_FAILED"), (403, "AUTH_FAILED"), (429, "RATE_LIMITED"), (500, "ERROR"),
])
def test_http_failures_are_classified_into_states_not_raw_errors(code, expected):
    assert sorani._classify(code)[0] == expected


def test_status_is_states_only_and_never_a_credential():
    stt, tts = sorani.build_routers(stt_key="stt-secret-000", tts_key="tts-secret-111")
    blob = json.dumps({"stt": stt.status(), "tts": tts.status()}, ensure_ascii=False)
    assert "stt-secret-000" not in blob
    assert "tts-secret-111" not in blob


# --- Fallback routing ---------------------------------------------------------


class _FakePrimary:
    name = "kurdishtts"

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def transcribe_audio(self, samples, sample_rate=16_000):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def health_check(self):
        return {"provider": "kurdishtts", "role": "stt", "status": "CONNECTED",
                "state": CapabilityState.AVAILABLE.value}


class _FakeFallback:
    def __init__(self, connected=True):
        self.connected = connected
        self.calls = 0

    def transcribe_audio(self, samples, sample_rate=16_000):
        self.calls += 1
        return {"text": "google said this", "language": "ckb"}

    def health_check(self):
        return {"provider": "google", "role": "stt",
                "status": "CONNECTED" if self.connected else "UNCONFIGURED",
                "state": CapabilityState.AVAILABLE.value, "language": "ckb-IQ"}


def test_a_surprising_transcript_does_not_trigger_the_fallback():
    """Only a provider failure is a fallback condition, never the content.

    Re-running because a transcript looked odd would silently swap in a second
    opinion about what the user said.
    """
    primary = _FakePrimary({"text": "شتێکی نامۆ", "language": "ckb"})
    fallback = _FakeFallback()
    router = sorani.SoraniSTTRouter(primary, fallback)
    result = router.transcribe(numpy.zeros(160, dtype=numpy.float32))
    assert result["route"] == "kurdishtts"
    assert fallback.calls == 0


def test_a_transport_failure_does_trigger_the_fallback():
    import httpx

    primary = _FakePrimary(httpx.ConnectError("down"))
    fallback = _FakeFallback()
    router = sorani.SoraniSTTRouter(primary, fallback)
    result = router.transcribe(numpy.zeros(160, dtype=numpy.float32))
    assert result["route"] == "google"
    assert fallback.calls == 1


def test_without_a_fallback_a_failure_is_reported_not_hidden():
    import httpx

    router = sorani.SoraniSTTRouter(_FakePrimary(httpx.ConnectError("down")), None)
    with pytest.raises(RuntimeError):
        router.transcribe(numpy.zeros(160, dtype=numpy.float32))


def test_a_disconnected_fallback_is_not_used():
    import httpx

    fallback = _FakeFallback(connected=False)
    router = sorani.SoraniSTTRouter(_FakePrimary(httpx.ConnectError("down")), fallback)
    with pytest.raises(RuntimeError):
        router.transcribe(numpy.zeros(160, dtype=numpy.float32))
    assert fallback.calls == 0


def test_the_tts_router_never_substitutes_another_language():
    _, tts = sorani.build_routers(stt_key=None, tts_key=None)
    assert "never substituted" in tts.status()["note"].lower()


# --- Streaming ----------------------------------------------------------------


def test_audio_is_streamed_in_pieces_so_playback_can_start_early():
    payload = b"x" * 100_000
    chunks = list(sorani.stream_chunks(payload, 32_768))
    assert b"".join(chunks) == payload
    assert len(chunks) == 4


# --- Recognition noise --------------------------------------------------------
#
# Every phrase below is real output captured from the live KurdishTTS recogniser
# while speaking the command in the comment. Written Sorani and recognised
# Sorani are not the same string, and the parser has to cope with the second.


@pytest.mark.parametrize("heard,action", [
    ("چووە بۆ پێنج خولەکی.", "set_timeframe"),    # said: بچۆ بۆ پێنج خولەکی
    ("دەچێ بۆ پێنج خولەکی.", "set_timeframe"),    # said: بچۆ بۆ پێنج خولەکی
    ("چی کاری زێڕ بکە!", "analyze"),              # said: شیکاری زێڕ بکە
    ("کێڵەکان بسڕەوە!", "clear_drawings"),        # said: هێڵەکان بسڕەوە
    ("هێمەکان بسڕەوە!", "clear_drawings"),        # said: هێڵەکان بسڕەوە
    ("وەستا!", "stop"),                            # said: وەستە
    ("تکایە وەستا.", "stop"),                      # said: تکایە وەستە
    ("کایە وەستا.", "stop"),                       # said: تکایە وەستە
    ("وەستا بەسە!", "stop"),                       # said: وەستە، بەسە
])
def test_commands_survive_real_recognition_noise(heard, action):
    parsed = sorani_intent.parse(heard)
    assert parsed is not None, heard
    assert parsed.action == action


def test_a_recognised_timeframe_keeps_its_value_through_the_noise():
    parsed = sorani_intent.parse("دەچێ بۆ پێنج خولەکی.")
    assert parsed.arguments["timeframe"] == "M5"


# --- The fuzzy matcher must not match everything ------------------------------


def test_fuzzy_matching_accepts_a_near_miss():
    assert sorani_intent.fuzzy_contains("چی کاری زێڕ بکە", "شیکاری") is True


def test_fuzzy_matching_rejects_a_different_word():
    assert sorani_intent.fuzzy_contains("کەشوهەوا چۆنە", "شیکاری") is False
    assert sorani_intent.fuzzy_contains("زێڕ", "بسڕەوە") is False


def test_short_keywords_are_never_fuzzy_matched():
    """Below four letters a near-match is more likely a different word."""
    assert sorani_intent.fuzzy_contains("بکە", "بچۆ") is False


def test_multi_word_keywords_are_matched_literally():
    assert sorani_intent.fuzzy_contains("smart monkey", "smart money") is False


def test_ordinary_conversation_still_maps_to_no_action():
    """Loosening the matching must not turn small talk into chart commands."""
    for phrase in ("ئەمڕۆ کەشوهەوا چۆنە", "سوپاس بۆ یارمەتیت", "چۆنی باشی"):
        assert sorani_intent.parse(phrase) is None, phrase


# --- The measured provider limit ----------------------------------------------


def test_the_short_utterance_limit_is_recorded():
    """Sub-second audio is unreliable, so stop cannot depend on recognising it."""
    assert sorani.MIN_RELIABLE_SECONDS > 0
    assert sorani.looks_sorani(sorani.MESSAGE_TOO_SHORT)


# --- Secret storage on Windows -------------------------------------------------


def test_the_secret_store_is_not_readable_by_other_accounts(tmp_path):
    """chmod does nothing on Windows, so the store is locked down by ACL.

    Before this was added the file inherited its directory's ACL, which on this
    machine granted a second account group read access to every API key.
    """
    import os

    from sam_backend.secrets import SecretStore

    store = SecretStore(tmp_path)
    store.set("kurdishtts_tts_api_key", "a" * 40)
    summary = store.acl_summary()
    assert summary["exists"] is True
    assert summary["path_inside_web_root"] is False
    if os.name == "nt":
        # Exactly one principal: the account that owns the file.
        assert len(summary["principals"]) == 1, summary["principals"]


def test_the_secret_store_never_returns_a_value_in_its_public_status(tmp_path):
    from sam_backend.secrets import SecretStore

    store = SecretStore(tmp_path)
    secret = "b" * 40
    store.set("kurdishtts_stt_api_key", secret)
    blob = json.dumps(store.public_status())
    assert secret not in blob
    assert store.public_status()["kurdishtts_stt_api_key"]["configured"] is True
