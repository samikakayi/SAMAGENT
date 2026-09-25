"""Live check: one real strategy ingest through the free ``extract`` ladder.

    set SAM_HOME=C:\\Users\\samit\\Desktop\\SAM-Agent
    .venv\\Scripts\\python.exe acceptance\\engine_strategy_live.py

Keys are read only by SAM's own SecretStore/LLM client (never printed). The
card is written to a temporary database under work/, not to the user's
strategy memory; usage counters still go to the real sam2.sqlite3 so the
daily quota count stays right. Exactly ONE model request (plus the ladder's
own fallbacks if a rung fails).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sam.app import App  # noqa: E402
from sam.db import Database  # noqa: E402
from sam.textnorm import is_arabic_script  # noqa: E402
from sam.trading.strategies import StrategyStore  # noqa: E402

TEXT = ("ستراتیژی ئاسیا سویپ: لە چارتی یەک کاتژمێر ترێند دیاری دەکەم. دوای ئەوەی نرخ نزمایی سیشنی ئاسیا "
        "ڕادەماڵێت، لەسەر پازدە خولەک چاوەڕێی شکانی پێکهاتە دەکەم و لە ناو FVG دەچمە ژوورەوە. ستۆپ لە ژێر "
        "نزمایی ڕاماڵین، ئامانج یەک بە دوو. تەنها لە سیشنی لەندەن.")


async def main() -> int:
    app = App(os.environ.get("SAM_HOME"))
    scratch = ROOT / "work" / "acceptance-strategy.sqlite3"
    scratch.unlink(missing_ok=True)
    store = StrategyStore(app)
    store.app = type("ScratchApp", (), {"db": Database(scratch), "llm": app.llm, "config": app.config})()
    began = time.perf_counter()
    try:
        result = await store.ingest(TEXT)
    finally:
        await app.llm.aclose()
    card = result["card"]
    print(json.dumps({
        "seconds": round(time.perf_counter() - began, 2), "model": result["model"], "id": card["id"],
        "title_ckb": card["title_ckb"], "markets": card["markets"], "timeframes": card["timeframes"],
        "sessions": card["sessions"], "risk": card["risk"],
        "rules": [{"kind": r["kind"], "text_ckb": r["text_ckb"], "check": r["check"]} for r in card["rules"]],
        "missing": result["missing"], "readback_ckb": result["readback_ckb"],
        "readback_is_sorani": is_arabic_script(result["readback_ckb"]),
        "raw_predicates": [(r.get("predicate"), r.get("params")) for r in result.get("raw_rules") or []],
    }, ensure_ascii=False, indent=1))
    app.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
