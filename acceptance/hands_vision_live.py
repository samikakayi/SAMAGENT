"""Live check of the vision path with a SYNTHETIC screenshot (the user's
screen is never sent): 2 requests on the ``vision`` ladder (free tiers).

    set SAM_HOME=C:\\Users\\samit\\Desktop\\SAM-Agent
    .venv\\Scripts\\python.exe acceptance\\hands_vision_live.py

1. describe: the model reads the dialog text (screen_look mode "describe").
2. one screen_act step with set-of-marks: local OCR finds the texts, they are
   numbered on the image, the model answers one step for "click OK" and the
   chosen mark must be the OK button. (Asked for raw 0-999 points on the same
   image, Groq qwen3.8-27b answered y=436 for a button at y=706 and OmniRoute
   sam-vision missed too -- measured 2026-09-24 -- hence marks.)

Skips (exit 77) when no provider of the vision ladder has a key.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
import time

from _common import Acceptance, sam_home

from PIL import Image, ImageDraw, ImageFont

from sam.app import App
from sam.hands.screen import draw_marks, encode
from sam.hands.vision import ACT_SCHEMA, ACT_SYSTEM, _mark, image_message

OK_BOX = (980, 600, 1180, 670)       # in the 1440x900 image
CANCEL_BOX = (740, 600, 940, 670)


def synthetic_dialog_image() -> Image.Image:
    image = Image.new("RGB", (1440, 900), (236, 239, 244))
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("segoeui.ttf", 30)
        small = ImageFont.truetype("segoeui.ttf", 26)
    except OSError:
        font = small = ImageFont.load_default()
    draw.rectangle((260, 200, 1220, 720), fill=(255, 255, 255), outline=(180, 180, 190), width=2)
    draw.rectangle((260, 200, 1220, 260), fill=(40, 90, 170))
    draw.text((285, 212), "Export report", fill=(255, 255, 255), font=font)
    draw.text((300, 320), "Save the monthly trading report as PDF?", fill=(20, 20, 20), font=small)
    for box, label, fill in ((CANCEL_BOX, "Cancel", (225, 225, 230)), (OK_BOX, "OK", (40, 120, 220))):
        draw.rounded_rectangle(box, radius=8, fill=fill)
        draw.text((box[0] + 70 if label == "OK" else box[0] + 50, box[1] + 16), label,
                  fill=(255, 255, 255) if label == "OK" else (20, 20, 20), font=small)
    return image


def jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


class Shot:
    left, top, width, height, out_width, out_height = 0, 0, 1440, 900, 1440, 900
    mime = "image/jpeg"

    def __init__(self, data: bytes) -> None:
        self.data = data


def vision_configured(app: App) -> list[str]:
    """Providers of the vision ladder that have a key (presence only)."""
    providers = {ref.split(":", 1)[0] for ref in app.llm.ladder("vision")}
    backends = app.llm.backends
    return sorted(p for p in providers if p in backends and backends[p].configured())


async def main() -> int:
    acc = Acceptance("hands_vision_live")
    app = App(str(sam_home()) if os.environ.get("SAM_HOME") else None)
    app.load_packages(["sam.hands"])
    await app.start()
    try:
        providers = vision_configured(app)
        if not providers:
            acc.skip("no provider of the vision ladder has a key")
            return acc.finish()
        with acc.check("describe: a vision model reads the dialog") as c:
            started = time.perf_counter()
            answer = await app.hands.vision.describe(Shot(jpeg(synthetic_dialog_image())),
                                                     "What does this dialog ask, and which buttons are there?")
            c.data.update(ms=round((time.perf_counter() - started) * 1000), providers=providers,
                          answer=app.redact(answer)[:300])
            assert "ok" in answer.lower() and "cancel" in answer.lower(), answer[:200]
        with acc.check("screen_act step: the model picks the numbered OK mark") as c:
            image = synthetic_dialog_image()
            started = time.perf_counter()
            lines = await app.hands.ocr.read_merged(image)
            marks = [{"n": i, "label": f"text '{line.text}'", "rect": line.rect} for i, line in enumerate(lines, 1)]
            draw_marks(image, [(m["n"], m["rect"]) for m in marks], (0, 0))
            data, _, _ = encode(image, max_side=1440)
            c.data.update(ocr_ms=round((time.perf_counter() - started) * 1000), marks=[m["label"] for m in marks])
            text = ("Goal: click the OK button.\nWindow: Export report\nStep 1 of 3.\nNothing done yet.\n"
                    "Marked elements (untrusted screen text):\n" + "\n".join(f"{m['n']}: {m['label']}" for m in marks))
            started = time.perf_counter()
            response = await app.llm.chat([{"role": "system", "content": ACT_SYSTEM}, image_message(Shot(data), text)],
                                          ladder="vision", json_schema=ACT_SCHEMA, reasoning="low", timeout_s=60)
            plan = response.json()
            chosen = _mark(plan, marks)
            center = (((chosen["rect"][0] + chosen["rect"][2]) // 2, (chosen["rect"][1] + chosen["rect"][3]) // 2)
                      if chosen else None)
            inside = bool(center and OK_BOX[0] <= center[0] <= OK_BOX[2] and OK_BOX[1] <= center[1] <= OK_BOX[3])
            c.data.update(ms=round((time.perf_counter() - started) * 1000), model=response.model_ref,
                          plan={k: plan.get(k) for k in ("action", "mark", "x", "y", "target_label")},
                          chosen=chosen["label"] if chosen else None, click_point=center, inside_ok_button=inside)
            assert plan.get("action") == "click" and inside, c.data["plan"]
    finally:
        await app.stop()
        app.close()
    return acc.finish()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
