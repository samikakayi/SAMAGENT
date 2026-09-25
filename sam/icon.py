"""SAM 2's application icon, drawn with Pillow (no image files in the repo).

The mark: a dark rounded square with a glowing orb -- the island's avatar --
and a white "S" monogram. Every ICO size is drawn separately at 4x and
downsampled, because one 256 px image shrunk to 16 px turns the "S" into a
smudge; below 32 px the ring and glow are dropped so the letter stays legible
in the taskbar and the tray.

``python -m sam.icon <path.ico>`` writes the icon (used by
``scripts/install.ps1`` through ``SAM.pyw --write-icon``). No Qt here: the UI
can turn :func:`make_icon_image` into a QIcon itself.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

ICO_SIZES: tuple[int, ...] = (16, 20, 24, 32, 40, 48, 64, 128, 256)
_SUPERSAMPLE = 4

BG_TOP = (23, 32, 72)        # deep indigo
BG_BOTTOM = (9, 12, 30)      # near black
ORB_INNER = (94, 234, 212)   # teal glow
ORB_OUTER = (79, 70, 229)    # indigo edge
RING = (125, 211, 252)       # light cyan ring
LETTER = (255, 255, 255)

_FONTS = ("segoeuib.ttf", "seguisb.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf", "arial.ttf")


def _font(px: int) -> Any:
    from PIL import ImageFont

    for name in _FONTS:
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))  # type: ignore[return-value]


def make_icon_image(size: int = 256) -> Any:
    """One RGBA ``PIL.Image`` of the SAM mark at ``size`` x ``size`` px."""
    from PIL import Image, ImageDraw, ImageFilter

    big = size * _SUPERSAMPLE
    small = size < 32
    image = Image.new("RGBA", (big, big), (0, 0, 0, 0))

    # Background: vertical gradient clipped to a rounded square.
    gradient = Image.new("RGBA", (big, big))
    draw = ImageDraw.Draw(gradient)
    for y in range(big):
        draw.line([(0, y), (big, y)], fill=_lerp(BG_TOP, BG_BOTTOM, y / max(1, big - 1)) + (255,))
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, big - 1, big - 1), radius=int(big * 0.23), fill=255)
    image.paste(gradient, (0, 0), mask)

    # Orb: radial gradient disc (+ soft glow and a thin ring at larger sizes).
    centre = big / 2
    radius = big * (0.40 if small else 0.34)
    if not small:
        glow = Image.new("RGBA", (big, big), (0, 0, 0, 0))
        ImageDraw.Draw(glow).ellipse((centre - radius * 1.25, centre - radius * 1.25,
                                      centre + radius * 1.25, centre + radius * 1.25),
                                     fill=ORB_INNER + (90,))
        glow = glow.filter(ImageFilter.GaussianBlur(big * 0.06))
        image = Image.alpha_composite(image, glow)
    orb = Image.new("RGBA", (big, big), (0, 0, 0, 0))
    orb_draw = ImageDraw.Draw(orb)
    steps = 48
    for i in range(steps):
        t = i / (steps - 1)
        r = radius * (1 - t)
        # The highlight drifts up-left so the orb reads as a lit sphere.
        cx = cy = centre - radius * 0.28 * t
        orb_draw.ellipse((cx - r, cy - r, cx + r, cy + r),
                         fill=_lerp(ORB_OUTER, ORB_INNER, t ** 1.4) + (255,))
    image = Image.alpha_composite(image, orb)
    draw = ImageDraw.Draw(image)
    if not small:
        width = max(2, int(big * 0.012))
        draw.ellipse((centre - radius, centre - radius, centre + radius, centre + radius),
                     outline=RING + (200,), width=width)

    # Monogram "S", optically centred on the orb.
    font = _font(int(radius * (1.55 if small else 1.35)))
    box = draw.textbbox((0, 0), "S", font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    draw.text((centre - w / 2 - box[0], centre - h / 2 - box[1]), "S", font=font, fill=LETTER + (255,))

    return image.resize((size, size), Image.Resampling.LANCZOS)


def write_icon(path: Path | str, sizes: tuple[int, ...] = ICO_SIZES) -> Path:
    """Write a multi-size Windows ``.ico`` (each size drawn separately)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frames = [make_icon_image(s) for s in sorted(sizes, reverse=True)]
    frames[0].save(target, format="ICO", sizes=[(s, s) for s in sizes], append_images=frames[1:])
    return target


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: python -m sam.icon <path.ico>", file=sys.stderr)
        return 2
    write_icon(args[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["make_icon_image", "write_icon", "ICO_SIZES"]
