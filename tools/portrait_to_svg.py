#!/usr/bin/env python3
"""Turn a studio portrait into the animated pixel-art hero card used in this README.

    python3 tools/portrait_to_svg.py photo.jpg --svg assets/hero.svg --ascii ascii.txt

Pipeline: flood-fill the (bright, colourless) studio background away, lift the near-black
clothing so it reads on a dark card, quantise to a small palette, run-length encode every
row into one <path> per colour, then wrap it in a boot-up animation that plays once and
freezes. No <image> tag, no raster payload — just rectangles.

Requires: pillow, numpy.
"""
import argparse
from collections import defaultdict, deque

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

GW, GH, S = 64, 84, 4          # pixel grid + on-canvas scale -> 256 x 336
PX, PY = 40, 42                # portrait origin inside the card
W, H = 1000, 420               # card size
CYAN, DIM = "#00d4ff", "#8fa8b8"
INTRO = "4.2s"                 # boot sequence: plays once, then freezes assembled

LUM_MIN, CHROMA_MAX = 140, 20  # studio grey is bright and colourless; skin sits at chroma ~80
RAMP = "@%#*+=~-:'. "          # index 0 = darkest -> densest glyph (ink-on-paper)
BAYER = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0


# ---------------------------------------------------------------- source pixels
def load(src, gw, gh, bottom_crop=0.97):
    im = Image.open(src).convert("RGB")
    w, h = im.size
    return im.crop((0, 0, w, int(h * bottom_crop))).resize((gw, gh), Image.LANCZOS)


def bg_mask(arr):
    """Flood fill the light studio background inward from the borders."""
    gh, gw = arr.shape[:2]
    lum = arr @ np.array([0.299, 0.587, 0.114])
    cand = (lum > LUM_MIN) & (arr.max(2) - arr.min(2) < CHROMA_MAX)
    seen = np.zeros((gh, gw), bool)
    q = deque()
    for y in range(gh):
        for x in range(gw):
            if (y in (0, gh - 1) or x in (0, gw - 1)) and cand[y, x]:
                seen[y, x] = True
                q.append((y, x))
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < gh and 0 <= nx < gw and not seen[ny, nx] and cand[ny, nx]:
                seen[ny, nx] = True
                q.append((ny, nx))
    return seen  # True == background


def build(src, gw, gh, colors=30, lift=(26, 30, 40), contrast=1.16, sat=1.14):
    small = load(src, gw, gh)
    fg = ~bg_mask(np.asarray(small).astype(float))
    small = ImageEnhance.Color(ImageEnhance.Contrast(small).enhance(contrast)).enhance(sat)
    arr = np.asarray(small).astype(float)

    lum = arr @ np.array([0.299, 0.587, 0.114])
    t = np.clip(1.0 - lum / 90.0, 0, 1)[..., None]           # only the darkest zone
    arr = arr * (1 - t) + np.maximum(arr, np.array(lift, float)) * t
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    q = Image.fromarray(arr).quantize(colors=colors, method=Image.MEDIANCUT, dither=Image.NONE)
    pal = np.array(q.getpalette()[: colors * 3]).reshape(-1, 3)
    return pal[np.asarray(q)], fg


def rings(fg):
    """Two rings of rim light just outside the silhouette."""
    def dilate(m):
        out = m.copy()
        out[1:, :] |= m[:-1, :]; out[:-1, :] |= m[1:, :]
        out[:, 1:] |= m[:, :-1]; out[:, :-1] |= m[:, 1:]
        return out
    r1 = dilate(fg) & ~fg
    return r1, dilate(dilate(fg)) & ~fg & ~r1


def runs_by_color(px, fg, gw, gh):
    """Run-length encode each row, grouped by colour."""
    out = defaultdict(list)
    for y in range(gh):
        x = 0
        while x < gw:
            if not fg[y, x]:
                x += 1
                continue
            c, x0 = tuple(px[y, x]), x
            while x + 1 < gw and fg[y, x + 1] and tuple(px[y, x + 1]) == c:
                x += 1
            out["#%02x%02x%02x" % c].append((y, x0, x - x0 + 1))
            x += 1
    return out


def cells_path(cells):
    return "".join(f"M{PX + x * S} {PY + y * S}h{w * S}v{S}h-{w * S}z" for y, x, w in cells)


def mask_path(mask):
    return "".join(f"M{PX + int(x) * S} {PY + int(y) * S}h{S}v{S}h-{S}z" for y, x in zip(*np.where(mask)))


# ---------------------------------------------------------------- text portraits
def text_grid(src, cols, aspect=0.42):
    w, h = Image.open(src).size
    h2 = int(h * 0.97)
    rows = int(round(cols * h2 / w * aspect))
    mw = 192
    m = ~bg_mask(np.asarray(load(src, mw, int(round(mw * h2 / w)))).astype(float))
    fg = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).resize((cols, rows), Image.BILINEAR)) > 115
    g = Image.open(src).convert("L").crop((0, 0, w, h2))
    g = g.filter(ImageFilter.UnsharpMask(radius=45, percent=120, threshold=2))
    a = np.asarray(g.resize((cols, rows), Image.LANCZOS)).astype(float)
    lo, hi = np.percentile(a[fg], 1.5), np.percentile(a[fg], 99)
    return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1), fg, cols, rows


def ascii_art(src, cols=84):
    n, fg, w, h = text_grid(src, cols)
    idx = (n * (len(RAMP) - 1)).round().astype(int)
    return "\n".join("".join(RAMP[idx[y, x]] if fg[y, x] else " " for x in range(w)).rstrip()
                     for y in range(h))


def binary_art(src, cols=84):
    n, fg, w, h = text_grid(src, cols)
    th = np.array([[BAYER[y % 4, x % 4] for x in range(w)] for y in range(h)])
    bits = n > th
    return "\n".join("".join(("1" if bits[y, x] else "0") if fg[y, x] else " " for x in range(w)).rstrip()
                     for y in range(h))


# ---------------------------------------------------------------- the card
def svg(src):
    px, fg = build(src, GW, GH)
    r1, r2 = rings(fg)
    pix = "".join(f'<path fill="{c}" d="{cells_path(v)}"/>'
                  for c, v in runs_by_color(px, fg, GW, GH).items())
    ring1 = f'<path fill="{CYAN}" opacity=".55" d="{mask_path(r1)}"/>'
    ring2 = f'<path fill="{CYAN}" opacity=".16" d="{mask_path(r2)}"/>'

    RX, PW, PH = 330, GW * S, GH * S
    lines = [
        "&#9656; agentic pipelines  ::  n8n &#183; LangGraph &#183; MCP &#183; RAG",
        "&#9656; production backends ::  Django &#183; FastAPI &#183; Docker &#183; PostgreSQL",
        "&#9656; AI Engineer @ BhivesAI  ::  Amsterdam, NL",
    ]
    chips = ["Python", "PyTorch", "Azure AI", "n8n", "Docker", "RAG", "LLMs"]

    def fade(begin):
        """opacity 0 -> 1 at `begin` fraction of the intro, then frozen visible."""
        return (f'<animate attributeName="opacity" values="0;0;1;1" '
                f'keyTimes="0;{begin};{begin + 0.03};1" dur="{INTRO}" repeatCount="1" fill="freeze"/>')

    out = []
    a = out.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
      f'font-family="ui-monospace,SFMono-Regular,Menlo,Consolas,monospace" role="img" '
      f'aria-label="Yassine Ghilani - AI Engineer">')
    a('<defs>')
    a('<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0b1016"/>'
      '<stop offset=".55" stop-color="#0a0f15"/><stop offset="1" stop-color="#0d1a22"/></linearGradient>')
    a(f'<linearGradient id="sweep" x1="0" y1="0" x2="0" y2="1">'
      f'<stop offset="0" stop-color="{CYAN}" stop-opacity="0"/>'
      f'<stop offset=".65" stop-color="{CYAN}" stop-opacity=".35"/>'
      f'<stop offset="1" stop-color="#bff4ff" stop-opacity=".95"/></linearGradient>')
    a(f'<linearGradient id="name" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#ffffff"/>'
      f'<stop offset=".6" stop-color="#d8f4ff"/><stop offset="1" stop-color="{CYAN}"/></linearGradient>')
    a(f'<linearGradient id="rule" x1="0" y1="0" x2="1" y2="0">'
      f'<stop offset="0" stop-color="{CYAN}" stop-opacity=".9"/>'
      f'<stop offset="1" stop-color="{CYAN}" stop-opacity="0"/></linearGradient>')
    a(f'<pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse">'
      f'<path d="M24 0H0V24" fill="none" stroke="{CYAN}" stroke-opacity=".055" stroke-width="1"/></pattern>')
    a('<pattern id="scan" width="4" height="4" patternUnits="userSpaceOnUse">'
      '<rect width="4" height="2" fill="#000" fill-opacity=".16"/></pattern>')
    a(f'<clipPath id="reveal"><rect x="{PX}" y="{PY}" width="{PW}" height="{PH}">'
      f'<animate attributeName="height" values="0;{PH};{PH}" keyTimes="0;.36;1" '
      f'dur="{INTRO}" repeatCount="1" fill="freeze"/></rect></clipPath>')
    a(f'<clipPath id="frame"><rect x="{PX}" y="{PY}" width="{PW}" height="{PH}"/></clipPath>')
    a('<filter id="soft" x="-30%" y="-30%" width="160%" height="160%">'
      '<feGaussianBlur stdDeviation="7" result="b"/>'
      '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>')
    a('</defs>')

    a(f'<rect width="{W}" height="{H}" rx="22" fill="url(#bg)"/>')
    a(f'<rect width="{W}" height="{H}" rx="22" fill="url(#grid)"/>')
    a(f'<rect x=".75" y=".75" width="{W - 1.5}" height="{H - 1.5}" rx="21.5" fill="none" '
      f'stroke="{CYAN}" stroke-opacity=".28" stroke-width="1.5">'
      f'<animate attributeName="stroke-opacity" values=".14;.5;.14" dur="4s" repeatCount="indefinite"/></rect>')

    a(f'<rect x="{PX - 8}" y="{PY - 8}" width="{PW + 16}" height="{PH + 16}" rx="12" fill="#060a0e" '
      f'stroke="{CYAN}" stroke-opacity=".22"/>')
    a(f'<g clip-path="url(#reveal)">{ring2}{ring1}{pix}</g>')
    a(f'<rect x="{PX}" y="{PY}" width="{PW}" height="{PH}" fill="url(#scan)" clip-path="url(#frame)"/>')
    a(f'<rect x="{PX}" y="{PY - 30}" width="{PW}" height="30" fill="url(#sweep)" opacity="0" '
      f'clip-path="url(#frame)">'
      f'<animate attributeName="y" values="{PY - 30};{PY + PH};{PY + PH}" keyTimes="0;.36;1" '
      f'dur="{INTRO}" repeatCount="1" fill="freeze"/>'
      f'<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;.34;.40;1" '
      f'dur="{INTRO}" repeatCount="1" fill="freeze"/></rect>')
    for cx, cy, sx, sy in ((PX - 8, PY - 8, 1, 1), (PX + PW + 8, PY - 8, -1, 1),
                           (PX - 8, PY + PH + 8, 1, -1), (PX + PW + 8, PY + PH + 8, -1, -1)):
        a(f'<path d="M{cx} {cy + sy * 26}V{cy}H{cx + sx * 26}" fill="none" stroke="{CYAN}" '
          f'stroke-width="2.5" stroke-linecap="round" opacity=".85"/>')

    a(f'<text x="{RX}" y="{PY + 34}" font-size="13" fill="{CYAN}" fill-opacity=".85" '
      f'letter-spacing="1.5">$ whoami{fade(0.05)}</text>')
    a(f'<text x="{RX}" y="{PY + 92}" font-size="46" font-weight="700" letter-spacing="1" '
      f'fill="url(#name)" filter="url(#soft)" font-family="Segoe UI,system-ui,Helvetica,Arial,sans-serif">'
      f'YASSINE GHILANI{fade(0.12)}</text>')
    # ambient chromatic-split flicker, only after the name has landed
    a(f'<text x="{RX}" y="{PY + 92}" font-size="46" font-weight="700" letter-spacing="1" fill="{CYAN}" '
      f'fill-opacity=".45" opacity="0" font-family="Segoe UI,system-ui,Helvetica,Arial,sans-serif">'
      f'<animate attributeName="x" values="{RX};{RX + 3};{RX - 2};{RX};{RX}" keyTimes="0;.02;.05;.08;1" '
      f'dur="7s" begin="{INTRO}" repeatCount="indefinite"/>'
      f'<animate attributeName="opacity" values="0;.5;0;0" keyTimes="0;.03;.08;1" '
      f'dur="7s" begin="{INTRO}" repeatCount="indefinite"/>YASSINE GHILANI</text>')
    a(f'<text x="{RX}" y="{PY + 124}" font-size="14.5" fill="{CYAN}" letter-spacing="3.2">'
      f'AI ENGINEER &#183; LLM &amp; MULTI-AGENT SYSTEMS{fade(0.20)}</text>')
    a(f'<rect x="{RX}" y="{PY + 142}" width="630" height="1.6" fill="url(#rule)">'
      f'<animate attributeName="width" values="0;630;630" keyTimes="0;.40;1" dur="{INTRO}" '
      f'repeatCount="1" fill="freeze"/></rect>')

    for i, ln in enumerate(lines):
        a(f'<text x="{RX}" y="{PY + 184 + i * 30}" font-size="14.5" fill="{DIM}">{ln}{fade(0.46 + i * 0.08)}</text>')
    a(f'<rect x="{RX}" y="{PY + 264}" width="10" height="18" fill="{CYAN}" opacity="0">{fade(0.70)}'
      f'<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;.5;.5;1" dur="1.1s" '
      f'begin="{INTRO}" repeatCount="indefinite"/></rect>')

    x = RX
    for i, c in enumerate(chips):
        w = 15 + len(c) * 8.1
        a(f'<g>{fade(0.74 + i * 0.025)}'
          f'<rect x="{x:.0f}" y="{PY + 300}" width="{w:.0f}" height="26" rx="13" fill="{CYAN}" '
          f'fill-opacity=".09" stroke="{CYAN}" stroke-opacity=".45"/>'
          f'<text x="{x + w / 2:.0f}" y="{PY + 317}" font-size="12" fill="#bfe9f7" text-anchor="middle" '
          f'letter-spacing=".4">{c}</text></g>')
        x += w + 9

    a(f'<text x="{W - 34}" y="{H - 26}" font-size="11.5" fill="{DIM}" fill-opacity=".55" '
      f'text-anchor="end">github.com/GhilaniYassine{fade(0.92)}</text>')
    a('</svg>')
    return "".join(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("photo", help="source portrait (plain bright background works best)")
    p.add_argument("--svg", default="assets/hero.svg")
    p.add_argument("--ascii", help="also write the ASCII portrait here")
    p.add_argument("--binary", help="also write the 0/1 portrait here")
    args = p.parse_args()

    open(args.svg, "w").write(svg(args.photo))
    print("wrote", args.svg)
    if args.ascii:
        open(args.ascii, "w").write(ascii_art(args.photo))
        print("wrote", args.ascii)
    if args.binary:
        open(args.binary, "w").write(binary_art(args.photo))
        print("wrote", args.binary)
