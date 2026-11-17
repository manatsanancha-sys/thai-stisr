"""Synthetic Thai word dataset for STISR (HR / LR / 3-zone mask).

Per sample we write:
  <out>/<split>/hr/<id>.png    sharp image, black text on white
  <out>/<split>/lr/<id>.png    degraded + downscaled image (HR / scale)
  <out>/<split>/mask/<id>.png  3-channel HR-sized mask  R=upper  G=middle  B=lower
and one meta.csv with the word, split, font and degradation parameters.

Zones (by Unicode, as in the proposal):
  upper : U+0E31, U+0E34-U+0E37 (upper vowels), U+0E47-U+0E4E (tone marks, thanthakhat, ...)
  lower : U+0E38-U+0E3A (lower vowels)
  middle: everything else (consonants, leading/trailing vowels, sara am U+0E33 ...)

Reference masks are built from rendering differences: render the full word, render
it again without the upper marks / without the lower marks, and the ink that only
exists in the full render belongs to that zone. Nothing is hand-labelled.

Splits are made by WORD, so one word never appears in two splits.
"""
import argparse
import csv
import glob
import hashlib
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features
from pythainlp.corpus import thai_words

UPPER = {0x0E31, *range(0x0E34, 0x0E38), *range(0x0E47, 0x0E4F)}
LOWER = {0x0E38, 0x0E39, 0x0E3A}
THAI_ONLY = re.compile(r"^[ก-๎]+$")
BIG = 160  # font size used for the high-resolution render
PAD = 40


def has(word, zone):
    return any(ord(c) in zone for c in word)


def strip(word, zone):
    return "".join(c for c in word if ord(c) not in zone)


def load_fonts(font_dir):
    fonts = []
    for path in sorted(glob.glob(os.path.join(font_dir, "*.ttf"))):
        try:
            f = ImageFont.truetype(path, BIG, layout_engine=ImageFont.Layout.RAQM)
            if "NotoSansThai" in path:  # variable font; axis order is [Weight, Width]
                try:
                    f.set_variation_by_axes([400, 100])
                except Exception:
                    pass
            fonts.append((os.path.basename(path), path, f))
        except OSError as e:
            print(f"skip font {path}: {e}")
    if not fonts:
        raise SystemExit(f"no usable fonts in {font_dir}")
    return fonts


def render_ink(text, font, size):
    """Binary ink map (uint8 0/1) rendered at fixed origin so renders align."""
    img = Image.new("L", size, 0)
    ImageDraw.Draw(img).text((PAD, PAD), text, font=font, fill=255)
    return (np.asarray(img) > 127).astype(np.uint8)


def zone_inks(word, font):
    """Return (upper, middle, lower) binary ink maps at the big resolution."""
    w = int(font.getlength(word)) + 2 * PAD
    size = (w, int(BIG * 2.2) + 2 * PAD)
    full = render_ink(word, font, size)
    no_up = render_ink(strip(word, UPPER), font, size)
    no_lo = render_ink(strip(word, LOWER), font, size)
    # ink that exists only in the full render => belongs to the removed zone
    upper = (full & (1 - no_up)).astype(np.uint8)
    lower = (full & (1 - no_lo)).astype(np.uint8)
    middle = np.clip(full.astype(int) - upper - lower, 0, 1).astype(np.uint8)
    return full, upper, middle, lower


def fit(big, hr_h, hr_w, box):
    """Crop to the shared bbox, resize with aspect ratio kept, centre on white/0."""
    y0, y1, x0, x1 = box
    crop = big[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    scale = min((hr_h - 8) / h, (hr_w - 8) / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    res = cv2.resize(crop.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((hr_h, hr_w), np.float32)
    oy, ox = (hr_h - nh) // 2, (hr_w - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = res
    return out


def make_sample(word, font, hr_h, hr_w, scale, rng):
    full, up, mid, lo = zone_inks(word, font)
    ys, xs = np.where(full > 0)
    if len(ys) == 0:
        return None
    box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    ink = fit(full, hr_h, hr_w, box)  # soft coverage 0..1
    masks = np.stack([fit(z, hr_h, hr_w, box) for z in (up, mid, lo)], -1)
    hr = np.round(255 * (1 - ink)).astype(np.uint8)  # black text on white

    # --- degradation: blur -> downscale -> noise -> JPEG ------------------------
    sigma = float(rng.uniform(0.5, 2.0))
    interp_name = str(rng.choice(["bicubic", "bilinear", "area"]))
    interp = {"bicubic": cv2.INTER_CUBIC, "bilinear": cv2.INTER_LINEAR,
              "area": cv2.INTER_AREA}[interp_name]
    noise = float(rng.uniform(0, 6))
    quality = int(rng.integers(35, 91))
    blurred = cv2.GaussianBlur(hr, (0, 0), sigma)
    lr = cv2.resize(blurred, (hr_w // scale, hr_h // scale), interpolation=interp)
    lr = np.clip(lr.astype(np.float32) + rng.normal(0, noise, lr.shape), 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", lr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    lr = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    params = dict(blur_sigma=round(sigma, 2), interp=interp_name,
                  noise_std=round(noise, 2), jpeg_quality=quality)
    return hr, lr, (np.clip(masks, 0, 1) * 255).round().astype(np.uint8), params


def pick_words(n, seed):
    """Stratified pick so upper marks / lower marks / plain words are all covered."""
    rng = np.random.default_rng(seed)
    words = sorted({w for w in thai_words() if THAI_ONLY.match(w) and 2 <= len(w) <= 8})
    groups = {"upper": [], "lower": [], "both": [], "plain": []}
    for w in words:
        u, l = has(w, UPPER), has(w, LOWER)
        groups["both" if u and l else "upper" if u else "lower" if l else "plain"].append(w)
    share = {"upper": 0.45, "lower": 0.15, "both": 0.15, "plain": 0.25}
    chosen = []
    for g, frac in share.items():
        k = min(len(groups[g]), int(n * frac))
        chosen += list(rng.choice(groups[g], size=k, replace=False))
    rng.shuffle(chosen)
    return chosen


def split_of(word, ratios):
    """Deterministic by-word split (hash of the word), so no leakage across splits."""
    h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % 1000 / 1000
    return "train" if h < ratios[0] else "val" if h < ratios[0] + ratios[1] else "test"


def main():
    if not features.check("raqm"):
        raise SystemExit("Pillow has no Raqm: Thai marks will render wrong. Use Linux/Colab.")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="number of distinct words")
    ap.add_argument("--scale", type=int, default=4, choices=[2, 4])
    ap.add_argument("--hr-h", type=int, default=64)
    ap.add_argument("--hr-w", type=int, default=256)
    ap.add_argument("--fonts", default="fonts")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    fonts = load_fonts(args.fonts)
    words = pick_words(args.n, args.seed)
    rows = []
    for d in ("train", "val", "test"):
        for sub in ("hr", "lr", "mask"):
            os.makedirs(os.path.join(args.out, d, sub), exist_ok=True)

    for i, word in enumerate(words):
        split = split_of(word, (0.8, 0.1, 0.1))
        # seed depends on the word => the same test image every run, for every model
        rng = np.random.default_rng(int(hashlib.md5(word.encode()).hexdigest()[:8], 16))
        name, _, font = fonts[int(rng.integers(len(fonts)))]
        out = make_sample(word, font, args.hr_h, args.hr_w, args.scale, rng)
        if out is None:
            continue
        hr, lr, mask, params = out
        sid = f"{i:06d}"
        cv2.imwrite(os.path.join(args.out, split, "hr", sid + ".png"), hr)
        cv2.imwrite(os.path.join(args.out, split, "lr", sid + ".png"), lr)
        cv2.imwrite(os.path.join(args.out, split, "mask", sid + ".png"), mask[..., ::-1])  # RGB
        rows.append(dict(id=sid, split=split, text=word, font=name,
                         has_upper=int(has(word, UPPER)), has_lower=int(has(word, LOWER)), **params))
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(words)}")

    with open(os.path.join(args.out, "meta.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "val", "test")}
    print("done:", counts)


if __name__ == "__main__":
    main()
