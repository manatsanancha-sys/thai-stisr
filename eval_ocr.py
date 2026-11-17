"""OCR evaluation for Thai STISR.

Reads HR/LR pairs made by make_dataset.py and measures how well an OCR reads:
  hr       the sharp image (upper bound)
  bicubic  the LR image upscaled with bicubic (the "no super-resolution" baseline)
  <name>   any extra folder of restored images, e.g. TSRN output (--extra tsrn=out/tsrn)

Metrics (per condition, micro-averaged over words):
  word_acc      exact match of the whole word
  cer           character error rate (Levenshtein / len(gt))
  mark_err      errors on vowel/tone marks: gt marks that were substituted or dropped,
                plus marks the OCR inserted, divided by the number of gt marks
  cons_err      the same for every other character (consonants, leading vowels, ...)
We report everything twice: on all words, and on the words the OCR reads correctly
from the HR image (so OCR mistakes that have nothing to do with blur are excluded).

Usage:
  python eval_ocr.py --data data --split test --out results/baseline
  python eval_ocr.py --data data --split test --extra tsrn=out/tsrn --out results/tsrn
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

UPPER = {0x0E31, *range(0x0E34, 0x0E38), *range(0x0E47, 0x0E4F)}
LOWER = {0x0E38, 0x0E39, 0x0E3A}
MARKS = UPPER | LOWER


def is_mark(ch):
    return ord(ch) in MARKS


def align_errors(gt, pred):
    """Levenshtein alignment. Returns (S, D, I, mark_err, mark_total, cons_err, cons_total).

    mark_err counts gt marks that were substituted/deleted and marks the OCR inserted.
    cons_err counts the same for all non-mark characters.
    """
    n, m = len(gt), len(pred)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if gt[i - 1] == pred[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + cost)
    s = dl = ins = mark_err = cons_err = 0
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (gt[i - 1] != pred[j - 1]):
            if gt[i - 1] != pred[j - 1]:
                s += 1
                if is_mark(gt[i - 1]):
                    mark_err += 1
                else:
                    cons_err += 1
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dl += 1
            if is_mark(gt[i - 1]):
                mark_err += 1
            else:
                cons_err += 1
            i -= 1
        else:
            ins += 1
            if is_mark(pred[j - 1]):
                mark_err += 1
            else:
                cons_err += 1
            j -= 1
    mark_total = sum(is_mark(c) for c in gt)
    return s, dl, ins, mark_err, mark_total, cons_err, len(gt) - mark_total


def clean(text):
    return "".join(text.split())


def make_reader(backend):
    if backend == "easyocr":
        import easyocr
        reader = easyocr.Reader(["th"], gpu=False)

        def read(img):
            h, w = img.shape[:2]
            # skip the detector: our images are already tight crops of one word
            out = reader.recognize(img, horizontal_list=[[0, w, 0, h]], free_list=[], detail=0)
            return clean("".join(out))
        return read
    raise SystemExit(f"unknown backend {backend!r} (only 'easyocr' is wired up so far)")


def summarize(rows, key):
    """Micro-average over a list of per-sample dicts for one condition."""
    if not rows:
        return {"n": 0}
    n = len(rows)
    tot = lambda f: sum(r[f"{key}_{f}"] for r in rows)
    gt_len = sum(len(r["text"]) for r in rows)
    return {
        "n": n,
        "word_acc": round(sum(r[f"{key}_ok"] for r in rows) / n, 4),
        "cer": round((tot("S") + tot("D") + tot("I")) / max(gt_len, 1), 4),
        "mark_err": round(tot("mark_err") / max(tot("mark_total"), 1), 4),
        "cons_err": round(tot("cons_err") / max(tot("cons_total"), 1), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", default="test")
    ap.add_argument("--backend", default="easyocr")
    ap.add_argument("--limit", type=int, default=0, help="only the first N words (0 = all)")
    ap.add_argument("--extra", action="append", default=[],
                    help="name=dir with restored images <id>.png (e.g. TSRN output)")
    ap.add_argument("--out", default="results/baseline")
    args = ap.parse_args()

    meta = [r for r in csv.DictReader(open(os.path.join(args.data, "meta.csv"), encoding="utf-8"))
            if r["split"] == args.split]
    if args.limit:
        meta = meta[:args.limit]
    extras = dict(e.split("=", 1) for e in args.extra)
    conditions = ["hr", "bicubic", *extras]
    read = make_reader(args.backend)

    rows = []
    for k, r in enumerate(meta):
        gt = r["text"]
        base = os.path.join(args.data, args.split)
        hr = cv2.imread(f"{base}/hr/{r['id']}.png", 0)
        lr = cv2.imread(f"{base}/lr/{r['id']}.png", 0)
        h, w = hr.shape
        imgs = {"hr": hr, "bicubic": cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)}
        for name, folder in extras.items():
            imgs[name] = cv2.imread(os.path.join(folder, f"{r['id']}.png"), 0)
        row = {"id": r["id"], "text": gt, "has_upper": int(r["has_upper"]), "has_lower": int(r["has_lower"])}
        for c in conditions:
            pred = read(imgs[c])
            s, d, i, me, mt, ce, ct = align_errors(gt, pred)
            row.update({f"{c}_pred": pred, f"{c}_ok": int(pred == gt), f"{c}_S": s, f"{c}_D": d,
                        f"{c}_I": i, f"{c}_mark_err": me, f"{c}_mark_total": mt,
                        f"{c}_cons_err": ce, f"{c}_cons_total": ct})
        rows.append(row)
        if (k + 1) % 50 == 0:
            print(f"{k + 1}/{len(meta)}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "per_sample.csv"), "w", newline="", encoding="utf-8") as f:
        w_ = csv.DictWriter(f, fieldnames=list(rows[0]))
        w_.writeheader()
        w_.writerows(rows)

    subsets = {
        "all words": rows,
        "words OCR reads correctly from HR": [r for r in rows if r["hr_ok"]],
        "words with upper marks": [r for r in rows if r["has_upper"]],
    }
    summary = {s: {c: summarize(rs, c) for c in conditions} for s, rs in subsets.items()}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2)

    lines = []
    for s, per in summary.items():
        lines += [f"\n### {s} (n={per['hr']['n']})", "",
                  "| condition | word acc | CER | mark err | consonant err |", "|---|---|---|---|---|"]
        for c, m in per.items():
            if m["n"]:
                lines.append(f"| {c} | {m['word_acc']:.1%} | {m['cer']:.1%} | {m['mark_err']:.1%} | {m['cons_err']:.1%} |")
    text = "\n".join(lines)
    open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
