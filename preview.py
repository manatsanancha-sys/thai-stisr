"""Contact sheet: HR | LR (nearest-upscaled) | bicubic-upscaled LR | mask overlay."""
import csv
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

data = sys.argv[1] if len(sys.argv) > 1 else "data"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
split = sys.argv[3] if len(sys.argv) > 3 else "train"
font = ImageFont.truetype("fonts/Sarabun-Regular.ttf", 18)

rows = [r for r in csv.DictReader(open(f"{data}/meta.csv", encoding="utf-8")) if r["split"] == split][:n]
tiles = []
for r in rows:
    p = f"{data}/{split}/%s/{r['id']}.png"
    hr = cv2.imread(p % "hr", 0)
    lr = cv2.imread(p % "lr", 0)
    m = cv2.imread(p % "mask")[..., ::-1]  # RGB: R=upper G=mid B=lower
    h, w = hr.shape
    nearest = cv2.resize(lr, (w, h), interpolation=cv2.INTER_NEAREST)
    bicubic = cv2.resize(lr, (w, h), interpolation=cv2.INTER_CUBIC)
    # overlay: white background, zone colours (orange upper, grey mid, blue lower)
    ov = np.full((h, w, 3), 255, np.uint8)
    for k, col in enumerate([(230, 90, 20), (60, 60, 60), (30, 90, 230)]):
        a = (m[..., k:k + 1] / 255.0)
        ov = (ov * (1 - a) + np.array(col) * a).astype(np.uint8)
    row = np.concatenate([np.stack([x] * 3, -1) for x in (hr, nearest, bicubic)] + [ov], 1)
    label = Image.new("RGB", (row.shape[1], 24), (245, 245, 245))
    ImageDraw.Draw(label).text((4, 1), f"{r['text']}  [{r['font']}] blur={r['blur_sigma']} jpg={r['jpeg_quality']}",
                               font=font, fill=(0, 0, 0))
    tiles += [np.asarray(label), row]
cv2.imwrite("preview.png", np.concatenate(tiles, 0)[..., ::-1])
print("wrote preview.png")
