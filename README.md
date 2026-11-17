# Thai STISR — vowel/tone-aware masking (work in progress)

Goal: test whether a 3-zone mask (upper / middle / lower, from Unicode) helps
Thai scene-text super-resolution + OCR, on top of a TSRN baseline.

**Status:** proposal + synthetic data pipeline done. Model training and OCR
evaluation are in progress — no results claimed yet.

## Quick start
```bash
pip install -r requirements.txt
# fonts: put Sarabun-Regular.ttf, Sarabun-Medium.ttf, NotoSansThai.ttf in ./fonts (OFL)
python make_dataset.py --n 3000 --scale 4 --out data
python preview.py data 8 train      # writes preview.png
```

## Data
- Words from PyThaiNLP, stratified: upper-mark / lower-mark / both / plain.
- HR 64x256 rendered with open fonts; LR = blur -> downscale -> noise -> JPEG.
- Reference 3-zone mask comes from render differences (no hand labels).
- Splits are by word (hash), so no word appears in two splits.
