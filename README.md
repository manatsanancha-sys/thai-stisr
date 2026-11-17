# Thai STISR — vowel/tone-aware masking (work in progress)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manatsanancha-sys/thai-stisr/blob/main/colab_generate.ipynb)

Goal: test whether a 3-zone mask (upper / middle / lower, from Unicode) helps
Thai scene-text super-resolution + OCR, on top of a TSRN baseline.

**Status:** proposal, synthetic data pipeline and OCR baseline done (see
Baseline below). TSRN training and the 3-zone mask are in progress.

## Quick start
```bash
pip install -r requirements.txt
# fonts: put Sarabun-Regular.ttf, Sarabun-Medium.ttf, NotoSansThai.ttf in ./fonts (OFL)
python make_dataset.py --n 3000 --scale 4 --out data
python preview.py data 8 train      # writes preview.png
```

## Fonts
Fonts are not included in this repo and must be downloaded into `fonts/`.
They are Sarabun and Noto Sans Thai, both under the SIL Open Font License (OFL),
from the [google/fonts](https://github.com/google/fonts) repo:
```bash
mkdir -p fonts
B=https://raw.githubusercontent.com/google/fonts/main/ofl
curl -L -o fonts/Sarabun-Regular.ttf $B/sarabun/Sarabun-Regular.ttf
curl -L -o fonts/Sarabun-Medium.ttf  $B/sarabun/Sarabun-Medium.ttf
curl -L -o fonts/NotoSansThai.ttf   "$B/notosansthai/NotoSansThai%5Bwdth,wght%5D.ttf"
```

## Data
- Words from PyThaiNLP, stratified: upper-mark / lower-mark / both / plain.
- HR 64x256 rendered with open fonts; LR = blur -> downscale -> noise -> JPEG.
- Reference 3-zone mask comes from render differences (no hand labels).
- Splits are by word (hash), so no word appears in two splits.

## Evaluation
`eval_ocr.py` runs EasyOCR (Thai) on each test word and compares the result
with the ground-truth text:
```bash
python eval_ocr.py --data data --split test --out results/baseline
```
It scores the HR image (upper bound) and the bicubic-upscaled LR image
(no-SR baseline). To score restored images such as TSRN output, add
`--extra tsrn=out/tsrn`. Use `--limit N` for a quick run on the first N words.
```bash
python eval_ocr.py --data data --split test --crop --out results/baseline_crop
```
--crop removes the white border around the word before OCR, applied the same way to every input.

Metrics are micro-averaged over words:
- **word acc**: share of words read exactly right.
- **CER**: character error rate, edit distance / ground-truth length.
- **mark err**: errors on upper/lower vowels and tone marks: marks that were
  dropped or substituted, plus marks the OCR added, divided by the number of
  ground-truth marks.
- **consonant err**: the same count for all other characters (consonants,
  leading vowels, ...).

Results are reported for all words, for words the OCR reads correctly from HR
(so errors that have nothing to do with blur are left out), and for words with
upper marks. Output goes to `--out`: `per_sample.csv`, `summary.json` and
`summary.md`. Only `summary.md` is tracked in git.

## Baseline: how blur hurts Thai OCR (synthetic test set, n=302, EasyOCR, --crop)
Evaluated on words EasyOCR reads correctly from the HR image (n=280):

| input | word acc | CER | mark err | consonant err |
|---|---|---|---|---|
| HR | 100.0% | 0.0% | 0.0% | 0.0% |
| LR x4, bicubic | 76.1% | 6.8% | 12.1% | 5.5% |
| LR x2, bicubic | 97.5% | 0.4% | 0.7% | 0.4% |

At x2 almost nothing is lost, so x4 is the main setting. At x4, vowel/tone marks
are lost about twice as often per character as other characters. All numbers use
--crop (white border removed before OCR); without it EasyOCR adds ghost characters.
Single run, one OCR engine, synthetic data: indicative only.
