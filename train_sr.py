"""TSRN-lite for Thai scene-text super-resolution, with optional zone-weighted loss.

Reads the pairs made by make_dataset.py (data/<split>/{lr,hr,mask}), trains a small
TSRN-style network (conv -> residual blocks with GRUs along width/height -> PixelShuffle),
and writes the restored test images to runs/<name>/sr_test/<id>.png so they can be scored with
    python eval_ocr.py --data data --split test --crop --extra <name>=runs/<name>/sr_test

Variants (the experiment grid of the proposal):
  plain          python train_sr.py --name tsrn_plain
  zone loss      python train_sr.py --name tsrn_zoneloss --w-up 5 --w-lo 5
                 (the zone mask is used ONLY to weight the loss at training time,
                  so nothing extra is needed at test time)
  oracle input   python train_sr.py --name tsrn_oracle_zone --mask-in zone   (upper bound only:
                 the mask is taken from the ground truth, not predicted from the LR image)
                 --mask-in binary  = generic text/background mask (oracle too)

Differences from the original TSRN: no STN alignment module (our synthetic pairs are
already aligned), grayscale input, GRU in place of BLSTM, smaller by default.
"""
import argparse
import csv
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ----------------------------------------------------------------------------- data
class SRData(Dataset):
    def __init__(self, root, split, limit=0):
        meta = [r for r in csv.DictReader(open(os.path.join(root, "meta.csv"), encoding="utf-8"))
                if r["split"] == split]
        self.ids = [r["id"] for r in meta][:limit or None]
        self.dir = os.path.join(root, split)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        rd = lambda sub, flag: cv2.imread(f"{self.dir}/{sub}/{sid}.png", flag)
        lr = torch.from_numpy(rd("lr", 0).astype(np.float32) / 255)[None]
        hr = torch.from_numpy(rd("hr", 0).astype(np.float32) / 255)[None]
        m = rd("mask", 1)[..., ::-1].astype(np.float32) / 255      # RGB = upper, middle, lower
        return lr, hr, torch.from_numpy(m.copy()).permute(2, 0, 1), sid


# ----------------------------------------------------------------------------- model
class GruBlock(nn.Module):
    """3x3 conv, then a bidirectional GRU that runs along the last (width) axis of every row."""

    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)
        self.gru = nn.GRU(c, c // 2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * h, w, c)
        x, _ = self.gru(x)
        return x.reshape(b, h, w, c).permute(0, 3, 1, 2)


class SRB(nn.Module):
    """Sequential residual block: conv-BN-PReLU-conv-BN, GRU over height, add, GRU over width."""

    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.PReLU(),
                                  nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c))
        self.gru_h = GruBlock(c)
        self.gru_w = GruBlock(c)

    def forward(self, x):
        r = self.body(x)
        r = self.gru_h(r.transpose(-1, -2)).transpose(-1, -2)   # recurrence along the height axis
        return self.gru_w(x + r)


class TSRNLite(nn.Module):
    def __init__(self, in_ch=1, c=64, n_srb=5, scale=4):
        super().__init__()
        self.head = nn.Sequential(nn.Conv2d(in_ch, c, 9, padding=4), nn.PReLU())
        self.srbs = nn.Sequential(*[SRB(c) for _ in range(n_srb)])
        self.mid = nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c))
        up = []
        for _ in range(int(np.log2(scale))):
            up += [nn.Conv2d(c, c * 4, 3, padding=1), nn.PixelShuffle(2), nn.PReLU()]
        self.up = nn.Sequential(*up, nn.Conv2d(c, 1, 9, padding=4))

    def forward(self, x):
        h = self.head(x)
        y = self.mid(self.srbs(h)) + h        # global skip, as in TSRN
        return torch.sigmoid(self.up(y))


# ----------------------------------------------------------------------------- helpers
def make_input(lr, m, mask_in, scale):
    if mask_in == "none":
        return lr
    ml = F.avg_pool2d(m, scale)               # ground-truth mask at LR size (oracle!)
    if mask_in == "binary":
        ml = ml.sum(1, keepdim=True).clamp(0, 1)
    return torch.cat([lr, ml], 1)


def weight_map(m, w_up, w_mid, w_lo):
    return 1 + (w_up - 1) * m[:, 0:1] + (w_mid - 1) * m[:, 1:2] + (w_lo - 1) * m[:, 2:3]


def psnr(a, b):
    mse = ((a - b) ** 2).flatten(1).mean(1).clamp_min(1e-10)
    return (10 * torch.log10(1 / mse))


@torch.no_grad()
def evaluate(model, loader, dev, args):
    model.eval()
    ps, pb = [], []
    for lr, hr, m, _ in loader:
        lr, hr, m = lr.to(dev), hr.to(dev), m.to(dev)
        sr = model(make_input(lr, m, args.mask_in, args.scale))
        ps.append(psnr(sr, hr).cpu())
        pb.append(psnr(F.interpolate(lr, scale_factor=args.scale, mode="bicubic").clamp(0, 1), hr).cpu())
    return torch.cat(ps).mean().item(), torch.cat(pb).mean().item()


@torch.no_grad()
def dump(model, ds, dev, args, out_dir):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    for lr, _, m, sid in DataLoader(ds, batch_size=args.bs):
        sr = model(make_input(lr.to(dev), m.to(dev), args.mask_in, args.scale)).cpu()
        for img, s in zip(sr, sid):
            cv2.imwrite(os.path.join(out_dir, f"{s}.png"), (img[0].numpy() * 255).round().astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--name", default="tsrn_plain")
    ap.add_argument("--scale", type=int, default=4, choices=[2, 4])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--n-srb", type=int, default=5)
    ap.add_argument("--w-up", type=float, default=1.0, help="loss weight of upper-zone pixels")
    ap.add_argument("--w-mid", type=float, default=1.0)
    ap.add_argument("--w-lo", type=float, default=1.0, help="loss weight of lower-zone pixels")
    ap.add_argument("--mask-in", default="none", choices=["none", "binary", "zone"])
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N training words")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = os.path.join("runs", args.name)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out, "config.json"), "w"), indent=2)

    tr, va, te = (SRData(args.data, s, args.limit if s == "train" else 0) for s in ("train", "val", "test"))
    ltr = DataLoader(tr, batch_size=args.bs, shuffle=True, drop_last=len(tr) > args.bs)
    lva = DataLoader(va, batch_size=args.bs)
    in_ch = 1 + {"none": 0, "binary": 1, "zone": 3}[args.mask_in]
    model = TSRNLite(in_ch, args.ch, args.n_srb, args.scale).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    print(f"{dev} | params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | train {len(tr)} val {len(va)} test {len(te)}")

    best, log = -1, []
    for ep in range(1, args.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        for lr, hr, m, _ in ltr:
            lr, hr, m = lr.to(dev), hr.to(dev), m.to(dev)
            sr = model(make_input(lr, m, args.mask_in, args.scale))
            w = weight_map(m, args.w_up, args.w_mid, args.w_lo)
            loss = (w * (sr - hr).abs()).mean() / w.mean()     # zone-weighted L1, normalised
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(lr)
        sched.step()
        vp, vb = evaluate(model, lva, dev, args)
        log.append(dict(epoch=ep, train_loss=tot / len(tr), val_psnr=vp, val_psnr_bicubic=vb))
        if vp > best:
            best = vp
            torch.save(model.state_dict(), os.path.join(out, "best.pt"))
        print(f"ep {ep:3d} loss {tot / len(tr):.4f} val PSNR {vp:.2f} (bicubic {vb:.2f}) {time.time() - t0:.0f}s")

    with open(os.path.join(out, "log.csv"), "w", newline="") as f:
        w_ = csv.DictWriter(f, fieldnames=list(log[0]))
        w_.writeheader()
        w_.writerows(log)
    model.load_state_dict(torch.load(os.path.join(out, "best.pt"), map_location=dev))
    tp, tb = evaluate(model, DataLoader(te, batch_size=args.bs), dev, args)
    dump(model, te, dev, args, os.path.join(out, "sr_test"))
    json.dump({"test_psnr": tp, "test_psnr_bicubic": tb, "best_val_psnr": best},
              open(os.path.join(out, "result.json"), "w"), indent=2)
    print(f"best val {best:.2f} | test PSNR {tp:.2f} (bicubic {tb:.2f}) | images in {out}/sr_test")


if __name__ == "__main__":
    main()
