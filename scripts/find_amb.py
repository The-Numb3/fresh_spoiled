# -*- coding: utf-8 -*-
"""
Val 셋에서 모델이 가장 애매하게 본(0.5에 가까운) 샘플 상위 K를 CSV/이미지로 저장.
- 입력: --config, --ckpt (예: runs/best.pth)
- 출력: runs/analysis/ambiguous_val.csv, ambiguous_val_topK.png (옵션)
"""
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import argparse
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from tqdm import tqdm

# ---- 너의 프로젝트 모듈 재사용 ----
from meatscan.utils import load_config, set_seed
from meatscan.datasets import build_dataloaders
from meatscan.augment import build_transforms
from meatscan.models import build_model  # 모델 팩토리(이미 사용 중인 함수)




# ---------- 유틸 ----------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

@torch.no_grad()
def scan_split(model: nn.Module, loader, device, return_topk=30):
    """
    loader(보통 val) 전체를 훑어서, 0.5에 가장 가까운(=가장 애매한) 샘플 상위 return_topk개를 리턴.
    """
    model.eval()
    rows = []
    for imgs, labels, infos in tqdm(loader, desc="scan val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs1 = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # class=1 확률
        labels = labels.detach().cpu().numpy()

        # infos는 {"path": str or None, "meta": dict} 형태(현재 Dataset 반환 규약)
        for p, y, info in zip(probs1, labels, infos):
            rows.append({
                "path": info.get("path"),
                "label": int(y),
                "prob": float(p),
                "margin": float(abs(p - 0.5)),
                "conf": float(max(p, 1.0 - p)),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values("margin", ascending=True).reset_index(drop=True)  # margin↑가 애매
    if return_topk is not None and return_topk > 0:
        df_top = df.head(return_topk).copy()
    else:
        df_top = df
    return df, df_top

def make_contact_sheet(df_top: pd.DataFrame, out_png: str, title="Ambiguous (closest to 0.5)", ncols=6, thumb=224):
    """
    df_top에 있는 path들을 타일로 저장. 파일이 없거나 열기 실패하면 스킵.
    """
    paths = [p for p in df_top["path"].tolist() if p and os.path.isfile(p)]
    if not paths:
        print("[WARN] No valid image path to render contact sheet; skip.")
        return

    n = len(paths)
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols

    # 캔버스 생성
    W, H = ncols * thumb, nrows * thumb
    canvas = Image.new("RGB", (W, H), (240, 240, 240))

    # 각 타일 붙이기
    i = 0
    for r in range(nrows):
        for c in range(ncols):
            if i >= n: break
            p = paths[i]
            try:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    im = im.resize((thumb, thumb), Image.BILINEAR)
                canvas.paste(im, (c * thumb, r * thumb))
            except Exception as e:
                # 실패해도 계속
                pass
            i += 1

    # 위쪽에 간단한 타이틀 붙이기(선택) → 간단히 파일명만 저장
    ensure_dir(os.path.dirname(out_png))
    canvas.save(out_png, quality=92)
    print(f"[OUT] contact sheet: {out_png}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mobilenetv2.yaml", help="실험에 사용한 동일 config")
    ap.add_argument("--ckpt",   default="runs/best.pth", help="state_dict 경로")
    ap.add_argument("--topk",   type=int, default=30, help="가장 애매한 상위 K")
    ap.add_argument("--outdir", default="runs/analysis", help="결과 저장 폴더")
    ap.add_argument("--save_png", action="store_true", help="타일 이미지 저장")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    # 디바이스
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 변환/로더 (val만 필요)
    transforms = build_transforms(cfg)
    _, dl_val, _, _, _, _ = build_dataloaders(cfg, transforms)  # (train, val, test 반환 중 val만 사용)

    # 모델 구성 & 체크포인트 로드 (안전모드)
    model = build_model(cfg).to(device)
    try:
        state = torch.load(args.ckpt, map_location=device, weights_only=True)
    except TypeError:
        # torch<2.5 호환: weights_only 지원 안 하면 일반 로드
        state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # 스캔
    outdir = ensure_dir(args.outdir)
    df_all, df_top = scan_split(model, dl_val, device, return_topk=args.topk)

    # CSV 저장
    csv_all = os.path.join(outdir, "ambiguous_val_all.csv")
    csv_top = os.path.join(outdir, f"ambiguous_val_top{args.topk}.csv")
    df_all.to_csv(csv_all, index=False, encoding="utf-8-sig")
    df_top.to_csv(csv_top, index=False, encoding="utf-8-sig")
    print(f"[OUT] {csv_top}  (total={len(df_top)})")
    print(f"[OUT] {csv_all}  (total={len(df_all)})")

    # 타일 저장(옵션)
    if args.save_png:
        png = os.path.join(outdir, f"ambiguous_val_top{args.topk}.png")
        make_contact_sheet(df_top, png, ncols=6, thumb=224)

if __name__ == "__main__":
    main()
