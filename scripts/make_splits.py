import os, argparse, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit

FRESH_DIR = "Fresh_CowMeat"
SPOIL_DIR = "Spoiled_CowMeat"
CLASS_TO_DIR = {0: FRESH_DIR, 1: SPOIL_DIR}
DIR_TO_CLASS = {FRESH_DIR: 0, SPOIL_DIR: 1}

def robust_read_csv(path):
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            raise
    # 마지막 시도: 에러 무시
    return pd.read_csv(path, encoding="utf-8", encoding_errors="ignore")

def build_manifest_from_dirs(root):
    rows = []
    for sub, y in [(FRESH_DIR, 0), (SPOIL_DIR, 1)]:
        d = os.path.join(root, sub)
        if not os.path.isdir(d): 
            continue
        for fn in os.listdir(d):
            # 이미지 확장자만
            ext = os.path.splitext(fn.lower())[1]
            if ext in {".jpg",".jpeg",".png",".bmp",".webp"}:
                rows.append({"filename": fn, "label": y})
    df = pd.DataFrame(rows).drop_duplicates(subset=["filename"])
    return df

def normalize_meta(df, root, col_hint=None):
    # filename 컬럼 찾기
    if col_hint and col_hint in df.columns:
        col_file = col_hint
    else:
        cand = [c for c in df.columns if str(c).lower() in ("filename","image","file","img","path")]
        col_file = cand[0] if cand else None

    if col_file is None:
        return pd.DataFrame()  # 신호: 폴더 스캔으로 대체

    out = df.copy()
    out["filename"] = out[col_file].astype(str).map(lambda x: os.path.basename(x).strip())
    # label 없으면 폴더로부터 유추
    if "label" not in out.columns:
        fn2y = {}
        for sub, y in [(FRESH_DIR,0),(SPOIL_DIR,1)]:
            d = os.path.join(root, sub)
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    fn2y[fn] = y
        out["label"] = out["filename"].map(lambda x: fn2y.get(x, None))

    out = out[["filename","label"]]
    out = out.dropna(subset=["filename","label"])
    out["label"] = out["label"].astype(int)
    out = out.drop_duplicates(subset=["filename"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./MeatScan_Dataset")
    ap.add_argument("--meta_csv", default="MeatScan_Metadata.filtered.csv")
    ap.add_argument("--out_dir", default="./splits")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--test_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group_key", default=None)  # 예: location / date
    ap.add_argument("--meta_col_hint", default=None)  # filename 컬럼명 힌트
    ap.add_argument("--exclude_quarantine", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) 메타 로드 시도
    meta_path = os.path.join(args.root, args.meta_csv)
    use_dirs_fallback = False
    try:
        meta_raw = robust_read_csv(meta_path)
    except FileNotFoundError:
        print(f"[WARN] metadata CSV not found: {meta_path} → will scan directories.")
        use_dirs_fallback = True
        meta_raw = pd.DataFrame()

    # 2) 정규화
    if not use_dirs_fallback:
        meta = normalize_meta(meta_raw, args.root, col_hint=args.meta_col_hint)
        if len(meta) == 0:
            print("[WARN] metadata has no usable rows → will scan directories.")
            use_dirs_fallback = True

    # 3) 폴더 스캔 대체
    if use_dirs_fallback:
        meta = build_manifest_from_dirs(args.root)

    # 4) quarantine 제외(옵션)
    if args.exclude_quarantine:
        qdir = os.path.join(args.root, "_quarantine")
        if os.path.isdir(qdir):
            bad = set(os.listdir(qdir))
            before = len(meta)
            meta = meta[~meta["filename"].isin(bad)].copy()
            print(f"[INFO] excluded quarantine: {before-len(meta)} rows")

    # 최종 유효성
    meta = meta.dropna(subset=["filename","label"])
    meta["label"] = meta["label"].astype(int)
    meta = meta.drop_duplicates(subset=["filename"]).reset_index(drop=True)

    if len(meta) == 0:
        raise RuntimeError("No samples found after metadata normalization / directory scan.")

    print(f"[INFO] total usable samples = {len(meta)} (Fresh={sum(meta.label==0)}, Spoiled={sum(meta.label==1)})")

    # 5) split
    assert 0 < args.val_ratio < 1 and 0 < args.test_ratio < 1 and (args.val_ratio + args.test_ratio) < 1

    idx = np.arange(len(meta))
    y = meta["label"].values

    if args.group_key and args.group_key in (c.lower() for c in meta.columns):
        gcol = next(c for c in meta.columns if c.lower()==args.group_key)
        groups = meta[gcol].fillna("unknown").astype(str).values
        gss_test = GroupShuffleSplit(n_splits=1, test_size=args.test_ratio, random_state=args.seed)
        trainval_idx, test_idx = next(gss_test.split(idx, y, groups))
        adj_val = args.val_ratio / (1.0 - args.test_ratio)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=adj_val, random_state=args.seed)
        tr_idx, va_idx = next(gss_val.split(trainval_idx, y[trainval_idx], groups[trainval_idx]))
        tr_idx = trainval_idx[tr_idx]; va_idx = trainval_idx[va_idx]
    else:
        trainval_idx, test_idx = train_test_split(idx, test_size=args.test_ratio, random_state=args.seed, stratify=y)
        adj_val = args.val_ratio / (1.0 - args.test_ratio)
        tr_idx, va_idx = train_test_split(trainval_idx, test_size=adj_val, random_state=args.seed, stratify=y[trainval_idx])

    # 6) 저장
    def _save(name, indices):
        df = meta.iloc[indices][["filename","label"]].copy()
        outp = os.path.join(args.out_dir, f"{name}.csv")
        df.to_csv(outp, index=False, encoding="utf-8-sig")
        return df, outp

    tr, trp = _save("train", tr_idx)
    va, vap = _save("val",   va_idx)
    te, tep = _save("test",  test_idx)

    # 7) 교집합 0 검증
    assert set(tr.filename).isdisjoint(va.filename) \
        and set(tr.filename).isdisjoint(te.filename) \
        and set(va.filename).isdisjoint(te.filename), "Split overlap detected!"

    print(f"[OK] train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    print(f"[OUT] {trp}\n[OUT] {vap}\n[OUT] {tep}")

if __name__ == "__main__":
    main()
