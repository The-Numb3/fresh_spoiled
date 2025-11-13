# scripts/make_test_from_train_onehot.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(".")                         # 프로젝트 루트
DATA_ROOT = ROOT / "meat_data"             # 이미지 루트 (./dataset)
TRAIN_DIR  = DATA_ROOT / "train"         # 실제 이미지가 있는 폴더
VALID_DIR  = DATA_ROOT / "valid"

IN_TRAIN = ROOT / "meat_data" / "train" / "_classes.csv" # 현재 one-hot 라벨의 train.csv
IN_VAL   = ROOT / "meat_data" / "valid" / "_classes.csv"   # 현재 one-hot 라벨의 val.csv
OUT_DIR  = ROOT / "splits_new"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 라벨 칼럼 이름(대소문자/하이픈/공백 허용, 순서: Fresh=0, Half-fresh=1, Spoiled=2)
LABEL_COLS_CANON = ["Fresh", "Half-fresh", "Spoiled"]
LABEL_MAP = {"Fresh":0, "Half-fresh":1, "Spoiled":2}

def find_cols(df, names):
    # 대소문자/공백/하이픈 무시 매칭
    norm = {c:"".join(c.lower().split()).replace("-", "") for c in df.columns}
    want = ["fresh", "halffresh", "spoiled"]
    out = {}
    for w, canon in zip(want, names):
        for c, nc in norm.items():
            if nc == w:
                out[canon] = c
                break
    if len(out) != 3:
        raise ValueError(f"라벨 컬럼 탐색 실패. 찾은 매핑: {out}. CSV 헤더를 확인하세요.")
    return out

def pick_rel_path(name: str, prefer="train"):
    """filename값(이름 또는 경로)을 받아 실제 존재하는 파일을 찾아
       DATA_ROOT 기준 상대경로('train/xxx.jpg' 또는 'valid/xxx.jpg')로 반환"""
    p = Path(name)
    # 이미 상대경로 형태면 그대로 유효성만 확인
    if p.suffix:  # 확장자 있음
        cand = ([TRAIN_DIR / p.name, VALID_DIR / p.name] 
                if prefer == "train" else 
                [VALID_DIR / p.name, TRAIN_DIR / p.name])
        for c in cand:
            if c.exists():
                return c.relative_to(DATA_ROOT).as_posix()
        # 마지막 수단: 원래 주어진 경로가 dataset 하위면 그대로
        if (DATA_ROOT / p).exists():
            return p.as_posix()
        # 못 찾으면 우선순위로 강제
        return f"{prefer}/{p.name}"
    else:
        # 확장자 모르면 후보 탐색
        exts = [".jpg",".jpeg",".png",".bmp",".webp"]
        for ext in exts:
            c1, c2 = TRAIN_DIR / f"{p.name}{ext}", VALID_DIR / f"{p.name}{ext}"
            if c1.exists(): return c1.relative_to(DATA_ROOT).as_posix()
            if c2.exists(): return c2.relative_to(DATA_ROOT).as_posix()
        return f"{prefer}/{p.name}.jpg"

def onehot_to_id(df, label_cols_map):
    # one-hot 유효성 점검
    onehot = df[[label_cols_map[c] for c in LABEL_COLS_CANON]].astype(int)
    sums = onehot.sum(axis=1)
    bad = df.index[(sums != 1)]
    if len(bad) > 0:
        print(f"[WARN] one-hot 합이 1이 아닌 행 {len(bad)}개를 제외합니다.")
        df = df.drop(index=bad)

    # argmax로 ID 생성
    id_series = onehot.values.argmax(axis=1)
    df = df.copy()
    df["label_id"] = id_series
    return df

def normalize_train(df):
    # filename 열 찾아 표준화
    fcol = [c for c in df.columns if c.lower() in ("filename","image","file","path")]
    assert fcol, "filename(또는 image/file/path) 컬럼이 필요합니다."
    fcol = fcol[0]

    # 라벨 컬럼 찾기 & 변환
    colmap = find_cols(df, LABEL_COLS_CANON)
    df = onehot_to_id(df, colmap)

    # 경로 정리
    df["filename"] = df[fcol].astype(str).apply(lambda s: pick_rel_path(s, prefer="train"))
    return df[["filename","label_id"]]

def normalize_val(df):
    fcol = [c for c in df.columns if c.lower() in ("filename","image","file","path")]
    assert fcol, "filename(또는 image/file/path) 컬럼이 필요합니다."
    fcol = fcol[0]

    colmap = find_cols(df, LABEL_COLS_CANON)
    df = onehot_to_id(df, colmap)

    df["filename"] = df[fcol].astype(str).apply(lambda s: pick_rel_path(s, prefer="valid"))
    return df[["filename","label_id"]]

# 1) 입력 로드 및 정규화
train_raw = pd.read_csv(IN_TRAIN, encoding="utf-8")
val_raw   = pd.read_csv(IN_VAL,   encoding="utf-8")

train_df = normalize_train(train_raw)
val_df   = normalize_val(val_raw)

# 2) train에서만 test 분리 (예: 10%)
train_rest, test_df = train_test_split(
    train_df, test_size=0.10, random_state=42, stratify=train_df["label_id"]
)

# 3) 저장
train_rest.to_csv(OUT_DIR/"train.csv", index=False, encoding="utf-8")
val_df.to_csv(      OUT_DIR/"val.csv",   index=False, encoding="utf-8")
test_df.to_csv(     OUT_DIR/"test.csv",  index=False, encoding="utf-8")

print("saved:", len(train_rest), len(val_df), len(test_df))
print("train dist:", train_rest["label_id"].value_counts().to_dict())
print("val   dist:",   val_df["label_id"].value_counts().to_dict())
print("test  dist:",   test_df["label_id"].value_counts().to_dict())
