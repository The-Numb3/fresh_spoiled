# meatscan/datasets.py
import os
from typing import Tuple, Dict, Any, List
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASS_TO_DIR = {0: "Fresh_CowMeat", 1: "Spoiled_CowMeat"}

# --- CSV 파일기반 데이터셋 ---
class FileListDataset(Dataset):
    """
    CSV 형식: filename,label_id
    - filename: data.root_dir 기준 상대경로 (예: "train/xxx.jpg", "valid/yyy.jpg")
    - label_id: 0/1/2 (fresh/half-fresh/spoiled)
    CSV에 이미 경로가 들어있으므로 CLASS_TO_DIR를 절대 쓰지 않는다.
    """
    def __init__(self, csv_path: str, root_dir: str, transform=None, img_size_fallback: int = 224):
        self.root = root_dir
        self.transform = transform
        self.img_size_fallback = img_size_fallback

        df = pd.read_csv(csv_path)
        # 컬럼명 정규화
        fcol = [c for c in df.columns if c.lower() in ("filename","image","file","path")]
        lcol = [c for c in df.columns if c.lower() in ("label_id","label","class","target")]
        assert fcol and lcol, "CSV에는 filename, label_id 컬럼(또는 동치)이 필요합니다."
        fcol, lcol = fcol[0], lcol[0]

        self.samples = []
        for _, r in df.iterrows():
            rel = str(r[fcol]).strip().replace("\\", "/")  # 윈도우 대비
            lab = int(r[lcol])
            full = os.path.join(self.root, rel)            # ← CSV 경로를 그대로 사용
            self.samples.append((full, lab))

    def __len__(self):
        return len(self.samples)

    def _infer_out_size(self) -> int:
        size = self.img_size_fallback
        tf = self.transform
        try:
            for t in getattr(tf, "transforms", []):
                if hasattr(t, "size"):
                    s = getattr(t, "size")
                    if isinstance(s, int): size = s
                    elif isinstance(s, (list, tuple)) and len(s)>0: size = s[0]
                    break
        except Exception:
            pass
        return size

    def __getitem__(self, idx):
        tries, cur = 3, idx
        for _ in range(tries):
            path, label = self.samples[cur]
            try:
                with Image.open(path) as im:
                    im.load()
                    img = im.convert("RGB")
                if self.transform:
                    img = self.transform(img)
                return img, label, {"path": path}
            except Exception as e:
                print(f"Error loading image {path}: {e}")
                cur = np.random.randint(0, len(self.samples))
        # 실패 시 더미 반환
        from PIL import Image as PILImage
        out_size = self._infer_out_size()
        dummy = PILImage.new("RGB", (out_size, out_size), (0,0,0))
        if self.transform:
            dummy = self.transform(dummy)
        return dummy, 0, {"path": None}

