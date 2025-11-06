# meatscan/datasets.py
import os
from typing import Tuple, Dict, Any, List
from PIL import Image, ImageFile
from torch.utils.data import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASS_TO_DIR = {0: "Fresh_CowMeat", 1: "Spoiled_CowMeat"}

class FileListDataset(Dataset):
    def __init__(self, root_dir: str, csv_path: str, transform=None, img_size_fallback: int = 224):
        import pandas as pd
        self.root = root_dir
        self.transform = transform
        self.img_size = img_size_fallback

        # robust read
        for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
            try:
                df = pd.read_csv(csv_path, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            df = pd.read_csv(csv_path, encoding="utf-8", encoding_errors="ignore")

        assert "filename" in df.columns and "label" in df.columns, "CSV must have filename,label"
        self.items: List[Tuple[str,int]] = []
        for fn, y in zip(df["filename"], df["label"]):
            y = int(y)
            p = os.path.join(self.root, CLASS_TO_DIR[y], str(fn).strip())
            self.items.append((p, y))

        self.items = [(p,y) for (p,y) in self.items if os.path.isfile(p)]
        if len(self.items) == 0:
            print(f"[WARN] No valid items from {csv_path}")

    def __len__(self): return len(self.items)

    def _dummy(self):
        from PIL import Image as PILImage
        return PILImage.new("RGB", (self.img_size, self.img_size), (0,0,0))

    def __getitem__(self, i):
        path, y = self.items[i]
        try:
            with Image.open(path) as im:
                im.load()
                img = im.convert("RGB")
        except Exception:
            img = self._dummy()
        if self.transform: img = self.transform(img)
        return img, y, {"path": path}
