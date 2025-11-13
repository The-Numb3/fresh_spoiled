# meatscan/engine.py
import torch
from tqdm import tqdm
import numpy as np
from .metrics import compute_metrics

def train_one_epoch(model, loader, optimizer, device, criterion, scaler=None):
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc="train", leave=False)
    for imgs, labels, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs); loss = criterion(logits, labels)
            loss.backward(); optimizer.step()

        total += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=float(loss))
    return total / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """
    - 로짓의 shape로 binary(2-class) vs multiclass 자동 판별.
    - 반환: (metrics_dict, probs_array, labels_array)
      * binary: probs shape (N,)
      * multiclass: probs shape (N, C)
    """
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    for imgs, labels, _ in tqdm(loader, desc="valid", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device)

        logits = model(imgs)
        loss = criterion(logits, labels)
        total_loss += loss.item() * imgs.size(0)

        # 클래스 수 자동 추론
        if logits.ndim >= 2:
            C = logits.shape[-1]
        else:
            C = 1

        if C == 2:
            probs = torch.softmax(logits, dim=1)[:, 1]  # (N,)
            all_probs.append(probs.detach().cpu().numpy())
        elif C > 2:
            probs = torch.softmax(logits, dim=1)        # (N, C)
            all_probs.append(probs.detach().cpu().numpy())
        else:
            # 드문 케이스(스칼라 로짓)
            probs = torch.sigmoid(logits).view(-1)
            all_probs.append(probs.detach().cpu().numpy())

        all_labels.append(labels.detach().cpu().numpy())

    all_probs  = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    mets = compute_metrics(all_labels, all_probs)  # ← 형태로 바이너리/멀티 자동 처리
    mets["loss"] = total_loss / len(loader.dataset)
    return mets, all_probs, all_labels