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
def evaluate(model, loader, device, criterion, return_arrays=False):
    model.eval()
    total = 0.0
    probs_all, labels_all = [], []

    for imgs, labels, _ in tqdm(loader, desc="eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        total += loss.item() * imgs.size(0)
        probs = torch.softmax(logits, dim=1)[:,1]
        probs_all.append(probs.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    probs_all = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all)
    mets = compute_metrics(labels_all, probs_all)  # threshold는 main에서 결정해 재계산 가능
    mets["loss"] = total / len(loader.dataset)

    if return_arrays:
        return mets, probs_all, labels_all
    return mets
