import os, time, torch
from torch import nn, optim
from torch.utils.data import DataLoader
from meatscan.utils import load_config, set_seed, ensure_dir, optimize_threshold
from meatscan.augment import build_transforms
from meatscan.datasets import FileListDataset
from meatscan.models import create_model
from meatscan.engine import train_one_epoch, evaluate

def build_criterion(cfg):
    ls = float(cfg["train"].get("label_smoothing", 0.0))
    name = cfg["train"]["criterion"]
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=ls)
    elif name == "bce_logits":
        # 이진 전용일 때만 권장(여기선 CE 권장)
        return nn.BCEWithLogitsLoss()
    raise ValueError(name)

def build_optimizer(cfg, params):
    opt = cfg["train"]["optimizer"]["name"].lower()
    lr = float(cfg["train"]["optimizer"]["lr"])
    wd = float(cfg["train"]["optimizer"]["weight_decay"])
    if opt == "adam":
        return optim.Adam(params, lr=lr, weight_decay=wd)
    elif opt == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    raise ValueError(opt)

def build_scheduler(cfg, optimizer):
    name = cfg["train"]["scheduler"]["name"].lower()
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    return None

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mobilenetv2.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf, eval_tf = build_transforms(cfg)
    root = cfg["data"]["root_dir"]

    ds_train = FileListDataset(root, cfg["data"]["train_csv"], transform=train_tf, img_size_fallback=cfg["data"]["img_size"])
    ds_val   = FileListDataset(root, cfg["data"]["val_csv"],   transform=eval_tf,   img_size_fallback=cfg["data"]["img_size"])
    ds_test  = FileListDataset(root, cfg["data"]["test_csv"],  transform=eval_tf,   img_size_fallback=cfg["data"]["img_size"])

    dl_train = DataLoader(ds_train, batch_size=cfg["data"]["batch_size"], shuffle=True,
                          num_workers=cfg["data"]["num_workers"],
                          pin_memory=cfg["data"]["pin_memory"],
                          persistent_workers=cfg["data"]["persistent_workers"],
                          prefetch_factor=cfg["data"]["prefetch_factor"])
    dl_val   = DataLoader(ds_val, batch_size=cfg["data"]["batch_size"], shuffle=False,
                          num_workers=cfg["data"]["num_workers"], pin_memory=True)
    dl_test  = DataLoader(ds_test, batch_size=cfg["data"]["batch_size"], shuffle=False,
                          num_workers=cfg["data"]["num_workers"], pin_memory=True)

    model = create_model(cfg["model"]["name"], cfg["model"]["num_classes"],
                         pretrained=bool(cfg["model"].get("pretrained", True)),
                         dropout=float(cfg["model"].get("dropout", 0.0))).to(device)

    criterion = build_criterion(cfg)
    optimizer = build_optimizer(cfg, model.parameters())
    scheduler = build_scheduler(cfg, optimizer)

    out_dir = cfg["train"]["out_dir"]; ensure_dir(out_dir)
    scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg["train"].get("mixed_precision", True)) and device.type=="cuda")

    # Early Stopping
    es = cfg["train"]["early_stopping"]
    es_on = bool(es.get("enabled", True))
    monitor = es.get("monitor", cfg["train"].get("save_best_by", "f1"))
    mode = es.get("mode", "max")
    patience = int(es.get("patience", 5)); min_delta = float(es.get("min_delta", 0.001))
    best = float("-inf") if mode=="max" else float("inf"); bad = 0
    best_path = os.path.join(out_dir, "best.pth")

    max_epochs = int(cfg["train"]["epochs"])
    for epoch in range(max_epochs):
        train_loss = train_one_epoch(model, dl_train, optimizer, device, criterion, scaler=scaler)
        val_mets, val_probs, val_labels = evaluate(model, dl_val, device, criterion, return_arrays=True)
        cur = val_mets[monitor]

        # 스케줄러
        if scheduler is not None:
            scheduler.step()

        improved = (cur >= best + min_delta) if mode=="max" else (cur <= best - min_delta)
        if improved:
            best = cur; bad = 0
            if monitor == cfg["train"]["save_best_by"]:
                torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if es_on and bad >= patience:
                print(f"[EARLY STOP] No improvement in {patience} epochs on val {monitor}.")
                break

        print(f"[E{epoch:03d}] train_loss={train_loss:.4f} | val_{monitor}={cur:.4f} | val_loss={val_mets['loss']:.4f}")

    # best 로드 & threshold 최적화
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    # 임계값: val에서 최적화 후 test에 고정 적용
    th = float(cfg["eval"].get("default_threshold", 0.5))
    if cfg["eval"].get("optimize_threshold_on", "val") == "val":
        _, val_probs, val_labels = evaluate(model, dl_val, device, criterion, return_arrays=True)
        th, _ = optimize_threshold(val_labels, val_probs, metric=monitor)
        print(f"[VAL] optimized threshold for {monitor}: {th:.3f}")

    # 최종 테스트
    test_mets, test_probs, test_labels = evaluate(model, dl_test, device, criterion, return_arrays=True)
    from meatscan.metrics import compute_metrics
    test_mets = compute_metrics(test_labels, test_probs, threshold=th) | {"loss": test_mets["loss"]}
    print("[TEST]", test_mets)

if __name__ == "__main__":
    main()
