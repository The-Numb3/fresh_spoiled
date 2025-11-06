# meatscan/augment.py
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transforms(cfg):
    size = cfg["data"]["img_size"]
    mode = cfg["augment"]["mode"]
    cj   = cfg["augment"].get("color_jitter", [0.2,0.2,0.2,0.05])

    if mode == "standard":
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(*cj),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:  # "paper" (보수적인 증강)
        train_tf = transforms.Compose([
            transforms.Resize(int(size*1.14)),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    eval_tf = transforms.Compose([
        transforms.Resize(int(size*1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf
