import os
import zipfile
import random, time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
from tqdm import tqdm


# =========================
# CONFIG
# =========================
ZIP_PATH = os.environ.get("DOG_CAT_ZIP", os.path.join("data", "Datos.zip"))
EXTRACT_DIR = os.environ.get("DOG_CAT_EXTRACT_DIR", os.path.join("data", "Datos_extracted"))
BEST_PATH = os.environ.get("DOG_CAT_BEST_PATH", os.path.join("models", "best_effnet_dogcat.pth"))

IMG_SIZE = 224
BATCH_SIZE = 256

NUM_WORKERS = 6
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4
PIN_MEMORY = True

USE_AMP = True

EPOCHS = 30
PATIENCE = 5
LR_HEAD = 3e-4
LR_FINE = 1e-4
FINE_TUNE_EPOCH = 5
SEED = 42

USE_DATAPARALLEL = False


# =========================
# UTILIDADES
# =========================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_zip_once(zip_path, extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    marker = os.path.join(extract_dir, ".extracted_ok")
    if os.path.exists(marker):
        return

    print(f"Extrayendo ZIP una sola vez a:\n{extract_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok")


def find_data_dir(root_dir):
    for r, dirs, _ in os.walk(root_dir):
        if "cats" in dirs and "dogs" in dirs:
            return r
    raise FileNotFoundError("No encontré carpetas 'cats' y 'dogs' dentro del ZIP extraído.")


def set_requires_grad(module, requires_grad: bool):
    for p in module.parameters():
        p.requires_grad = requires_grad


def warmup_gpu(model, device):
    if device.type != "cuda":
        return
    model.eval()
    x = torch.randn(16, 3, IMG_SIZE, IMG_SIZE, device=device)
    with torch.no_grad():
        for _ in range(10):
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=USE_AMP):
                _ = model(x)
    torch.cuda.synchronize()


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(USE_AMP and x.is_cuda)):
            logits = model(x)
            loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs

    return total_loss / max(total, 1), correct / max(total, 1)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(USE_AMP and x.is_cuda)):
            logits = model(x)
            loss = criterion(logits, y)

        if scaler is not None and x.is_cuda and USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        total += bs

    return total_loss / max(total, 1), correct / max(total, 1)


# =========================
# VISUALIZACIÓN
# =========================
def denormalize(img_tensor, mean, std):
    """img_tensor: (3,H,W) normalizado -> vuelve a [0,1] aprox para mostrar."""
    mean = torch.tensor(mean).view(3,1,1)
    std  = torch.tensor(std).view(3,1,1)
    x = img_tensor.cpu() * std + mean
    return x.clamp(0, 1)

def show_grid(images, titles, ncols=4, figsize=(12, 8)):
    n = len(images)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(images[i])
        plt.title(titles[i], fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def show_predictions(model, dataset, class_names, device, mean, std, k=12):
    """Muestra k imágenes con predicción vs etiqueta real."""
    model.eval()
    idxs = np.random.choice(len(dataset), size=min(k, len(dataset)), replace=False)

    imgs, titles = [], []
    for idx in idxs:
        x, y = dataset[idx]  # x ya normalizado
        x_in = x.unsqueeze(0).to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(USE_AMP and device.type=="cuda")):
            logits = model(x_in)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        pred = int(np.argmax(probs))
        conf = float(probs[pred])

        img_np = denormalize(x, mean, std).permute(1,2,0).numpy()
        ok = "✅" if pred == y else "❌"
        titles.append(f"{ok} true={class_names[y]} | pred={class_names[pred]} ({conf*100:.1f}%)")
        imgs.append(img_np)

    show_grid(imgs, titles, ncols=4, figsize=(14, 9))

def show_original_samples(raw_dataset, class_names, k=12):
    """Muestra imágenes sin augment/normalize (solo resize+centercrop+ToTensor)"""
    idxs = np.random.choice(len(raw_dataset), size=min(k, len(raw_dataset)), replace=False)
    imgs, titles = [], []
    for idx in idxs:
        x, y = raw_dataset[idx]
        img_np = x.permute(1,2,0).numpy()
        imgs.append(img_np)
        titles.append(f"{class_names[y]}")
    show_grid(imgs, titles, ncols=4, figsize=(14, 9))

def show_augment_effects(train_dataset, class_names, mean, std, idx=0, k=8):
    """Muestra la MISMA imagen varias veces para ver augmentations diferentes."""
    imgs, titles = [], []
    # truco: llamamos varias veces dataset[idx] -> cada vez aplica aug diferente
    for i in range(k):
        x, y = train_dataset[idx]
        img_np = denormalize(x, mean, std).permute(1,2,0).numpy()
        imgs.append(img_np)
        titles.append(f"aug {i+1} | {class_names[y]}")
    show_grid(imgs, titles, ncols=4, figsize=(14, 7))


# =========================
# MAIN
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Torch:", torch.__version__)
    print("GPUs:", torch.cuda.device_count() if torch.cuda.is_available() else 0)

    torch.backends.cudnn.benchmark = True
    seed_everything(SEED)

    extract_zip_once(ZIP_PATH, EXTRACT_DIR)
    data_dir = find_data_dir(EXTRACT_DIR)
    print("Directorio de datos:", data_dir)

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    # dataset para ver imágenes "bonitas" sin normalize
    raw_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ])

    base_ds = datasets.ImageFolder(root=data_dir)
    class_names = base_ds.classes
    print("Clases:", class_names)
    print("class_to_idx:", base_ds.class_to_idx)

    trainview = datasets.ImageFolder(root=data_dir, transform=train_tfms)
    evalview  = datasets.ImageFolder(root=data_dir, transform=val_tfms)
    rawview   = datasets.ImageFolder(root=data_dir, transform=raw_tfms)

    N = len(trainview)
    n_train = int(0.70 * N)
    n_val   = int(0.15 * N)
    n_test  = N - n_train - n_val

    g = torch.Generator().manual_seed(SEED)
    train_subset, val_subset, test_subset = random_split(trainview, [n_train, n_val, n_test], generator=g)

    val_ds  = Subset(evalview, val_subset.indices)
    test_ds = Subset(evalview, test_subset.indices)
    train_ds = train_subset

    print(f"Split -> train:{len(train_ds)}, val:{len(val_ds)}, test:{len(test_ds)}")

    # class weights
    train_targets = [trainview.samples[i][1] for i in train_ds.indices]
    counts = np.bincount(train_targets, minlength=len(class_names)).astype(np.float32)
    counts[counts == 0] = 1.0
    weights_np = (counts.sum() / counts)
    class_weights = torch.tensor(weights_np, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    loader_kwargs = dict(
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(PERSISTENT_WORKERS and NUM_WORKERS > 0),
        prefetch_factor=(PREFETCH_FACTOR if NUM_WORKERS > 0 else None),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, **loader_kwargs)

    # Modelo
    w = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=w)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 2)
    model = model.to(device)

    if USE_DATAPARALLEL and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
        model = nn.DataParallel(model)
        print("Usando DataParallel en GPUs:", torch.cuda.device_count())

    warmup_gpu(model, device)

    # =========================
    # 1) MUESTRA IMÁGENES REALES (SIN AUGMENT)
    # =========================
    print("\nMostrando ejemplos del dataset (sin augment)...")
    show_original_samples(rawview, class_names, k=12)

    # =========================
    # 2) MUESTRA AUGMENTATIONS EN LA MISMA IMAGEN
    # =========================
    # Elegimos un índice del train original (no subset) para que sea determinístico
    idx_example = 0
    print("\nMostrando augmentations (misma imagen, varias transformaciones)...")
    show_augment_effects(trainview, class_names, imagenet_mean, imagenet_std, idx=idx_example, k=8)

    # =========================
    # 3) CARGA BEST MODEL Y MUESTRA PREDICCIONES
    # =========================
    if os.path.exists(BEST_PATH):
        print("\nCargando best model y mostrando predicciones...")
        state = torch.load(BEST_PATH, map_location=device, weights_only=True)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state)
        else:
            model.load_state_dict(state)

        # Mostramos predicciones en TEST (sin data augmentation)
        show_predictions(model, test_ds, class_names, device, imagenet_mean, imagenet_std, k=12)

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f"TEST | loss={test_loss:.4f} acc={test_acc:.3f}")
    else:
        print(f"\nNo encontré el archivo BEST_PATH:\n{BEST_PATH}\nEntrena primero para guardarlo.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
