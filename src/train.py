import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import wandb

from src.data.loader import GSNDataset
from src.models.networks import MultitaskModel
from src.utils.seed import set_seed


def get_config_mapping() -> dict[tuple[int, ...], int]:
    """Generates a mapping from 135 possible combinations of shapes to a class ID."""
    all_possible_configs = []
    num_shape_types = 6
    total_shapes = 10

    for i in range(num_shape_types):
        for j in range(i + 1, num_shape_types):
            for k in range(1, total_shapes):
                config = [0] * num_shape_types
                config[i], config[j] = k, total_shapes - k
                all_possible_configs.append(tuple(config))

    return {cfg: idx for idx, cfg in enumerate(all_possible_configs)}


class EarlyStopping:
    """Stops training if the validation loss does not improve."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module, path: str = "best_model.pth") -> None:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion_cls: nn.Module,
        criterion_reg: nn.Module,
        lambda_cls: float,
        lambda_cnt: float,
        device: torch.device
) -> float:
    model.train()
    running_loss = 0.0

    for inputs, counts, class_ids in tqdm(loader, desc="Training", leave=False):
        inputs, counts, class_ids = inputs.to(device), counts.to(device), class_ids.to(device)

        optimizer.zero_grad()
        log_probs, cnt_preds = model(inputs)

        loss_cls = criterion_cls(log_probs, class_ids)
        loss_reg = criterion_reg(cnt_preds, counts)
        loss = (lambda_cls * loss_cls) + (lambda_cnt * loss_reg)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        wandb.log({
            "batch/train_loss": loss.item(),
            "batch/cls_loss": loss_cls.item(),
            "batch/reg_loss": loss_reg.item()
        })

    return running_loss / len(loader)


@torch.no_grad()
def validate_epoch(
        model: nn.Module,
        loader: DataLoader,
        criterion_cls: nn.Module,
        criterion_reg: nn.Module,
        lambda_cls: float,
        lambda_cnt: float,
        device: torch.device
) -> tuple[float, float]:
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    for inputs, counts, class_ids in tqdm(loader, desc="Validation", leave=False):
        inputs, counts, class_ids = inputs.to(device), counts.to(device), class_ids.to(device)

        log_probs, cnt_preds = model(inputs)

        loss_cls = criterion_cls(log_probs, class_ids)
        loss_reg = criterion_reg(cnt_preds, counts)
        loss = (lambda_cls * loss_cls) + (lambda_cnt * loss_reg)

        val_loss += loss.item()

        preds = log_probs.argmax(dim=1)
        correct += preds.eq(class_ids).sum().item()
        total += class_ids.size(0)

    return val_loss / len(loader), 100.0 * correct / total


def main(args: argparse.Namespace) -> None:
    set_seed(42)

    wandb.init(project="geometric-multitask", config=vars(args))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[*] Using hardware accelerator: {device}")

    import torchvision.transforms as transforms
    transform = transforms.Compose([transforms.ToTensor()])
    config_to_id = get_config_mapping()

    full_train_dataset = GSNDataset(
        csv_file='data/labels.csv', root_dir='data/',
        config_to_id=config_to_id, transform=transform, is_train=True
    )
    full_val_dataset = GSNDataset(
        csv_file='data/labels.csv', root_dir='data/',
        config_to_id=config_to_id, transform=transform, is_train=False
    )

    indices = list(range(10000))
    train_dataset = Subset(full_train_dataset, indices[:9000])
    val_dataset = Subset(full_val_dataset, indices[9000:])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=0)

    model = MultitaskModel(dropout_rate=0.4).to(device)
    criterion_cls = nn.NLLLoss()
    criterion_reg = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=args.patience)

    print(f"[*] Starting training loop. Lambda CLS: {args.lambda_cls}, Lambda CNT: {args.lambda_cnt}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion_cls, criterion_reg,
            args.lambda_cls, args.lambda_cnt, device
        )
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion_cls, criterion_reg,
            args.lambda_cls, args.lambda_cnt, device
        )

        wandb.log({
            "epoch": epoch,
            "epoch/train_loss": train_loss,
            "epoch/val_loss": val_loss,
            "epoch/val_acc": val_acc
        })

        print(
            f"Epoch {epoch:03d}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("[!] Early stopping triggered.")
            break

    wandb.finish()
    print("[*] Training completed. Best model saved as 'best_model.pth'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multitask Network")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_batch_size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_cnt", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=10)

    args = parser.parse_args()
    main(args)