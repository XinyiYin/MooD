import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

from affect_encoder.encoder import EmotionEncoder
from model.dataset import EmoSetVADataset, get_transforms

# TODO：supplement the file path
class Config:
    image_dir = ' '
    annotation_dir = ' '
    save_dir = ' '
    batch_size = 128
    num_workers = 8
    lr = 1e-3
    weight_decay = 1e-4
    epochs = 100
    val_ratio = 0.2
    seed = 42
    patience = 10
    min_delta = 1e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ccc(pred, true):
    pred = pred.flatten()
    true = true.flatten()
    mean_p, mean_t = np.mean(pred), np.mean(true)
    var_p, var_t = np.var(pred), np.var(true)
    cov = np.mean((pred - mean_p) * (true - mean_t))
    return 2 * cov / (var_p + var_t + (mean_p - mean_t) ** 2 + 1e-8)


def split_train_val(annotation_dir, val_ratio, seed):
    all_ids = []
    for fname in sorted(os.listdir(annotation_dir)):
        if fname.endswith('.json'):
            all_ids.append(fname.replace('.json', ''))

    rng = random.Random(seed)
    rng.shuffle(all_ids)

    n_val = int(len(all_ids) * val_ratio)
    val_ids = all_ids[:n_val]
    train_ids = all_ids[n_val:]

    print(f"Total: {len(all_ids)}, Train: {len(train_ids)}, Val: {len(val_ids)}")
    return train_ids, val_ids


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    n_samples = 0

    for images, va_targets, _ in dataloader:
        images = images.to(device)
        va_targets = va_targets.to(device)

        va_pred = model(images)
        loss = criterion(va_pred, va_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

    return total_loss / n_samples


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    n_samples = 0
    all_pred_v, all_pred_a = [], []
    all_true_v, all_true_a = [], []

    for images, va_targets, _ in dataloader:
        images = images.to(device)
        va_targets = va_targets.to(device)

        va_pred = model(images)
        loss = criterion(va_pred, va_targets)

        total_loss += loss.item() * images.size(0)
        n_samples += images.size(0)

        all_pred_v.append(va_pred[:, 0].cpu().numpy())
        all_pred_a.append(va_pred[:, 1].cpu().numpy())
        all_true_v.append(va_targets[:, 0].cpu().numpy())
        all_true_a.append(va_targets[:, 1].cpu().numpy())

    avg_loss = total_loss / n_samples

    pred_v = np.concatenate(all_pred_v)
    pred_a = np.concatenate(all_pred_a)
    true_v = np.concatenate(all_true_v)
    true_a = np.concatenate(all_true_a)

    metrics = {
        'loss': avg_loss,
        'mae_v': np.mean(np.abs(pred_v - true_v)),
        'mae_a': np.mean(np.abs(pred_a - true_a)),
        'pearson_v': pearsonr(pred_v, true_v)[0],
        'pearson_a': pearsonr(pred_a, true_a)[0],
        'ccc_v': ccc(pred_v, true_v),
        'ccc_a': ccc(pred_a, true_a),
    }
    return metrics


def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    train_ids, val_ids = split_train_val(cfg.annotation_dir, cfg.val_ratio, cfg.seed)

    train_dataset = EmoSetVADataset(
        cfg.image_dir, cfg.annotation_dir,
        image_ids=train_ids,
        transform=get_transforms('train')
    )
    val_dataset = EmoSetVADataset(
        cfg.image_dir, cfg.annotation_dir,
        image_ids=val_ids,
        transform=get_transforms('val')
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size,
        shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size,
        shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True
    )

    model = EmotionEncoder(freeze_backbone=True).to(cfg.device)


    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nStarting training on {cfg.device}")
    print(f"{'Epoch':>6s} | {'Train Loss':>10s} | {'Val Loss':>10s} | "
          f"{'MAE_V':>6s} {'MAE_A':>6s} | {'PCC_V':>6s} {'PCC_A':>6s} | "
          f"{'CCC_V':>6s} {'CCC_A':>6s} | {'LR':>8s} | {'Time':>6s}")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, cfg.device)

        val_metrics = validate(model, val_loader, criterion, cfg.device)
        val_loss = val_metrics['loss']

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        elapsed = time.time() - t0

        print(f"{epoch:>6d} | {train_loss:>10.6f} | {val_loss:>10.6f} | "
              f"{val_metrics['mae_v']:>6.4f} {val_metrics['mae_a']:>6.4f} | "
              f"{val_metrics['pearson_v']:>6.4f} {val_metrics['pearson_a']:>6.4f} | "
              f"{val_metrics['ccc_v']:>6.4f} {val_metrics['ccc_a']:>6.4f} | "
              f"{current_lr:>8.1e} | {elapsed:>5.1f}s")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
        }
        torch.save(checkpoint, os.path.join(cfg.save_dir, 'last_checkpoint.pth'))

        if val_loss < best_val_loss - cfg.min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(checkpoint, os.path.join(cfg.save_dir, 'best_model.pth'))
            print(f"       >> Best model saved (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {cfg.patience} epochs)")
            break

    best_ckpt = torch.load(os.path.join(cfg.save_dir, 'best_model.pth'))
    model.load_state_dict(best_ckpt['model_state_dict'])
    final_metrics = validate(model, val_loader, criterion, cfg.device)

    print(f"\nFinal results (best model, epoch {best_ckpt['epoch']}):")
    print(f"  Val Loss:   {final_metrics['loss']:.6f}")
    print(f"  MAE:        V={final_metrics['mae_v']:.4f}, A={final_metrics['mae_a']:.4f}")
    print(f"  Pearson:    V={final_metrics['pearson_v']:.4f}, A={final_metrics['pearson_a']:.4f}")
    print(f"  CCC:        V={final_metrics['ccc_v']:.4f}, A={final_metrics['ccc_a']:.4f}")


if __name__ == '__main__':
    main()