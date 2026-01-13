"""
CGFormer Training with PyTorch Lightning

Usage:
    # Energy prediction training
    python train_lightning.py --mode energy --data_dir ./STFO_data --epochs 100

    # Swap + REINFORCE training (requires pretrained energy model)
    python train_lightning.py --mode swap --energy_ckpt ./checkpoints/best.ckpt

    # Test only
    python train_lightning.py --mode test --ckpt ./checkpoints/best.ckpt --data_dir ./STFO_data
"""

import argparse
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model import CrystalGraphConvNet
from data import CIFData, collate_pool, get_train_val_test_loader
from swap_utils import (
    parse_poscar_string, poscar_to_tensors,
    sample_sublattice_swap, apply_n_swaps,
    log_prob_sublattice_swap
)


# =============================================================================
# Normalizer
# =============================================================================

class Normalizer:
    """Normalize targets to zero mean and unit variance."""

    def __init__(self, tensor=None):
        if tensor is not None:
            self.mean = tensor.mean().item()
            self.std = tensor.std().item()
        else:
            self.mean = 0.0
            self.std = 1.0

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, tensor):
        return tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


# =============================================================================
# Data Module
# =============================================================================

class CrystalDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for crystal data."""

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 16,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        num_workers: int = 0,
        max_num_nbr: int = 12,
        radius: float = 8.0,
        random_seed: int = 123,
    ):
        super().__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_workers = num_workers
        self.max_num_nbr = max_num_nbr
        self.radius = radius
        self.random_seed = random_seed

        self.dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def setup(self, stage: Optional[str] = None):
        if self.dataset is None:
            self.dataset = CIFData(
                self.root_dir,
                max_num_nbr=self.max_num_nbr,
                radius=self.radius,
                random_seed=self.random_seed
            )

            self.train_loader, self.val_loader, self.test_loader = get_train_val_test_loader(
                dataset=self.dataset,
                collate_fn=collate_pool,
                batch_size=self.batch_size,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                return_test=True,
                num_workers=self.num_workers,
                pin_memory=torch.cuda.is_available(),
                train_size=None,
                val_size=None,
                test_size=None,
            )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def get_train_targets(self):
        """Collect all training targets for normalization."""
        self.setup()
        targets = []
        for batch in self.train_loader:
            _, target, _ = batch
            targets.append(target)
        return torch.cat(targets, dim=0)


# =============================================================================
# Lightning Module - Energy Prediction
# =============================================================================

class CGFormerModule(pl.LightningModule):
    """PyTorch Lightning module for CGFormer energy prediction."""

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int = 64,
        n_conv: int = 3,
        h_fea_len: int = 128,
        n_h: int = 1,
        graphormer_layers: int = 1,
        num_heads: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        normalizer_mean: float = 0.0,
        normalizer_std: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = CrystalGraphConvNet(
            orig_atom_fea_len=orig_atom_fea_len,
            nbr_fea_len=nbr_fea_len,
            atom_fea_len=atom_fea_len,
            n_conv=n_conv,
            h_fea_len=h_fea_len,
            n_h=n_h,
            graphormer_layers=graphormer_layers,
            num_heads=num_heads,
            classification=False,
        )

        self.normalizer = Normalizer()
        self.normalizer.mean = normalizer_mean
        self.normalizer.std = normalizer_std

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.MSELoss()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        return self.model(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)

    def _shared_step(self, batch, batch_idx):
        (atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx), target, _ = batch

        output = self(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx)
        target_normed = self.normalizer.norm(target)
        loss = self.criterion(output, target_normed)

        pred_denorm = self.normalizer.denorm(output)
        mae = F.l1_loss(pred_denorm, target)

        return loss, mae

    def training_step(self, batch, batch_idx):
        loss, mae = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_mae', mae, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mae = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_mae', mae, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, mae = self._shared_step(batch, batch_idx)
        self.log('test_loss', loss)
        self.log('test_mae', mae)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_mae'
            }
        }


# =============================================================================
# Lightning Module - Swap + REINFORCE
# =============================================================================

class SwapScoreNet(nn.Module):
    """Network that outputs swap scores per atom."""

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SwapREINFORCEModule(pl.LightningModule):
    """REINFORCE training for swap-based structure optimization."""

    def __init__(
        self,
        energy_model: Optional[nn.Module] = None,
        input_dim: int = 5,
        hidden_dim: int = 128,
        learning_rate: float = 1e-4,
        n_swaps_per_step: int = 10,
        entropy_reg: float = 0.01,
        baseline_ema: float = 0.99,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['energy_model'])

        self.energy_model = energy_model
        if energy_model is not None:
            self.energy_model.eval()
            for p in self.energy_model.parameters():
                p.requires_grad = False

        self.score_net = SwapScoreNet(input_dim, hidden_dim)

        self.learning_rate = learning_rate
        self.n_swaps = n_swaps_per_step
        self.entropy_reg = entropy_reg
        self.baseline_ema = baseline_ema

        self.register_buffer('baseline', torch.tensor(0.0))

    def forward(self, atom_types_onehot):
        return self.score_net(atom_types_onehot)

    def compute_energy(self, atom_types, tensors):
        """Compute energy. Override for actual energy model."""
        # Placeholder: use negative sum as "energy"
        return -atom_types.float().sum(dim=-1)

    def training_step(self, batch, batch_idx):
        atom_types, tensors = batch
        batch_size, N = atom_types.shape
        device = atom_types.device

        n_types = 5
        atom_types_onehot = F.one_hot(atom_types, n_types).float()
        scores = self(atom_types_onehot)

        b_site_mask = tensors['b_site_mask']
        type_map = tensors['type_map']
        ti, fe = type_map['Ti'], type_map['Fe']

        total_log_prob = 0.0
        current = atom_types.clone()

        for _ in range(self.n_swaps):
            swapped, indices = sample_sublattice_swap(
                current, b_site_mask, ti, fe, scores
            )
            log_prob = log_prob_sublattice_swap(
                scores, b_site_mask, ti, fe, current, indices
            )
            total_log_prob = total_log_prob + log_prob
            current = swapped

        energy = self.compute_energy(current, tensors)

        advantage = energy - self.baseline
        reinforce_loss = (advantage.detach() * total_log_prob).mean()
        entropy_loss = -self.entropy_reg * total_log_prob.mean()

        loss = reinforce_loss + entropy_loss

        self.baseline = (
            self.baseline_ema * self.baseline +
            (1 - self.baseline_ema) * energy.mean().detach()
        )

        self.log('train_loss', loss, prog_bar=True)
        self.log('energy', energy.mean(), prog_bar=True)
        self.log('baseline', self.baseline)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.score_net.parameters(), lr=self.learning_rate)


# =============================================================================
# Training Functions
# =============================================================================

def train_energy(args):
    """Train energy prediction model."""
    print("="*60)
    print("Training Energy Prediction Model")
    print("="*60)

    # Data
    data_module = CrystalDataModule(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        num_workers=args.num_workers,
    )
    data_module.setup()

    # Get sample for model init
    sample_batch = next(iter(data_module.train_dataloader()))
    (atom_fea, nbr_fea, _, _), _, _ = sample_batch

    # Normalizer
    train_targets = data_module.get_train_targets()
    normalizer = Normalizer(train_targets)
    print(f"Target mean: {normalizer.mean:.4f}, std: {normalizer.std:.4f}")

    # Model
    model = CGFormerModule(
        orig_atom_fea_len=atom_fea.shape[-1],
        nbr_fea_len=nbr_fea.shape[-1],
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        graphormer_layers=args.graphormer_layers,
        num_heads=args.num_heads,
        learning_rate=args.lr,
        normalizer_mean=normalizer.mean,
        normalizer_std=normalizer.std,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_mae',
        dirpath=args.ckpt_dir,
        filename='cgformer-{epoch:02d}-{val_mae:.4f}',
        save_top_k=3,
        mode='min',
    )

    early_stop = EarlyStopping(
        monitor='val_mae',
        patience=args.patience,
        mode='min',
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices=1,
        callbacks=[checkpoint_callback, early_stop],
        enable_progress_bar=True,
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, data_module)

    # Test
    print("\n" + "="*60)
    print("Testing Best Model")
    print("="*60)
    trainer.test(model, data_module)

    print(f"\nBest checkpoint: {checkpoint_callback.best_model_path}")
    return checkpoint_callback.best_model_path


def test_model(args):
    """Test a trained model."""
    print("="*60)
    print("Testing Model")
    print("="*60)

    # Load model
    model = CGFormerModule.load_from_checkpoint(args.ckpt)
    print(f"Loaded checkpoint: {args.ckpt}")

    # Data
    data_module = CrystalDataModule(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Trainer
    trainer = pl.Trainer(
        accelerator='auto',
        devices=1,
    )

    # Test
    trainer.test(model, data_module)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='CGFormer Training')

    # Mode
    parser.add_argument('--mode', type=str, default='energy',
                        choices=['energy', 'swap', 'test'],
                        help='Training mode')

    # Data
    parser.add_argument('--data_dir', type=str, default='./STFO_data',
                        help='Path to crystal data directory')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)

    # Model
    parser.add_argument('--atom_fea_len', type=int, default=64)
    parser.add_argument('--n_conv', type=int, default=3)
    parser.add_argument('--h_fea_len', type=int, default=128)
    parser.add_argument('--n_h', type=int, default=1)
    parser.add_argument('--graphormer_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')

    # Checkpoint (for test/swap mode)
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Checkpoint path for testing')
    parser.add_argument('--energy_ckpt', type=str, default=None,
                        help='Energy model checkpoint for swap training')

    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    if args.mode == 'energy':
        train_energy(args)
    elif args.mode == 'test':
        if args.ckpt is None:
            raise ValueError("--ckpt required for test mode")
        test_model(args)
    elif args.mode == 'swap':
        print("Swap training not yet fully implemented")
        # TODO: Implement swap training with dataloader-free approach


if __name__ == '__main__':
    main()
