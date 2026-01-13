from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn

class CheckpointManager:
    """
    Manages loading and saving of Pytorch Checkpoints

    Example usage:
        # Save Checkpoints
        manager = CheckpointManager(checkpoint_dir="./checkpoints")
        manager.save(model, optimizer, epoch=10, loss=0.05)

        # Load Checkpoints
        checkpoint = manag er.load(".pt)
        model.load_state_dict(checkpoint["model_state_dict"])
    """

    def __init__(self, checkpoint_dir = "./checkpoints"):
        
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents = True, exist_ok = True)

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        filename: Optional[str] = None,
        **kwargs
    ) -> Path:
        
        """
        Save model checkpoint.
        Args:
            model: PyTorch model to save
            optimizer: Optimizer state (optional)
            epoch: Current epoch number
            loss: Current loss value
            metrics: Dictionary of additional metrics
            filename: Custom filename (default: model_epoch_{epoch}.pt)
            **kwargs: Additional data to save
        Returns:
            Path to saved checkpoint
        """

        checkpoint = {
            "model_state_dict": model.state_dict(),
            'model_config': self._extract_model_config(model),
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if epoch is not None:
            checkpoint["epoch"] = epoch

        if loss is not None:
            checkpoint["loss"] = loss

        if metrics is not None:
            checkpoint["metrics"] = metrics

        checkpoint.update(kwargs)

        if filename is None:
            epoch_str = f"epoch_{epoch}" if epoch is not None else "latest"
            filename = f"model_{epoch_str}.pt"

        save_path = self.checkpoint_dir / filename

        torch.save(self.chekcpoint_dir, save_path)

        return save_path
    
    def load(
        self,
        filename: str,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """
        Load checkpoint from file.
        Args:
            filename: Checkpoint filename or full path
            map_location: Device to load tensors to (e.g., 'cpu', 'cuda:0')
            strict: Whether to strictly enforce state_dict keys match
        Returns:
            Dictionary containing checkpoint data
        """

        if os.path.isabs(filename) or os.path.exists(filename):
            load_path = Path(filename)

        else:
            load_path = self.checkpoint_dir / filename

        if map_location is None:
            map_location = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(load_path, map_location=map_location, weights_only=False)

        return checkpoint
    
    def load_model(
        self,
        model: nn.Module,
        filename: str,
        map_location: Optional[Union[str, torch.device]] = None,
        strict: bool = True,
    ):
        
        checkpoint = self.load(filename, map_location)
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        return model
    
    def _extract_model_config(self, model: nn.Module) -> Dict[str, Any]:

        config = {
            "class_name": model.__class__.__name__
        }

        if hasattr(model, "atom_fea_len"):
            config["atom_fea_len"] = model.atom_fea_len

        if hasattr(model, "nbr_fea_len"):
            config["nbr_fea_len"] = model.nbr_fea_len

        if hasattr(model, "n_conv"):
            config["n_conv"] = model.n_conv

        if hasattr(model, "classification"):
            config["classification"] = model.classification

        return config
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints in the directory."""
        return sorted([
            f.name for f in self.checkpoint_dir.glob("*.pt")
        ] + [
            f.name for f in self.checkpoint_dir.glob("*.pth")
        ])

    def get_best_checkpoint(
        self,
        metric_key: str = 'loss',
        mode: str = 'min'
    ) -> Optional[str]:

        """
        Find the best checkpoint based on a metric.

        Args:
            metric_key: Key to compare (in metrics dict or top-level)
            mode: 'min' or 'max'
        Returns:
            Filename of best checkpoint or None
        """
        best_value = float('inf') if mode == 'min' else float('-inf')
        best_checkpoint = None

        for ckpt_name in self.list_checkpoints():
            try:
                ckpt = self.load(ckpt_name, map_location='cpu')
                # Look for metric in metrics dict or top-level
                if 'metrics' in ckpt and metric_key in ckpt['metrics']:
                    value = ckpt['metrics'][metric_key]

                elif metric_key in ckpt:
                    value = ckpt[metric_key]
                else:
                    continue

                if mode == 'min' and value < best_value:
                    best_value = value
                    best_checkpoint = ckpt_name
                elif mode == 'max' and value > best_value:
                    best_value = value
                    best_checkpoint = ckpt_name
            except Exception:
                continue
        return best_checkpoint


def load_pretrained(
    model: nn.Module,
    checkpoint_path: str,
    map_location: Optional[str] = None,
    strict: bool = False
) -> nn.Module:

    """
    Convenience function to load pre-trained weights.
    Args:
        model: Model instance
        checkpoint_path: Path to .pt file
        map_location: Device mapping
        strict: Strict loading mode

    Returns:
        Model with loaded weights
    """

    if map_location is None:
        map_location = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']

    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # Assume it's a raw state dict
        state_dict = checkpoint

    # Remove 'module.' prefix if saved with DataParallel

    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=strict)

    return model


def convert_checkpoint_format(
    input_path: str,
    output_path: str,
    output_format: str = 'state_dict'
) -> None:
    """
    Convert checkpoint between formats.

    Args:
        input_path: Input checkpoint path
        output_path: Output path
        output_format: 'state_dict', 'full', or 'torchscript'
    """
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']

    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    if output_format == 'state_dict':
        torch.save(state_dict, output_path)

    elif output_format == 'full':
        torch.save({
            'model_state_dict': state_dict,
            'format_version': '1.0'
        }, output_path)

    else:
        raise ValueError(f"Unknown output format: {output_format}")