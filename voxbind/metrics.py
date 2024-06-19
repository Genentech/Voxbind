import numpy as np
import torch
import torchmetrics


def create_metrics_for_training(device: str = "cuda"):
    """
    Create metrics for training.

    Args:
        device (str): The device to use for computation. Defaults to "cuda".

    Returns:
        metrics_denoise (MetricsDenoise): Metrics for denoising task.
        metrics_sampling (MetricsSampling): Metrics for sampling task.
    """
    metrics_denoise = MetricsDenoise(
        loss=torchmetrics.MeanMetric(),
        miou=torchmetrics.classification.BinaryJaccardIndex(),
    )
    metrics_denoise.to(device)
    metrics_sampling = None
    return metrics_denoise, metrics_sampling


class MetricsDenoise(torchmetrics.MetricCollection):
    def __init__(self, **kwargs):
        """Class containing all metrics for denoising task.

        Args:
            **kwargs: Keyword arguments representing different metrics.

        """
        self.metrics = {k: v for k, v in kwargs.items()}

    def apply_threshold(self, y: torch.Tensor, threshold: float = 0.5):
        """Applies a threshold to the predicted values.

        Args:
            y (torch.Tensor): Predicted values.
            threshold (float, optional): Threshold value. Defaults to 0.5.

        Returns:
            torch.Tensor: Thresholded values.

        """
        return (y > threshold).to(torch.uint8)

    def reset(self):
        """Resets all metrics to their initial state."""
        for metric in self.metrics.values():
            metric.reset()

    def update(self, loss: torch.Tensor, pred: torch.Tensor, y: torch.Tensor):
        """Updates the metrics with the given loss, predicted values, and ground truth values.

        Args:
            loss (torch.Tensor): Loss value.
            pred (torch.Tensor): Predicted values.
            y (torch.Tensor): Ground truth values.

        """
        pred_th = self.apply_threshold(pred)
        y_th = self.apply_threshold(y)

        for metric_name in self.metrics.keys():
            if metric_name == "loss":
                self.metrics["loss"].update(loss)
            elif metric_name == "miou":
                self.metrics["miou"].update(pred_th, y_th)

    def compute(self):
        """Computes the computed metrics.

        Returns:
            dict: Dictionary containing the computed metrics.

        """
        return {k: v.compute().item() for k, v in self.metrics.items() if not np.isnan(v.compute().item())}

    def to(self, device: str):
        """Moves all metrics to the specified device.

        Args:
            device (torch.device): Device to move the metrics to.

        """
        self.metrics = {k: v.to(device) for k, v in self.metrics.items()}
