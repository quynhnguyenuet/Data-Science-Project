"""
Triển khai mô hình DeepAnt cho dữ liệu log DNS
"""

import logging
import torch
import torch.nn as nn
import pytorch_lightning as pl
#  logging
logger = logging.getLogger(__name__)


class DeepAntPredictor(nn.Module):
    def __init__(self, feature_dim: int, window_size: int, hidden_size: int = 256) -> None:
        """
        Mô hình dự đoán DeepAnt (CNN-based) dùng cho chuỗi thời gian.

        Tham số:
            feature_dim (int): Số lượng feature mỗi bước thời gian (VD: 34 cho log DNS).
            window_size (int): Số bước thời gian trong mỗi cửa sổ (sliding window).
            hidden_size (int): Số node trong lớp fully-connected ẩn.
        """
        super(DeepAntPredictor, self).__init__()
        # Khối CNN để trích đặc trưng chuỗi thời gian
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim, out_channels=64, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            # nn.Flatten(),
            # nn.Linear(in_features=(window_size - 2) // 4 * 128, out_features=hidden_size),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.4),
            # nn.Linear(in_features=hidden_size, out_features=feature_dim),
        )
        # Dùng AdaptiveAvgPool để không bị phụ thuộc window_size
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # Khối fully-connected để dự đoán bước tiếp theo
        self.fc_block = nn.Sequential(
            nn.Flatten(),                             # → (batch_size, 128)
            nn.Linear(128, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(hidden_size, feature_dim)       # → đầu ra dự đoán toàn bộ vector feature
        )

        logger.info("Khởi tạo mô hình DeepAnt với feature_dim=%d, window_size=%d", feature_dim, window_size)
        # logger.info("DeepAntPredictor model initialized.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hàm lan truyền xuôi của mô hình DeepAnt.

        Tham số:
            x (torch.Tensor): tensor đầu vào dạng (batch_size, feature_dim, window_size)

        Trả về:
            torch.Tensor: Dự đoán bước kế tiếp (batch_size, feature_dim)
        """
        x = self.model(x)            # → Trích đặc trưng qua CNN
        x = self.global_pool(x)      # → Gom thành 1 điểm thời gian (batch, 128, 1)
        x = self.fc_block(x)         # → Fully-connected → (batch, feature_dim)
        return x

class AnomalyDetector(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, lr: float) -> None:
        """
        Anomaly Detector based on DeepAnt model.

        Args:
            model (nn.Module): The DeepAnt predictor model.
            lr (float): Learning rate for the optimizer.
        """
        super(AnomalyDetector, self).__init__()
        
        self.model = model
        self.criterion = torch.nn.L1Loss()
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DeepAnt model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, feature_dim, window_size).

        Returns:
            torch.Tensor: Model prediction of shape (batch_size, feature_dim).
        """
        return self.model(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Defines a single step in the training loop.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): (input_sequence, target_value).
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The loss value for this batch.
        """
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)

        self.log("train_loss", loss.item(), on_epoch=True)
        logger.info(f"Epoch {self.current_epoch} - Training step {batch_idx} - Loss: {loss.item()}")
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Defines a single step in the validation loop.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): (input_sequence, target_value).
            batch_idx (int): Index of the current validation batch.

        Returns:
            torch.Tensor: The validation loss for this batch.
        """
        x, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)

        self.log("val_loss", loss.item(), on_epoch=True)
        logger.info(f"Epoch {self.current_epoch} - Validation step {batch_idx} - Loss: {loss.item()}")
        return loss

    def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
        """
        Defines the step for prediction.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor]): (input_sequence, target_value).
            batch_idx (int): Index of the current batch in prediction.

        Returns:
            torch.Tensor: Model predictions for this batch.
        """
        x, y = batch
        y_pred = self(x)
        return y_pred

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: The Adam optimizer.
        """
        logger.info(f"Configuring optimizer with learning rate: {self.lr}")
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)
