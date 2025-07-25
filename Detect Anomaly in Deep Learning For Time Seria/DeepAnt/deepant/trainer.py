"""
Trainer pipeline containing a PyTorch Lightning-based implementation 
for training and inference using the DeepAnT model

"""

import os
import json
import logging
from typing import Dict, Tuple, List
from deepant.model import AnomalyDetector
import numpy as np
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from deepant.model import DeepAntPredictor
from utils.utils import calculate_thresholds,identify_anomalies
# Set up logging
logger = logging.getLogger(__name__)


# class AnomalyDetector(pl.LightningModule):
#     def __init__(self, model: torch.nn.Module, lr: float) -> None:
#         """
#         Anomaly Detector based on DeepAnt model.

#         Args:
#             model (nn.Module): The DeepAnt predictor model.
#             lr (float): Learning rate for the optimizer.
#         """
#         super(AnomalyDetector, self).__init__()
        
#         self.model = model
#         self.criterion = torch.nn.L1Loss()
#         self.lr = lr

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the DeepAnt model.

#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, feature_dim, window_size).

#         Returns:
#             torch.Tensor: Model prediction of shape (batch_size, feature_dim).
#         """
#         return self.model(x)

#     def training_step(self, batch, batch_idx: int) -> torch.Tensor:
#         """
#         Defines a single step in the training loop.

#         Args:
#             batch (Tuple[torch.Tensor, torch.Tensor]): (input_sequence, target_value).
#             batch_idx (int): Index of the current batch.

#         Returns:
#             torch.Tensor: The loss value for this batch.
#         """
#         x, y = batch
#         y_pred = self(x)
#         loss = self.criterion(y_pred, y)

#         self.log("train_loss", loss.item(), on_epoch=True)
#         logger.info(f"Epoch {self.current_epoch} - Training step {batch_idx} - Loss: {loss.item()}")
#         return loss

#     def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
#         """
#         Defines a single step in the validation loop.

#         Args:
#             batch (Tuple[torch.Tensor, torch.Tensor]): (input_sequence, target_value).
#             batch_idx (int): Index of the current validation batch.

#         Returns:
#             torch.Tensor: The validation loss for this batch.
#         """
#         x, y = batch
#         y_pred = self(x)
#         loss = self.criterion(y_pred, y)

#         self.log("val_loss", loss.item(), on_epoch=True)
#         logger.info(f"Epoch {self.current_epoch} - Validation step {batch_idx} - Loss: {loss.item()}")
#         return loss

#     def predict_step(self, batch, batch_idx: int) -> torch.Tensor:
#         """
#         Defines the step for prediction.

#         Args:
#             batch (Tuple[torch.Tensor, torch.Tensor]): (input_sequence, target_value).
#             batch_idx (int): Index of the current batch in prediction.

#         Returns:
#             torch.Tensor: Model predictions for this batch.
#         """
#         x, y = batch
#         y_pred = self(x)
#         return y_pred

#     def configure_optimizers(self) -> torch.optim.Optimizer:
#         """
#         Configure the optimizer for training.

#         Returns:
#             torch.optim.Optimizer: The Adam optimizer.
#         """
#         logger.info(f"Configuring optimizer with learning rate: {self.lr}")
#         return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-4)


class DeepAnT:
    def __init__(
            self, 
            config: Dict, 
            train_dataset: torch.utils.data.Dataset, 
            val_dataset: torch.utils.data.Dataset,
            test_dataset: torch.utils.data.Dataset,
            feature_dim: int
    ) -> None:
        """
        DeepAnT manager class for anomaly petection procedure.

        Args:
            config (Dict): Configuration dictionary.
            train_dataset (Dataset): Training dataset.
            val_dataset (Dataset): Validation dataset.
            test_dataset (Dataset): Test dataset.
            feature_dim (int): Number of channels in the input data.
        """
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.feature_dim = feature_dim
        self.device = config["device"]
        self.z = config.anomaly_threshold_z

        # Initialize the model
        self.deepant_predictor = DeepAntPredictor(
            feature_dim=feature_dim,
            window_size=config["window_size"],
            hidden_size=config["hidden_size"]
        ).to(self.device)
        self.anomaly_detector = AnomalyDetector(
            model=self.deepant_predictor,
            lr=float(config["lr"])
        ).to(self.device)

        # Initial trainer configuration
        self.initial_trainer = pl.Trainer(
            max_epochs=config["max_initial_steps"],
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=config["run_dir"],
                    filename="initial_model",
                    monitor="epoch",
                    save_top_k=1,
                    mode="max"
                )
            ],
            default_root_dir=config["run_dir"],
            accelerator=self.device,
            devices=1 if self.device == "cuda" else "auto",
        )

        # Main trainer configuration
        self.trainer = pl.Trainer(
            max_epochs=config["max_steps"],
            callbacks=[
                pl.callbacks.ModelCheckpoint(
                    dirpath=config["run_dir"],
                    filename="best_model",
                    monitor="val_loss",
                    save_top_k=1,
                    mode="min"
                ),
                pl.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=config["patience"],
                    mode="min"
                )
            ],
            default_root_dir=config["run_dir"],
            accelerator=self.device,
            devices=1 if self.device == "cuda" else "auto",
        )

        logger.info("DeepAnT model initialized.")

    def train(self) -> None:
        """
        Train the DeepAnT model in two phases:
            1) Initial short training,.
            2) Main training, with early stopping based on validation loss.
        """
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=True
        )
        val_loader = DataLoader(self.val_dataset, 
                                batch_size=self.config["batch_size"], 
                                shuffle=False
        )

        # Phase 1: initial training
        logger.info("Starting initial training phase...")
        self.initial_trainer.fit(self.anomaly_detector, train_loader)

        # Phase 2: Main training
        initial_checkpoint_path = os.path.join(self.config["run_dir"], "initial_model.ckpt")
        self.anomaly_detector = AnomalyDetector.load_from_checkpoint(
            checkpoint_path=initial_checkpoint_path,
            model=self.deepant_predictor,
            lr=float(self.config["lr"])
        )
        logger.info("Starting main training phase...")
        self.trainer.fit(self.anomaly_detector, train_loader, val_loader)
        logger.info("Training completed.")

    def detect_anomaly(self) -> None:
        """
        Detect anomalies on the test dataset using the best saved model.
        """
        logger.info("Starting anomaly detection process...")
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=self.test_dataset.data_x.shape[0], 
            shuffle=False
        )
        best_model = AnomalyDetector.load_from_checkpoint(
            checkpoint_path=os.path.join(self.config["run_dir"], "best_model.ckpt"),
            model=self.deepant_predictor,
            lr=float(self.config["lr"])
        )
        output = self.trainer.predict(best_model, test_loader)

        ground_truth = test_loader.dataset.data_y.squeeze()
        predictions = output[0].numpy().squeeze()
        if ground_truth.ndim == 1:
            ground_truth = ground_truth.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)

        anomaly_scores = np.abs(predictions - ground_truth)
        if anomaly_scores.ndim == 1:
            anomaly_scores = anomaly_scores.reshape(-1, 1)
        feature_names = getattr(self.test_dataset, "feature_names", [f"feature_{i}" for i in range(anomaly_scores.shape[1])])
        thresholds = calculate_thresholds(anomaly_scores,std_rate=self.z)
        logger.info(f"Calculated thresholds are: {thresholds}")

        anomalies_dict = identify_anomalies(anomaly_scores, thresholds,feature_names)
        with open(os.path.join(self.config["run_dir"], "anomalies_by_feature.json"), "w") as f:
            json.dump(anomalies_dict, f, indent=2)
        anomaly_events = []
        for feature, indices in anomalies_dict.items():
            for t in indices:
                anomaly_events.append({"feature": feature, "timestamp_index": int(t)})

        with open(os.path.join(self.config["run_dir"], "anomaly_events.json"), "w") as f:
            json.dump(anomaly_events, f, indent=2 )  
        with open(os.path.join(self.config["run_dir"], "anomalies_indices.json"), "w") as json_file:
            json.dump(anomalies_dict, json_file)
        logger.info(f"Anomalies detected: {anomalies_dict}")

        self.visualize_results(ground_truth, predictions, anomalies_dict, thresholds)
        logger.info("Anomaly detection process completed.")

    # def calculate_thresholds(self, anomaly_scores: np.ndarray, std_rate: int = 2) -> List[float]:
    #     """
    #     Calculate dynamic thresholds for anomaly detection (One threshold per feature).

    #     Args:
    #         anomaly_scores (np.ndarray): Anomaly scores of shape (num_samples, feature_dim).
    #         std_rate (int, optional): Standard deviation multiplier for thresholds (Defaults to 2).

    #     Returns:
    #         List[float]: A list of thresholds, one per feature.
    #     """
    #     thresholds = []
    #     for feature_idx in range(self.feature_dim):
    #         feature_scores = anomaly_scores[:, feature_idx]
    #         mean_scores = np.mean(feature_scores)
    #         std_scores = np.std(feature_scores)
    #         thresholds.append(mean_scores + std_rate * std_scores)
    #     return thresholds
    
    # def identify_anomalies(self, anomaly_scores: np.ndarray, thresholds: List[float]) -> dict:
    #     """
    #     Identify anomalies based on the per-feature calculated thresholds.

    #     Args:
    #         anomaly_scores (np.ndarray): Anomaly scores with shape (num_samples, feature_dim)
    #         thresholds (List[float]): List of threshold values for each feature.

    #     Returns:
    #         dict: Dictionary of anomalies.
    #     """
    #     anomalies_dict = {}
    #     for f_idx in range(self.feature_dim):
    #         feature_scores = anomaly_scores[:, f_idx]
    #         anomalies_dict[f"Feature_{f_idx + 1}"] = [
    #             i for i, score in enumerate(feature_scores) if score > thresholds[f_idx]
    #         ]
    #     return anomalies_dict
    
    def reconstruct_original_sequence(self) -> Tuple[np.ndarray, int]:
        """
        Reconstruct the original time series from overlapping windows in the dataset (validation + test).

        Returns:
            Tuple[np.ndarray, int]:
                - full_seq: shape (val_length + test_length, feature_dim).
                - boundary_idx: the index separating validation data and test data.
        """
        val_windows = self.val_dataset.data_x
        val_reconstructed_list = []
        for i in range(len(val_windows)):
            window = val_windows[i]
            if i == 0:
                val_reconstructed_list.append(window)
            else:
                val_reconstructed_list.append(window[-1:])

        val_reconstructed = np.concatenate(val_reconstructed_list, axis=0)

        test_windows = self.test_dataset.data_x
        test_reconstructed_list = []
        for i in range(len(test_windows)):
            window = test_windows[i]
            test_reconstructed_list.append(window[-1:])

        test_reconstructed = np.concatenate(test_reconstructed_list, axis=0)

        boundary_idx = len(val_reconstructed)
        full_seq = np.concatenate([val_reconstructed, test_reconstructed], axis=0)

        return full_seq, boundary_idx


    def visualize_results(
        self,
        target_seq: np.ndarray,
        pred_seq: np.ndarray,
        anomalies: Dict[str, List[int]],
        thresholds: List[float]
    ) -> None:
        """
        Visualize prediction vs. target and anomaly detection for each feature.

        Args:
            target_seq (np.ndarray): Target data, shape (num_samples, feature_dim)
            pred_seq (np.ndarray): Predicted data, same shape
            anomalies (dict): Keys like "Feature_1", values = list of anomaly indices
            thresholds (List[float]): Thresholds per feature (optional)
        """
        if len(thresholds) == 1 and self.feature_dim > 1:
            logger.warning("A single threshold was provided for multi-feature data.")

        original_data, boundary_idx = self.reconstruct_original_sequence()
        time_steps = range(original_data.shape[0])

        fig, axs = plt.subplots(
            self.feature_dim, 2,
            figsize=(14, 4 * self.feature_dim),
            sharex=False, sharey=False
        )
        if self.feature_dim == 1:
            axs = axs.reshape(1, 2)

        axs[0, 0].annotate(
            "Target vs. Prediction", xy=(0.5, 1.15), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=14, weight='bold'
        )
        axs[0, 1].annotate(
            "Detected Anomalies", xy=(0.5, 1.15), xycoords='axes fraction',
            ha='center', va='bottom', fontsize=14, weight='bold'
        )

        handles_left, labels_left = [], []
        handles_right, labels_right = [], []

        for f_idx in range(self.feature_dim):
            # === Plot Left (Target vs. Prediction) ===
            axs[f_idx, 0].plot(
                time_steps[:boundary_idx], original_data[:boundary_idx, f_idx],
                label="Validation Sequence" if f_idx == 0 else "_nolegend_",
                color="blue", linewidth=1.5
            )
            axs[f_idx, 0].plot(
                time_steps[boundary_idx:], target_seq[:, f_idx],
                label="Target Sequence" if f_idx == 0 else "_nolegend_",
                color="green", linestyle='--', linewidth=1.5
            )
            axs[f_idx, 0].plot(
                time_steps[boundary_idx:], pred_seq[:, f_idx],
                label="Predicted Sequence" if f_idx == 0 else "_nolegend_",
                color="orange", linestyle='-.', linewidth=1.5
            )
            axs[f_idx, 0].set_title(f"Feature {f_idx + 1}")
            axs[f_idx, 0].set_xlabel("Time")
            axs[f_idx, 0].set_ylabel("Value")
            axs[f_idx, 0].grid(True, linestyle='--', alpha=0.6)

            if f_idx == 0:
                handles_left.extend(axs[f_idx, 0].get_lines())
                labels_left.extend([line.get_label() for line in axs[f_idx, 0].get_lines()])

            # === Plot Right (Anomalies) ===
            axs[f_idx, 1].plot(
                time_steps[:boundary_idx], original_data[:boundary_idx, f_idx],
                label="Data Sequence" if f_idx == 0 else "_nolegend_",
                color="blue", linewidth=1.5
            )
            axs[f_idx, 1].plot(
                time_steps[boundary_idx:], target_seq[:, f_idx],
                color="blue", linewidth=1.5
            )

            # Vẽ đường chia train/test
            split_line = axs[f_idx, 1].axvline(
                x=boundary_idx, color="red", linestyle="--",
                linewidth=1, label="Val/Test Split" if f_idx == 0 else "_nolegend_"
            )

            # Vẽ ngưỡng nếu có
            if thresholds and len(thresholds) > f_idx:
                axs[f_idx, 1].axhline(
                    y=thresholds[f_idx],
                    color="purple", linestyle=":", linewidth=1.5,
                    label="Threshold" if f_idx == 0 else "_nolegend_"
                )

            # Vẽ anomaly points
            feature_key = f"Feature_{f_idx + 1}"
            if feature_key in anomalies:
                anomaly_indices = [i for i in anomalies[feature_key] if i < len(target_seq)]
                axs[f_idx, 1].scatter(
                    [time_steps[boundary_idx + i] for i in anomaly_indices],
                    target_seq[anomaly_indices, f_idx],
                    color="red", edgecolors="black", s=60, marker='o',
                    label="Anomaly Points" if f_idx == 0 else "_nolegend_",
                    zorder=5
                )

            axs[f_idx, 1].set_title(f"Feature {f_idx + 1}")
            axs[f_idx, 1].set_xlabel("Time")
            axs[f_idx, 1].set_ylabel("Value")
            axs[f_idx, 1].grid(True, linestyle='--', alpha=0.6)

            if f_idx == 0:
                handles_right.extend(axs[f_idx, 1].get_lines())
                labels_right.extend([line.get_label() for line in axs[f_idx, 1].get_lines()])
                handles_right.append(split_line)
                labels_right.append(split_line.get_label())

        # === Legend ===
        if handles_left:
            axs[0, 0].legend(handles_left, labels_left, loc="upper right", fontsize=10)
        if handles_right:
            unique_right = dict(zip(labels_right, handles_right))
            axs[0, 1].legend(unique_right.values(), unique_right.keys(), loc="upper right", fontsize=10)

        plt.tight_layout()
        save_path = os.path.join(self.config["run_dir"], "anomalies_visualization.png")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Visualization saved to {save_path}")
