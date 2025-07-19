"""
Utilities for Configuration and Environment Setup

"""

import argparse
import logging
import os
import random
import torch
from omegaconf import DictConfig, OmegaConf
import numpy as np
from typing import List, Dict
# Set up logging
logger = logging.getLogger(__name__)


def load_config(config_file: str) -> DictConfig:
    """
    Load the configuration data from the YAML file.

    Args:
        config_file (str): Path to the configuration YAML file.

    Returns:
        DictConfig: The combined configuration dictionary.
    """
    config = OmegaConf.load(config_file)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default=config.defaults[0].dataset, 
        help="Name of the dataset to use (must be defined in the YAML config)"
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    if dataset_name not in config.dataset:
        raise ValueError(f"Dataset '{dataset_name}' not found in configuration file.")
    
    cfg = OmegaConf.merge(config.common, config.dataset[dataset_name])
    return prepare_config(cfg)

def prepare_config(cfg: DictConfig) -> DictConfig:
    """
    Prepare the configuration by:
      1. Ensuring the required directory exists.
      2. Setting the random seed.
      3. Detecting the compute device.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        DictConfig: The updated configuration dictionary.
    """
    logger.info("Preparing configuration...")
    
    check_dir(cfg.run_dir)
    check_seed(cfg.seed)
    cfg.device = check_device()

    # Log hyper-parameters
    logger.info("*******  Hyper-parameters  ********")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("***********************************")

    logger.info("Configuration prepared successfully")
    return cfg

def check_dir(dir_name: str) -> None:
    """
    Check if a directory exists; creates it if missing.
    """
    os.makedirs(dir_name, exist_ok=True)
    logger.info(f"Directory verified: {dir_name}")

def check_seed(seed: int) -> None:
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Random seed set to: {seed}")

def check_device() -> str:
    """
    Detects CUDA availability; returns 'cuda' if available, else 'cpu'.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device set to: {device}")
    return device

def calculate_thresholds(anomaly_scores: np.ndarray, std_rate: int) -> List[float]:
    """
    Calculate dynamic thresholds for anomaly detection (one threshold per feature).

    Args:
        anomaly_scores (np.ndarray): Anomaly scores of shape (num_samples, feature_dim).
        std_rate (int, optional): Multiplier for standard deviation (default = 2).

    Returns:
        List[float]: Threshold for each feature.
    """
    thresholds = []
    for feature_idx in range(anomaly_scores.shape[1]):
        feature_scores = anomaly_scores[:, feature_idx]
        mean_scores = np.mean(feature_scores)
        std_scores = np.std(feature_scores)
        thresholds.append(mean_scores + std_rate * std_scores)
    return thresholds


def identify_anomalies(anomaly_scores: np.ndarray, thresholds: List[float], feature_names: List[str]) -> Dict[str, List[int]]:
    """
    Identify anomalies based on calculated thresholds.

    Args:
        anomaly_scores (np.ndarray): Anomaly scores of shape (num_samples, feature_dim).
        thresholds (List[float]): List of thresholds for each feature.

    Returns:
        Dict[str, List[int]]: Dictionary of anomaly indices per feature.
    """
    anomalies = {}
    for idx, (feature_score, threshold) in enumerate(zip(anomaly_scores.T, thresholds)):
        anomaly_indices = np.where(feature_score > threshold)[0].tolist()
        anomalies[feature_names[idx]] = anomaly_indices
    return anomalies
