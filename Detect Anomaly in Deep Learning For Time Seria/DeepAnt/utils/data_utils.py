"""
Data Utilities for Time-Series Datasets

"""

import os
import logging
import numpy as np
import pandas as pd
import torch
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler

# Set up logging
logger = logging.getLogger(__name__)


class DataModule(torch.utils.data.Dataset):
    def __init__(self, data_x: np.ndarray, data_y: np.ndarray, device: str) -> None:
        """
        Dataset cho dữ liệu chuỗi thời gian dùng trong PyTorch.

        Mỗi mẫu huấn luyện là một tuple gồm:
            (input_sequence, target_value)
        Trong đó:
            - input_sequence: có shape (num_features, window_size)
                              → là chuỗi đầu vào gồm nhiều bước thời gian (sliding window)
            - target_value: có shape (num_features,)
                            → là vector giá trị thực tại bước thời gian kế tiếp

        Tham số:
            data_x (np.ndarray): Mảng chứa các chuỗi đầu vào, 
                                 shape = (num_samples, window_size, num_features)
                                 → cần được transpose lại thành (num_features, window_size) khi dùng cho Conv1D.
            data_y (np.ndarray): Mảng chứa các giá trị mục tiêu (1 bước thời gian tiếp theo), 
                                 shape = (num_samples, 1, num_features)
            device (str): Thiết bị tính toán ('cuda' hoặc 'cpu')
        """
        self.data_x = data_x
        self.data_y = data_y
        self.device = device

    def __len__(self) -> int:
        """
        Trả về:
            int: Tổng số mẫu (số lượng cửa sổ sliding window) trong tập dữ liệu.
        """
        return len(self.data_x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Lấy một mẫu dữ liệu chuỗi thời gian và giá trị mục tiêu tương ứng tại chỉ số `idx`.

        Tham số:
            idx (int): Vị trí (index) của mẫu trong tập dữ liệu.

        Trả về:
            Tuple[torch.Tensor, torch.Tensor]:
                - input_sequence (torch.Tensor): Dữ liệu đầu vào có shape (num_features, window_size)
                                                sau khi chuyển đổi từ (window_size, num_features).
                - target_value (torch.Tensor): Giá trị dự đoán tại bước kế tiếp,
                                            có shape (num_features,)
        """
        return (
            torch.tensor(self.data_x[idx], device=self.device, dtype=torch.float32).transpose(0, 1),
            torch.tensor(self.data_y[idx], device=self.device, dtype=torch.float32).squeeze(0),
        )

def classify_by_sparsity(sparsity_value: float) -> str:
    if sparsity_value >= 0.95:
        return 'sparse_high'
    elif sparsity_value >= 0.5:
        return 'sparse_medium'
    else:
        return 'dense'

def split_dataframe_by_sparsity(df: pd.DataFrame):
    non_feature_cols = ['time_window']
    df_features = df.drop(columns=non_feature_cols, errors='ignore')


    sparsity_series = (df_features == 0).sum() / len(df_features)
    feature_groups = sparsity_series.apply(classify_by_sparsity)

    df_sparse_high = df[['time_window'] + feature_groups[feature_groups == 'sparse_high'].index.tolist()]
    df_dense_medium = df[['time_window'] + feature_groups[feature_groups == 'dense'].index.tolist() + feature_groups[feature_groups == 'sparse_medium'].index.tolist() ]
    # df_dense = df[['time_window'] + feature_groups[feature_groups == 'dense'].index.tolist() ]

    return df_sparse_high, df_dense_medium
def load_data(
        dataset_name: str, 
        window_size: int, 
        device: str, 
        val_rate: float = 0.1, 
        test_rate: float = 0.1
) -> Tuple[DataModule, DataModule, DataModule, int]:
    """
    Tải và tiền xử lý một bộ dữ liệu chuỗi thời gian.

    Tham số:
        dataset_name (str): Tên bộ dữ liệu cần sử dụng 
                            (ví dụ: 'NAB', 'AirQuality', hoặc 'DNS_LOGS').
        window_size (int): Số bước thời gian trong mỗi cửa sổ trượt (sliding window).
        device (str): Thiết bị tính toán để huấn luyện mô hình ('cuda' hoặc 'cpu').
        val_rate (float, tùy chọn): Tỷ lệ dữ liệu dành cho tập validation (mặc định = 0.1).
        test_rate (float, tùy chọn): Tỷ lệ dữ liệu dành cho tập kiểm thử (mặc định = 0.1).

    Trả về:
        Tuple[DataModule, DataModule, DataModule, int]:
            - train_dataset (DataModule): Tập dữ liệu huấn luyện.
            - val_dataset (DataModule): Tập dữ liệu dùng để validation.
            - test_dataset (DataModule): Tập dữ liệu kiểm thử.
            - num_features (int): Số lượng đặc trưng (features) trong dữ liệu đầu vào.
    """

    logger.info(f"Loading dataset: {dataset_name}")
    path = os.path.join("./data", dataset_name)

    if dataset_name == "NAB":
        file_name = "TravelTime_451.csv"
        data = pd.read_csv(
            os.path.join(path, file_name),
            index_col="timestamp",
            parse_dates=["timestamp"]
        )

    elif dataset_name == "AirQuality":
        file_name = "AirQualityUCI.csv"
        data = pd.read_csv(
            os.path.join(path, file_name),
            sep=";",
            decimal=".",
            na_values=-200      # As per dataset docs: -200 indicates missing data
        )
        data["timestamp"] = pd.to_datetime(
            data["Date"] + " " + data["Time"], format="%d/%m/%Y %H.%M.%S"
        )
        data.drop(columns=["Date", "Time"], inplace=True)
        data.set_index("timestamp", inplace=True)
        data.dropna(axis=1, how="all", inplace=True)        # Remove columns fully NaN
        data.ffill(inplace=True)                            # Forward-fill missing values 
        data.dropna(axis=0, how="any", inplace=True)        # Drop rows that still have NaNs
        data = data.select_dtypes(include=[np.number])      # Keep only numeric columns (sensor data)
    elif dataset_name == "DNS_LOGS":
        file_name = "detection_anomaly_log_dns.csv"
        data = pd.read_csv(
            os.path.join(path,file_name)
        )
        # features_to_binarize = [
        #     'txt_query',
        #     'nxdomain_query',
        #     'count_cat_icmp',
        #     'format_error_count',
        #     'count_cat_blacklist',
        # ]
        # for col in features_to_binarize:
        #     if col in data.columns:
        #         data[col] = (data[col] > 0).astype(int)
        # features_to_smooth = [
        #     'count_cat_protocol_anomaly',
        #     'count_cat_msg_types',
        #     'count_cat_default_drop',
        #     'early_drop_count',
        #     'multiple_questions_count'
        # ]
        
        # for col in features_to_smooth:
        #     if col in data.columns:
        #         data[col] = data[col].rolling(window=5, min_periods=1).mean()
        # df_sparse_high,df_dense_medium = split_dataframe_by_sparsity(data)
        feature_drop = ['nxdomain_query','txt_query','count_cat_icmp']
        data.drop(columns=feature_drop,inplace=True)
        # timestamps = df_dense_medium["time_window"].tolist()
        data["time_window"] = pd.to_datetime(data["time_window"])
        data.set_index("time_window", inplace=True)
        # data = data[selected_features]
        # data['txt_query'] = data['txt_query'].rolling(window=10, min_periods=1).mean()
  
        data = data.select_dtypes(include=[np.number])
        logger.info(f"Columns after selection: {data.columns.tolist()}")
        logger.info(f"Data shape: {data.shape}")
        
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    logger.info(f"Dataset shape after loading: {data.shape}")

    sc = MinMaxScaler()
    data_scaled = sc.fit_transform(data)
    data_x, data_y = split_data(data_scaled, window_size)

    train_slice = slice(None, int((1 - val_rate - test_rate) * len(data_x)))
    val_slice = slice(int((1 - val_rate - test_rate) * len(data_x)), int((1 - test_rate) * len(data_x)))
    test_slice = slice(int((1 - test_rate) * len(data_x)), None)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    train_dataset = DataModule(data_x[train_slice], data_y[train_slice], device)
    val_dataset = DataModule(data_x[val_slice], data_y[val_slice], device)
    test_dataset = DataModule(data_x[test_slice], data_y[test_slice], device)

    logger.info(f"Train dataset shape: {train_dataset.data_x.shape}")
    logger.info(f"Validation dataset shape: {val_dataset.data_x.shape}")
    logger.info(f"Test dataset shape: {test_dataset.data_x.shape}")

    return train_dataset, val_dataset, test_dataset, data_x.shape[-1]

def split_data(
        data: np.ndarray, 
        window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cắt dữ liệu thành các chuỗi con theo kiểu sliding window, với bước tiếp theo làm nhãn dự đoán.

    Tham số:
        data (np.ndarray): Dữ liệu đã được chuẩn hóa (scaled), có shape (số_mẫu, số_đặc_trưng).
        window_size (int): Số bước thời gian trong mỗi cửa sổ trượt.

    Trả về:
        Tuple[np.ndarray, np.ndarray]:
            - data_x (np.ndarray): Mảng các cửa sổ trượt (sliding windows), 
                                   có shape (số_cửa_sổ, window_size, số_đặc_trưng).
            - data_y (np.ndarray): Mảng các nhãn tương ứng là bước kế tiếp, 
                                   có shape (số_cửa_sổ, 1, số_đặc_trưng).
    """
    data_x, data_y = [], []
    for i in range(window_size, data.shape[0]):
        if (i + 1) >= data.shape[0]:
            break
        window = data[i - window_size:i]     # cửa sổ đầu vào
        target = data[i:i + 1]               # bước tiếp theo làm nhãn
        data_x.append(window)
        data_y.append(target)

    return np.array(data_x), np.array(data_y)
