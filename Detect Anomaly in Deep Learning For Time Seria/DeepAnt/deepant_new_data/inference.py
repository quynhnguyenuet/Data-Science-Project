import os
import json
import logging
from glob import glob
import plotly.graph_objects as go
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from deepant.model import AnomalyDetector
from deepant.model import DeepAntPredictor
from utils.data_utils import split_data
from utils.utils import load_config, calculate_thresholds, identify_anomalies

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class InferenceRunner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        self.window_size = cfg.window_size
        #self.num_features = cfg.num_features
        self.batch_size = cfg.batch_size
        self.z = cfg.anomaly_threshold_z
        self.run_dir = cfg.run_dir
        self.output_dir = cfg.output_dir
        self.new_data_path = cfg.new_data_path
        self.ckpt_path = self._find_checkpoint(cfg.ckpt_path)
        df = pd.read_csv(self.new_data_path)
        df = df.select_dtypes(include=[np.number])
        self.num_features = df.shape[1]


    def _find_checkpoint(self, ckpt_cfg):
        if ckpt_cfg == "auto" or ckpt_cfg is None:
            # Ưu tiên checkpoint dạng best_model-v*.ckpt
            ckpt_list = glob(os.path.join(self.run_dir, "best_model-v*.ckpt"))
            if ckpt_list:
                ckpt = max(ckpt_list, key=os.path.getmtime)
                logger.info(f"Dùng checkpoint mới nhất: {ckpt}")
                return ckpt

            default_ckpt = os.path.join(self.run_dir, "best_model.ckpt")
            if os.path.exists(default_ckpt):
                logger.warning(f"Không tìm thấy best_model-v*.ckpt — fallback dùng: {default_ckpt}")
                return default_ckpt

            # Không có checkpoint nào phù hợp
            raise FileNotFoundError(f"Không tìm thấy checkpoint nào trong {self.run_dir} (best_model-v*.ckpt hoặc best_model.ckpt)")
        
        else:
            # Nếu người dùng chỉ định checkpoint cụ thể
            if not os.path.exists(ckpt_cfg):
                raise FileNotFoundError(f"Checkpoint chỉ định không tồn tại: {ckpt_cfg}")
            logger.info(f"Dùng checkpoint chỉ định: {ckpt_cfg}")
            return ckpt_cfg
    @staticmethod
    def classify_by_sparsity(sparsity_value: float) -> str:
        if sparsity_value >= 0.95:
            return 'sparse_high'
        elif sparsity_value >= 0.5:
            return 'sparse_medium'
        else:
            return 'dense'

    @staticmethod
    def split_dataframe_by_sparsity(df: pd.DataFrame):
        non_feature_cols = ['time_window']
        df_features = df.drop(columns=non_feature_cols, errors='ignore')

        sparsity_series = (df_features == 0).sum() / len(df_features)
        feature_groups = sparsity_series.apply(InferenceRunner.classify_by_sparsity)

        df_sparse_high = df[['time_window'] + feature_groups[feature_groups == 'sparse_high'].index.tolist()]
        df_dense_medium = df[['time_window'] + feature_groups[feature_groups.isin(['dense', 'sparse_medium'])].index.tolist()]

        return df_sparse_high, df_dense_medium
    def run(self):
        # === Load model ===


        # === Load and preprocess data ===
        logger.info(f"Đọc dữ liệu từ: {self.new_data_path}")
        df = pd.read_csv(self.new_data_path)
        # selected_features = [
        #     'rpz_block_count',
        #     'unique_ip_rpz_hit_count',
        #     'max_queries_hit_count_from_one_ip',
        #     'format_error_count',
        #     'txt_query',
        #     'flood_event_count_udp',
        #     'nxdomain_query',
        #     'resolver_error_count',
        #     'count_act_drop',
        #     'count_cat_default_drop',
        #     'count_cat_msg_types',
        #     'count_cat_protocol_anomaly'
        #     # ,'fqdn_entropy_max'
        # ]
        # features_to_binarize = [
        #     'txt_query',
        #     'nxdomain_query',
        #     'count_cat_icmp',
        #     'format_error_count',
        #     'count_cat_blacklist',
        # ]
        # for col in features_to_binarize:
        #     if col in df.columns:
        #         df[col] = (df[col] > 0).astype(int)
        # features_to_smooth = [
        #     'count_cat_protocol_anomaly',
        #     'count_cat_msg_types',
        #     'count_cat_default_drop',
        #     'early_drop_count',
        #     'multiple_questions_count'
        # ]
        # feature_drop = ['fqdn_entropy_max', 'fqdn_entropy_avg']
        # df.drop(columns=feature_drop, inplace=True)
        # for col in features_to_smooth:
        #     if col in df.columns:
        #         df[col] = df[col].rolling(window=5, min_periods=1).mean()
        timestamps = df["time_window"].tolist()
        # df = df[selected_features]
        # df['txt_query'] = df['txt_query'].rolling(window=10, min_periods=1).mean()
        # print(df.head()) 
        # df_sparse_high,df_dense_medium = self.split_dataframe_by_sparsity(df)
        feature_drop = ['nxdomain_query','txt_query','count_cat_icmp']
        df.drop(columns=feature_drop,inplace=True)
        timestamps = df["time_window"].tolist()
        df["time_window"] = pd.to_datetime(df["time_window"])
        df.set_index("time_window", inplace=True)
        df = df.select_dtypes(include=[np.number])
        self.num_features = df.shape[1]
        feature_names = df.columns.tolist()
        model_core = DeepAntPredictor(
            feature_dim=self.num_features,
            window_size=self.window_size,
            hidden_size=self.cfg.hidden_size  
        )
        model = AnomalyDetector.load_from_checkpoint(self.ckpt_path, model=model_core, lr=self.cfg.lr)
        model.to(self.device)
        model.eval()
        # df = df.iloc[:, :self.num_features]

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(df.values)

        data_x, data_y = split_data(data_scaled, self.window_size)
        dataset = TensorDataset(
            torch.tensor(data_x, dtype=torch.float32).transpose(1, 2),  # (batch, features, window)
            torch.tensor(data_y[:, 0, :], dtype=torch.float32)
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)

        # === Predict ===
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = model(x)
                all_preds.append(y_hat.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        errors = np.abs(preds - targets)

        # === Anomaly detection ===
        thresholds = calculate_thresholds(errors, std_rate=self.z)
        anomalies_dict = identify_anomalies(errors, thresholds,feature_names)
        all_anomaly_indices = sorted(set(idx for indices in anomalies_dict.values() for idx in indices))

        # === Save results ===
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(self.output_dir, "anomalies_by_feature.json"), "w") as f:
            json.dump(anomalies_dict, f, indent=2)
        anomaly_records = []
        for feat_name, indices in anomalies_dict.items():
            for idx in indices:
                if 0 <= idx < len(timestamps):
                    anomaly_records.append({
                        "feature": feat_name,
                        "timestamp_index": idx,
                        "timestamp": str(timestamps[idx])
                    })

        # Ghi toàn bộ điểm bất thường ra file
        with open(os.path.join(self.output_dir, "anomaly_events.json"), "w") as f:
            json.dump(anomaly_records, f, indent=2)
        with open(os.path.join(self.output_dir, "anomalies_all_indices.json"), "w") as f:
            json.dump(all_anomaly_indices, f)

        mean_error = np.mean(errors, axis=1)
        # plt.figure(figsize=(12, 5))
        # plt.plot(mean_error, label='Mean Reconstruction Error')
        # plt.scatter(all_anomaly_indices, mean_error[all_anomaly_indices], color='red', marker='x', label='Anomalies')
        # plt.title("Anomaly Detection Result (Feature-wise Thresholding)")
        # plt.xlabel("Time step")
        # plt.ylabel("Mean Error")
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(os.path.join(self.output_dir, "anomaly_plot.png"))
        trace_mean = go.Scatter(
            y=mean_error,
            mode='lines',
            name='Mean Reconstruction Error'
    )

        trace_anomalies = go.Scatter(
            x=all_anomaly_indices,
            y=mean_error[all_anomaly_indices],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=6, symbol='x')
        )

        layout = go.Layout(
            title="Anomaly Detection Result (Interactive)",
            xaxis=dict(title='Time step'),
            yaxis=dict(title='Mean Error'),
            height=500,
            width=1200,
            legend=dict(x=0, y=1.1, orientation='h'),
            margin=dict(t=50, b=40)
        )

        fig = go.Figure(data=[trace_mean, trace_anomalies], layout=layout)

        # Hiển thị khi chạy Jupyter hoặc trực tiếp
        # fig.show()

        # Lưu ra file HTML
        plot_path = os.path.join(self.output_dir, "anomaly_plot_interactive.html")
        # fig.write_image(os.path.join(self.output_dir, "anomaly_plot_interactive.png"))
        fig.write_html(plot_path)
        logger.info(f"Đã lưu biểu đồ tương tác tại: {plot_path}")
        logger.info(f"Đã phát hiện {len(all_anomaly_indices)} điểm bất thường.")
        logger.info(f"Kết quả lưu tại: {self.output_dir}")
                # === Ghi file Excel đánh dấu điểm bất thường ===
        logger.info("Đang xuất file Excel với đánh dấu điểm bất thường...")

        # Đọc lại dữ liệu gốc (vì split_data làm mất window_size đầu tiên)
        df_raw = pd.read_csv(self.new_data_path)
        df_raw = df_raw.iloc[self.window_size:].reset_index(drop=True)
        df_raw["time_window"] = timestamps[self.window_size:]

        # Tạo cột đánh dấu bất thường (1 = bất thường, 0 = bình thường)
        is_anomaly_flags = np.zeros(len(df_raw), dtype=int)
        for idx in all_anomaly_indices:
            if 0 <= idx < len(is_anomaly_flags):
                is_anomaly_flags[idx] = 1
        df_raw["is_anomaly"] = is_anomaly_flags

        # Xuất ra file Excel
        excel_path = os.path.join(self.output_dir, "anomaly_flagged_data.xlsx")
        df_raw.to_excel(excel_path, index=False)
        logger.info(f"Đã lưu file Excel đánh dấu điểm bất thường tại: {excel_path}")



if __name__ == "__main__":
    cfg = load_config("config.yaml")
    if cfg.inference_mode:
        runner = InferenceRunner(cfg)
        runner.run()
    else:
        logger.warning("Đặt `inference_mode: true` trong config.yaml để chạy inference.")
