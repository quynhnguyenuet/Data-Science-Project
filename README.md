# Phát hiện bất thường không giám sát trong chuỗi thời gian bằng Deep Learning
## DeepAnT(Deep Learning for Unsupervised Anomaly Detection in Time Series)

> 📄 Dựa theo bài báo: [DeepAnT: A Deep Learning Approach for Unsupervised Anomaly Detection in Time Series](https://ieeexplore.ieee.org/document/8581424)
---
## Giới thiệu

**DeepAnT** là một mô hình deep learning sử dụng mạng tích chập (CNN) để phát hiện bất thường trong chuỗi thời gian theo phương pháp **không giám sát** (unsupervised). Mô hình có thể áp dụng cho cả **dữ liệu đơn biến** và **đa biến**, và không yêu cầu nhãn bất thường trong quá trình huấn luyện.

---

## Cấu trúc mô hình

- Sử dụng **CNN** để phát hiện các mẫu cục bộ.
- Kết hợp với **fully connected layers** để học các quan hệ dài hạn.
- Phát hiện bất thường dựa trên sai số giữa giá trị **dự đoán** và **giá trị thực tế**.
- Hình minh họa từ bài báo:

  ![Cấu trúc DeepAnT](https://github.com/quynhnguyenuet/Data-Science-Project/blob/main/Detect%20Anomaly%20in%20Deep%20Learning%20For%20Time%20Seria/DeepAnt/images/structure.png)

---

## Đặc điểm chính

- **Unsupervised Learning** – không yêu cầu dữ liệu được gán nhãn.
- **Tiền xử lý Sliding Window** – tạo cặp (input, target) từ chuỗi thời gian.
- **Dự đoán giá trị kế tiếp** – bất thường nếu sai số lớn.
- **Early Stopping & Best Model Checkpoint** – tránh overfitting.
- **Trực quan hóa** – biểu đồ cho cả dữ liệu gốc và điểm bất thường.

---

## Hỗ trợ dữ liệu đa biến

- Tự động nhận diện số chiều (features).
- Tính **ngưỡng phát hiện riêng** cho từng chiều bằng:

  **Ngưỡng = Trung bình ± N × Độ lệch chuẩn**

- Phát hiện và hiển thị bất thường theo từng chiều.
- Tạo **subplot riêng biệt** cho từng feature.



## LSTM-Autoencoder với Attention cho Chuỗi Thời Gian Đa Biến

<p align="center">
    <img src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg">
</p>

Kho lưu trữ này chứa mô hình **autoencoder cho dự báo chuỗi thời gian đa biến**.

Mô hình tích hợp hai cơ chế attention được mô tả trong bài báo  
📄 *[A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction](https://arxiv.org/abs/1704.02971)*  
và lấy cảm hứng từ [repo của Seanny123](https://github.com/Seanny123/da-rnn).

![Autoencoder architecture](https://github.com/quynhnguyenuet/Data-Science-Project/blob/main/Detect%20Anomaly%20in%20Deep%20Learning%20For%20Time%20Seria/Autoencoder-LSTM/autoenc_architecture.png)