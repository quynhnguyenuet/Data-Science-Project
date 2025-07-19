# 🛡️ Phát hiện Bất Thường trong Log DNS Inforblox bằng Học Sâu Không Giám Sát

Dự án xây dựng một hệ thống phát hiện bất thường từ **log hệ thống bảo mật (security logs)** của thiết bị **DNS Inforblox** bằng kỹ thuật **học sâu không giám sát** (unsupervised deep learning).

## Các Hành Vi Bất Thường Được Ghi Nhận

Dưới đây là một số mẫu hành vi bất thường tiêu biểu được hệ thống phát hiện trong quá trình phân tích log từ thiết bị **DNS Inforblox**:

- 🔁 **UDP/TCP DNS Flood**  
  Lượng lớn truy vấn DNS được gửi trong thời gian ngắn từ nhiều nguồn hoặc một nguồn duy nhất.  
  → Dấu hiệu rõ ràng của **tấn công DDoS** vào hệ thống DNS.

- 📈 **Truy vấn từ một IP duy nhất với tần suất đột biến**  
  - Có thể là truy vấn tự động, dò quét dịch vụ DNS, hoặc từ **malware điều khiển từ xa**.
  - Tần suất truy vấn vượt mức bình thường trong thời gian ngắn.

- ❌ **Truy vấn bất thường theo phản hồi DNS**:
  - **NXDOMAIN**: Truy vấn tới domain không tồn tại
  - **REFUSED**: Bị hệ thống từ chối truy vấn
  - **FORMAT ERROR**: Lỗi cú pháp định dạng DNS  
  → Đây thường là **dấu hiệu của botnet hoặc công cụ tấn công DNS**.

- 🔐 **Truy vấn có cấu trúc bất thường / entropy cao**  
  - Domain dài, nhiều ký tự ngẫu nhiên → nghi vấn **DNS Tunneling** hoặc **DGA (Domain Generation Algorithm)**.
  - Entropy cao → không giống các domain bình thường.

- ⚠️ **Truy vấn bị drop ngay hoặc cảnh báo nhưng chưa chặn**  
  - Những truy vấn này không được xử lý hoặc chỉ cảnh báo (alert-only)  
  → Đây có thể là **hành vi nguy hiểm tiềm ẩn**, cần theo dõi thêm hoặc nâng mức cảnh báo.

---

> 📌 Ghi chú: Việc xác định bất thường không chỉ dựa vào tần suất, mà còn dựa vào **ngữ cảnh, nguồn phát**, và **mẫu hành vi theo thời gian**.
mà **không cần dữ liệu gán nhãn**.

---

> ⚠️ **Lưu ý bảo mật**:
>
> Dự án sử dụng dữ liệu **thực tế từ hệ thống mạng nội bộ của một ngân hàng**.  
> Do vậy, **dữ liệu không được phép chia sẻ công khai** dưới bất kỳ hình thức nào nhằm tuân thủ chính sách bảo mật và quy định nội bộ.
>
> Mọi thử nghiệm, huấn luyện và đánh giá đều được thực hiện trong môi trường kiểm soát, tuân thủ nghiêm ngặt yêu cầu bảo mật thông tin.

---
## Đặc trưng (Features) Đưa Vào Mô Hình

### 🗂️ Nguồn dữ liệu:
- **Thiết bị**: Log từ hệ thống **DNS Infoblox**
- **Số lượng bản ghi**: `12,960` dòng log
- **Số trường dữ liệu (features)**: `30` trường
- **Khoảng thời gian ghi nhận**:  
  Dữ liệu được ghi **liên tục theo từng phút**, từ:  
  ⏱️ `'2025-06-21 23:59:00'` → `'2025-06-30 23:58:00'`  
  Tổng thời gian: **10 ngày**

| STT | Tên feature                     | Trường liên quan        | Định nghĩa                                                        | Ý nghĩa & ảnh hưởng đến mô hình                                                                 |
|-----|----------------------------------|--------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| 1   | time_window                     | Date, time               | Khoảng thời gian 1 phút ghi nhận log                              | Đơn vị thời gian cơ bản để gom nhóm log theo chuỗi thời gian                                   |
| 2   | rpz_block_count                 | msg                      | Tổng số truy vấn bị chặn bởi DNS firewall                         | Cho thấy mức độ truy cập đến domain nguy hiểm                                                   |
| 3   | rpz_qname_count                | log_type                 | Truy vấn bị chặn dựa trên tên miền (QNAME)                        | Phản ánh truy cập đến các domain độc hại                                                        |
| 4   | rpz_ip_count                   | msg                      | Truy vấn bị chặn dựa trên IP đích                                 | Dấu hiệu IP đích nằm trong blacklist                                                            |
| 5   | unique_ip_rpz_hit_count       | msg, client_ip           | Số lượng truy vấn lớn nhất từ 1 IP trong 1 phút                   | Đánh giá mức độ phân tán tấn công                                                               |
| 6   | max_queries_hit_count_from_one_ip | msg, client_ip        | Số lượng truy vấn lớn nhất từ 1 IP                                | Dấu hiệu DDoS hoặc quét port                                                                    |
| 7   | refused_count                  | log_type                 | Truy vấn bị từ chối xử lý                                         | Có thể do cấu hình sai hoặc truy cập trái phép                                                  |
| 8   | format_error_count            | log_type                 | Truy vấn DNS sai định dạng                                        | Dấu hiệu của fuzzing hoặc công cụ tấn công                                                      |
| 9   | resolver_error_count          | log_type                 | Truy vấn đến DNS không được ủy quyền                              | Có thể là tấn công hoặc lỗi hệ thống                                                            |
| 10  | resolver_priming_count        | msg                      | Truy vấn khởi tạo lại root zone                                   | Đột biến có thể là tấn công hoặc hệ thống bị can thiệp                                         |
| 11  | nxdomain_query                | msg                      | Truy vấn đến domain không tồn tại                                 | Dấu hiệu domain giả trong DGA hoặc malware                                                      |
| 12  | txt_query                     | qtype                    | Truy vấn bản ghi TXT                                              | Có thể bị dùng cho DNS tunneling                                                                |
| 13  | early_drop_count             | event_name               | Truy vấn bị chặn ngay khi bắt đầu                                 | Phản ánh cơ chế phòng thủ chủ động mạnh                                                         |
| 14  | multiple_questions_count     | event_name               | Truy vấn chứa nhiều câu hỏi DNS                                   | Vi phạm RFC, dấu hiệu malware/scanner                                                           |
| 15  | flood_event_count_udp        | category, event_name     | Ghi nhận tấn công UDP Flood                                       | Loại tấn công volumetric phổ biến                                                               |
| 16  | flood_event_count_tcp        | category, event_name     | Ghi nhận tấn công TCP Flood                                       | Tinh vi hơn, lợi dụng kết nối hợp lệ                                                            |
| 17  | unique_ips_flagged_flood     | category, src_ip         | Số IP bị gắn cờ flood                                             | Đánh giá mức phân tán của tấn công (botnet)                                                    |
| 18  | ntp_drop_count               | event_name, category, act| Gói NTP bị loại bỏ                                                | Dấu hiệu tấn công NTP reflection                                                               |
| 19  | max_hit_count                | hit_count                | Số truy vấn bất thường cao nhất trong 1 phút                      | Phát hiện đỉnh điểm tấn công                                                                    |
| 20  | avg_hit_count                | hit_count                | Trung bình số truy vấn bất thường                                 | Phát hiện hành vi nghi vấn kéo dài                                                              |
| 21  | count_act_drop               | act                      | Truy vấn bị chặn (DROP)                                           | Biểu thị phản ứng bảo mật mạnh                                                                  |
| 22  | count_act_alert              | act                      | Truy vấn bị cảnh báo (ALERT)                                      | Cần theo dõi thêm, nghi vấn                                                                     |
| 23  | count_cat_default_drop       | category                 | Truy vấn bị chặn theo rule mặc định                               | Vi phạm chính sách cơ bản                                                                       |
| 24  | count_cat_icmp               | category                 | Truy vấn/dữ liệu ICMP                                             | Dễ bị lợi dụng để dò quét hoặc tấn công                                                         |
| 25  | count_cat_blacklist          | category                 | Truy vấn đến domain trong blacklist                               | Dấu hiệu nguy hiểm (malware, phishing)                                                         |
| 26  | count_cat_msg_types          | category                 | Loại bản tin DNS vi phạm chuẩn                                    | Dấu hiệu hệ thống lỗi hoặc mã độc                                                               |
| 27  | count_fqns_na_or_null        | fqdn                     | Truy vấn không có FQDN hợp lệ                                     | Dấu hiệu giả mạo, ẩn danh, bypass                                                               |
| 28  | count_cat_protocol_anomaly   | category                 | Gói DNS sai chuẩn giao thức                                       | Dấu hiệu tấn công hoặc tunneling                                                               |
| 29  | fqdn_entropy_max             | fqdn                     | Độ ngẫu nhiên cao nhất của domain                                 | Dấu hiệu DGA, tunneling, malware                                                                |
| 30  | fqdn_entropy_avg             | fqdn                     | Độ ngẫu nhiên trung bình của domain                               | Dùng đánh giá rủi ro tổng thể                                                                  |

### Phân phối và mối quan hệ tương quan của các đặc trưng

<div align="center" style="font-size:14px; color: gray;">
  <div>
    <p><em>Biểu đồ phân phối đặc trưng (Histogram)</em></p>
    <img src="https://github.com/quynhnguyenuet/Data-Science-Project/blob/main/Detect%20Anomaly%20in%20Deep%20Learning%20For%20Time%20Seria/DeepAnt/images/image.png" alt="Histogram" width="100100%" />
  </div>

  <br/>

  <div>
    <p><em>Ma trận tương quan giữa các đặc trưng</em></p>
    <img src="https://github.com/quynhnguyenuet/Data-Science-Project/blob/main/Detect%20Anomaly%20in%20Deep%20Learning%20For%20Time%20Seria/DeepAnt/images/image_1.png" alt="Correlation Matrix" width="100%" />
  </div>
</div>
### 📉 Quá trình huấn luyện mô hình

<div align="center">
  <img src="https://github.com/quynhnguyenuet/Data-Science-Project/blob/main/Detect%20Anomaly%20in%20Deep%20Learning%20For%20Time%20Seria/DeepAnt/train.png" alt="Train Loss" width="4545%" />
  <p style="font-size:14px; color:gray;"><em>Biểu đồ Train Loss</em></p>
</div>

**Nhận xét Train Loss:**
- Đường cong `train_loss` có xu hướng giảm đều, từ khoảng `0.0484` xuống `0.0471`.
- Cho thấy quá trình huấn luyện ổn định, mô hình học tốt từ dữ liệu huấn luyện.
- Không có dấu hiệu overfitting hoặc dao động bất thường.

---

<div align="center">
  <img src="https://github.com/quynhnguyenuet/Data-Science-Project/blob/main/Detect%20Anomaly%20in%20Deep%20Learning%20For%20Time%20Seria/DeepAnt/vaid.png" alt="Validation Loss" width="45%" />
  <p style="font-size:14px; color:gray;"><em>Biểu đồ Validation Loss</em></p>
</div>

**Nhận xét Validation Loss:**
- Biểu đồ dao động mạnh, không mượt mà như `train_loss` → thể hiện độ biến động của dữ liệu kiểm tra.
- Tuy nhiên, xu hướng tổng thể vẫn là giảm dần, từ ~`0.045` về ~`0.0433`.
- Cho thấy mô hình có cải thiện hiệu năng trên tập validation dù dữ liệu có nhiễu.

---

<div align="center">
  <img src="https://github.com/quynhnguyenuet/Data-Science-Project/blob/main/Detect%20Anomaly%20in%20Deep%20Learning%20For%20Time%20Seria/DeepAnt/images/anomalies_visualization.png" alt="Training Loss - All Features" width="70%" />
  <p style="font-size:14px; color:gray;"><em>Biểu đồ Train/Validation Loss cho tất cả các feature</em></p>
</div>
📎 [Xem PowerPoint trình bày](https://github.com/quynhnguyenuet/Data-Science-Project/blob/main/Detect%20Anomaly%20in%20Deep%20Learning%20For%20Time%20Seria/DeepAnt/DeepAnt%20For%20Detect%20Anomaly%20Time%20Serial.pptx)



### Kết quả
### 📊 Kết quả: Biểu đồ điểm bất thường

<div align="center">
  <a href="https://github.com/quynhnguyenuet/Data-Science-Project/blob/main/Detect%20Anomaly%20in%20Deep%20Learning%20For%20Time%20Seria/DeepAnt/results/anomaly_plot_interactive.html" target="_blank">
    <img src="https://github.com/quynhnguyenuet/Data-Science-Project/blob/main/Detect%20Anomaly%20in%20Deep%20Learning%20For%20Time%20Seria/DeepAnt/results/Screenshot.png" alt="Biểu đồ điểm bất thường" width="80%" />
  </a>
  <br/>
  <sub><em>Nhấn vào ảnh để xem biểu đồ tương tác (HTML)</em></sub>
</div>
