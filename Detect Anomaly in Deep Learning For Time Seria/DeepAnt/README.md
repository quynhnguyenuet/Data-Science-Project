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


