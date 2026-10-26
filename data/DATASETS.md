# Nguồn dữ liệu

`focus_dataset.csv` hiện có 5 session, 14.5k dòng, trong đó 81% đến từ một session. Nhãn `is_distracted_label` được sinh từ chính luật mà reward dựa vào, nên không dùng để đánh giá được. Tài liệu này liệt kê nơi lấy thêm dữ liệu, ưu tiên dữ liệu có **nhãn độc lập** (`gt_label`).

## 1. Tự thu thập (ưu tiên số 1 cho bài toán học tập)

Mở bản web, nhập *Mã người tham gia* trong Cài đặt, rồi làm việc thật 20–30 phút. Mỗi khi thật sự mất tập trung, bấm **D** (bấm lại khi quay về). Cuối buổi bấm *Tải CSV đặc trưng*.

- File chỉ chứa đặc trưng số, không có hình ảnh, cùng schema với `focus_dataset.csv`, thêm `subject_id`, `source`, `gt_label`.
- Mục tiêu tối thiểu: 15–20 người, mỗi người 2 buổi ở hai thời điểm khác nhau trong ngày, có người đeo kính và người không.
- Xin đồng ý bằng văn bản: ghi rõ chỉ lưu đặc trưng, mục đích là đồ án, ai được xem.
- Nhãn tự báo có trễ. Khi đánh giá, nới cửa sổ ±2 s quanh mỗi lần bấm.

## 2. Bộ dữ liệu công khai

Link dẫn tới bài báo hoặc trang giới thiệu; trang tải và form đăng ký nằm trong đó. Tất cả đều yêu cầu đăng ký hoặc điền form và **chỉ cho phép dùng cho nghiên cứu**. Đọc kỹ điều khoản của từng bộ trước khi tải, và không đưa video gốc lên repo.

| Bộ | Bài toán | Nhãn | Adapter |
|---|---|---|---|
| [UTA-RLDD](https://arxiv.org/abs/1904.07312) | Buồn ngủ, webcam/điện thoại, 60 người | Theo video: tỉnh / hơi buồn ngủ / buồn ngủ | `uta-rldd` |
| [NTHU-DDD](https://conferences.visionbib.com/2016/accv-ddd-11-16-call.html) | Buồn ngủ khi lái (mô phỏng, có IR ban đêm, kính) | Theo từng frame: buồn ngủ, nhắm mắt, ngáp, gật đầu | `nthu-ddd` |
| [DMD](https://dmd.vicomtech.org/) | Lái xe: phân tâm, buồn ngủ, cử chỉ; RGB + depth + IR | Theo khoảng thời gian (OpenLABEL) | cần viết thêm |
| [DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html) | Mức độ tham gia khi học online, 112 người | Theo clip 10 s: Engagement 0–3 | `daisee` |
| [EngageNet](https://arxiv.org/abs/2302.00431) | Tham gia khi học online, quy mô lớn | Theo clip, 4 mức | cần viết thêm |
| YawDD | Ngáp khi lái | Theo video | `folder` |

Gợi ý dùng:
- **Học tập:** dữ liệu tự thu + DAiSEE (Engagement 0–1 coi là mất tập trung).
- **Lái xe:** UTA-RLDD để chỉnh ngưỡng PERCLOS, NTHU-DDD để chỉnh ngưỡng nhắm mắt và ngáp theo từng frame, DMD cho phân tâm (điện thoại, nhìn đi chỗ khác).

## 3. Trích đặc trưng

```bash
python scripts/extract_features.py --dataset uta-rldd --root D:/UTA-RLDD --out data/uta_rldd.csv
python scripts/extract_features.py --dataset nthu-ddd --root D:/NTHU-DDD --out data/nthu.csv --phone
python scripts/extract_features.py --dataset daisee  --root D:/DAiSEE  --out data/daisee.csv --max-seconds 10
```

Script lấy mẫu về 3 fps như dữ liệu train, dùng đúng các hàm đặc trưng của `demo_webcam.py`, gấp góc đầu về [-90°, 90°] để bỏ lỗi nhảy ±180°, và ghi thêm `subject_id`, `source`, `gt_label`. Bộ dữ liệu có cấu trúc thư mục khác thì dùng `--dataset folder` rồi tự ghép nhãn.

## 4. Chia dữ liệu

Chia theo **người** (leave-subject-out), không chia theo dòng hay theo session. Các frame liền nhau gần như giống hệt nhau, chia ngẫu nhiên sẽ cho kết quả đẹp giả. Báo cáo chỉ số trên `gt_label`, không trên `is_distracted_label`.
