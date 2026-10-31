# Giao thức nghiên cứu người dùng

**Câu hỏi:** so với không nhắc (A) và nhắc theo luật v2.1 (B), nhắc bằng bandit (C) có giúp người học quay lại tập trung nhanh hơn mà không bị làm phiền hơn không?

## Giả thuyết (chốt trước khi thu dữ liệu)

- H1: Thời gian quay lại trung bình ở C ngắn hơn B.
- H2: Tỉ lệ nhắc có ích ở C cao hơn B, với số lần nhắc mỗi giờ không cao hơn.
- H3: Điểm "bị làm phiền" (1–7) ở C không cao hơn B.
- H4: Số phút tập trung ở B và C cao hơn A.

## Thiết kế

- Within-subject, 3 điều kiện, thứ tự theo hình vuông Latin: ABC, BCA, CAB, xoay vòng theo mã người tham gia.
- 18 người (bội của 3), ít nhất 6 người đeo kính. Mỗi người 3 buổi × 25 phút, cách nhau ít nhất 1 ngày, cùng khung giờ.
- Điều kiện C: model nhắc khởi đầu từ `web/models/nudge_prior.json` và tiếp tục học trong 3 buổi của chính người đó.

## Chuẩn bị mỗi buổi

1. Mở https://picuisme.github.io/realtime-focus-tracking/web/ trên laptop của người tham gia.
2. Cài đặt và nghiên cứu → nhập **Mã người tham gia** (p01…p18), chọn **Điều kiện** theo bảng thứ tự, Mức nhắc = Vừa.
3. Bật camera, ngồi đúng tư thế làm việc trong 4 giây hiệu chỉnh.
4. Nhiệm vụ: đọc một bài (khoảng 2 500 từ) rồi làm 10 câu trắc nghiệm. Điện thoại để trên bàn như thường ngày.
5. Hướng dẫn người tham gia bấm **D** khi tự nhận ra mình mất tập trung, bấm lại khi quay về.
6. Hết 25 phút: bấm **Dừng và xem tổng kết**, trả lời 2 câu hỏi, bấm **Lưu câu trả lời**, rồi tải cả 3 file.

## Đo lường

| Biến | Nguồn |
| --- | --- |
| Phút tập trung | `summary.focused_minutes` |
| Thời gian quay lại trung bình | `summary.mean_return_s` |
| Số lần nhắc mỗi giờ, tỉ lệ nhắc có ích, số lần "Không hữu ích" | `summary`, `decisions_*.csv` |
| Tự đánh giá mức tập trung, mức bị làm phiền (1–7) | `summary.self_rated_*` |
| Điểm trắc nghiệm | Nhập tay vào `quiz.csv` (participant, condition, score) |
| Nhãn tự báo | Cột `gt_label` trong `features_*.csv` |
| Nhóm tuổi, ngành học | Lấy từ form đăng ký, người thực hiện nhập vào `participants.csv` (participant, age_group, field); không nhập vào app |

## Phân tích

```
python scripts/analyze_study.py --dir study_data/ --out reports/study.json
```

- So sánh cặp trong từng người (C − B, B − A, C − A) cho mỗi biến, khoảng tin cậy 95% bằng bootstrap theo người.
- Nếu có `statsmodels`: mô hình hỗn hợp `biến ~ điều kiện + thứ tự buổi + (1 | người)`.
- Báo cáo mọi biến đã chốt ở trên, kể cả khi không có ý nghĩa thống kê.
- Phân tích khám phá: chênh lệch C − B theo nhóm tuổi và ngành học, khi có `participants.csv`. Với 18 người, kết quả này chỉ để gợi ý hướng nghiên cứu, không dùng để kết luận.
- Off-policy evaluation trên log của điều kiện C: ước lượng giá trị của luật v2.1 và của chính sách "không nhắc" bằng IPS, SNIPS và DR (`focus/bandit.py`).

## Đạo đức và dữ liệu

- Giấy đồng ý: `docs/CONSENT_FORM.md`. Hỏi quy trình duyệt đạo đức của trường trước khi bắt đầu.
- Không quay video, không chụp ảnh. Chỉ lưu các file số liệu do người tham gia tự tải và gửi.
- Người tham gia có thể dừng bất cứ lúc nào và yêu cầu xoá dữ liệu theo mã của mình.
- Dữ liệu lưu trong thư mục có giới hạn quyền truy cập, không đưa lên repo công khai.
