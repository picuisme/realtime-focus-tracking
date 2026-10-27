# Model card

## 1. Mô hình buồn ngủ (`web/models/drowsiness.json`)

**Dùng để làm gì.** Ước lượng mức buồn ngủ từ 0 đến 100% trong cả hai chế độ. Ở chế độ Lái xe, mức trên 70% chỉ hiện cảnh báo; báo động an toàn vẫn do luật cố định quyết định.

**Đầu vào.** Tỉ lệ thời gian nhắm mắt trong 60 giây gần nhất (PERCLOS). Mắt được coi là nhắm khi độ mở dưới 70% độ mở bình thường của chính người đó, đo trong 20 giây đầu buổi.

**Mô hình.** Hồi quy logistic một biến, chuyển PERCLOS sang thang 0–1. Thứ tự xếp hạng giống hệt PERCLOS thô.

**Dữ liệu huấn luyện.** UTA-RLDD fold 1–2: 24 người, 48 video tỉnh táo hoặc buồn ngủ, 28 896 cửa sổ 60 giây. Nhãn là mức tự báo của người tham gia cho cả video. Video "hơi buồn ngủ" không dùng để huấn luyện.

**Kết quả đánh giá chéo theo người, trên tập phát triển.**

| Mô hình | AUROC theo cửa sổ | Khoảng tin cậy 95% | AUROC theo video |
| --- | --- | --- | --- |
| PERCLOS | 0,741 | 0,667–0,813 | 0,806 |
| Logistic, 5 đặc trưng mắt | 0,709 | 0,637–0,784 | 0,783 |
| LightGBM, 13 đặc trưng | 0,711 | 0,638–0,778 | 0,790 |
| Logistic, 13 đặc trưng | 0,677 | 0,612–0,745 | 0,740 |

Chi tiết: `reports/drowsiness_dev_variants.txt`.

**Đánh giá trên người chưa thấy.** Fold 3–5, gồm 36 người, được giữ kín làm tập kiểm tra một lần. Luật chọn mô hình đã được chốt trước trong `scripts/evaluate_holdout.py`. Tại thời điểm viết, Google Drive đang giới hạn lượt tải các fold này; kết quả sẽ được cập nhật vào đây.

**Giới hạn.**
- Nhãn gán cho cả video 10 phút, không theo từng thời điểm, nên điểm số theo từng cửa sổ chắc chắn có nhiễu.
- Mức chuẩn được lấy từ 20 giây đầu buổi. Người đã buồn ngủ ngay từ đầu sẽ bị đánh giá thấp hơn thực tế.
- Video UTA-RLDD được quay bằng điện thoại hoặc webcam trong nhà, ban ngày. Chưa kiểm chứng khi lái xe thật, ban đêm hay khi đeo kính râm.
- Con số 0–100% là xác suất trong điều kiện số video tỉnh táo và buồn ngủ bằng nhau, không phải xác suất tuyệt đối.

## 2. Bộ quyết định nhắc nhở (`web/models/nudge_prior.json`)

**Dùng để làm gì.** Ở điều kiện C của chế độ Học tập, chọn một trong năm hành động mỗi khi người dùng mất tập trung từ 5 giây trở lên: chờ, nhắc nhẹ, âm thanh, gợi ý nghỉ, làm tối màn hình.

**Thuật toán.** LinUCB với 12 đặc trưng ngữ cảnh: thời gian đã mất tập trung, mức buồn ngủ, thời gian từ lần nhắc trước, số lần nhắc trong 10 phút, độ dài buổi, giờ trong ngày, lý do mất tập trung. 10% số lần chọn ngẫu nhiên để tiếp tục học và để đánh giá ngoại tuyến được.

**Phần thưởng.** 1 nếu người dùng quay lại trong 10 giây, trừ chi phí làm phiền của hành động (0 đến 0,3), trừ thêm 0,5 nếu người dùng bấm "Không hữu ích".

**Điểm khởi đầu.** Học trên **người học mô phỏng** (`focus/simulator.py`), không phải người thật. Trên mô phỏng, bộ quyết định đạt phần thưởng trung bình 0,40 mỗi quyết định, so với 0,31 của luật phiên bản 2.1, với số lần nhắc gần bằng nhau. Con số này chỉ cho thấy thuật toán hoạt động đúng như thiết kế, không chứng minh hiệu quả với người thật.

**Giới hạn và kế hoạch.** Điểm khởi đầu cần được thay bằng mô hình học từ nhật ký thật (`scripts/fit_policy_from_logs.py`) sau giai đoạn tự thu dữ liệu. Hiệu quả thật được đo bằng nghiên cứu trong `docs/STUDY_PROTOCOL.md`. Bản cá nhân hoá theo từng người hiện để tắt, vì trên mô phỏng nó chưa vượt bản chung khi mỗi người chỉ có vài chục quyết định.
