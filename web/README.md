# Focus Tracker Web

Bản web của Focus Tracker: chạy hoàn toàn trong trình duyệt, không gửi hình ảnh đi đâu.

- **Trên GitHub Pages:** https://picuisme.github.io/realtime-focus-tracking/web/
- **Trên máy:** ở thư mục gốc repo chạy `python -m http.server 8000`, mở `http://localhost:8000/web/`. Camera chỉ hoạt động trên HTTPS hoặc `localhost`.

## Tệp

| Tệp | Nội dung |
|---|---|
| `core.js` | Bản JS của `focus/`: đặc trưng khung hình, đặc trưng cửa sổ, model buồn ngủ, LinUCB, token bucket. `tests/test_parity.py` kiểm tra nó cho ra đúng kết quả như Python. |
| `app.js` | Vòng lặp camera, hai chế độ, nhắc nhở, tổng kết, xuất dữ liệu. Logic buổi học giống `focus/session.py`. |
| `models/drowsiness.json` | Logistic regression từ `scripts/train_drowsiness.py` (UTA-RLDD). |
| `models/nudge_prior.json` | Điểm khởi đầu của bandit, học trên người học mô phỏng; tiếp tục học trên máy người dùng. |
| `privacy.html` | Chính sách quyền riêng tư. |

## Dùng cho nghiên cứu

Cài đặt và nghiên cứu → nhập mã người tham gia, chọn điều kiện A/B/C. Trong buổi: phím **D** để tự báo mất tập trung. Cuối buổi: bảng hỏi 2 câu và 3 nút tải (tổng kết JSON, log nhắc nhở CSV, đặc trưng CSV). Xem `docs/STUDY_PROTOCOL.md`.
