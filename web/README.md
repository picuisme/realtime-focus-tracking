# Focus Tracker Web

Bản web của Focus Tracker. Toàn bộ xử lý chạy trong trình duyệt (MediaPipe Tasks + mạng DQN viết lại bằng JS), hình ảnh camera không gửi đi đâu.

## Chạy thử

- **GitHub Pages:** Settings → Pages → Deploy from branch `main`, thư mục `/ (root)`. Trang sẽ ở `https://picuisme.github.io/realtime-focus-tracking/web/`.
- **Trên máy:** ở thư mục gốc repo chạy `python -m http.server 8000` rồi mở `http://localhost:8000/web/`.

Camera chỉ hoạt động trên HTTPS hoặc `localhost`; mở trực tiếp file `index.html` sẽ không được.

## Hai chế độ

| | Học tập · DQN | Lái xe · DMS |
|---|---|---|
| Quyết định | `dqn_focus_agent_v2.1` + luật ghi đè giống bản desktop | Luật xác định, không dùng RL |
| Tín hiệu | EAR, MAR, tư thế đầu, gaze, điện thoại | PERCLOS 60 s, nhắm mắt liên tục, rời mắt khỏi đường, ngáp, điện thoại |
| Cảnh báo | Toast sau 9 s, toàn màn hình sau 90 s | Báo động khi nhắm mắt > 1.5 s, rời mắt > 2 s, điện thoại > 1 s, PERCLOS > 30% |

## Tệp

- `core.js`: port của `extract_window_features`, `encode_context`, scaler và mạng Q. Đã đối chiếu với Python: sai lệch đặc trưng < 3e-6, Q-value trùng khớp.
- `policy.bin`: trọng số `q_net` (float32, 33→256→256→6) lấy từ `models/dqn_focus_agent_v2.1.zip`.
- `app.js`: vòng lặp camera, hiệu chỉnh, hai chế độ, xuất CSV đặc trưng cùng schema với `data/focus_dataset.csv` kèm nhãn tự báo (phím D) ở cột `gt_label`. Xem `data/DATASETS.md`.

## Khác với bản desktop

- Ngưỡng tính theo **giây**, không theo số frame, nên không phụ thuộc FPS của máy. Chính sách DQN được gọi 3 lần/giây để khớp với dữ liệu train.
- Tư thế đầu lấy từ ma trận biến đổi của MediaPipe nên không bị lỗi nhảy ±180° như `cv2.decomposeProjectionMatrix`. Góc được trừ tư thế trung tính sau 4 giây hiệu chỉnh, ngưỡng nhắm mắt = 70% EAR khi mở mắt (tốt hơn cho người đeo kính).
- Phát hiện điện thoại dùng EfficientDet-Lite0 của MediaPipe thay cho YOLOv8n.
