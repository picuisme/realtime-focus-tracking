# Data card

Repo này chỉ chứa **đặc trưng số** trích từ video, không chứa video hay ảnh. Mọi bộ công khai chỉ dùng cho nghiên cứu, theo điều khoản của từng bộ.

## Định dạng chung

Mỗi dòng là một khung hình lấy mẫu 3 lần/giây (`scripts/extract_features.py`).

| Cột | Ý nghĩa |
| --- | --- |
| `session_id` | Mã ngẫu nhiên cho một video hoặc một buổi |
| `timestamp` | Giây tính từ đầu video |
| `head_pitch`, `head_yaw`, `head_roll` | Góc đầu theo cách của v2.1 (solvePnP), đã gấp về [-90°, 90°] |
| `mp_pitch`, `mp_yaw`, `mp_roll` | Góc đầu từ ma trận biến đổi của MediaPipe; dùng cho model v3 và bản web |
| `ear_score`, `mar_score`, `brow_dist`, `gaze_ratio` | Độ mở mắt, độ mở miệng, khoảng cách lông mày–mí (px trên khung 640×480), vị trí mống mắt (0.5 = giữa) |
| `person_detected`, `phone_count` | Có mặt người; số điện thoại (chỉ khi chạy với `--phone`) |
| `is_distracted_label`, `consecutive_frames`, `fatigue_score` | Do **luật** sinh ra, giữ để tương thích v2.1. Không dùng làm nhãn đánh giá. |
| `subject_id`, `source`, `gt_label` | Người, bộ dữ liệu, **nhãn của bộ dữ liệu** (độc lập với luật) |

`ear_score = 0` nghĩa là không tìm thấy khuôn mặt ở khung đó.

## Các bộ

### `focus_dataset.csv` (v2.1)
- 5 session, 14 506 dòng. 81% đến từ một session không có timestamp.
- Nhãn sinh từ luật; 31% số dòng có EAR = 0 trong khi `person_detected = 1`, tức là mất khuôn mặt nhưng vẫn bị coi là mất tập trung. 27% số dòng có |roll| > 150° do lỗi góc.
- Chỉ dùng để so sánh với v2.1.

### UTA-RLDD (`uta_rldd_features.csv`)
- 60 người, 180 video (mỗi người 3 video, khoảng 10 phút), tự quay bằng điện thoại hoặc webcam của người tham gia. Nguồn: [Ghoddoosian và cộng sự, 2019](https://arxiv.org/abs/1904.07312).
- `gt_label`: 0 = tỉnh táo, 1 = hơi buồn ngủ, 2 = buồn ngủ (người tham gia tự báo theo thang KSS, gán cho cả video).
- `subject_id` = `FoldK/NN` theo 5 fold của bộ dữ liệu.
- Giới hạn: nhãn theo cả video chứ không theo từng thời điểm; điều kiện quay rất khác nhau; ít video ban đêm.

### DAiSEE (`daisee_features.csv`)
- 112 người, khoảng 9 000 clip 10 giây, webcam laptop khi học online. Nguồn: [trang chủ DAiSEE](https://people.iith.ac.in/vineethnb/resources/daisee/index.html).
- `gt_label` = mức Engagement 0–3 do người gán nhãn đánh giá theo clip; có thêm cột `clip_id`.
- Giới hạn: phân bố lệch mạnh về mức tham gia cao; clip 10 s quá ngắn cho các đặc trưng 60 s.

### DMD (chưa có)
- Đã gửi yêu cầu truy cập phần Distraction và Drowsiness. Khi có, dùng để đánh giá theo sự kiện cho chế độ Lái xe.

### Dữ liệu tự thu (chưa có)
- Thu bằng bản web theo `docs/STUDY_PROTOCOL.md`, có phiếu đồng ý. `gt_label` = nhãn tự báo bằng phím D.

## Quy tắc dùng

- Chia train/test theo người, không theo dòng.
- Không đưa video gốc vào repo (`.gitignore` chặn `.mp4`, `.avi`, `.mov`).
- Trích dẫn bài báo gốc của từng bộ khi báo cáo kết quả.
