# Focus Tracker

Phát hiện lúc người dùng mất tập trung hoặc buồn ngủ qua webcam và nhắc họ quay lại. Toàn bộ xử lý chạy trên thiết bị; không ảnh nào rời máy.

**Dùng thử:** https://picuisme.github.io/realtime-focus-tracking/web/ (Chrome, Edge, Safari; cần cho phép camera)

## Hai chế độ

| | Học tập | Lái xe |
|---|---|---|
| Phát hiện | Luật hiệu chỉnh theo từng người (quay đầu, cúi đầu, nhắm mắt, điện thoại, vắng mặt) + điểm buồn ngủ từ model | PERCLOS 60 s, nhắm mắt liên tục, rời mắt khỏi đường, ngáp, điện thoại, điểm buồn ngủ |
| Quyết định | Contextual bandit (LinUCB) chọn chờ / nhắc nhẹ / âm thanh / gợi ý nghỉ / làm tối màn hình, giới hạn ≤6 lần/giờ | Luật cố định, không qua model học |
| Ngoại lệ | Chế độ họp, chế độ ghi chép | — |

## Cấu trúc

```
focus/            lõi Python dùng chung
  features.py     đặc trưng từng khung hình (EAR, MAR, gaze, tư thế đầu)
  temporal.py     đặc trưng cửa sổ 20 s / 60 s, không dùng nhãn làm đầu vào
  rules.py        luật học tập (hiệu chỉnh theo người) và luật cảnh báo lái xe
  bandit.py       LinUCB, phần thưởng, IPS / SNIPS / doubly robust
  simulator.py    người học mô phỏng để kiểm thử bandit, token bucket
  session.py      logic một buổi học (bản sao của web/app.js)
  metrics.py      recall theo sự kiện, báo động sai/giờ, độ trễ, bootstrap theo người
web/              bản web (GitHub Pages); core.js là bản JS của focus/
gui_app.py        app desktop (Tkinter); --headless --video để chạy lại một file
scripts/
  extract_features.py   video → CSV đặc trưng (UTA-RLDD, NTHU-DDD, DAiSEE, thư mục bất kỳ)
  train_drowsiness.py   train + đánh giá model buồn ngủ, xuất web/models/drowsiness.json
  simulate_bandit.py    so sánh chính sách nhắc trên người học mô phỏng, kiểm tra OPE
  analyze_study.py      phân tích nghiên cứu within-subject
tests/            pytest, gồm kiểm tra Python ↔ JS cho ra cùng kết quả
docs/             giao thức nghiên cứu, phiếu đồng ý, data card, model card
legacy/           bản v2.1 (DQN) và lý do thay thế
```

## Chạy

```bash
pip install -r requirements.txt

# desktop
python gui_app.py                          # điều kiện C (bandit)
python gui_app.py --condition B            # A: không nhắc, B: luật v2.1, C: bandit
python gui_app.py --headless --video clip.mp4 --no-phone

# web trên máy
python -m http.server 8000                 # rồi mở http://localhost:8000/web/

# dữ liệu và model
python scripts/extract_features.py --dataset uta-rldd --root D:/UTA-RLDD --out data/uta_rldd_features.csv
python scripts/train_drowsiness.py --csv data/uta_rldd_features.csv
python scripts/simulate_bandit.py --export-prior web/models/nudge_prior.json

# test
pip install -r requirements-dev.txt && python -m pytest -q
```

## Tài liệu

- [Data card](docs/DATA_CARD.md) · [Model card](docs/MODEL_CARD.md) · [Nguồn dữ liệu](data/DATASETS.md)
- [Giao thức nghiên cứu](docs/STUDY_PROTOCOL.md) · [Phiếu đồng ý](docs/CONSENT_FORM.md) · [Quyền riêng tư](web/privacy.html)
- [Vì sao bỏ DQN của v2.1](legacy/README.md)

Bản demo, không phải thiết bị an toàn đã kiểm định. License: MIT.
