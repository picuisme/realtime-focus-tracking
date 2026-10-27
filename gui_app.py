"""Focus Tracker desktop (v3).

    python gui_app.py                    # webcam + window
    python gui_app.py --condition B      # study condition: A none, B rule v2.1, C bandit (default)
    python gui_app.py --headless --video clip.mp4   # replay a file without a window, print the summary

Same logic as the web app (focus/session.py mirrors web/app.js). Nothing leaves
the computer: sessions and the nudge model are saved under ~/FocusTracker/.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

import cv2

ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from focus.bandit import LinUCB  # noqa: E402
from focus.features import frame_features, matrix_to_colmajor  # noqa: E402
from focus.session import StudySession, load_json  # noqa: E402

HOME = Path.home() / "FocusTracker"
FACE_MODEL = ROOT / "models" / "face_landmarker.task"
DROWSY_MODEL = ROOT / "web" / "models" / "drowsiness.json"
NUDGE_PRIOR = ROOT / "web" / "models" / "nudge_prior.json"
REASON_TEXT = {"absent": "Không thấy người", "no_face": "Không thấy khuôn mặt", "eyes_closed": "Mắt nhắm",
               "head_turned": "Quay đầu sang bên", "head_down": "Cúi đầu quá lâu", "looking_up": "Nhìn lên",
               "phone": "Có điện thoại"}
NUDGE_TEXT = {"toast": "Bạn đang rời khỏi việc đang làm. Quay lại nhé.", "sound": "Quay lại tập trung nào.",
              "break": "Có vẻ bạn đang mệt. Nghỉ 5 phút rồi quay lại sẽ hiệu quả hơn.",
              "dim": "Màn hình tối đi một chút để nhắc bạn quay lại."}


# ── vision ───────────────────────────────────────────────────────────────
class Vision:
    def __init__(self, phone: bool = True):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(FACE_MODEL)), num_faces=1,
            output_facial_transformation_matrixes=True))
        self.yolo = None
        if phone:
            try:
                from ultralytics import YOLO
                self.yolo = YOLO(str(ROOT / "models" / "yolov8n.pt"))
            except Exception as e:  # ultralytics is optional
                print(f"[phone detection off] {e}")
        self.last_phone_t, self.phone, self.person = 0.0, 0, 0

    def __call__(self, frame_bgr) -> tuple[dict, int]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.face.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb))
        now = time.time()
        if self.yolo is not None and now - self.last_phone_t > 0.4:
            self.last_phone_t = now
            r = self.yolo(frame_bgr, verbose=False)[0]
            self.phone = sum(1 for b in r.boxes if int(b.cls) == 67 and float(b.conf) > 0.45)
            self.person = int(any(int(b.cls) == 0 and float(b.conf) > 0.45 for b in r.boxes))
        f = {"ear_score": 0.0, "mar_score": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "gaze_ratio": 0.5, "person": self.person}
        if res.face_landmarks:
            mat = matrix_to_colmajor(res.facial_transformation_matrixes[0]) if res.facial_transformation_matrixes else None
            f.update(frame_features(res.face_landmarks[0], w, h, mat), person=1)
        return f, self.phone


# ── persistence ──────────────────────────────────────────────────────────
def load_bandit(participant: str) -> LinUCB:
    saved = HOME / f"bandit_{participant or 'anon'}.json"
    seed = int.from_bytes(os.urandom(4), "little")     # fresh exploration each session (OPE needs it)
    for p in (saved, NUDGE_PRIOR):
        if p.exists():
            return LinUCB.from_json(p.read_text(), seed=seed)
    return LinUCB(seed=seed)


def save_session(sess: StudySession, participant: str, extra: dict | None = None) -> Path:
    HOME.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    (HOME / f"bandit_{participant or 'anon'}.json").write_text(sess.bandit.to_json())
    summ = {**sess.summary(), "participant": participant or "anon", "saved_at": stamp, **(extra or {})}
    out = HOME / f"session_{stamp}.json"
    out.write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    with open(HOME / f"decisions_{stamp}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["t", "action", "p", "why", "condition", "refocused", "not_helpful", "reward", "distracted_s", "reasons", "x"])
        for d in sess.decisions:
            wr.writerow([round(d.t, 2), d.action, round(d.p, 4), d.why, d.condition, int(d.refocused), int(d.not_helpful),
                         round(d.reward, 3), round(d.distracted_s, 1), " ".join(d.reasons), " ".join(f"{v:.4f}" for v in d.x)])
    return out


def make_session(args) -> StudySession:
    return StudySession(condition=args.condition, level=args.level, bandit=load_bandit(args.participant),
                        drowsy_model=load_json(DROWSY_MODEL))


# ── headless replay ──────────────────────────────────────────────────────
def headless(args) -> None:
    sess, vis = make_session(args), Vision(phone=not args.no_phone)
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps / 10)))          # analyse ~10 frames per second
    i = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        i += 1
        if i % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        f, phone = vis(frame)
        for d in sess.step(step / fps, f, phone=phone, hour=20):
            print(f"t={sess.t:6.1f}s  {d.action:5s} p={d.p:.2f} why={d.why} reasons={d.reasons}")
    cap.release()
    vis.face.close()
    print(json.dumps(sess.summary(), indent=2))
    if args.save:
        print("saved", save_session(sess, args.participant))


# ── GUI ──────────────────────────────────────────────────────────────────
def gui(args) -> None:
    import tkinter as tk
    from tkinter import ttk

    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("Focus Tracker")
    root.geometry("1120x680")
    root.configure(bg="#151d1c")
    style = ttk.Style(root)
    style.theme_use("clam")

    state = {"frame": None, "f": None, "phone": 0, "running": True, "err": None, "closing": False}
    sess = make_session(args)
    lock = threading.Lock()
    events: queue.Queue = queue.Queue()    # worker -> Tk thread; Tk is only touched from the main thread

    def worker():
        cap = vis = None
        try:
            vis = Vision(phone=not args.no_phone)
            cap = cv2.VideoCapture(args.camera)
            last = time.time()
            while state["running"]:
                ok, frame = cap.read()
                if not ok:
                    state["err"] = "Không mở được camera (có thể ứng dụng khác đang dùng)."
                    time.sleep(1)
                    continue
                frame = cv2.flip(frame, 1)
                f, phone = vis(frame)
                now = time.time()
                with lock:
                    made = sess.step(min(0.5, now - last), f, phone=phone)
                last = now
                for d in made:
                    events.put(d)
                state.update(frame=frame, f=f, phone=phone, err=None)
                time.sleep(max(0.0, 1 / 15 - (time.time() - now)))
        except Exception as e:  # show the error in the window instead of dying silently
            state["err"] = f"{type(e).__name__}: {e}"
        finally:
            if cap is not None:
                cap.release()
            if vis is not None:
                vis.face.close()

    # layout
    left = tk.Frame(root, bg="#1c2625", padx=18, pady=18)
    left.pack(side="left", fill="y")
    right = tk.Frame(root, bg="#151d1c")
    right.pack(side="right", fill="both", expand=True)
    big = lambda parent, size, color="#e3ecea", weight="bold": tk.Label(parent, font=("Segoe UI", size, weight), bg="#1c2625", fg=color, anchor="w", justify="left")
    tk.Label(left, text="FOCUS TRACKER", font=("Segoe UI", 18, "bold"), bg="#1c2625", fg="#e3ecea").pack(anchor="w")
    status = big(left, 26)
    status.pack(anchor="w", pady=(16, 0))
    reason = big(left, 11, "#8fa19d", "normal")
    reason.pack(anchor="w")
    timer = big(left, 11, "#8fa19d", "normal")
    timer.pack(anchor="w", pady=(0, 12))
    drowsy = big(left, 11, "#e3ecea", "normal")
    drowsy.pack(anchor="w")
    stats = big(left, 11, "#e3ecea", "normal")
    stats.pack(anchor="w", pady=(0, 12))
    tk.Label(left, text=f"Điều kiện: {args.condition} · Mức nhắc: {args.level}", bg="#1c2625", fg="#8fa19d").pack(anchor="w")

    def toggle(attr, btn, label):
        with lock:
            setattr(sess, attr, not getattr(sess, attr))
            on = getattr(sess, attr)
        btn.config(text=f"Tắt chế độ {label}" if on else f"Chế độ {label}", bg="#d12f2f" if on else "#2a3634")

    for attr, label in (("meeting", "họp"), ("notes", "ghi chép")):
        b = tk.Button(left, text=f"Chế độ {label}", bg="#2a3634", fg="white", relief="flat", padx=10, pady=6)
        b.config(command=lambda a=attr, bt=b, lb=label: toggle(a, bt, lb))
        b.pack(anchor="w", fill="x", pady=3)
    def recalibrate():
        with lock:
            sess.recalibrate()
    tk.Button(left, text="Hiệu chỉnh lại", bg="#2a3634", fg="white", relief="flat", padx=10, pady=6,
              command=recalibrate).pack(anchor="w", fill="x", pady=3)
    video = tk.Label(right, bg="#151d1c")
    video.pack(expand=True)

    dim_win = {"w": None}

    def perform(d):
        if d.action == "wait":
            return
        if d.action == "sound":
            try:
                import winsound
                winsound.Beep(660, 250)
            except Exception:
                root.bell()
        if d.action == "dim" and dim_win["w"] is None:
            w = tk.Toplevel(root)
            w.attributes("-fullscreen", True)
            w.attributes("-alpha", 0.45)
            w.attributes("-topmost", True)
            w.configure(bg="black")
            dim_win["w"] = w
            root.after(5000, lambda: (w.destroy(), dim_win.update(w=None)))
        toast(NUDGE_TEXT[d.action], d)

    def toast(text, d):
        t = tk.Toplevel(root)
        t.overrideredirect(True)
        t.attributes("-topmost", True)
        t.configure(bg="#b36f05")
        x, y = root.winfo_screenwidth() - 440, root.winfo_screenheight() - 170
        t.geometry(f"420x110+{x}+{y}")
        tk.Label(t, text=text, wraplength=390, bg="#b36f05", fg="black", font=("Segoe UI", 11, "bold")).pack(expand=True, padx=12, pady=(10, 0))

        def not_helpful():
            with lock:
                d.not_helpful = True
            t.destroy()
        tk.Button(t, text="Không hữu ích", command=not_helpful, relief="flat").pack(pady=(0, 8))
        root.after(20000, lambda: t.winfo_exists() and t.destroy())

    def refresh():
        if not state["running"]:
            return
        while not events.empty():
            perform(events.get_nowait())
        with lock:
            s = sess.distracted_s
            reasons = list(sess.reasons)
            summ = sess.summary()
            calibrating = not sess.calib["done"]
            dz = sess.drowsy
        if state["err"]:
            status.config(text="LỖI", fg="#ff5a5a")
            reason.config(text=state["err"])
        elif calibrating:
            status.config(text="Đang hiệu chỉnh", fg="#3ccf7a")
            reason.config(text="Nhìn vào màn hình như lúc đang làm việc")
        else:
            txt, col = ("Tập trung", "#3ccf7a") if s < 3 else ("Cảnh báo sớm", "#e7a73a") if s < 9 else ("Mất tập trung", "#ff5a5a")
            if sess.meeting:
                txt, col = "Đang họp", "#3ccf7a"
            status.config(text=txt, fg=col)
            reason.config(text=" · ".join(REASON_TEXT.get(r, r) for r in reasons) or "Đang nhìn vào màn hình")
        timer.config(text=f"Mất tập trung: {s:4.1f} s")
        drowsy.config(text="Buồn ngủ: đang đo…" if dz is None else f"Buồn ngủ: {dz * 100:.0f}%")
        useful = summ["useful_nudge_rate"]
        stats.config(text=f"Tập trung {summ['focused_minutes']:.1f} / {summ['minutes']:.1f} phút\n"
                          f"Đã nhắc {summ['nudges']} lần · có ích {'–' if useful is None else f'{useful * 100:.0f}%'}")
        if state["frame"] is not None:
            img = Image.fromarray(cv2.cvtColor(state["frame"], cv2.COLOR_BGR2RGB))
            img.thumbnail((800, 600))
            video.img = ImageTk.PhotoImage(img)
            video.config(image=video.img)
        root.after(100, refresh)

    def on_close():
        if state["closing"]:
            return
        state["closing"] = True
        state["running"] = False
        worker_thread.join(timeout=3)
        with lock:
            path = save_session(sess, args.participant)
            summ = sess.summary()
        w = tk.Toplevel(root)
        w.title("Tổng kết buổi")
        w.configure(bg="#1c2625", padx=20, pady=16)
        lines = [f"Tập trung {summ['focused_minutes']} / {summ['minutes']} phút",
                 f"Lần mất tập trung ≥5 s: {summ['distraction_episodes']}",
                 f"Thời gian quay lại trung bình: {summ['mean_return_s'] or '–'} s",
                 f"Số lần nhắc: {summ['nudges']}", f"Đã lưu: {path}"]
        for ln in lines:
            tk.Label(w, text=ln, bg="#1c2625", fg="#e3ecea", anchor="w", font=("Segoe UI", 11)).pack(anchor="w")
        tk.Button(w, text="Đóng", command=root.destroy).pack(pady=(12, 0))
        w.protocol("WM_DELETE_WINDOW", root.destroy)

    root.protocol("WM_DELETE_WINDOW", on_close)
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()
    root.after(100, refresh)
    root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Focus Tracker desktop")
    ap.add_argument("--condition", choices="ABC", default="C")
    ap.add_argument("--level", choices=["low", "medium", "high"], default="medium")
    ap.add_argument("--participant", default="")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--no-phone", action="store_true", help="skip YOLO phone detection")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--video", default=None)
    ap.add_argument("--save", action="store_true", help="headless: save the session like the GUI does")
    args = ap.parse_args()
    headless(args) if args.headless else gui(args)


if __name__ == "__main__":
    main()
