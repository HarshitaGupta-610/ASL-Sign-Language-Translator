import os
from collections import Counter, deque
from pathlib import Path
import tkinter as tk
from tkinter import font as tkfont

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
import pickle
from PIL import Image, ImageTk, ImageDraw


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════════
BG_COLOR = "#0b0c10"
ACCENT_COLOR = "#66fcf1"
RECT_TEXT_COLOR = "#ffffff"
CARD_COLOR = "#e8f5f3"
LOGO_PATH = r"C:\Users\harsh\Downloads\Picture.png"
WIN_W, WIN_H = 800, 500

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT_DIR / "model" / "asl_model.pkl"
ENCODER_PATH = ROOT_DIR / "model" / "label_encoder.pkl"


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def make_circle_image(size, bg, fg, symbol="?", font_size=22):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([0, 0, size - 1, size - 1], fill=bg)
    try:
        from PIL import ImageFont

        fnt = ImageFont.truetype("arialbd.ttf", font_size)
        bbox = d.textbbox((0, 0), symbol, font=fnt)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.text(((size - tw) / 2, (size - th) / 2 - 2), symbol, fill=fg, font=fnt)
    except Exception:
        d.text((size // 2 - 6, size // 2 - 10), symbol, fill=fg)
    return img


def make_play_icon(size=70, bg="#ffffff", fg="#0b0c10"):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([0, 0, size - 1, size - 1], fill=bg)
    margin = size // 4
    d.polygon([(margin + 4, margin), (size - margin, size // 2), (margin + 4, size - margin)], fill=fg)
    return img


def make_stop_icon(size=64, bg="#807c7c", fg="#2d2c2c"):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([0, 0, size - 1, size - 1], fill=bg)
    sq = size // 3
    cx, cy = size // 2, size // 2
    d.rectangle([cx - sq // 2, cy - sq // 2, cx + sq // 2, cy + sq // 2], fill=fg)
    return img


def draw_rounded_rect(canvas, x0, y0, x1, y1, r=16, outline="#9fccc3", width=3, dash=(10, 6)):
    kw = dict(style="arc", outline=outline, width=width, dash=dash)
    canvas.create_arc(x0, y0, x0 + 2 * r, y0 + 2 * r, start=90, extent=90, **kw)
    canvas.create_arc(x1 - 2 * r, y0, x1, y0 + 2 * r, start=0, extent=90, **kw)
    canvas.create_arc(x1 - 2 * r, y1 - 2 * r, x1, y1, start=270, extent=90, **kw)
    canvas.create_arc(x0, y1 - 2 * r, x0 + 2 * r, y1, start=180, extent=90, **kw)
    lkw = dict(fill=outline, width=width, dash=dash)
    canvas.create_line(x0 + r, y0, x1 - r, y0, **lkw)
    canvas.create_line(x1, y0 + r, x1, y1 - r, **lkw)
    canvas.create_line(x1 - r, y1, x0 + r, y1, **lkw)
    canvas.create_line(x0, y1 - r, x0, y0 + r, **lkw)


def load_logo(path, height=48):
    if not os.path.isfile(path):
        return None
    try:
        img = Image.open(path).convert("RGBA")
        ratio = height / img.height
        img = img.resize((int(img.width * ratio), height), Image.LANCZOS)
        return ImageTk.PhotoImage(img)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
class ASLApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ASL - American Sign Language")
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.resizable(False, False)
        self.configure(bg=BG_COLOR)
        self.protocol("WM_DELETE_WINDOW", self._quit)

        self._logo_img = None
        self._circle_img = None
        self._play_photo = None
        self._stop_photo = None
        self._cam_photo = None

        self._cap = None
        self._cam_job = None
        self._prediction_buffer = deque(maxlen=10)

        self.predicted_letter = tk.StringVar(value="-")
        self.prediction_state = tk.StringVar(value="Press start to begin")

        self.model, self.encoder = self._load_model()
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
        )

        self._load_fonts()
        self._build_skeleton()
        self.show_start_page()

    def _load_fonts(self):
        avail = list(tkfont.families())
        display = "Archivo Black" if "Archivo Black" in avail else "Arial Black"
        self.font_title = tkfont.Font(family=display, size=20, weight="bold")
        self.font_logo = tkfont.Font(family=display, size=26, weight="bold")
        self.font_box = tkfont.Font(family=display, size=14, weight="bold")
        self.font_pred = tkfont.Font(family=display, size=22, weight="bold")

    def _load_model(self):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)

        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)

        return model, encoder

    def _build_skeleton(self):
        top = tk.Frame(self, bg=BG_COLOR)
        top.pack(fill="x", padx=16, pady=(14, 0))

        self._logo_img = load_logo(LOGO_PATH, height=48)
        if self._logo_img:
            tk.Label(top, image=self._logo_img, bg=BG_COLOR).pack(side="left")
        else:
            tk.Label(top, text="ABC", font=self.font_logo, fg=ACCENT_COLOR, bg=BG_COLOR).pack(side="left")

        tk.Label(
            top,
            text="ASL - American Sign Language",
            font=self.font_title,
            fg=ACCENT_COLOR,
            bg=BG_COLOR,
        ).pack(side="left", expand=True)

        raw_c = make_circle_image(44, bg="#6b6b6b", fg="#ffffff")
        self._circle_img = ImageTk.PhotoImage(raw_c)
        help_lbl = tk.Label(top, image=self._circle_img, bg=BG_COLOR, cursor="hand2")
        help_lbl.pack(side="right")
        help_lbl.bind("<Button-1>", self._on_help)

        self.content = tk.Frame(self, bg=BG_COLOR)
        self.content.pack(expand=True, fill="both", padx=24, pady=(8, 18))

    def _clear_content(self):
        self._stop_camera()
        for widget in self.content.winfo_children():
            widget.destroy()

    def show_start_page(self):
        self._clear_content()

        canvas = tk.Canvas(self.content, bg=BG_COLOR, highlightthickness=0)
        canvas.pack(expand=True, fill="both")

        def draw(event=None):
            canvas.delete("all")
            cw, ch = canvas.winfo_width(), canvas.winfo_height()
            if cw < 10 or ch < 10:
                return

            draw_rounded_rect(canvas, 10, 10, cw - 10, ch - 10)

            mid_x = cw / 2
            canvas.create_text(
                mid_x,
                ch * 0.38,
                text="Start ASL Sign Predictor",
                font=self.font_box,
                fill=RECT_TEXT_COLOR,
            )

            raw_play = make_play_icon(size=72)
            self._play_photo = ImageTk.PhotoImage(raw_play)
            canvas.create_image(mid_x, ch * 0.65, image=self._play_photo, anchor="center", tags="play")
            canvas.tag_bind("play", "<Button-1>", lambda e: self.show_camera_page())
            canvas.tag_bind("play", "<Enter>", lambda e: canvas.config(cursor="hand2"))
            canvas.tag_bind("play", "<Leave>", lambda e: canvas.config(cursor=""))

        canvas.bind("<Configure>", draw)

    def show_camera_page(self):
        self._clear_content()

        outer_canvas = tk.Canvas(self.content, bg=BG_COLOR, highlightthickness=0)
        outer_canvas.pack(expand=True, fill="both")

        card = tk.Frame(self.content, bg=CARD_COLOR, bd=0)
        card.configure(highlightthickness=0)

        lbl_top = tk.Label(card, text="Show sign with hand", font=self.font_box, fg="#1a1a1a", bg=CARD_COLOR)
        lbl_top.pack(pady=(4, 0))

        lbl_pred = tk.Label(
            card,
            textvariable=self.predicted_letter,
            font=self.font_pred,
            fg="#0b0c10",
            bg=CARD_COLOR,
        )
        self._pred_label = lbl_pred
        self._pred_label_visible = False

        lbl_state = tk.Label(card, textvariable=self.prediction_state, font=("Arial", 10), fg="#444444", bg=CARD_COLOR)
        lbl_state.pack(pady=(0, 0))

        self._cam_label = tk.Label(card, bg=CARD_COLOR)
        self._cam_label.pack(expand=True, fill="both", padx=6, pady=6)

        raw_stop = make_stop_icon(size=58, bg="#111111", fg="#ffffff")
        self._stop_photo = ImageTk.PhotoImage(raw_stop)

        self._stop_btn = tk.Label(
            card,
            image=self._stop_photo,
            bg=CARD_COLOR,
            cursor="hand2",
        )
        self._stop_btn.place(relx=0.5, rely=0.86, anchor="center")
        self._stop_btn.bind("<Button-1>", lambda e: self.show_start_page())

        def redraw(event=None):
            outer_canvas.delete("all")
            cw = outer_canvas.winfo_width()
            ch = outer_canvas.winfo_height()
            if cw < 10 or ch < 10:
                return

            pad, r = 10, 16
            draw_rounded_rect(outer_canvas, pad, pad, cw - pad, ch - pad, r=r)

            inner_pad = pad + 20
            card.place(
                x=inner_pad,
                y=inner_pad,
                width=cw - 2 * inner_pad,
                height=ch - 2 * inner_pad,
            )

            self._stop_btn.lift()

        outer_canvas.bind("<Configure>", redraw)
        self._start_camera()

    def _start_camera(self):
        self.predicted_letter.set("")
        self.prediction_state.set("Camera starting...")
        if hasattr(self, "_pred_label"):
            self._pred_label.pack_forget()
            self._pred_label_visible = False
        if hasattr(self, "_stop_btn"):
            self._stop_btn.lift()
        self._prediction_buffer.clear()

        try:
            self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                raise RuntimeError("Could not open webcam")
            self.prediction_state.set("Show your hand sign in the frame")
            self._poll_camera()
        except Exception as exc:
            self.prediction_state.set(f"Camera error: {exc}")
            if hasattr(self, "_cam_label") and self._cam_label.winfo_exists():
                self._cam_label.config(text="Camera unavailable", fg="red", font=("Arial", 12))

    def _predict_from_hand(self, hand):
        landmarks = []
        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        if len(landmarks) != 63:
            return None, None

        X = np.array(landmarks, dtype=np.float32).reshape(1, -1)
        prediction = self.model.predict(X)
        label = self.encoder.inverse_transform(prediction)[0]

        if hasattr(self.model, "predict_proba"):
            confidence = float(np.max(self.model.predict_proba(X)))
        else:
            confidence = None

        self._prediction_buffer.append(label)
        smoothed_label = Counter(self._prediction_buffer).most_common(1)[0][0]
        return smoothed_label, confidence

    def _draw_overlay(self, frame, label, confidence=None):
        text = label if label else ""

        if text:
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (180, 90), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            cv2.putText(frame, text, (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 252, 241), 2, cv2.LINE_AA)

    def _poll_camera(self):
        if self._cap is None or not self._cap.isOpened():
            return

        ret, frame = self._cap.read()
        if not ret:
            self.prediction_state.set("Unable to read from camera")
            self._cam_job = self.after(30, self._poll_camera)
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        label = ""
        confidence = None
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            hand = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
            predicted = self._predict_from_hand(hand)
            if predicted[0] is not None:
                label, confidence = predicted

        self.predicted_letter.set(label)
        if not hand_detected:
            self.prediction_state.set("Show your hand sign in the frame")
            if hasattr(self, "_pred_label"):
                self._pred_label.pack_forget()
                self._pred_label_visible = False
        else:
            self.prediction_state.set("Prediction updating live")
            if hasattr(self, "_pred_label") and not self._pred_label_visible:
                self._pred_label.pack(pady=(0, 0))
                self._pred_label_visible = True

        self._draw_overlay(frame, label, confidence)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lw = self._cam_label.winfo_width() if hasattr(self, "_cam_label") else 0
        lh = self._cam_label.winfo_height() if hasattr(self, "_cam_label") else 0
        if lw > 10 and lh > 10:
            frame = cv2.resize(frame, (lw, lh))

        self._cam_photo = ImageTk.PhotoImage(Image.fromarray(frame))
        if hasattr(self, "_cam_label") and self._cam_label.winfo_exists():
            self._cam_label.config(image=self._cam_photo, text="")

        self._cam_job = self.after(30, self._poll_camera)

    def _stop_camera(self):
        if self._cam_job:
            self.after_cancel(self._cam_job)
            self._cam_job = None
        if self._cap:
            self._cap.release()
            self._cap = None

    def _on_help(self, event=None):
        w = tk.Toplevel(self, bg=BG_COLOR)
        w.title("Help")
        w.geometry("340x170")
        tk.Label(
            w,
            text="ASL Sign Predictor\n\nPage 1 → Press ▶ to open camera\nPage 2 → Show your hand sign\nPress ⏹ to go back",
            font=("Arial", 11),
            fg=ACCENT_COLOR,
            bg=BG_COLOR,
            justify="center",
        ).pack(expand=True)

    def _quit(self):
        self._stop_camera()
        try:
            if hasattr(self, "hands") and self.hands:
                self.hands.close()
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = ASLApp()
    app.mainloop()
