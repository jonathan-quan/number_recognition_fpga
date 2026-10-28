"""Run with: python -m digit_recognition.app"""

import argparse
from datetime import datetime
import json
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from PIL import Image, ImageDraw, ImageTk

from .download_model import ROOT
from .preprocess import prepare_drawing
from .serial_fpga import FPGAClient, available_ports


class DigitApp:
    CANVAS_SIZE = 336
    BRUSH = 22

    def __init__(self, root, model=None, min_confidence=0.90, min_margin=0.50, center=True,
                 mode="fpga", port=None, baud=115200):
        self.root = root
        self.model = model
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.center = center
        self.pending = False
        self.fpga_client = None
        self.responses = queue.Queue()
        self.last = None
        self.last_prediction = None
        self.last_prepared = None
        root.title("Basys 3 Digit Recognition - FPGA inference")
        root.resizable(False, False)
        box = ttk.Frame(root, padding=16)
        box.pack()
        ttk.Label(box, text="Draw one digit", font=("Segoe UI", 18, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(box, text="White ink on black. Use the left mouse button.").grid(row=1, column=0, sticky="w", pady=(0, 10))
        self.canvas = tk.Canvas(box, width=self.CANVAS_SIZE, height=self.CANVAS_SIZE,
                                bg="black", highlightthickness=0, cursor="crosshair")
        connection = ttk.Frame(box)
        connection.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 10))
        self.mode = tk.StringVar(value=mode)
        self.mode_box = ttk.Combobox(connection, textvariable=self.mode, values=("fpga", "python"), width=8, state="readonly")
        self.mode_box.pack(side="left", padx=(0, 8))
        self.mode_box.bind("<<ComboboxSelected>>", self.mode_changed)
        self.port = tk.StringVar(value=port or "")
        self.port_box = ttk.Combobox(connection, textvariable=self.port, width=9, state="readonly")
        self.port_box.pack(side="left", padx=(0, 8))
        self.refresh_button = ttk.Button(connection, text="Refresh ports", command=self.refresh_ports)
        self.refresh_button.pack(side="left", padx=(0, 8))
        self.baud = tk.StringVar(value=str(baud))
        self.baud_box = ttk.Combobox(connection, textvariable=self.baud, values=("115200", "57600", "230400"), width=9)
        self.baud_box.pack(side="left")
        self.canvas.grid(row=3, column=0, rowspan=3)
        self.canvas.bind("<Button-1>", self.start)
        self.canvas.bind("<B1-Motion>", self.draw)
        root.bind("<ButtonRelease-1>", self.stop)
        ttk.Label(box, text="Model input (28 x 28)").grid(row=3, column=1, padx=20, sticky="n")
        self.preview = ttk.Label(box)
        self.preview.grid(row=4, column=1, padx=20, sticky="n")
        self.display_text = tk.StringVar(value="----")
        ttk.Label(box, textvariable=self.display_text, font=("Consolas", 32, "bold")).grid(row=5, column=1, padx=20)
        buttons = ttk.Frame(box)
        buttons.grid(row=6, column=0, columnspan=2, sticky="w", pady=12)
        self.predict_button = ttk.Button(buttons, text="Predict", command=self.predict)
        self.predict_button.pack(side="left", padx=(0, 8))
        self.clear_button = ttk.Button(buttons, text="Clear", command=self.clear)
        self.clear_button.pack(side="left", padx=(0, 8))
        self.save_button = ttk.Button(buttons, text="Save drawing + result", command=self.save)
        self.save_button.pack(side="left")
        self.result_text = tk.StringVar()
        ttk.Label(box, textvariable=self.result_text, wraplength=540, font=("Segoe UI", 11)).grid(row=7, column=0, columnspan=2, sticky="w")
        self.help_text = tk.StringVar()
        ttk.Label(box, textvariable=self.help_text, wraplength=540).grid(row=8, column=0, columnspan=2, sticky="w", pady=(12, 0))
        root.bind("<Return>", lambda event: self.predict())
        root.bind("<Escape>", lambda event: self.clear())
        self.clear()
        self.refresh_ports()
        self.mode_changed()
        root.after(50, self.poll_response)

    def refresh_ports(self):
        ports = available_ports()
        self.port_box.configure(values=[p.device for p in ports])
        ftdi = [p for p in ports if p.vid == 0x0403 and p.pid == 0x6010]
        usb_names = [p.device for p in ports if "USB" in p.hwid]
        if len(ftdi) == 1 and (not self.port.get() or self.port.get() not in usb_names):
            self.port.set(ftdi[0].device)

    def mode_changed(self, event=None):
        self.last_prediction = None
        self.display_text.set("----")
        self.help_text.set("9999 = unknown heuristic; letters and scribbles may still pass.\n" +
                           ("FPGA mode: laptop only prepares/sends pixels. No local model inference."
                            if self.mode.get() == "fpga" else "Python validation mode: pretrained model inference runs on the laptop."))

    def set_busy(self, pending):
        self.pending = pending
        for button in (self.predict_button, self.clear_button, self.save_button, self.refresh_button):
            button.configure(state="disabled" if pending else "normal")
        for combo in (self.mode_box, self.port_box):
            combo.configure(state="disabled" if pending else "readonly")
        self.baud_box.configure(state="disabled" if pending else "normal")

    def show_preview(self, pixels):
        self.preview_image = ImageTk.PhotoImage(Image.fromarray(pixels).resize((140, 140), Image.Resampling.NEAREST))
        self.preview.configure(image=self.preview_image)

    def clear(self):
        if self.pending:
            return
        self.canvas.delete("all")
        self.image = Image.new("L", (self.CANVAS_SIZE, self.CANVAS_SIZE), 0)
        self.painter = ImageDraw.Draw(self.image)
        self.last = None
        self.last_prediction = None
        self.last_prepared = None
        self.display_text.set("----")
        self.result_text.set("Draw a digit, then click Predict (or press Enter).")
        self.show_preview(prepare_drawing(self.image).pixels)

    def start(self, event):
        if self.pending:
            return
        self.last = (event.x, event.y)
        self.draw(event)

    def draw(self, event):
        if self.last is None or self.pending:
            return
        x = max(0, min(self.CANVAS_SIZE - 1, event.x))
        y = max(0, min(self.CANVAS_SIZE - 1, event.y))
        self.canvas.create_line(*self.last, x, y, width=self.BRUSH, fill="white", capstyle=tk.ROUND)
        self.painter.line([self.last, (x, y)], fill=255, width=self.BRUSH)
        r = self.BRUSH / 2
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.painter.ellipse((x-r, y-r, x+r, y+r), fill=255)
        self.last = (x, y)
        self.last_prediction = None
        self.last_prepared = None
        self.display_text.set("----")
        self.result_text.set("Drawing changed. Click Predict to update.")

    def stop(self, event=None):
        self.last = None

    def predict(self):
        if self.pending:
            return
        prepared = prepare_drawing(self.image, center=self.center)
        self.last_prepared = prepared
        self.show_preview(prepared.pixels)
        if self.mode.get() == "fpga":
            try:
                baud = int(self.baud.get())
                if self.fpga_client is None or self.fpga_client.port != self.port.get() or self.fpga_client.baud != baud:
                    self.fpga_client = FPGAClient(self.port.get(), baud)
                client = self.fpga_client
            except ValueError as exc:
                self.result_text.set(str(exc))
                return
            self.set_busy(True)
            self.result_text.set("Sending 784 pixels; waiting for FPGA acknowledgment and result...")
            def transfer():
                try:
                    self.responses.put(client.predict(prepared.pixels))
                except Exception as exc:
                    self.responses.put(exc)
            threading.Thread(target=transfer, daemon=True).start()
            return
        # Explicit Python validation mode only. FPGA mode never imports these.
        try:
            import torch
            from .model import load_model
            from .predict import predict
            torch.set_num_threads(1)
            if self.model is None:
                self.model = load_model()
            result = predict(self.model, prepared, self.min_confidence, self.min_margin)
        except Exception as exc:
            self.result_text.set(f"Python validation failed: {exc}")
            return
        self.show_result(result)
        return result

    def show_result(self, result):
        self.last_prediction = result
        self.display_text.set(result.display)
        if self.mode.get() == "fpga":
            status = f"Unknown (FPGA rejection; top digit: {result.digit})" if result.unknown else f"FPGA prediction: {result.digit}"
            self.result_text.set(f"{status}\nInteger score gap: {result.margin}; "
                                 f"inference: {result.cycles} clocks ({result.cycles / 100000:.3f} ms at 100 MHz).")
            return
        status = f"Unknown: {result.reason}" if result.unknown else f"Prediction: {result.digit}"
        self.result_text.set(f"{status}\nTop digit: {result.digit}; softmax confidence: {result.confidence:.1%}; "
                             f"probability margin: {result.probability_margin:.1%}")

    def poll_response(self):
        try:
            result = self.responses.get_nowait()
        except queue.Empty:
            pass
        else:
            self.set_busy(False)
            if isinstance(result, Exception):
                self.display_text.set("----")
                self.last_prediction = None
                self.result_text.set(f"FPGA request failed: {result}\nNo local inference fallback was used.")
            else:
                self.show_result(result)
        self.root.after(50, self.poll_response)

    def save(self):
        if self.pending:
            return
        if self.last_prepared is None:
            self.last_prepared = prepare_drawing(self.image, center=self.center)
        folder = ROOT / "drawings"
        try:
            folder.mkdir(exist_ok=True)
            stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.image.save(folder / f"{stem}_canvas.png")
            Image.fromarray(self.last_prepared.pixels).save(folder / f"{stem}_28x28.png")
            metadata = {**(self.last_prediction.as_dict() if self.last_prediction else {}),
                        "backend": self.mode.get(), "center": self.center}
            if self.mode.get() == "fpga":
                metadata["rejection_rule"] = "FPGA: blank image or integer softmax confidence lower bound < 0.90"
            else:
                metadata.update(min_confidence=self.min_confidence, min_margin=self.min_margin)
            (folder / f"{stem}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            self.result_text.set(self.result_text.get() + f"\nSaved locally: drawings/{stem}")
        except OSError as exc:
            messagebox.showerror("Could not save drawing", str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-confidence", type=float, default=0.90)
    parser.add_argument("--min-margin", type=float, default=0.50)
    parser.add_argument("--direct-resize", action="store_true", help="Use the upstream GUI's geometry instead of centering")
    parser.add_argument("--mode", choices=("fpga", "python"), default="fpga")
    parser.add_argument("--port", help="Basys 3 USB serial COM port, e.g. COM6")
    parser.add_argument("--baud", type=int, default=115200, help="Must match the programmed FPGA bitstream")
    args = parser.parse_args()
    if not 0 <= args.min_confidence <= 1 or not 0 <= args.min_margin <= 1:
        parser.error("thresholds must be in [0, 1]")
    root = tk.Tk()
    DigitApp(root, min_confidence=args.min_confidence, min_margin=args.min_margin, center=not args.direct_resize,
             mode=args.mode, port=args.port, baud=args.baud)
    root.mainloop()


if __name__ == "__main__":
    main()
