import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
from gcode_sender import send_gcode


class PlotterUI:
    """
    纯 UI 层：
      - 参数面板
      - Run All（后台线程跑算法）
      - 预览（原图/边缘/叠加）
      - 播放：Polyline 播放 / Gcode 播放
      - 180° 翻转（动画坐标翻转）
    """

    def __init__(self, algorithm):
        self.alg = algorithm
        self.root = tk.Tk()
        self.root.title("Plotter Demo - Stage Presentation")
        self.root.geometry("1280x760")
        self.root.minsize(1200, 700)

        self.image_path = None
        self.last_polylines = None
        self.last_gcode = None
        self.last_stats = None
        self.last_size = None

        self.worker_q = queue.Queue()
        self.worker_thread = None

        # animation
        self.anim_job = None
        self.anim_stream = None
        self.anim_i = 0
        self.anim_mode = tk.StringVar(value="polyline")  # 'polyline' or 'gcode'
        self.flip180 = tk.BooleanVar(value=True)

        # preview tk images refs
        self.tkimg_orig = None
        self.tkimg_edges = None
        self.tkimg_overlay = None

        self._build_ui()
        self._poll_worker_queue()

    # Send gcode
    def on_send_gcode(self):
        if not self.last_gcode:
            messagebox.showwarning("No G-code", "Run All first.")
            return

        ip = "192.168.4.1"  # default ESP32 AP IP
        port = 23  # GRBL-ESP32 Telnet

        try:
            send_gcode(ip, port, self.last_gcode)
            messagebox.showinfo("Success", "G-code sent!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---------------- UI Layout ----------------

    def _build_ui(self):
        left = ttk.Frame(self.root, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(self.root, padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Controls row
        ttk.Label(left, text="Controls", font=("Segoe UI", 14, "bold")).pack(anchor="w", pady=(0, 8))

        ttk.Button(left, text="Select Image", command=self.on_select_image).pack(fill=tk.X, pady=3)
        ttk.Button(left, text="Run All (Non-blocking)", command=self.on_run_all).pack(fill=tk.X, pady=3)

        ttk.Separator(left).pack(fill=tk.X, pady=8)

        ttk.Label(left, text="Playback", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))

        mode_frame = ttk.Frame(left)
        mode_frame.pack(fill=tk.X, pady=3)
        ttk.Radiobutton(mode_frame, text="Polyline", variable=self.anim_mode, value="polyline").pack(anchor="w")
        ttk.Radiobutton(mode_frame, text="G-code", variable=self.anim_mode, value="gcode").pack(anchor="w")

        ttk.Checkbutton(left, text="Flip 180° (Animation)", variable=self.flip180).pack(anchor="w", pady=4)

        pb_row = ttk.Frame(left)
        pb_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(pb_row, text="Play", command=self.on_play).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(pb_row, text="Stop", command=self.on_stop).pack(side=tk.LEFT, expand=True, fill=tk.X)

        ttk.Button(left, text="Export G-code", command=self.on_export_gcode).pack(fill=tk.X, pady=(8, 0))
        ttk.Button(left, text="Send to ESP32", command=self.on_send_gcode).pack(fill=tk.X, pady=3)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        # Parameters panel
        ttk.Label(left, text="Parameters", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))

        self.var_size = tk.IntVar(value=self.alg.params.target_size)
        self.var_c1 = tk.IntVar(value=self.alg.params.canny1)
        self.var_c2 = tk.IntVar(value=self.alg.params.canny2)
        self.var_minlen = tk.IntVar(value=self.alg.params.min_contour_len)
        self.var_eps = tk.DoubleVar(value=self.alg.params.simplify_eps)
        self.var_mm = tk.DoubleVar(value=self.alg.params.mm_per_pixel)
        self.var_f0 = tk.DoubleVar(value=self.alg.params.feed_g0)
        self.var_f1 = tk.DoubleVar(value=self.alg.params.feed_g1)

        self._add_slider(left, "Target Size", self.var_size, 128, 1024, step=16, is_float=False)
        self._add_slider(left, "Canny 1", self.var_c1, 0, 255, step=1, is_float=False)
        self._add_slider(left, "Canny 2", self.var_c2, 0, 255, step=1, is_float=False)
        self._add_slider(left, "Min Contour Len", self.var_minlen, 10, 500, step=1, is_float=False)
        self._add_slider(left, "Simplify EPS", self.var_eps, 0.5, 5.0, step=0.1, is_float=True)
        self._add_slider(left, "mm / pixel", self.var_mm, 0.05, 1.0, step=0.01, is_float=True)
        self._add_slider(left, "Feed G0 (mm/min)", self.var_f0, 500, 8000, step=50, is_float=True)
        self._add_slider(left, "Feed G1 (mm/min)", self.var_f1, 100, 5000, step=50, is_float=True)

        # Stats box
        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Stats", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 6))
        self.stats = tk.Text(left, width=34, height=10)
        self.stats.pack(fill=tk.X)
        self._set_stats("Select an image, click Run All.\n")

        # Right: previews + animation canvas
        top = ttk.Frame(right)
        top.pack(fill=tk.BOTH, expand=True)

        self.p_orig = self._make_panel(top, "Original")
        self.p_edges = self._make_panel(top, "Edges")
        self.p_overlay = self._make_panel(top, "Overlay")

        self.p_orig.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self.p_edges.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self.p_overlay.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        bottom = ttk.Frame(right)
        bottom.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        ttk.Label(bottom, text="Plot Animation", font=("Segoe UI", 12, "bold")).pack(anchor="w")

        self.anim_canvas = tk.Canvas(bottom, bg="white")
        self.anim_canvas.pack(fill=tk.BOTH, expand=True)

    def _make_panel(self, parent, title):
        f = ttk.Frame(parent)
        ttk.Label(f, text=title, font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 6))
        body = ttk.Frame(f, relief=tk.GROOVE, padding=6)
        body.pack(fill=tk.BOTH, expand=True)
        lbl = ttk.Label(body)
        lbl.pack(fill=tk.BOTH, expand=True)
        f.body = body
        f.lbl = lbl
        return f

    def _add_slider(self, parent, label, var, lo, hi, step, is_float):
        box = ttk.Frame(parent)
        box.pack(fill=tk.X, pady=3)
        ttk.Label(box, text=label).pack(anchor="w")
        s = ttk.Scale(box, from_=lo, to=hi, orient=tk.HORIZONTAL)
        s.pack(fill=tk.X)
        ent = ttk.Entry(box, width=10, textvariable=var)
        ent.pack(anchor="e", pady=(2, 0))

        def on_move(v):
            val = float(v)
            if is_float:
                val = round(val / step) * step
                var.set(float(val))
            else:
                val = int(round(val / step) * step)
                var.set(int(val))

        s.configure(command=on_move)
        s.set(var.get())

    # ---------------- Helpers ----------------

    def _set_stats(self, text: str):
        self.stats.config(state="normal")
        self.stats.delete("1.0", "end")
        self.stats.insert("end", text)
        self.stats.config(state="disabled")

    def _apply_params_to_alg(self):
        # basic sanity
        size = int(self.var_size.get())
        size = max(128, min(1024, size))
        size = int(round(size / 16) * 16)

        c1 = int(self.var_c1.get())
        c2 = int(self.var_c2.get())
        if c2 < c1:
            c2 = c1 + 10

        self.alg.params.target_size = size
        self.alg.params.canny1 = max(0, min(255, c1))
        self.alg.params.canny2 = max(0, min(255, c2))
        self.alg.params.min_contour_len = max(5, int(self.var_minlen.get()))
        self.alg.params.simplify_eps = float(self.var_eps.get())
        self.alg.params.mm_per_pixel = float(self.var_mm.get())
        self.alg.params.feed_g0 = float(self.var_f0.get())
        self.alg.params.feed_g1 = float(self.var_f1.get())

        # reflect
        self.var_size.set(self.alg.params.target_size)
        self.var_c2.set(self.alg.params.canny2)

    def _to_tkimg(self, img_bgr_or_gray, max_w, max_h):
        if img_bgr_or_gray.ndim == 2:
            arr = np.stack([img_bgr_or_gray]*3, axis=2)
        else:
            # BGR -> RGB
            arr = img_bgr_or_gray[:, :, ::-1]
        h, w = arr.shape[:2]
        scale = min(max_w / w, max_h / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        arr = np.array(Image.fromarray(arr).resize((nw, nh), resample=Image.BILINEAR))
        return ImageTk.PhotoImage(Image.fromarray(arr))

    def _update_previews(self, img, edges, overlay):
        self.root.update_idletasks()
        panels = [
            (self.p_orig, img, "tkimg_orig"),
            (self.p_edges, edges, "tkimg_edges"),
            (self.p_overlay, overlay, "tkimg_overlay"),
        ]
        for p, src, name in panels:
            w = max(220, p.body.winfo_width() - 20)
            h = max(220, p.body.winfo_height() - 20)
            tkimg = self._to_tkimg(src, w, h)
            setattr(self, name, tkimg)
            p.lbl.configure(image=tkimg)

    # ---------------- Events ----------------

    def on_select_image(self):
        fp = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not fp:
            return
        self.image_path = fp
        self._set_stats(f"Selected:\n{fp}\n\nClick Run All.\n")

    def on_run_all(self):
        if not self.image_path:
            messagebox.showwarning("No image", "Please Select Image first.")
            return
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "Processing is still running...")
            return

        self.on_stop()
        self._apply_params_to_alg()
        self._set_stats("Processing in background...\n")

        def worker():
            try:
                result = self.alg.run(self.image_path)
                self.worker_q.put(("ok", result))
            except Exception as e:
                self.worker_q.put(("err", str(e)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def on_export_gcode(self):
        if not self.last_gcode:
            messagebox.showwarning("No G-code", "Run All first to generate G-code.")
            return
        fp = filedialog.asksaveasfilename(
            title="Save G-code",
            defaultextension=".gcode",
            filetypes=[("G-code", "*.gcode"), ("Text", "*.txt")]
        )
        if not fp:
            return
        with open(fp, "w", encoding="utf-8") as f:
            f.write(self.last_gcode)
        messagebox.showinfo("Saved", f"Saved:\n{fp}")

    def on_play(self):
        if not self.last_polylines or not self.last_size:
            messagebox.showwarning("No data", "Run All first.")
            return
        self._build_anim_stream()
        self._reset_anim_canvas()
        self.anim_i = 0
        self._animate_tick()

    def on_stop(self):
        if self.anim_job is not None:
            try:
                self.root.after_cancel(self.anim_job)
            except Exception:
                pass
        self.anim_job = None

    # ---------------- Worker poll ----------------

    def _poll_worker_queue(self):
        try:
            msg, payload = self.worker_q.get_nowait()
        except queue.Empty:
            self.root.after(50, self._poll_worker_queue)
            return

        if msg == "err":
            self._set_stats("Error:\n" + payload + "\n")
            messagebox.showerror("Run failed", payload)
        else:
            img, edges, overlay, polylines, gcode, stats = payload
            self.last_polylines = polylines
            self.last_gcode = gcode
            self.last_stats = stats
            self.last_size = self.alg.params.target_size

            self._update_previews(img, edges, overlay)

            est = stats["est_time_sec"]
            est_str = f"{est:.2f} s  (~{est/60.0:.2f} min)"
            text = (
                f"Time: {stats['time_ms']:.2f} ms\n"
                f"Polylines: {stats['polylines']}\n"
                f"Total Points: {stats['points_total']}\n"
                f"Gcode lines: {stats['gcode_lines']}\n"
                f"Estimated motion time: {est_str}\n"
            )
            self._set_stats(text)

        self.root.after(50, self._poll_worker_queue)

    # ---------------- Animation ----------------

    def _reset_anim_canvas(self):
        self.anim_canvas.delete("all")
        self.root.update_idletasks()
        cw = max(200, self.anim_canvas.winfo_width())
        ch = max(200, self.anim_canvas.winfo_height())
        self.anim_scale = min((cw - 20) / self.last_size, (ch - 20) / self.last_size)
        self.anim_offx = 10
        self.anim_offy = 10

    def _map_xy_px_to_canvas(self, x_px, y_px):
        """
        注意：polylines 是图像坐标（y向下）。
        我们在显示时做：
          - 先翻成笛卡尔感觉（y -> size-1-y）
          - 若 flip180: 再做 180° 旋转（x->size-1-x, y->size-1-y）
        """
        s = self.last_size
        # convert to cartesian-like
        y = (s - 1 - y_px)
        x = x_px

        if self.flip180.get():
            x = (s - 1 - x)
            y = (s - 1 - y)

        cx = self.anim_offx + x * self.anim_scale
        cy = self.anim_offy + y * self.anim_scale
        return cx, cy

    def _build_anim_stream(self):
        """
        两种播放：
          polyline: 用 polylines 逐段画，段间 gap 视为抬笔
          gcode: 解析 gcode -> segments（move/draw），只画 draw 段，可选画 move 段为浅灰(这里默认不画)
        """
        mode = self.anim_mode.get()
        stream = []

        if mode == "polyline":
            for pts in self.last_polylines:
                for i in range(1, len(pts)):
                    x1, y1 = pts[i - 1]
                    x2, y2 = pts[i]
                    stream.append(("draw", x1, y1, x2, y2))
                stream.append(("gap", None, None, None, None))

        else:
            segs = self.alg.parse_gcode_to_segments(self.last_gcode)
            # gcode 是 mm 坐标；我们要映射回像素坐标用于同一个 Canvas
            # 简化：用 mm_per_pixel 反推 px，且 y 需要从 Gcode 的“笛卡尔”转回图像坐标
            mm = float(self.alg.params.mm_per_pixel)
            s = self.last_size

            for sg in segs:
                if sg["type"] != "draw":
                    # 你想展示空移也可以改成画虚线
                    continue
                x1_mm, y1_mm, x2_mm, y2_mm = sg["x1"], sg["y1"], sg["x2"], sg["y2"]
                x1_px = x1_mm / mm
                x2_px = x2_mm / mm
                # gcode y 是笛卡尔方向，转回图像坐标 y_img = (s-1 - y_cart)
                y1_px = (s - 1) - (y1_mm / mm)
                y2_px = (s - 1) - (y2_mm / mm)
                stream.append(("draw", x1_px, y1_px, x2_px, y2_px))

        self.anim_stream = stream

    def _animate_tick(self):
        if not self.anim_stream or self.anim_i >= len(self.anim_stream):
            self.anim_job = None
            return

        # 每 tick 画多少段（越大越快，越像一口气画完）
        segs_per_tick = 80
        count = 0

        while count < segs_per_tick and self.anim_i < len(self.anim_stream):
            typ, x1, y1, x2, y2 = self.anim_stream[self.anim_i]
            self.anim_i += 1

            if typ == "gap":
                continue

            c1x, c1y = self._map_xy_px_to_canvas(x1, y1)
            c2x, c2y = self._map_xy_px_to_canvas(x2, y2)
            self.anim_canvas.create_line(c1x, c1y, c2x, c2y, width=1)
            count += 1

        # ~60fps
        self.anim_job = self.root.after(16, self._animate_tick)

    # ----------------

    def start(self):
        self.root.mainloop()