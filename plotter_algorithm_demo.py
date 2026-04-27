import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np


@dataclass
class AlgoParams:
    target_size: int = 512
    canny1: int = 60
    canny2: int = 150
    min_contour_len: int = 80
    simplify_eps: float = 1.5
    mm_per_pixel: float = 0.2

    # G-code / time estimate parameters
    feed_g0: float = 3000.0   # mm/min
    feed_g1: float = 1200.0   # mm/min
    pen_up_z: float = 5.0
    pen_down_z: float = 0.0


class PlotterAlgorithm:
    """
    纯算法层：图像->边缘->轮廓(polyline)->Gcode，并提供 Gcode 时间估算 & 解析。
    """

    def __init__(self, params: Optional[AlgoParams] = None):
        self.params = params or AlgoParams()

    def _load_and_resize(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        s = int(self.params.target_size)
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray

    def _edges(self, gray: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, int(self.params.canny1), int(self.params.canny2))
        # 让断线更连贯（展示更像线稿）
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        return edges

    def _contours_to_polylines(self, edges: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        polylines: List[np.ndarray] = []
        for c in contours:
            if len(c) < int(self.params.min_contour_len):
                continue
            approx = cv2.approxPolyDP(c, epsilon=float(self.params.simplify_eps), closed=False)
            pts = approx.reshape(-1, 2).astype(np.int32)
            if pts.shape[0] >= max(10, int(self.params.min_contour_len) // 8):
                polylines.append(pts)
        return polylines

    @staticmethod
    def sort_polylines_nearest(polylines: List[np.ndarray]) -> List[np.ndarray]:
        """减少空移：按最近邻把 polylines 串起来（展示更像真实绘图）。"""
        if not polylines:
            return []
        remaining = polylines[:]
        remaining.sort(key=lambda p: -len(p))
        ordered = [remaining.pop(0)]
        cur = ordered[0]
        while remaining:
            end = cur[-1]
            best_i, best_flip, best_dist = None, False, 1e18
            for i, p in enumerate(remaining):
                d1 = np.sum((p[0] - end) ** 2)
                d2 = np.sum((p[-1] - end) ** 2)
                if d1 < best_dist:
                    best_i, best_flip, best_dist = i, False, d1
                if d2 < best_dist:
                    best_i, best_flip, best_dist = i, True, d2
            nxt = remaining.pop(best_i)
            if best_flip:
                nxt = nxt[::-1].copy()
            ordered.append(nxt)
            cur = nxt
        return ordered

    def generate_gcode(self, polylines: List[np.ndarray]) -> str:
        s = int(self.params.target_size)
        mm = float(self.params.mm_per_pixel)

        g = []
        g.append("$X")
        g.append("G21")  # mm
        g.append("G90")  # absolute positioning
        g.append("G92 X0 Y0")  # current pen position = paper top-left
        g.append(f"G0 F{self.params.feed_g0:.0f}")
        g.append(f"G1 F{self.params.feed_g1:.0f}")
        g.append(f"G0 Z{self.params.pen_up_z:.3f}")

        for pts in polylines:
            x0, y0 = pts[0]

            x_mm = x0 * mm
            y_mm = (s - 1 - y0) * mm

            g.append(f"G0 X{x_mm:.3f} Y{y_mm:.3f}")
            g.append(f"G0 Z{self.params.pen_down_z:.3f}")

            for x, y in pts:
                x_mm = x * mm
                y_mm = (s - 1 - y) * mm
                g.append(f"G1 X{x_mm:.3f} Y{y_mm:.3f}")

            g.append(f"G0 Z{self.params.pen_up_z:.3f}")

        g.append("G0 X0 Y0")
        g.append("M2")
        return "\n".join(g)

    # --------- G-code parsing & time estimate ---------

    @staticmethod
    def parse_gcode_to_segments(gcode: str) -> List[Dict]:
        """
        把 G-code 解析成段：
        segment: {type:'move'/'draw', x1,y1,x2,y2, feed_mm_min}
        规则：
          - G0 视为 move
          - G1 在 pen_down 时视为 draw（否则 move）
          - pen_down 依据 Z 值（Z==0 近似落笔，Z>0 抬笔）
        """
        def get_float(tok: str) -> Optional[float]:
            try:
                return float(tok[1:])
            except Exception:
                return None

        x = y = z = 0.0
        feed_g0 = 3000.0
        feed_g1 = 1200.0
        pen_down = False

        segments: List[Dict] = []

        for raw in gcode.splitlines():
            line = raw.strip()
            if not line or line.startswith(";"):
                continue
            # remove inline comment after ';'
            if ";" in line:
                line = line.split(";", 1)[0].strip()
            parts = line.split()
            cmd = parts[0].upper()

            # update feed if "F"
            f_in_line = None
            x_new = y_new = z_new = None

            for p in parts[1:]:
                up = p.upper()
                if up.startswith("X"):
                    x_new = get_float(up)
                elif up.startswith("Y"):
                    y_new = get_float(up)
                elif up.startswith("Z"):
                    z_new = get_float(up)
                elif up.startswith("F"):
                    f_in_line = get_float(up)

            if cmd in ("G0", "G00"):
                if f_in_line is not None:
                    feed_g0 = f_in_line
                x2 = x if x_new is None else x_new
                y2 = y if y_new is None else y_new
                z2 = z if z_new is None else z_new

                # update pen state if Z move
                if z_new is not None:
                    pen_down = (abs(z2) < 1e-6)

                # create motion segment if XY changes
                if (x2 != x) or (y2 != y):
                    segments.append({
                        "type": "move",
                        "x1": x, "y1": y, "x2": x2, "y2": y2,
                        "feed": feed_g0
                    })
                x, y, z = x2, y2, z2

            elif cmd in ("G1", "G01"):
                if f_in_line is not None:
                    feed_g1 = f_in_line
                x2 = x if x_new is None else x_new
                y2 = y if y_new is None else y_new
                z2 = z if z_new is None else z_new

                if z_new is not None:
                    pen_down = (abs(z2) < 1e-6)

                if (x2 != x) or (y2 != y):
                    segments.append({
                        "type": "draw" if pen_down else "move",
                        "x1": x, "y1": y, "x2": x2, "y2": y2,
                        "feed": feed_g1
                    })
                x, y, z = x2, y2, z2

            else:
                # handle standalone feed setup lines like "G0 F3000" already in G0/G1
                if cmd.startswith("G0") and f_in_line is not None:
                    feed_g0 = f_in_line
                if cmd.startswith("G1") and f_in_line is not None:
                    feed_g1 = f_in_line
                # ignore others (M2 etc.)
                continue

        return segments

    @staticmethod
    def estimate_motion_time_seconds(segments: List[Dict]) -> float:
        """
        按段距离 / 速度估算时间。feed 以 mm/min。
        """
        t = 0.0
        for s in segments:
            dx = s["x2"] - s["x1"]
            dy = s["y2"] - s["y1"]
            dist = float(np.hypot(dx, dy))  # mm
            feed = max(1e-6, float(s["feed"]))  # mm/min
            speed = feed / 60.0  # mm/s
            t += dist / speed
        return t

    # --------- End-to-end run ---------

    def run(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[np.ndarray], str, Dict]:
        """
        返回：
          img_bgr, edges(0/255), overlay_bgr, polylines, gcode, stats
        """
        t0 = time.time()

        img, gray = self._load_and_resize(image_path)
        edges = self._edges(gray)
        polylines = self._contours_to_polylines(edges)
        polylines = self.sort_polylines_nearest(polylines)

        overlay = img.copy()
        for pts in polylines:
            cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], isClosed=False, color=(0, 0, 255), thickness=1)

        gcode = self.generate_gcode(polylines)
        segments = self.parse_gcode_to_segments(gcode)
        est_sec = self.estimate_motion_time_seconds(segments)

        dt_ms = (time.time() - t0) * 1000.0
        stats = {
            "time_ms": dt_ms,
            "polylines": int(len(polylines)),
            "points_total": int(sum(len(p) for p in polylines)) if polylines else 0,
            "gcode_lines": int(len(gcode.splitlines())),
            "est_time_sec": float(est_sec),
        }

        # edges for UI display: 0/255
        edges_255 = (edges > 0).astype(np.uint8) * 255
        return img, edges_255, overlay, polylines, gcode, stats