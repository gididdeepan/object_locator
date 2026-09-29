"""
Object Locator + Rotate GUI (all-in-one) -- with Matching Accuracy + Detection Time display
---------------------------------------------------------------------------------------------
Same as before, plus:
  - After detection, computes a Matching Accuracy % = (inliers / good matches) * 100
    This is the standard confidence measure for SIFT+homography matching:
    it tells you what percentage of the matched keypoints actually agree
    with the found position/orientation (vs. being outliers/noise).
  - Shows it as a big colored label in the UI (green/orange/red by confidence)
  - Times the full detection pipeline (preprocessing -> SIFT -> matching ->
    homography) and shows "Detection Time: X.XXX s" next to the accuracy label.
  - Also burns the accuracy text AND the detection time onto the saved result image
  - The crop window has scrollbars around the image, and the
    "Use This Crop" / "Cancel" buttons stay pinned and always visible,
    even when the source image is very large.
  - NEW: The crop window can ROTATE / FLIP the image first (-90 / +90, Flip H/V,
    fine-angle slider, Reset) and then crop from the rotated image.
  - NEW: The crop rectangle is ADJUSTABLE: drag corner/edge handles to resize it,
    drag inside it to move it (the mouse cursor changes to show what you can do).
  - NEW: "Target angle" box + "Align Object to This Angle" button. The program takes the CENTER of the
    scene image, draws a horizontal line through it, then a line from that center to the CENTER of the
    located object, measures the angle between them, and rotates the scene about its center so that
    line lands on the angle you typed (90 = object center straight above the image center).
    Both lines and the angle are drawn on the result image. After the rotation the object is NOT
    detected again: its box/center are moved with the scene by rotation maths (fast).
  - The crop window has ZOOM IN / ZOOM OUT (buttons, Fit, 100%,
    Ctrl+Mouse-wheel, +/-/0 keys). Pan with scrollbars, wheel, or right/middle-drag.

Install requirements first:
    pip install opencv-python numpy pillow

Run:
    python locator_rotate_gui.py
"""

import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np


class CropWindow(tk.Toplevel):
    """Popup window for cropping a template out of an image.

    Features
    --------
    * ROTATE first, then crop: -90 / +90 buttons, Flip H / Flip V, a fine
      angle slider (-180..180) and a Reset button. The crop is taken from
      the rotated/flipped image, so the template ends up "straightened".
    * ZOOM in / out: Zoom +/- buttons, Fit, 100%, Ctrl+Mouse-wheel (zooms
      around the mouse pointer) and the +/-/0 keys.
    * PAN: scrollbars, mouse wheel (vertical), Shift+wheel (horizontal),
      or drag with the right / middle mouse button.
    * The "Use This Crop" / "Cancel" buttons are pinned to the bottom of the
      window, so they are always visible even for very large images.

    Only the part of the image that is currently visible is rendered, so
    zooming into huge images stays fast and memory-friendly. The crop is
    always taken from the full-resolution rotated image.
    """

    ZOOM_STEP = 1.25
    MIN_ZOOM = 0.02
    MAX_ZOOM = 16.0
    HANDLE_SIZE = 5        # half-size of the resize handles (pixels on screen)
    HIT_TOLERANCE = 8      # how close (screen px) the mouse must be to grab an edge/handle
    FILL_COLOR = (30, 30, 30)  # background used for the empty corners after a fine rotation

    def __init__(self, parent, pil_image, on_crop_done):
        super().__init__(parent)
        self.title("Rotate / zoom, drag to select crop area, then click 'Use This Crop'")
        self.on_crop_done = on_crop_done
        self.source_image = pil_image      # untouched original
        self.work_image = pil_image        # after flips / rotation (this is what gets cropped)

        # ---- transform state ----
        self.flip_h = False
        self.flip_v = False
        self.base_angle = 0                # multiples of 90 (clockwise)
        self.fine_angle = 0.0              # slider angle (clockwise, degrees)
        self._rotate_job = None

        # ---- view state ----
        self.zoom = 1.0
        self.sel_rect = None               # [x1, y1, x2, y2] in work_image pixel coords
        self.drag_mode = None              # None / 'new' / 'move' / 'resize'
        self._drag_handle = None
        self._drag_origin = None
        self._anchor = None
        self._rect_at_start = None
        self._view_img = None
        self._redraw_job = None

        # Window size: never larger than the screen
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        win_w = min(1200, screen_w - 80)
        win_h = min(850, screen_h - 120)
        self.geometry(f"{win_w}x{win_h}+40+30")
        self.minsize(640, 420)

        # ---- Bottom controls FIRST so they are always visible ----
        btn_frame = tk.Frame(self)
        btn_frame.pack(side="bottom", fill="x", pady=8)
        tk.Button(btn_frame, text="Use This Crop", bg="#4f8cff", fg="white", width=16,
                  command=self.confirm_crop).pack(side="left", padx=6)
        tk.Button(btn_frame, text="Cancel", width=10,
                  command=self.destroy).pack(side="left", padx=6)

        self.info_label = tk.Label(
            self, text="Drag to draw the crop box; drag its handles/edges to resize, drag inside to move.  "
                       "Ctrl+Wheel = zoom,  Right/Middle-drag = pan.")
        self.info_label.pack(side="bottom", pady=(0, 4))

        # ---- Rotation toolbar ----
        rot = tk.Frame(self)
        rot.pack(side="top", pady=(6, 2))
        tk.Button(rot, text="⟲ -90°", width=8, command=lambda: self.quick_rotate(-90)).grid(row=0, column=0, padx=3)
        tk.Button(rot, text="⟳ +90°", width=8, command=lambda: self.quick_rotate(90)).grid(row=0, column=1, padx=3)
        tk.Button(rot, text="Flip H", width=7, command=self.flip_horizontal).grid(row=0, column=2, padx=3)
        tk.Button(rot, text="Flip V", width=7, command=self.flip_vertical).grid(row=0, column=3, padx=3)
        tk.Button(rot, text="Reset", width=8, command=self.reset_transform).grid(row=0, column=4, padx=3)
        tk.Label(rot, text="Fine angle:").grid(row=0, column=5, padx=(16, 4))
        self.angle_slider = tk.Scale(rot, from_=-180, to=180, orient="horizontal",
                                      length=300, command=self.on_slider_change)
        self.angle_slider.grid(row=0, column=6, padx=4)
        self.angle_label = tk.Label(rot, text="0°", width=5)
        self.angle_label.grid(row=0, column=7, padx=4)

        # ---- Zoom toolbar ----
        zbar = tk.Frame(self)
        zbar.pack(side="top", pady=2)
        tk.Button(zbar, text="Zoom -", width=8, command=self.zoom_out).grid(row=0, column=0, padx=3)
        tk.Button(zbar, text="Zoom +", width=8, command=self.zoom_in).grid(row=0, column=1, padx=3)
        tk.Button(zbar, text="Fit", width=7, command=self.fit_to_window).grid(row=0, column=2, padx=3)
        tk.Button(zbar, text="100%", width=7, command=lambda: self.set_zoom(1.0)).grid(row=0, column=3, padx=3)
        self.zoom_label = tk.Label(zbar, text="100%", width=7, font=("Arial", 10, "bold"))
        self.zoom_label.grid(row=0, column=4, padx=8)

        # ---- Scrollable canvas area ----
        canvas_frame = tk.Frame(self)
        canvas_frame.pack(side="top", fill="both", expand=True, padx=8, pady=6)
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        self.v_scroll = tk.Scrollbar(canvas_frame, orient="vertical")
        self.h_scroll = tk.Scrollbar(canvas_frame, orient="horizontal")
        self.canvas = tk.Canvas(canvas_frame, cursor="cross", bg="#111111", highlightthickness=0,
                                xscrollcommand=self._on_xscroll, yscrollcommand=self._on_yscroll)
        self.v_scroll.config(command=self.canvas.yview)
        self.h_scroll.config(command=self.canvas.xview)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        # Selection
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_motion)

        # Wheel: scroll / zoom
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)                 # Windows / macOS
        self.canvas.bind("<Shift-MouseWheel>", self._on_shift_mousewheel)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self.canvas.yview_scroll(-1, "units"))   # Linux
        self.canvas.bind("<Button-5>", lambda e: self.canvas.yview_scroll(1, "units"))
        self.canvas.bind("<Control-Button-4>", lambda e: self.set_zoom(self.zoom * self.ZOOM_STEP, e.x, e.y))
        self.canvas.bind("<Control-Button-5>", lambda e: self.set_zoom(self.zoom / self.ZOOM_STEP, e.x, e.y))

        # Pan with right / middle mouse button
        for btn in (2, 3):
            self.canvas.bind(f"<ButtonPress-{btn}>", lambda e: self.canvas.scan_mark(e.x, e.y))
            self.canvas.bind(f"<B{btn}-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

        # Redraw visible part when the window is resized
        self.canvas.bind("<Configure>", lambda e: self.schedule_redraw())

        # Keyboard zoom
        self.bind("<plus>", lambda e: self.zoom_in())
        self.bind("<equal>", lambda e: self.zoom_in())
        self.bind("<minus>", lambda e: self.zoom_out())
        self.bind("<Key-0>", lambda e: self.fit_to_window())

        # Initial view: fit the whole image in the window
        self.update_idletasks()
        self.fit_to_window()

    # ------------------------------------------------------------------
    # Rotation / flip
    # ------------------------------------------------------------------
    def apply_transform(self, refit=False):
        """Rebuild work_image from the original using flips + rotations."""
        img = self.source_image
        if self.flip_h:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_v:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)

        turn = {0: None, 90: Image.ROTATE_270, 180: Image.ROTATE_180, 270: Image.ROTATE_90}[self.base_angle % 360]
        if turn is not None:
            img = img.transpose(turn)  # clockwise 90-degree steps

        if self.fine_angle != 0:
            img = img.rotate(-self.fine_angle, expand=True, resample=Image.BILINEAR,
                             fillcolor=self.FILL_COLOR)

        self.work_image = img
        # A selection made on the old orientation no longer makes sense
        self.sel_rect = None
        self.drag_mode = None
        self.canvas.delete("sel")
        self.info_label.config(text="Drag on the image to select the crop area.")

        if refit:
            self.fit_to_window()
        else:
            self.update_scrollregion()
            self.schedule_redraw()

    def quick_rotate(self, delta):
        self.base_angle = (self.base_angle + delta) % 360
        self.apply_transform(refit=True)

    def flip_horizontal(self):
        self.flip_h = not self.flip_h
        self.apply_transform(refit=True)

    def flip_vertical(self):
        self.flip_v = not self.flip_v
        self.apply_transform(refit=True)

    def reset_transform(self):
        if self._rotate_job:
            self.after_cancel(self._rotate_job)
            self._rotate_job = None
        self.flip_h = self.flip_v = False
        self.base_angle = 0
        self.fine_angle = 0.0
        self.angle_slider.set(0)
        self.angle_label.config(text="0°")
        self.apply_transform(refit=True)

    def on_slider_change(self, value):
        v = float(value)
        if v == self.fine_angle:
            return
        self.fine_angle = v
        self.angle_label.config(text=f"{int(v)}°")
        # Debounce so dragging the slider doesn't re-rotate a big image on every pixel
        if self._rotate_job:
            self.after_cancel(self._rotate_job)
        self._rotate_job = self.after(120, self._apply_fine_rotation)

    def _apply_fine_rotation(self):
        self._rotate_job = None
        self.apply_transform(refit=False)

    # ------------------------------------------------------------------
    # Zoom / scrolling / rendering
    # ------------------------------------------------------------------
    def display_size(self):
        w, h = self.work_image.size
        return max(1, int(w * self.zoom)), max(1, int(h * self.zoom))

    def update_scrollregion(self):
        dw, dh = self.display_size()
        self.canvas.config(scrollregion=(0, 0, dw, dh))

    def fit_to_window(self):
        self.update_idletasks()
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        w, h = self.work_image.size
        z = min(cw / w, ch / h, 1.0)
        self.zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, z))
        self.update_scrollregion()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self.zoom_label.config(text=f"{self.zoom * 100:.0f}%")
        self.draw_selection()
        self.schedule_redraw()

    def set_zoom(self, new_zoom, ax=None, ay=None):
        """Zoom to new_zoom, keeping the image point under (ax, ay) (widget coords) fixed."""
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, new_zoom))
        if abs(new_zoom - self.zoom) < 1e-9:
            return
        if ax is None:
            ax = self.canvas.winfo_width() / 2
            ay = self.canvas.winfo_height() / 2

        # image point currently under the anchor
        wx = self.canvas.canvasx(ax) / self.zoom
        wy = self.canvas.canvasy(ay) / self.zoom

        self.zoom = new_zoom
        self.update_scrollregion()
        dw, dh = self.display_size()
        self.canvas.xview_moveto(max(0.0, (wx * new_zoom - ax) / dw))
        self.canvas.yview_moveto(max(0.0, (wy * new_zoom - ay) / dh))

        self.zoom_label.config(text=f"{new_zoom * 100:.0f}%")
        self.draw_selection()
        self.schedule_redraw()

    def zoom_in(self):
        self.set_zoom(self.zoom * self.ZOOM_STEP)

    def zoom_out(self):
        self.set_zoom(self.zoom / self.ZOOM_STEP)

    def _on_xscroll(self, first, last):
        self.h_scroll.set(first, last)
        self.schedule_redraw()

    def _on_yscroll(self, first, last):
        self.v_scroll.set(first, last)
        self.schedule_redraw()

    def schedule_redraw(self):
        if self._redraw_job is None:
            self._redraw_job = self.after_idle(self._do_redraw)

    def _do_redraw(self):
        self._redraw_job = None
        if not self.winfo_exists():
            return
        self.redraw_view()

    def redraw_view(self):
        """Render only the visible part of the image (fast, even at high zoom)."""
        dw, dh = self.display_size()
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        cx0, cy0 = self.canvas.canvasx(0), self.canvas.canvasy(0)

        x0, y0 = max(0, int(cx0)), max(0, int(cy0))
        x1, y1 = min(dw, int(cx0) + cw), min(dh, int(cy0) + ch)
        if x1 <= x0 or y1 <= y0:
            return

        w, h = self.work_image.size
        box = (min(w, x0 / self.zoom), min(h, y0 / self.zoom),
               min(w, x1 / self.zoom), min(h, y1 / self.zoom))
        if box[2] <= box[0] or box[3] <= box[1]:
            return

        region = self.work_image.resize((x1 - x0, y1 - y0), Image.BILINEAR, box=box)
        self._view_img = ImageTk.PhotoImage(region)  # keep a reference!

        self.canvas.delete("view")
        self.canvas.create_image(x0, y0, anchor="nw", image=self._view_img, tags="view")
        self.canvas.tag_lower("view")

    def _on_mousewheel(self, event):
        direction = 1 if event.delta > 0 else -1
        self.canvas.yview_scroll(-direction, "units")

    def _on_shift_mousewheel(self, event):
        direction = 1 if event.delta > 0 else -1
        self.canvas.xview_scroll(-direction, "units")

    def _on_ctrl_mousewheel(self, event):
        factor = self.ZOOM_STEP if event.delta > 0 else 1 / self.ZOOM_STEP
        self.set_zoom(self.zoom * factor, event.x, event.y)

    # ------------------------------------------------------------------
    # Crop selection (stored in work_image pixel coordinates, so it
    # survives zooming and scrolling)
    # ------------------------------------------------------------------
    def _to_work_coords(self, ex, ey):
        w, h = self.work_image.size
        x = self.canvas.canvasx(ex) / self.zoom
        y = self.canvas.canvasy(ey) / self.zoom
        return max(0.0, min(float(w), x)), max(0.0, min(float(h), y))

    def _sel_display_rect(self):
        """Selection rectangle in canvas (display) coordinates, or None."""
        if self.sel_rect is None:
            return None
        return tuple(v * self.zoom for v in self.sel_rect)

    def draw_selection(self):
        self.canvas.delete("sel")
        r = self._sel_display_rect()
        if r is None:
            return
        x1, y1, x2, y2 = r
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="#00ff00", width=2, tags="sel")
        # 8 resize handles: 4 corners + 4 edge midpoints
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        s = self.HANDLE_SIZE
        for hx, hy in [(x1, y1), (mx, y1), (x2, y1), (x2, my),
                       (x2, y2), (mx, y2), (x1, y2), (x1, my)]:
            self.canvas.create_rectangle(hx - s, hy - s, hx + s, hy + s,
                                         fill="#00ff00", outline="#003300", tags="sel")

    def _hit_test(self, ex, ey):
        """Returns a handle name ('n','s','e','w','ne','nw','se','sw'), 'move' (inside), or None."""
        r = self._sel_display_rect()
        if r is None:
            return None
        x1, y1, x2, y2 = r
        cx, cy = self.canvas.canvasx(ex), self.canvas.canvasy(ey)
        tol = self.HIT_TOLERANCE

        in_x = (x1 - tol) <= cx <= (x2 + tol)
        in_y = (y1 - tol) <= cy <= (y2 + tol)
        if in_x and in_y:
            horiz = "w" if abs(cx - x1) <= tol else "e" if abs(cx - x2) <= tol else ""
            vert = "n" if abs(cy - y1) <= tol else "s" if abs(cy - y2) <= tol else ""
            if horiz or vert:
                return vert + horiz          # e.g. "ne", "s", "w"
            if x1 < cx < x2 and y1 < cy < y2:
                return "move"
        return None

    _CURSORS = {
        "nw": "top_left_corner", "se": "bottom_right_corner",
        "ne": "top_right_corner", "sw": "bottom_left_corner",
        "n": "sb_v_double_arrow", "s": "sb_v_double_arrow",
        "e": "sb_h_double_arrow", "w": "sb_h_double_arrow",
        "move": "fleur", None: "cross",
    }

    def _set_cursor(self, hit):
        try:
            self.canvas.config(cursor=self._CURSORS.get(hit, "cross"))
        except tk.TclError:
            self.canvas.config(cursor="cross")

    def on_motion(self, event):
        if self.drag_mode is None:
            self._set_cursor(self._hit_test(event.x, event.y))

    def _update_sel_info(self):
        if self.sel_rect is None:
            return
        x1, y1, x2, y2 = self.sel_rect
        self.info_label.config(
            text=f"Selection: {int(x2 - x1)} x {int(y2 - y1)} px at ({int(x1)}, {int(y1)})  "
                 f"-  drag handles to resize, drag inside to move")

    def on_press(self, event):
        x, y = self._to_work_coords(event.x, event.y)
        hit = self._hit_test(event.x, event.y)
        if hit == "move":
            self.drag_mode = "move"
            self._drag_origin = (x, y)
            self._rect_at_start = list(self.sel_rect)
        elif hit:
            self.drag_mode = "resize"
            self._drag_handle = hit
            self._rect_at_start = list(self.sel_rect)
        else:
            self.drag_mode = "new"
            self._anchor = (x, y)
            self.sel_rect = [x, y, x, y]
            self.draw_selection()

    def on_drag(self, event):
        if self.drag_mode is None:
            return
        W, H = self.work_image.size
        x, y = self._to_work_coords(event.x, event.y)
        MIN = 4.0  # minimum selection size in image pixels

        if self.drag_mode == "new":
            ax, ay = self._anchor
            self.sel_rect = [min(ax, x), min(ay, y), max(ax, x), max(ay, y)]

        elif self.drag_mode == "move":
            rx1, ry1, rx2, ry2 = self._rect_at_start
            rw, rh = rx2 - rx1, ry2 - ry1
            dx, dy = x - self._drag_origin[0], y - self._drag_origin[1]
            nx1 = max(0.0, min(W - rw, rx1 + dx))
            ny1 = max(0.0, min(H - rh, ry1 + dy))
            self.sel_rect = [nx1, ny1, nx1 + rw, ny1 + rh]

        elif self.drag_mode == "resize":
            x1, y1, x2, y2 = self._rect_at_start
            h = self._drag_handle
            if "w" in h:
                x1 = min(x, x2 - MIN)
            if "e" in h:
                x2 = max(x, x1 + MIN)
            if "n" in h:
                y1 = min(y, y2 - MIN)
            if "s" in h:
                y2 = max(y, y1 + MIN)
            self.sel_rect = [x1, y1, x2, y2]

        self.draw_selection()
        self._update_sel_info()

    def on_release(self, event):
        if self.drag_mode == "new" and self.sel_rect is not None:
            x1, y1, x2, y2 = self.sel_rect
            if (x2 - x1) < 3 or (y2 - y1) < 3:      # a plain click, not a real drag
                self.sel_rect = None
                self.draw_selection()
        self.drag_mode = None
        self._set_cursor(self._hit_test(event.x, event.y))

    def confirm_crop(self):
        if self.sel_rect is None:
            messagebox.showwarning("No selection", "Drag a rectangle on the image first.")
            return

        w, h = self.work_image.size
        x1, y1, x2, y2 = self.sel_rect

        ox1, oy1 = max(0, int(x1)), max(0, int(y1))
        ox2, oy2 = min(w, int(x2)), min(h, int(y2))

        if (ox2 - ox1) < 3 or (oy2 - oy1) < 3:
            messagebox.showwarning("Selection too small", "Drag a larger rectangle.")
            return

        cropped = self.work_image.crop((ox1, oy1, ox2, oy2))
        self.on_crop_done(cropped)
        self.destroy()


class LocatorRotateApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Locator + Rotate Tool")
        self.root.geometry("1250x820")

        self.template_pil = None
        self.scene_original_pil = None
        self.scene_angle = 0.0
        self.scene_base_angle = 0.0
        self.flip_h = False
        self.flip_v = False

        self.result_cv = None
        self.object_angle = None       # detected orientation of the object in the current scene view
        self.object_angle_key = None   # scene state the angle was measured for
        self.object_dist = 0.0         # distance image-center -> object-center (px)
        self.det = None                # geometry of the last detection (moved along with the scene on rotate)
        self.image_center = (0.0, 0.0)
        self.object_center = (0.0, 0.0)
        self.target_angle_var = tk.StringVar(value="90")

        self.build_ui()

    # ---------------- UI ----------------
    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(pady=6)

        tk.Button(top, text="1. Load Object (Template)", width=24,
                  command=self.load_template).grid(row=0, column=0, padx=5)
        tk.Button(top, text="Crop Image → Template", width=20, bg="#e8a33d",
                  command=self.crop_image_to_template).grid(row=0, column=1, padx=5)
        tk.Button(top, text="2. Load Scene Image", width=22,
                  command=self.load_scene).grid(row=0, column=2, padx=5)
        tk.Button(top, text="3. Detect Object", width=18, bg="#4f8cff", fg="white",
                  command=self.run_detect).grid(row=0, column=3, padx=5)
        tk.Button(top, text="Save Result", width=14,
                  command=self.save_result).grid(row=0, column=4, padx=5)

        rot = tk.Frame(self.root)
        rot.pack(pady=4)

        tk.Button(rot, text="⟲ -90°", width=8, command=lambda: self.quick_rotate(-90)).grid(row=0, column=0, padx=4)
        tk.Button(rot, text="⟳ +90°", width=8, command=lambda: self.quick_rotate(90)).grid(row=0, column=1, padx=4)
        tk.Button(rot, text="Flip H", width=8, command=self.flip_horizontal).grid(row=0, column=2, padx=4)
        tk.Button(rot, text="Flip V", width=8, command=self.flip_vertical).grid(row=0, column=3, padx=4)
        tk.Button(rot, text="Reset Rotation", width=14, command=self.reset_rotation).grid(row=0, column=4, padx=4)

        tk.Label(rot, text="Fine angle:").grid(row=0, column=5, padx=(20, 4))
        self.angle_slider = tk.Scale(rot, from_=-180, to=180, orient="horizontal",
                                      length=350, command=self.on_slider_change)
        self.angle_slider.grid(row=0, column=6, padx=4)
        self.angle_label = tk.Label(rot, text="0°")
        self.angle_label.grid(row=0, column=7, padx=4)

        # --- Object orientation: type an angle, click the button ---
        orient = tk.Frame(self.root)
        orient.pack(pady=4)
        tk.Label(orient, text="Target angle (°):", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=(0, 4))
        angle_entry = tk.Entry(orient, textvariable=self.target_angle_var, width=8, justify="center")
        angle_entry.grid(row=0, column=1, padx=4)
        angle_entry.bind("<Return>", lambda e: self.rotate_object_to_angle())
        tk.Button(orient, text="Align Object to This Angle", width=26, bg="#2e9e5b", fg="white",
                  command=self.rotate_object_to_angle).grid(row=0, column=2, padx=8)
        self.obj_angle_label = tk.Label(orient, text="Detected object angle: —", fg="#444444")
        self.obj_angle_label.grid(row=0, column=3, padx=12)
        tk.Label(orient, fg="gray",
                 text="Angle of the line  image center → object center,  measured from the horizontal line:  "
                      "0° = right,  90° = straight up (vertical),  180° = left,  -90° = straight down"
                 ).grid(row=1, column=0, columnspan=4, pady=(2, 0))

        # --- Status + Accuracy + Detection Time ---
        status_frame = tk.Frame(self.root)
        status_frame.pack(pady=4)

        self.status_label = tk.Label(status_frame, text="Load an object image and a scene image to begin.", fg="gray")
        self.status_label.pack(side="left", padx=10)

        self.accuracy_label = tk.Label(status_frame, text="", font=("Arial", 13, "bold"))
        self.accuracy_label.pack(side="left", padx=20)

        self.time_label = tk.Label(status_frame, text="", font=("Arial", 13, "bold"), fg="#2b6fd1")
        self.time_label.pack(side="left", padx=20)

        # --- Image display area ---
        display = tk.Frame(self.root)
        display.pack(fill="both", expand=True, padx=10, pady=8)

        tk.Label(display, text="Object (Template)", font=("Arial", 10, "bold")).grid(row=0, column=0)
        tk.Label(display, text="Scene (rotated preview)", font=("Arial", 10, "bold")).grid(row=0, column=1)
        tk.Label(display, text="Detection Result", font=("Arial", 10, "bold")).grid(row=0, column=2)

        self.template_label = tk.Label(display, bg="#222222", relief="sunken")
        self.template_label.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.scene_label = tk.Label(display, bg="#222222", relief="sunken")
        self.scene_label.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        self.result_label = tk.Label(display, bg="#222222", relief="sunken")
        self.result_label.grid(row=1, column=2, sticky="nsew", padx=5, pady=5)

        display.grid_columnconfigure(0, weight=1)
        display.grid_columnconfigure(1, weight=1)
        display.grid_columnconfigure(2, weight=1)

    # ---------------- Loading ----------------
    def crop_image_to_template(self):
        path = filedialog.askopenfilename(
            title="Select an image to crop the object/template from",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not path:
            return
        source_img = Image.open(path).convert("RGB")

        def on_crop_done(cropped_pil):
            self.template_pil = cropped_pil
            self.show_pil(self.template_pil, self.template_label, max_size=(360, 650))
            self.accuracy_label.config(text="")
            self.time_label.config(text="")
            self._clear_object_angle()

            # Also save the cropped template as an image file
            src_name = os.path.splitext(os.path.basename(path))[0]
            save_path = filedialog.asksaveasfilename(
                title="Save cropped template image",
                defaultextension=".png",
                initialfile=f"{src_name}_template.png",
                filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp")]
            )
            if save_path:
                try:
                    cropped_pil.save(save_path)
                    self.status_label.config(
                        text=f"Template set from crop and saved to: {os.path.basename(save_path)}")
                except Exception as e:
                    messagebox.showerror("Save failed", f"Could not save the template:\n{e}")
                    self.status_label.config(text="Template set from crop (not saved).")
            else:
                self.status_label.config(text="Template set from crop (not saved).")

        CropWindow(self.root, source_img, on_crop_done)

    def load_template(self):
        path = filedialog.askopenfilename(
            title="Select the object (cropped template) image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not path:
            return
        self.template_pil = Image.open(path).convert("RGB")
        self.show_pil(self.template_pil, self.template_label, max_size=(360, 650))
        self.status_label.config(text=f"Object loaded: {path.split('/')[-1]}")
        self.accuracy_label.config(text="")
        self.time_label.config(text="")
        self._clear_object_angle()

    def load_scene(self):
        path = filedialog.askopenfilename(
            title="Select the scene image to search in",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not path:
            return
        self.scene_original_pil = Image.open(path).convert("RGB")
        self.scene_angle = 0.0
        self.scene_base_angle = 0.0
        self.flip_h = False
        self.flip_v = False
        self.angle_slider.set(0)
        self.status_label.config(text=f"Scene loaded: {path.split('/')[-1]}")
        self.accuracy_label.config(text="")
        self.time_label.config(text="")
        self._clear_object_angle()
        self.update_scene_preview()

    # ---------------- Rotation ----------------
    def get_base_scene(self):
        img = self.scene_original_pil
        if self.flip_h:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flip_v:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if self.scene_base_angle != 0:
            img = img.rotate(-self.scene_base_angle, expand=True)
        return img

    def get_current_rotated_scene(self):
        if self.scene_original_pil is None:
            return None
        base = self.get_base_scene()
        rotated = base.rotate(-self.scene_angle, expand=True, fillcolor=(30, 30, 30))
        return rotated

    def update_scene_preview(self):
        rotated = self.get_current_rotated_scene()
        if rotated is None:
            return
        self.show_pil(rotated, self.scene_label, max_size=(430, 650))

    def on_slider_change(self, value):
        self.scene_angle = float(value)
        self.angle_label.config(text=f"{int(self.scene_angle)}°")
        self.update_scene_preview()

    def quick_rotate(self, delta):
        if self.scene_original_pil is None:
            messagebox.showwarning("No scene", "Load a scene image first.")
            return
        self.scene_base_angle = (self.scene_base_angle + delta) % 360
        self.update_scene_preview()

    def flip_horizontal(self):
        if self.scene_original_pil is None:
            return
        self.flip_h = not self.flip_h
        self.update_scene_preview()

    def flip_vertical(self):
        if self.scene_original_pil is None:
            return
        self.flip_v = not self.flip_v
        self.update_scene_preview()

    def reset_rotation(self):
        if self.scene_original_pil is None:
            return
        self.scene_angle = 0.0
        self.scene_base_angle = 0.0
        self.flip_h = False
        self.flip_v = False
        self.angle_slider.set(0)
        self.update_scene_preview()

    # ---------------- Rotate object to a chosen angle ----------------
    def _scene_state_key(self):
        return (id(self.template_pil), id(self.scene_original_pil),
                self.flip_h, self.flip_v, self.scene_base_angle, self.scene_angle)

    def _clear_object_angle(self):
        self.object_angle = None
        self.object_angle_key = None
        self.det = None
        self.obj_angle_label.config(text="Detected object angle: —")

    def rotate_object_to_angle(self):
        """Rotate the scene so the located object's CENTER sits at the typed angle.

        How the angle is defined
            * Take the center point of the scene image.
            * Draw a horizontal line through it.
            * Draw a line from the scene center to the center of the located object.
            * The angle between the two lines is the object's position angle:
                  0 = right,  90 = straight up (vertical),  180 = left,  -90 = straight down.
        The scene is rotated about its center by exactly the difference, so the object
        center ends up on the requested line (90 -> directly above the image center).
        """
        if self.template_pil is None or self.scene_original_pil is None:
            messagebox.showwarning("Missing images", "Load both an object image and a scene image first.")
            return
        try:
            target = float(self.target_angle_var.get().strip())
        except ValueError:
            messagebox.showwarning("Invalid angle", "Type a number for the target angle, e.g. 90 or -45.")
            return

        # 1) locate the object and measure its current position angle
        if self.object_angle is None or self.object_angle_key != self._scene_state_key():
            self.run_detect()
        if self.object_angle is None:
            messagebox.showwarning("Object not found",
                                   "The object could not be located, so its position angle is unknown.")
            return
        if self.object_dist < 3:
            messagebox.showwarning("Angle undefined",
                                   "The object's center is on the image center, so it has no direction.\n"
                                   "Use a scene where the object is off-center.")
            return

        # 2) rotate the scene (about its center). Rotating the scene clockwise by c lowers the
        #    object's maths-angle by c, so the clockwise rotation needed is (current - target).
        t0 = time.perf_counter()
        cw = self.object_angle - target
        cw = ((cw + 180.0) % 360.0) - 180.0                      # shortest way round
        total = self.scene_base_angle + self.scene_angle + cw
        total = ((total + 180.0) % 360.0) - 180.0                # keep in slider range [-180, 180)
        self.scene_base_angle = 0.0
        self.scene_angle = total

        # move the slider WITHOUT firing its callback (slider is integer-step, we keep the exact float)
        self.angle_slider.config(command="")
        self.angle_slider.set(total)
        self.angle_slider.config(command=self.on_slider_change)
        self.angle_label.config(text=f"{total:.1f}°")

        # same fast PIL rotation the preview uses - rotate ONCE and reuse it for preview + result
        rotated = self.get_current_rotated_scene()
        self.show_pil(rotated, self.scene_label, max_size=(430, 650))
        new_size = rotated.size

        # 3) NO new detection: the object was already located, so just move its geometry (box corners
        #    and center) with the scene using the rotation maths, then draw it on the rotated scene.
        det = dict(self.det)
        pts = np.vstack([det["dst"], np.array(det["obj_center"])[None, :]])
        moved = self._rotate_points(pts, det["scene_size"], new_size, cw)
        det["dst"] = moved[:4]
        det["obj_center"] = (float(moved[4][0]), float(moved[4][1]))
        det["scene_size"] = new_size
        self.det = det

        cx, cy = (new_size[0] - 1) / 2.0, (new_size[1] - 1) / 2.0
        ox, oy = det["obj_center"]
        self.image_center, self.object_center = (cx, cy), (ox, oy)
        self.object_dist = float(np.hypot(ox - cx, oy - cy))
        self.object_angle = float(np.degrees(np.arctan2(-(oy - cy), ox - cx)))
        self.object_angle_key = self._scene_state_key()      # geometry is valid for this new scene state
        self.obj_angle_label.config(text=f"Detected object angle: {self.object_angle:.1f}°")

        elapsed = time.perf_counter() - t0
        time_text = self._fmt_time("Rotation Time", elapsed)
        scene_gray = np.array(rotated.convert("L"))
        result_img = self.draw_result(scene_gray, det, time_text)
        self.result_cv = result_img
        self.show_cv(result_img, self.result_label, max_size=(430, 650))
        self.time_label.config(text=time_text + " (no re-detection)")
        self.status_label.config(
            text=f"Object center aligned to {target:g}° (now {self.object_angle:.1f}°). "
                 f"Scene rotated {cw:+.1f}° (clockwise +) - moved from the previous detection, no re-detection.",
            fg="green")

    # ---------------- Display helper ----------------
    def show_pil(self, pil_img, label_widget, max_size=(400, 600)):
        img = pil_img.copy()
        img.thumbnail(max_size)
        tk_img = ImageTk.PhotoImage(img)
        label_widget.config(image=tk_img, text="")
        label_widget.image = tk_img

    def show_cv(self, cv_img_bgr, label_widget, max_size=(400, 600)):
        rgb = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.show_pil(pil_img, label_widget, max_size)

    # ---------------- Detection ----------------
    def pil_to_gray_cv(self, pil_img):
        arr = np.array(pil_img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return gray

    def preprocess(self, img_gray):
        img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_gray)

    def set_accuracy_display(self, accuracy_pct):
        """Color-coded accuracy label: green (high), orange (medium), red (low)."""
        if accuracy_pct >= 80:
            color = "#1a9c1a"
        elif accuracy_pct >= 50:
            color = "#d98c00"
        else:
            color = "#cc0000"
        self.accuracy_label.config(text=f"Matching Accuracy: {accuracy_pct:.1f}%", fg=color)

    def set_time_display(self, elapsed_seconds):
        """Shows how long the detection pipeline took, in seconds (ms if very fast)."""
        if elapsed_seconds < 1.0:
            text = f"Detection Time: {elapsed_seconds * 1000:.0f} ms"
        else:
            text = f"Detection Time: {elapsed_seconds:.3f} s"
        self.time_label.config(text=text)

    @staticmethod
    def _fmt_time(label, seconds):
        return f"{label}: {seconds * 1000:.0f} ms" if seconds < 1.0 else f"{label}: {seconds:.3f} s"

    @staticmethod
    def _rotate_points(pts, old_size, new_size, cw_deg):
        """Where do points (N,2) of the OLD scene image land in the NEW scene image, after the scene
        was rotated clockwise by cw_deg about its center (canvas expanded, as PIL's rotate(expand=True))?
        Pure maths - no image processing, no detection."""
        th = np.radians(cw_deg)
        c, s = np.cos(th), np.sin(th)
        old_c = np.array([(old_size[0] - 1) / 2.0, (old_size[1] - 1) / 2.0])
        new_c = np.array([(new_size[0] - 1) / 2.0, (new_size[1] - 1) / 2.0])
        rel = pts - old_c
        x = rel[:, 0] * c - rel[:, 1] * s      # y axis points down, so +angle = clockwise on screen
        y = rel[:, 0] * s + rel[:, 1] * c
        return np.stack([x, y], axis=1) + new_c

    def draw_result(self, scene_gray, det, time_text):
        """Draw the located object (box), its center, the horizontal line through the image center,
        the image-center -> object-center line and the position angle on a gray scene image."""
        img_h, img_w = scene_gray.shape
        dst = det["dst"]
        ox, oy = det["obj_center"]
        cx, cy = (img_w - 1) / 2.0, (img_h - 1) / 2.0
        dx, dy = ox - cx, oy - cy
        angle = float(np.degrees(np.arctan2(-dy, dx)))
        dist = float(np.hypot(dx, dy))
        accuracy_pct, inliers, n_good = det["accuracy"], det["inliers"], det["n_good"]

        result_img = cv2.cvtColor(scene_gray, cv2.COLOR_GRAY2BGR)
        cv2.polylines(result_img, [np.int32(dst).reshape(-1, 1, 2)], True, (0, 255, 0), 3, cv2.LINE_AA)

        tr_x, tr_y = int(dst[3][0]), int(dst[3][1])
        cv2.circle(result_img, (tr_x, tr_y), 8, (0, 0, 255), -1)
        cv2.putText(result_img, "Top-Right", (tr_x + 12, tr_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # horizontal reference line through the image center + line from image center to object center
        p_c = (int(round(cx)), int(round(cy)))
        p_o = (int(round(ox)), int(round(oy)))
        cv2.line(result_img, (0, p_c[1]), (img_w - 1, p_c[1]), (255, 200, 0), 1, cv2.LINE_AA)
        cv2.line(result_img, p_c, p_o, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(result_img, p_c, 6, (255, 200, 0), -1)
        cv2.circle(result_img, p_o, 6, (0, 255, 255), -1)
        radius = int(max(15, min(60, dist * 0.5)))
        cv2.ellipse(result_img, p_c, (radius, radius), 0, min(-angle, 0), max(-angle, 0),
                    (0, 255, 255), 2, cv2.LINE_AA)

        # text overlay (also saved with the image)
        acc_color = (0, 200, 0) if accuracy_pct >= 80 else (0, 165, 255) if accuracy_pct >= 50 else (0, 0, 255)
        cv2.putText(result_img, f"Matching Accuracy: {accuracy_pct:.1f}%", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, acc_color, 2, cv2.LINE_AA)
        cv2.putText(result_img, f"Inliers: {inliers}/{n_good}", (15, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, acc_color, 2, cv2.LINE_AA)
        cv2.putText(result_img, time_text, (15, 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 220, 100), 2, cv2.LINE_AA)
        cv2.putText(result_img, f"Position angle: {angle:.1f} deg", (15, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        return result_img

    def run_detect(self):
        if self.template_pil is None or self.scene_original_pil is None:
            messagebox.showwarning("Missing images", "Load both an object image and a scene image first.")
            return

        self.status_label.config(text="Detecting...")
        self.accuracy_label.config(text="")
        self.time_label.config(text="")
        self._clear_object_angle()
        self.root.update_idletasks()
        start_time = time.perf_counter()

        rotated_scene_pil = self.get_current_rotated_scene()

        template_gray = self.pil_to_gray_cv(self.template_pil)
        target_gray = self.pil_to_gray_cv(rotated_scene_pil)

        template_p = self.preprocess(template_gray)
        target_p = self.preprocess(target_gray)

        sift = cv2.SIFT_create(nfeatures=0, contrastThreshold=0.02, edgeThreshold=15)
        kp1, des1 = sift.detectAndCompute(template_p, None)
        kp2, des2 = sift.detectAndCompute(target_p, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            elapsed = time.perf_counter() - start_time
            self.status_label.config(text="FAILED: not enough keypoints detected.", fg="red")
            self.result_label.config(image="", text="Not enough features")
            self.set_accuracy_display(0.0)
            self.set_time_display(elapsed)
            return

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=100)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        raw_matches = flann.knnMatch(des1, des2, k=2)

        good_matches = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        MIN_MATCHES = 8
        if len(good_matches) < MIN_MATCHES:
            elapsed = time.perf_counter() - start_time
            self.status_label.config(
                text=f"FAILED: only {len(good_matches)} good matches (need {MIN_MATCHES}).", fg="red")
            self.result_label.config(image="", text="Object not found")
            self.set_accuracy_display(0.0)
            self.set_time_display(elapsed)
            return

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            elapsed = time.perf_counter() - start_time
            self.status_label.config(text="FAILED: homography could not be computed.", fg="red")
            self.result_label.config(image="", text="Object not found")
            self.set_accuracy_display(0.0)
            self.set_time_display(elapsed)
            return

        inliers = int(mask.sum())
        # --- MATCHING ACCURACY: % of good matches that agree with the homography ---
        accuracy_pct = (inliers / len(good_matches)) * 100.0

        if inliers < MIN_MATCHES:
            elapsed = time.perf_counter() - start_time
            self.status_label.config(text=f"FAILED: only {inliers} inliers.", fg="red")
            self.result_label.config(image="", text="Object not found")
            self.set_accuracy_display(accuracy_pct)
            self.set_time_display(elapsed)
            return

        h, w = template_gray.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, H)

        # --- Object POSITION angle -------------------------------------------------
        # Line 1: horizontal line through the CENTER of the scene image.
        # Line 2: from the scene center to the CENTER of the located object.
        # Angle between them (maths convention): 0 = right, 90 = straight up,
        # 180 = left, -90 = straight down.
        img_h, img_w = target_gray.shape
        cx, cy = (img_w - 1) / 2.0, (img_h - 1) / 2.0
        oc = cv2.perspectiveTransform(np.float32([[[w / 2.0, h / 2.0]]]), H)[0][0]
        ox, oy = float(oc[0]), float(oc[1])
        dx, dy = ox - cx, oy - cy
        angle = float(np.degrees(np.arctan2(-dy, dx)))       # -dy because image y points down
        self.object_angle = angle
        self.object_dist = float(np.hypot(dx, dy))
        self.image_center = (cx, cy)
        self.object_center = (ox, oy)
        self.object_angle_key = self._scene_state_key()
        self.obj_angle_label.config(text=f"Detected object angle: {angle:.1f}°")

        # Detection pipeline is done at this point -- stop the clock before drawing
        elapsed = time.perf_counter() - start_time

        tr_x, tr_y = int(dst[3][0][0]), int(dst[3][0][1])
        # Keep the detected geometry so it can be MOVED with the scene later (no need to detect again)
        self.det = dict(dst=dst.reshape(4, 2).astype(np.float64), obj_center=(ox, oy),
                        inliers=inliers, n_good=len(good_matches), accuracy=accuracy_pct,
                        scene_size=(img_w, img_h))
        result_img = self.draw_result(target_gray, self.det, self._fmt_time("Detection Time", elapsed))

        self.result_cv = result_img
        self.show_cv(result_img, self.result_label, max_size=(430, 650))
        self.set_accuracy_display(accuracy_pct)
        self.set_time_display(elapsed)
        self.status_label.config(
            text=f"FOUND — {inliers}/{len(good_matches)} inliers. "
                 f"Top-right corner at ({tr_x},{tr_y}). "
                 f"Position angle: {angle:.1f}°. "
                 f"Scene angle used: {self.scene_base_angle + self.scene_angle}°",
            fg="green")

    def save_result(self):
        if self.result_cv is None:
            messagebox.showinfo("Nothing to save", "Run 'Detect Object' first and get a successful match.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg",
                                             filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
                                             initialfile="result.jpg")
        if path:
            cv2.imwrite(path, self.result_cv)
            messagebox.showinfo("Saved", f"Result saved to:\n{path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LocatorRotateApp(root)
    root.mainloop()