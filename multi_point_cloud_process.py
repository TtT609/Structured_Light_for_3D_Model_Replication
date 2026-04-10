import numpy as np
import cv2
import scipy.io
import os
import glob
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ==========================================
# CORE PROCESSING MATH (From process_cloud.py)
# ==========================================
def load_calibration(calib_path):
    data = scipy.io.loadmat(calib_path)
    return {
        "Nc": data["Nc"],
        "Oc": data["Oc"],
        "wPlaneCol": data["wPlaneCol"],
        "wPlaneRow": data["wPlaneRow"],
        "cam_K": data["cam_K"]
    }

def gray_decode(source, n_sets_col=11, n_sets_row=11, n_cols=1920, n_rows=1080,
                thresh_mode='otsu', shadow_val=40, contrast_val=10):
    """
    Decode structured-light Gray-code images.

    Parameters
    ----------
    source      : str (folder path) OR list[str] (sorted list of image paths)
    n_sets_col  : how many column pattern pairs to use  (default 11, max = ceil(log2(n_cols)))
    n_sets_row  : how many row pattern pairs to use     (default 11, max = ceil(log2(n_rows)))
    n_cols      : projector width  (pixels)
    n_rows      : projector height (pixels)

    Image order inside a scan folder / file list:
        index 0  : all-white  (reference bright)
        index 1  : all-black  (reference dark / shadow mask)
        index 2,3: column bit 0 positive + inverse
        index 4,5: column bit 1 positive + inverse
        ...
        index 2 + 2*(n_col_bits-1), 2 + 2*(n_col_bits-1)+1 : column bit n_col_bits-1
        next pairs: row bits (same pattern)
    """

    # ------------------------------------------------------------------
    # 1. Collect the file list
    # ------------------------------------------------------------------
    if isinstance(source, list):
        files = source                          # already a sorted list
    else:
        files = sorted(glob.glob(os.path.join(source, "*.bmp")))
        if not files:
            files = sorted(glob.glob(os.path.join(source, "*.png")))

    if len(files) < 4:
        raise ValueError(f"Not enough images (got {len(files)}, need at least 4).")

    # ------------------------------------------------------------------
    # 2. Load reference images
    # ------------------------------------------------------------------
    img_white = cv2.imread(files[0], 0).astype(np.float32)
    img_black = cv2.imread(files[1], 0).astype(np.float32)

    height, width = img_white.shape

    if thresh_mode == 'otsu':
        img_uint8 = img_white.astype(np.uint8)
        otsu_s_val, _ = cv2.threshold(img_uint8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask_shadow = img_white > otsu_s_val
        
        diff_img = np.clip(img_white - img_black, 0, 255).astype(np.uint8)
        otsu_c_val, _ = cv2.threshold(diff_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask_contrast = (img_white - img_black) > otsu_c_val
    else:
        # Manual thresholds
        mask_shadow   = img_white > shadow_val
        mask_contrast = (img_white - img_black) > contrast_val
        
    valid_mask = mask_shadow & mask_contrast

    # ------------------------------------------------------------------
    # 3. Clamp requested sets to the theoretical maximum
    # ------------------------------------------------------------------
    max_col_bits = int(np.ceil(np.log2(n_cols)))
    max_row_bits = int(np.ceil(np.log2(n_rows)))

    n_col_bits = min(int(n_sets_col), max_col_bits)
    n_row_bits = min(int(n_sets_row), max_row_bits)

    current_idx = 2          # first pattern image pair starts here

    # ------------------------------------------------------------------
    # 4. Generic Gray-code decoder – reads n_bits pairs from files
    # ------------------------------------------------------------------
    def decode_sequence(n_bits):
        nonlocal current_idx
        gray_val = np.zeros((height, width), dtype=np.int32)

        for b in range(n_bits):
            if current_idx + 1 >= len(files):
                break                              # ran out of images → stop early
            p_path = files[current_idx];     current_idx += 1
            i_path = files[current_idx];     current_idx += 1

            img_p = cv2.imread(p_path, 0).astype(np.float32)
            img_i = cv2.imread(i_path, 0).astype(np.float32)

            bit = np.zeros((height, width), dtype=np.int32)
            bit[img_p > img_i] = 1
            gray_val = np.bitwise_or(gray_val, np.left_shift(bit, (n_bits - 1 - b)))

        # Gray → binary
        mask = np.right_shift(gray_val, 1)
        while np.any(mask > 0):
            gray_val = np.bitwise_xor(gray_val, mask)
            mask = np.right_shift(mask, 1)

        return gray_val

    col_map = decode_sequence(n_col_bits)

    # Skip remaining column pairs so row decode starts at the right file index
    skipped_col_pairs = max_col_bits - n_col_bits
    current_idx += skipped_col_pairs * 2

    row_map = decode_sequence(n_row_bits)

    # CRITICAL: scale decoded values back to full projector coordinate range.
    # Without this, using 9 bits gives values 0-511 instead of 0-1919,
    # causing wrong plane-equation lookups and broken geometry.
    col_scale = 1 << (max_col_bits - n_col_bits)  # 2^(11-n_col_bits)
    row_scale = 1 << (max_row_bits - n_row_bits)
    col_map = col_map * col_scale
    row_map = row_map * row_scale

    return col_map, row_map, valid_mask, cv2.imread(files[0])


def reconstruct_point_cloud(col_map, row_map, mask, texture, calib,
                            row_mode=1, epipolar_tol=2.0):
    Nc = calib["Nc"]
    Oc = calib["Oc"]
    wPlaneCol = calib["wPlaneCol"]

    if wPlaneCol.shape[0] == 4: wPlaneCol = wPlaneCol.T

    h, w = col_map.shape
    col_flat  = col_map.flatten()
    mask_flat = mask.flatten()
    tex_flat  = texture.reshape(-1, 3)

    valid_indices = np.where(mask_flat)[0]

    if Nc.shape[1] == h * w:
        rays = Nc[:, valid_indices]
    else:
        K = calib["cam_K"]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        y_v, x_v = np.unravel_index(valid_indices, (h, w))
        x_n = (x_v - cx) / fx
        y_n = (y_v - cy) / fy
        z_n = np.ones_like(x_n)

        rays = np.stack((x_n, y_n, z_n))
        norms = np.linalg.norm(rays, axis=0)
        rays /= norms

    proj_cols = col_flat[valid_indices]
    proj_cols = np.clip(proj_cols, 0, wPlaneCol.shape[0] - 1)

    planes_col = wPlaneCol[proj_cols, :]
    N_col = planes_col[:, 0:3].T
    d_col = planes_col[:, 3]

    denom_col = np.sum(N_col * rays, axis=0)
    numer_col = np.dot(N_col.T, Oc).flatten() + d_col

    valid_intersect_col = np.abs(denom_col) > 1e-6

    # Avoid divide by zero by safely calculating t_col
    t_col = np.zeros_like(denom_col)
    t_col[valid_intersect_col] = -numer_col[valid_intersect_col] / denom_col[valid_intersect_col]

    if row_mode == 0:
        # Mode 0: Ignore Rows
        valid_final = valid_intersect_col
        t_final = t_col[valid_final]
        
        rays_valid = rays[:, valid_final]
        P = Oc + rays_valid * t_final
        C = tex_flat[valid_indices[valid_final]]
        return P.T, C
        
    # Mode 1 & 2 both need the row planes
    wPlaneRow = calib["wPlaneRow"]
    if wPlaneRow.shape[0] == 4: wPlaneRow = wPlaneRow.T
    row_flat = row_map.flatten()
    proj_rows = row_flat[valid_indices]
    proj_rows = np.clip(proj_rows, 0, wPlaneRow.shape[0] - 1)
    
    planes_row = wPlaneRow[proj_rows, :]
    N_row = planes_row[:, 0:3].T
    d_row = planes_row[:, 3]

    if row_mode == 1:
        # Mode 1: Epipolar Filter
        P_temp = Oc + rays * t_col
        dist_to_row = np.abs(np.sum(N_row * P_temp, axis=0) + d_row)
        valid_epipolar = dist_to_row < epipolar_tol
        
        valid_final = valid_intersect_col & valid_epipolar
        t_final = t_col[valid_final]
        
        rays_valid = rays[:, valid_final]
        P = Oc + rays_valid * t_final
        C = tex_flat[valid_indices[valid_final]]
        return P.T, C
        
    elif row_mode == 2:
        # Mode 2: Merge Point Clouds
        # Generate the column point cloud independently
        rays_col = rays[:, valid_intersect_col]
        t_col_valid = t_col[valid_intersect_col]
        P_col = Oc + rays_col * t_col_valid
        C_col = tex_flat[valid_indices[valid_intersect_col]]
        
        # Generate the row point cloud independently
        denom_row = np.sum(N_row * rays, axis=0)
        numer_row = np.dot(N_row.T, Oc).flatten() + d_row
        valid_intersect_row = np.abs(denom_row) > 1e-6
        
        # Use safety buffer to prevent divide by zero
        t_row = np.zeros_like(denom_row)
        t_row[valid_intersect_row] = -numer_row[valid_intersect_row] / denom_row[valid_intersect_row]
        
        rays_row = rays[:, valid_intersect_row]
        t_row_valid = t_row[valid_intersect_row]
        P_row = Oc + rays_row * t_row_valid
        C_row = tex_flat[valid_indices[valid_intersect_row]]
        
        # Merge both arrays
        P_merged = np.hstack((P_col, P_row))
        C_merged = np.vstack((C_col, C_row))
        return P_merged.T, C_merged


def save_ply(points, colors, filename):
    with open(filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")

        for i in range(len(points)):
            p = points[i]
            c = colors[i]
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[2]} {c[1]} {c[0]}\n")


# ==========================================
# HELPERS
# ==========================================

def _spinbox_clamp(var, lo, hi):
    """Clamp a StringVar that backs a Spinbox to [lo, hi]."""
    try:
        v = int(var.get())
    except ValueError:
        v = lo
    var.set(str(max(lo, min(hi, v))))


# ==========================================
# GUI APPLICATION
# ==========================================
class BatchCloudApp:
    MAX_COL_SETS = 11   # ceil(log2(1920))
    MAX_ROW_SETS = 11   # ceil(log2(1080))

    def __init__(self, root):
        self.root = root
        self.root.title("Batch Point Cloud Generator")
        self.root.geometry("660x680")

        # ---------- persistent variables ----------
        self.calib_file  = tk.StringVar()
        self.input_path  = tk.StringVar()
        self.mode        = tk.StringVar(value="folder")   # 'folder' | 'files'
        self.batch_mode  = tk.BooleanVar(value=False)     # single vs batch-parent

        # Pattern-set selectors
        self.col_sets_var = tk.StringVar(value=str(self.MAX_COL_SETS))
        self.row_sets_var = tk.StringVar(value=str(self.MAX_ROW_SETS))
        
        # Row Processing Mode
        self.row_mode_var = tk.IntVar(value=1) # 0=None, 1=Epipolar, 2=Merge
        self.epipolar_tol_var = tk.StringVar(value="2.0")

        # Thresholds
        self.thresh_mode_var = tk.StringVar(value="otsu")  # otsu or manual
        self.shadow_val_var = tk.StringVar(value="40")
        self.contrast_val_var = tk.StringVar(value="10")

        self.selected_files = []   # used when mode == 'files'

        self.setup_ui()

    # ------------------------------------------------------------------
    # UI BUILD
    # ------------------------------------------------------------------
    def setup_ui(self):
        pad = {"padx": 10, "pady": 6}

        # ── 1. Calibration ──────────────────────────────────────────────
        lf1 = ttk.LabelFrame(self.root, text="1. Calibration File (.mat)")
        lf1.pack(fill=tk.X, **pad)

        f1 = ttk.Frame(lf1)
        f1.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f1, text="Browse .mat", command=self.sel_calib).pack(side=tk.LEFT)
        ttk.Entry(f1, textvariable=self.calib_file).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # ── 2. Input mode ───────────────────────────────────────────────
        lf2 = ttk.LabelFrame(self.root, text="2. Input Source")
        lf2.pack(fill=tk.X, **pad)

        f_radio = ttk.Frame(lf2)
        f_radio.pack(fill=tk.X, padx=5, pady=4)
        ttk.Radiobutton(f_radio, text="Folder (contains images)",
                        variable=self.mode, value="folder",
                        command=self._on_mode_change).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(f_radio, text="Select specific image files",
                        variable=self.mode, value="files",
                        command=self._on_mode_change).pack(side=tk.LEFT, padx=10)

        # -- folder row --
        self.frame_folder = ttk.Frame(lf2)
        self.frame_folder.pack(fill=tk.X, padx=5, pady=2)

        ttk.Checkbutton(self.frame_folder, text="Batch (parent folder of scan subfolders)",
                        variable=self.batch_mode).pack(side=tk.LEFT, padx=4)

        f_folder_row = ttk.Frame(lf2)
        f_folder_row.pack(fill=tk.X, padx=5, pady=4)
        self.btn_folder = ttk.Button(f_folder_row, text="Select Folder",
                                     command=self.sel_input_folder)
        self.btn_folder.pack(side=tk.LEFT)
        ttk.Entry(f_folder_row, textvariable=self.input_path).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # -- files row (hidden by default) --
        self.frame_files = ttk.Frame(lf2)

        f_files_row = ttk.Frame(self.frame_files)
        f_files_row.pack(fill=tk.X, padx=5, pady=2)
        self.btn_files = ttk.Button(f_files_row, text="Select Image Files",
                                    command=self.sel_input_files)
        self.btn_files.pack(side=tk.LEFT)
        self.lbl_files_count = ttk.Label(f_files_row, text="No files selected")
        self.lbl_files_count.pack(side=tk.LEFT, padx=8)

        # ── 3. Pattern-Set Selector ──────────────────────────────────────
        lf3 = ttk.LabelFrame(
            self.root,
            text="3. Gray-Code Pattern Sets to Use  "
                 "(reduce to skip fine patterns the camera can't resolve)")
        lf3.pack(fill=tk.X, **pad)

        grid = ttk.Frame(lf3)
        grid.pack(padx=10, pady=8)

        # Column sets
        ttk.Label(grid, text="Column sets:").grid(row=0, column=0, sticky="w", padx=4)
        self._build_spinner(grid, self.col_sets_var, 1, self.MAX_COL_SETS, row=0, col=1)
        ttk.Label(grid, text=f"(max {self.MAX_COL_SETS}, uses first N column patterns)").grid(
            row=0, column=4, sticky="w", padx=6)

        # Row sets
        ttk.Label(grid, text="Row sets:").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self._build_spinner(grid, self.row_sets_var, 1, self.MAX_ROW_SETS, row=1, col=1)
        ttk.Label(grid, text=f"(max {self.MAX_ROW_SETS}, uses first N row patterns)").grid(
            row=1, column=4, sticky="w", padx=6)

        # Quick preset buttons
        preset_row = ttk.Frame(lf3)
        preset_row.pack(padx=10, pady=(0, 6))
        ttk.Label(preset_row, text="Quick presets:").pack(side=tk.LEFT, padx=4)
        for label, n in [("Use all (11)", 11), ("9", 9), ("8", 8), ("7", 7)]:
            ttk.Button(preset_row, text=label,
                       command=lambda c=n, r=n: self._apply_preset(c, r)).pack(
                side=tk.LEFT, padx=3)

        # ── 4. Row Processing Mode ───────────────────────────────────────
        lf_epio = ttk.LabelFrame(self.root, text="4. Row Processing Mode")
        lf_epio.pack(fill=tk.X, **pad)
        
        f_epi = ttk.Frame(lf_epio)
        f_epi.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(f_epi, text="Ignore (Fast)", variable=self.row_mode_var, value=0).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(f_epi, text="Epipolar Filter", variable=self.row_mode_var, value=1).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(f_epi, text="Merge Col+Row", variable=self.row_mode_var, value=2).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(f_epi, text=" |  Filter Tol (mm):").pack(side=tk.LEFT, padx=(5, 2))
        ttk.Spinbox(
            f_epi, textvariable=self.epipolar_tol_var, from_=0.1, to=50.0, increment=0.1,
            width=5, justify="center"
        ).pack(side=tk.LEFT, padx=2)

        # ── 5. Image Masking Thresholds  ─────────────────────────────────
        lf_thresh = ttk.LabelFrame(self.root, text="5. Image Masking Thresholds")
        lf_thresh.pack(fill=tk.X, **pad)
        
        f_thresh = ttk.Frame(lf_thresh)
        f_thresh.pack(fill=tk.X, padx=5, pady=5)
        
        def _toggle_manual_thresholds(*_):
            state = "normal" if self.thresh_mode_var.get() == "manual" else "disabled"
            sb_shadow.config(state=state)
            sb_contrast.config(state=state)
            
        ttk.Radiobutton(f_thresh, text="Otsu Auto (Recommended)", variable=self.thresh_mode_var, value="otsu", command=_toggle_manual_thresholds).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(f_thresh, text="Manual", variable=self.thresh_mode_var, value="manual", command=_toggle_manual_thresholds).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(f_thresh, text=" | Shadow (>):").pack(side=tk.LEFT, padx=(5, 2))
        sb_shadow = ttk.Spinbox(
            f_thresh, textvariable=self.shadow_val_var, from_=0, to=255, increment=1,
            width=4, justify="center", state="disabled"
        )
        sb_shadow.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(f_thresh, text="Contrast (>):").pack(side=tk.LEFT, padx=(10, 2))
        sb_contrast = ttk.Spinbox(
            f_thresh, textvariable=self.contrast_val_var, from_=0, to=255, increment=1,
            width=4, justify="center", state="disabled"
        )
        sb_contrast.pack(side=tk.LEFT, padx=2)

        # ── 6. Run ──────────────────────────────────────────────────────
        self.btn_run = ttk.Button(self.root, text="▶  START GENERATING PLY",
                                  command=self.start_processing)
        self.btn_run.pack(fill=tk.X, padx=20, pady=12)

        # ── 7. Log ──────────────────────────────────────────────────────
        lf4 = ttk.LabelFrame(self.root, text="Processing Log")
        lf4.pack(fill=tk.BOTH, expand=True, **pad)

        self.txt_log = tk.Text(lf4, state="disabled", height=10,
                               wrap="word", font=("Consolas", 9))
        sb = ttk.Scrollbar(lf4, orient="vertical", command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # SPINNER HELPER
    # ------------------------------------------------------------------
    def _build_spinner(self, parent, var, lo, hi, row, col):
        """Build a   [−] [value box] [+]  control."""
        ttk.Button(parent, text="−", width=2,
                   command=lambda: self._spin_step(var, lo, hi, -1)).grid(
            row=row, column=col, padx=2)

        sb = ttk.Spinbox(parent, textvariable=var, from_=lo, to=hi, width=4,
                         justify="center",
                         command=lambda: _spinbox_clamp(var, lo, hi))
        sb.grid(row=row, column=col + 1, padx=2)
        sb.bind("<FocusOut>", lambda e: _spinbox_clamp(var, lo, hi))
        sb.bind("<Return>",   lambda e: _spinbox_clamp(var, lo, hi))

        ttk.Button(parent, text="+", width=2,
                   command=lambda: self._spin_step(var, lo, hi, +1)).grid(
            row=row, column=col + 2, padx=2)

    def _spin_step(self, var, lo, hi, delta):
        try:
            v = int(var.get())
        except ValueError:
            v = lo
        var.set(str(max(lo, min(hi, v + delta))))

    def _apply_preset(self, n_col, n_row):
        self.col_sets_var.set(str(n_col))
        self.row_sets_var.set(str(n_row))

    # ------------------------------------------------------------------
    # MODE SWITCHING
    # ------------------------------------------------------------------
    def _on_mode_change(self):
        if self.mode.get() == "folder":
            self.frame_files.pack_forget()
            self.frame_folder.pack(fill=tk.X, padx=5, pady=2)
        else:
            self.frame_folder.pack_forget()
            self.frame_files.pack(fill=tk.X, padx=5, pady=2)

    # ------------------------------------------------------------------
    # FILE / FOLDER SELECTION
    # ------------------------------------------------------------------
    def sel_calib(self):
        f = filedialog.askopenfilename(filetypes=[("MAT Files", "*.mat")])
        if f:
            self.calib_file.set(f)

    def sel_input_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.input_path.set(d)

    def sel_input_files(self):
        files = filedialog.askopenfilenames(
            title="Select structured-light images (sorted order)",
            filetypes=[("Image files", "*.bmp *.png *.jpg *.jpeg"), ("All files", "*.*")]
        )
        if files:
            self.selected_files = sorted(list(files))
            self.lbl_files_count.config(
                text=f"{len(self.selected_files)} file(s) selected")
            self.log(f"Selected {len(self.selected_files)} files:")
            for p in self.selected_files:
                self.log(f"  {os.path.basename(p)}")

    # ------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------
    def log(self, message):
        self.root.after(0, self._append_log, message)

    def _append_log(self, message):
        self.txt_log.config(state="normal")
        self.txt_log.insert(tk.END, message + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state="disabled")

    # ------------------------------------------------------------------
    # PROCESSING – single scan
    # ------------------------------------------------------------------
    def process_single_folder(self, scan_dir, calib_data, n_col, n_row, row_mode, ep_tol, thresh_mode, shadow_val, contrast_val):
        ply_name = os.path.basename(scan_dir) + ".ply"
        out_path = os.path.join(scan_dir, ply_name)

        self.log(f"  -> Decoding folder: {os.path.basename(scan_dir)}  "
                 f"(col_sets={n_col}, row_sets={n_row})")
        c_map, r_map, mask, texture = gray_decode(scan_dir,
                                                   n_sets_col=n_col,
                                                   n_sets_row=n_row,
                                                   thresh_mode=thresh_mode,
                                                   shadow_val=shadow_val,
                                                   contrast_val=contrast_val)
        self._finish_point_cloud(c_map, r_map, mask, texture,
                                 calib_data, out_path, ply_name, row_mode, ep_tol)

    def process_single_files(self, file_list, calib_data, n_col, n_row, out_path, row_mode, ep_tol, thresh_mode, shadow_val, contrast_val):
        self.log(f"  -> Decoding {len(file_list)} selected files  "
                 f"(col_sets={n_col}, row_sets={n_row})")
        c_map, r_map, mask, texture = gray_decode(file_list,
                                                   n_sets_col=n_col,
                                                   n_sets_row=n_row,
                                                   thresh_mode=thresh_mode,
                                                   shadow_val=shadow_val,
                                                   contrast_val=contrast_val)
        ply_name = os.path.basename(out_path)
        self._finish_point_cloud(c_map, r_map, mask, texture,
                                 calib_data, out_path, ply_name, row_mode, ep_tol)

    def _finish_point_cloud(self, c_map, r_map, mask, texture, calib_data,
                            out_path, ply_name, row_mode, epipolar_tol):
        self.log("  -> Reconstructing 3D points...")
        points, colors = reconstruct_point_cloud(
            c_map, r_map, mask, texture, calib_data,
            row_mode=row_mode, epipolar_tol=epipolar_tol)

        self.log(f"  -> Saving {len(points)} points...")
        save_ply(points, colors, out_path)
        self.log(f"  ✔ Saved: {ply_name}\n")

    # ------------------------------------------------------------------
    # PROCESSING – main entry
    # ------------------------------------------------------------------
    def start_processing(self):
        calib   = self.calib_file.get().strip()
        mode    = self.mode.get()
        is_batch = self.batch_mode.get()

        # Validate calibration
        if not calib:
            messagebox.showerror("Error", "Please select a calibration file.")
            return
        if not os.path.exists(calib):
            messagebox.showerror("Error", "Calibration file not found.")
            return

        # Validate sets
        try:
            n_col = int(self.col_sets_var.get())
            n_row = int(self.row_sets_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid column/row set values.")
            return
        if not (1 <= n_col <= self.MAX_COL_SETS) or not (1 <= n_row <= self.MAX_ROW_SETS):
            messagebox.showerror(
                "Error",
                f"Column sets must be 1–{self.MAX_COL_SETS}, "
                f"row sets must be 1–{self.MAX_ROW_SETS}."
            )
            return

        # Validate Row Mode / Epipolar
        row_mode = self.row_mode_var.get()
        ep_tol = 2.0
        if row_mode == 1:
            try:
                ep_tol = float(self.epipolar_tol_var.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid epipolar tolerance value.")
                return

        # Validate Thresholds
        thresh_mode = self.thresh_mode_var.get()
        s_val, c_val = 40, 10
        if thresh_mode == "manual":
            try:
                s_val = int(self.shadow_val_var.get())
                c_val = int(self.contrast_val_var.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid threshold values.")
                return

        # Validate input source
        if mode == "folder":
            target = self.input_path.get().strip()
            if not target:
                messagebox.showerror("Error", "Please select an input folder.")
                return
            if not os.path.isdir(target):
                messagebox.showerror("Error", "Input folder not found.")
                return
        else:
            if not self.selected_files:
                messagebox.showerror("Error", "Please select image files.")
                return

        self.btn_run.config(state="disabled")
        self.log(f"=== Starting {mode.upper()} Processing  "
                 f"[col_sets={n_col}  row_sets={n_row}] ===")

        threading.Thread(
            target=self._process_thread,
            args=(calib, mode, is_batch, n_col, n_row, row_mode, ep_tol, thresh_mode, s_val, c_val),
            daemon=True
        ).start()

    # ------------------------------------------------------------------
    # BACKGROUND THREAD
    # ------------------------------------------------------------------
    def _process_thread(self, calib_path, mode, is_batch, n_col, n_row, row_mode, ep_tol, thresh_mode, shadow_val, contrast_val):
        try:
            self.log("Loading calibration data...")
            calib_data = load_calibration(calib_path)

            if mode == "files":
                # ── File-list mode ──────────────────────────────────────
                out_path = filedialog.asksaveasfilename(
                    title="Save PLY as",
                    defaultextension=".ply",
                    filetypes=[("PLY files", "*.ply")]
                )
                if not out_path:
                    self.log("Save cancelled.")
                    return
                self.process_single_files(
                    self.selected_files, calib_data, n_col, n_row, out_path, row_mode, ep_tol, thresh_mode, shadow_val, contrast_val)

            elif not is_batch:
                # ── Single folder mode ──────────────────────────────────
                target = self.input_path.get().strip()
                self.process_single_folder(target, calib_data, n_col, n_row, row_mode, ep_tol, thresh_mode, shadow_val, contrast_val)

            else:
                # ── Batch (parent) folder mode ──────────────────────────
                target = self.input_path.get().strip()
                subfolders = [f.path for f in os.scandir(target) if f.is_dir()]
                self.log(f"Found {len(subfolders)} subfolders.")

                success = 0
                for folder in subfolders:
                    has_imgs = (glob.glob(os.path.join(folder, "*.bmp")) or
                                glob.glob(os.path.join(folder, "*.png")))
                    if has_imgs:
                        try:
                            self.process_single_folder(folder, calib_data, n_col, n_row, row_mode, ep_tol, thresh_mode, shadow_val, contrast_val)
                            success += 1
                        except Exception as e:
                            self.log(f"  ❌ Error in {os.path.basename(folder)}: {e}\n")
                    else:
                        self.log(f"  Skipping {os.path.basename(folder)} "
                                 f"(no images found).")

                self.log(f"=== Batch complete: {success}/{len(subfolders)} succeeded ===")

            self.root.after(
                0, lambda: messagebox.showinfo("Done", "Processing complete!"))

        except Exception as e:
            self.log(f"CRITICAL ERROR: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.btn_run.config(state="normal"))


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    app = BatchCloudApp(root)
    root.mainloop()