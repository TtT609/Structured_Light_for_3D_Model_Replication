import os
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import scipy.io

from config import DEFAULT_ROOT
from sl_system import SLSystem
from processing import ProcessingLogic
from arduino import ArduinoController

# ==========================================
# GUI (Graphical User Interface)
# ==========================================
class ScannerGUI:
    # Class for creating the user interface (UI) using Tkinter
    def __init__(self, root):
        self.root = root
        self.root.title("Project 3D Scanner Suite") # Set window title
        self.root.geometry("800x700") # Initial size: 800x700 pixels
        
        # Bind functional scripts to variables within this GUI window
        self.sys = SLSystem()             # Light and scan control system
        self.processor = ProcessingLogic() # 3D graphics filtering (Open3D)
        self.arduino = ArduinoController() # Motor control via Arduino
        
        # --- State Variables (Scanner) ---
        # StringVar and IntVar used to bind values for real-time display and dynamic updates
        
        # Default save directory for calibration images
        self.calib_capture_dir = tk.StringVar(value=os.path.join(DEFAULT_ROOT, "calib"))
        # Path for the .mat file generated after calibration
        self.calib_file = tk.StringVar(value=os.path.join(DEFAULT_ROOT, "calib", "calib.mat"))
        # Number of checkerboard poses (default to 6)
        self.num_poses = tk.IntVar(value=6)
        
        # Name of the object to be scanned (used as folder name)
        self.scan_name = tk.StringVar(value="object_01")
        # Destination folder for individual scan bit images
        self.scan_capture_dir = tk.StringVar(value=os.path.join(DEFAULT_ROOT, "scans", "object_01"))
        
        # --- State Variables (Multi PLY Process - Tab 2) ---
        self.mpcp_calib_file  = tk.StringVar(value=os.path.join(DEFAULT_ROOT, "calib", "calib.mat"))
        self.mpcp_input_path  = tk.StringVar()
        self.mpcp_mode        = tk.StringVar(value="single")  # 'single' | 'files'
        self.mpcp_batch       = tk.BooleanVar(value=False)

        # How many of the FIRST (coarsest) bit-planes to use per axis (1-11, default 11 = all)
        self.mpcp_col_sets = tk.StringVar(value="11")
        self.mpcp_row_sets = tk.StringVar(value="11")
        
        # Row Processing Mode
        self.mpcp_row_mode = tk.IntVar(value=1) # 0=None, 1=Epipolar, 2=Merge
        self.mpcp_epipolar_tol = tk.StringVar(value="0.5")

        # Thresholds
        self.mpcp_thresh_mode = tk.StringVar(value="otsu") # otsu or manual
        self.mpcp_shadow_val = tk.StringVar(value="40")
        self.mpcp_contrast_val = tk.StringVar(value="10")

        self.mpcp_selected_files = []  # used when mode == 'files'
        
        # --- State Variables (Combined Processing - Tab 3) ---
        # Toggle: process a single file or an entire folder batch
        self.proc_mode = tk.StringVar(value="folder") # 'file' or 'folder'
        # Single-file mode paths
        self.proc_input_file = tk.StringVar()   # Single input .ply file path
        self.proc_output_file = tk.StringVar()  # Single output .ply file path
        # Folder-batch mode paths (original behaviour)
        self.proc_input_dir = tk.StringVar()   # Input folder (all .ply inside)
        self.proc_output_dir = tk.StringVar()  # Output folder
        
        # Checkboxes for toggling specific cleaning algorithms
        self.enable_bg_removal = tk.BooleanVar(value=True) # Enable/Disable background wall removal variable
        self.enable_outlier_removal = tk.BooleanVar(value=True) # Enable/Disable statistical noise removal variable
        self.enable_radius_outlier = tk.BooleanVar(value=False) # Enable/Disable radius noise removal variable
        self.enable_cluster = tk.BooleanVar(value=False) # Enable/Disable keeping only the largest cluster variable
        
        # BG Params (Background Removal Parameters)
        self.bg_dist_thresh = tk.DoubleVar(value=50.0) # Depth threshold from wall
        self.bg_ransac_n = tk.IntVar(value=3) # Number of random points
        self.bg_iterations = tk.IntVar(value=1000) # RANSAC iterations
        
        # Statistical Outlier Params
        self.proc_nb_neighbors = tk.IntVar(value=30)   # Number of neighbors for distance calculation
        self.proc_std_ratio = tk.DoubleVar(value=1.5)  # Standard deviation ratio for outlier threshold
        
        # Radius Outlier Params
        self.proc_radius_nb = tk.IntVar(value=100)
        self.proc_radius_r = tk.DoubleVar(value=5.0)
        
        # Cluster Params
        self.proc_cluster_eps = tk.DoubleVar(value=5.0)
        self.proc_cluster_min = tk.IntVar(value=200)
        
        # 360 Merge Params (Stitching models for 360-degree view)
        self.merge_input_dir = tk.StringVar()
        self.merge_output_file = tk.StringVar()
        self.merge_voxel = tk.DoubleVar(value=3) # Downsampling grid resolution
        
        # Merge 360 Advanced Algorithm Params
        self.merge_icp_dist = tk.DoubleVar(value=1.5) # ICP match distance multiplier
        self.merge_outlier_nb = tk.IntVar(value=20)   # Statistical outlier neighbor threshold
        self.merge_outlier_std = tk.DoubleVar(value=2.0) # Statistical outlier stddev ratio
        self.merge_sample_before = tk.IntVar(value=1) # Uniform down-sample before merge
        self.merge_sample_after = tk.IntVar(value=1)  # Uniform down-sample after merge
        self.merge_final_voxel = tk.DoubleVar(value=0.5) # Final overlapping point reduction
        # Checkbox: toggle step-by-step 3D preview popup (blocks merge between steps until window closed)
        self.merge_show_preview = tk.BooleanVar(value=False)
        # Checkbox: accumulative mode — align each scan against full merged cloud instead of previous scan only
        self.merge_accum_mode = tk.BooleanVar(value=False)
        # Checkbox: ICP fine pass — run a second tighter ICP after the coarse ICP for sub-voxel precision
        self.merge_icp_fine_pass = tk.BooleanVar(value=True)
        # Preview colour-coding: previous (accumulated) cloud and newly added cloud
        self.merge_prev_color = [0.8, 0.2, 0.2]   # default: red  (RGB 0-1)
        self.merge_new_color  = [0.2, 0.9, 0.3]   # default: green (RGB 0-1)
        # Toggle normal-based depth shading in the preview (makes point cloud look 3-D)
        self.merge_preview_shading = tk.BooleanVar(value=True)
        
        # 360 Meshing Params (Surface meshing)
        self.m360_input_ply = tk.StringVar()
        self.m360_output_stl = tk.StringVar()
        self.m360_depth = tk.IntVar(value=10) # Mesh grid calculation depth
        self.m360_trim = tk.DoubleVar(value=0.0) # Trimming level (0.0 = Watertight)
        self.m360_mode = tk.StringVar(value="radial") # Normal orientation mode (Default: Radial)
        
        # Advanced Poisson Reconstruct Params
        self.m360_width = tk.DoubleVar(value=0.0)
        self.m360_scale = tk.DoubleVar(value=1.1)
        self.m360_linear_fit = tk.BooleanVar(value=False)
        self.m360_threads = tk.IntVar(value=-1) # -1 = All cores
        
        # Normal Estimation Params
        self.m360_normal_radius = tk.DoubleVar(value=0.1)
        self.m360_normal_max_nn = tk.IntVar(value=30)
        # Save normals point cloud: checkbox + output path
        self.m360_save_normals = tk.BooleanVar(value=False)          # Enable/disable saving the normal-enriched PLY
        self.m360_normals_out = tk.StringVar()                        # Path to save the normals PLY

        # STL Reconstruction (Standard 3D modeling parameters)
        self.s_input_ply = tk.StringVar()
        self.s_output_stl = tk.StringVar()
        self.s_mode = tk.StringVar(value="watertight")
        self.s_depth = tk.IntVar(value=10)
        self.s_radii = tk.StringVar(value="1, 2, 4")
        # Centroid-based normal orientation (orients all normals to face outward from cloud center)
        self.s_centroid_orient = tk.BooleanVar(value=True)
        # Consistency pass: run orient_normals_consistent_tangent_plane(k) AFTER centroid orient
        # to propagate the outward direction through the neighborhood graph, fixing stray normals
        self.s_consistency_pass = tk.BooleanVar(value=False)   # Enable/disable consistency pass
        self.s_consistency_k = tk.IntVar(value=30)             # Number of neighbors for the pass
        # MeshLab post-processing via pymeshlab
        self.s_use_meshlab = tk.BooleanVar(value=False)
        self.s_ml_smooth_type = tk.StringVar(value="taubin")   # 'taubin' or 'laplacian'
        self.s_ml_smooth_iters = tk.IntVar(value=10)           # Number of smoothing iterations
        self.s_ml_close_holes = tk.BooleanVar(value=False)     # Fill small holes in mesh
        self.s_ml_close_max_size = tk.IntVar(value=30)         # Max hole size (edges) to close
        self.s_ml_simplify = tk.BooleanVar(value=False)        # Reduce polygon count
        self.s_ml_target_faces = tk.IntVar(value=50000)        # Target face count after simplification
        # Save normals point cloud (same feature as in 360 Meshing tab)
        self.s_save_normals = tk.BooleanVar(value=False)        # Enable/disable saving the normals PLY
        self.s_normals_out = tk.StringVar()                     # Path for the normals output PLY

        # --- State Variables (Unified Meshing & Reconstruction tab) ---
        # Normal orientation mode: 'radial', 'tangent', or 'centroid'
        self.unified_normal_mode = tk.StringVar(value="radial")
        # Reconstruction backend: 'poisson' (watertight) or 'ball_pivot' (surface)
        self.unified_recon_method = tk.StringVar(value="poisson")

        # --- State Variables (Turntable) ---
        self.tt_port = tk.StringVar() # COM Port selection
        self.tt_baud = tk.StringVar(value="115200") # Connection speed
        self.tt_degrees = tk.DoubleVar(value=30.0)# Degrees per rotation (e.g., 30)
        self.tt_turns = tk.IntVar(value=12) # Total scans (12 turns x 30 = 360 degrees)
        self.tt_status = tk.StringVar(value="Status: Idle")
        self.tt_base_name = tk.StringVar(value="Object_360")
        self.tt_save_dir = tk.StringVar(value=os.path.join(DEFAULT_ROOT, "scans_360"))

        # --- State Variables (Calib Check) ---
        self.chk_calib_file = tk.StringVar(value=os.path.join(DEFAULT_ROOT, "calib", "calib.mat"))

        # --- State Variables (Manual Merge) ---
        self.mm_input1 = tk.StringVar()
        self.mm_input2 = tk.StringVar()
        self.mm_output = tk.StringVar()
        self.mm_enable_icp = tk.BooleanVar(value=True)
        self.mm_match_mode = tk.StringVar(value="3")

        # --- Camera Mode (Web Frontend vs Android Native) ---
        self.camera_mode = tk.StringVar(value="web")  # 'web' or 'android'

        # --- TABS (Setting up program tab sheets) ---
        self.notebook = ttk.Notebook(root) # Create horizontal tab menu
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create frames for each of the 9 tabs
        self.tab_scan = ttk.Frame(self.notebook)
        self.tab_multiPCP = ttk.Frame(self.notebook)
        self.tab_proc = ttk.Frame(self.notebook)
        self.tab_merge = ttk.Frame(self.notebook)
        self.tab_mesh360 = ttk.Frame(self.notebook)   # unified Meshing & Reconstruction
        self.tab_turntable = ttk.Frame(self.notebook)
        self.tab_calib_check = ttk.Frame(self.notebook)
        self.tab_ply_inspect = ttk.Frame(self.notebook)
        self.tab_manual_merge = ttk.Frame(self.notebook)

        # Add frames to the menu
        self.notebook.add(self.tab_scan,         text="1. Scan & Generate")
        self.notebook.add(self.tab_multiPCP,     text="2. Multi .ply process")
        self.notebook.add(self.tab_proc,         text="3. Cleanup & Process")
        self.notebook.add(self.tab_merge,        text="4. Merge 360")
        self.notebook.add(self.tab_mesh360,      text="5. Meshing & Reconstruction")
        self.notebook.add(self.tab_turntable,    text="6. Auto-Scan 360")
        self.notebook.add(self.tab_calib_check,  text="7. Calib Check")
        self.notebook.add(self.tab_ply_inspect,  text="8. PLY Inspector")
        self.notebook.add(self.tab_manual_merge, text="9. Manual Merge")

        # Initialize UI components for each tab
        self.setup_scan_tab()
        self.setup_multiPCP_tab()
        self.setup_processing_tab()
        self.setup_merge_tab()
        self.setup_360_meshing_tab()   # unified meshing tab (replaces old Tab 5 + Tab 7)
        self.setup_turntable_tab()
        self.setup_calib_check_tab()
        self.setup_ply_inspect_tab()
        self.setup_manual_merge_tab()

    # ==========================================
    # GUI Layout Functions for Each Tab
    # ==========================================
    def setup_multiPCP_tab(self):
        # Tab 2: Multi .ply process — Batch Point Cloud Generator
        # Wrapped in a canvas so it can scroll if the window is small
        main_frame = self.tab_multiPCP

        canvas  = tk.Canvas(main_frame, highlightthickness=0)
        scrollb = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        root    = ttk.Frame(canvas)
        root.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        fid = canvas.create_window((0, 0), window=root, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(fid, width=e.width))
        canvas.configure(yscrollcommand=scrollb.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollb.pack(side="right", fill="y")

        def _mwheel(ev):
            try:
                if self.notebook.select() == str(self.tab_multiPCP):
                    canvas.yview_scroll(int(-1 * (ev.delta / 120)), "units")
            except Exception:
                pass
        canvas.bind_all("<MouseWheel>", _mwheel, add="+")

        ttk.Label(root, text="Batch Point Cloud Generator",
                  font=("Arial", 14, "bold")).pack(pady=10)

        # ── 1. Calibration ────────────────────────────────────────────────
        lf1 = ttk.LabelFrame(root, text="1. Calibration File (.mat)")
        lf1.pack(fill=tk.X, padx=10, pady=6)

        f1 = ttk.Frame(lf1); f1.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f1, text="Browse .mat",
                   command=lambda: self.sel_file_load(self.mpcp_calib_file, "MAT")).pack(side=tk.LEFT)
        ttk.Entry(f1, textvariable=self.mpcp_calib_file).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # ── 2. Input source ───────────────────────────────────────────────
        lf2 = ttk.LabelFrame(root, text="2. Input Source")
        lf2.pack(fill=tk.X, padx=10, pady=6)

        f_radio = ttk.Frame(lf2); f_radio.pack(fill=tk.X, padx=5, pady=4)
        ttk.Radiobutton(f_radio, text="Folder (contains images)",
                        variable=self.mpcp_mode, value="single",
                        command=self._mpcp_on_mode).pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(f_radio, text="Select specific image files",
                        variable=self.mpcp_mode, value="files",
                        command=self._mpcp_on_mode).pack(side=tk.LEFT, padx=8)

        # Folder sub-row
        self._mpcp_frame_folder = ttk.Frame(lf2)
        self._mpcp_frame_folder.pack(fill=tk.X, padx=5, pady=2)

        ttk.Checkbutton(self._mpcp_frame_folder,
                        text="Batch mode (parent folder containing multiple scan sub-folders)",
                        variable=self.mpcp_batch).pack(side=tk.LEFT, padx=4)

        f_frow = ttk.Frame(lf2); f_frow.pack(fill=tk.X, padx=5, pady=4)
        ttk.Button(f_frow, text="Select Folder",
                   command=lambda: self.sel_dir(self.mpcp_input_path)).pack(side=tk.LEFT)
        ttk.Entry(f_frow, textvariable=self.mpcp_input_path).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # File-list sub-row (hidden by default)
        self._mpcp_frame_files = ttk.Frame(lf2)
        f_files_row = ttk.Frame(self._mpcp_frame_files)
        f_files_row.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(f_files_row, text="Select Image Files",
                   command=self._mpcp_sel_files).pack(side=tk.LEFT)
        self._mpcp_lbl_files = ttk.Label(f_files_row, text="No files selected")
        self._mpcp_lbl_files.pack(side=tk.LEFT, padx=8)

        # ── 3. Pattern-Set Count ─────────────────────────────────────────
        MAX_SETS = 11

        lf3 = ttk.LabelFrame(
            root,
            text="3. Number of Pattern Sets to Use  (1 = coarsest only, 11 = all)")
        lf3.pack(fill=tk.X, padx=10, pady=6)

        desc = (
            "Gray code encodes position using ALL bit-planes together like binary digits:\n"
            "  Plane 1 (coarsest) splits the projector in half. Plane 11 (finest) into 2048 cells.\n"
            "Using FEWER planes skips the finest stripes the camera can't see.\n"
            "The coordinate is automatically scaled so 3D geometry stays correct."
        )
        ttk.Label(lf3, text=desc, foreground="#555", justify=tk.LEFT,
                  wraplength=640).pack(padx=8, pady=(4, 2))

        grid = ttk.Frame(lf3); grid.pack(padx=10, pady=6, anchor=tk.W)

        def _spin_count(parent, label, var, row):
            ttk.Label(parent, text=label, width=18).grid(
                row=row, column=0, sticky="w", padx=4)
            ttk.Button(parent, text="\u2212", width=2,
                       command=lambda: self._mpcp_step(var, 1, MAX_SETS, -1)
                       ).grid(row=row, column=1, padx=2)
            ttk.Spinbox(parent, textvariable=var, from_=1, to=MAX_SETS,
                        width=4, justify="center",
                        command=lambda v=var: self._mpcp_clamp(v, 1, MAX_SETS)
                        ).grid(row=row, column=2, padx=2)
            ttk.Button(parent, text="+", width=2,
                       command=lambda: self._mpcp_step(var, 1, MAX_SETS, +1)
                       ).grid(row=row, column=3, padx=2)
            ttk.Label(parent, text="/ 11", foreground="#777").grid(
                row=row, column=4, padx=(2, 12))

        _spin_count(grid, "Column patterns:", self.mpcp_col_sets, row=0)
        _spin_count(grid, "Row patterns:",    self.mpcp_row_sets, row=1)

        # Quick presets
        pf = ttk.Frame(lf3); pf.pack(padx=10, pady=(0, 8))
        ttk.Label(pf, text="Quick presets:").pack(side=tk.LEFT, padx=4)
        for label, n in [("All (11)", 11), ("9", 9), ("8", 8), ("7", 7), ("6", 6)]:
            ttk.Button(pf, text=label,
                       command=lambda _n=n: self._mpcp_preset(_n)
                       ).pack(side=tk.LEFT, padx=3)

        # ── 4. Row Processing Mode ───────────────────────────────────────
        lf_epio = ttk.LabelFrame(root, text="4. Row Processing Mode")
        lf_epio.pack(fill=tk.X, padx=10, pady=6)
        
        f_epi = ttk.Frame(lf_epio)
        f_epi.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Radiobutton(f_epi, text="Ignore (Fast)", variable=self.mpcp_row_mode, value=0).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(f_epi, text="Epipolar Filter", variable=self.mpcp_row_mode, value=1).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(f_epi, text="Merge Col+Row", variable=self.mpcp_row_mode, value=2).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(f_epi, text=" |  Filter Tol (mm):").pack(side=tk.LEFT, padx=(5, 2))
        ttk.Spinbox(
            f_epi, textvariable=self.mpcp_epipolar_tol, from_=0.1, to=50.0, increment=0.1,
            width=6, justify="center"
        ).pack(side=tk.LEFT, padx=2)

        # ── 5. Image Masking Thresholds ──────────────────────────────────
        lf_thresh = ttk.LabelFrame(root, text="5. Image Masking Thresholds")
        lf_thresh.pack(fill=tk.X, padx=10, pady=6)
        
        f_thresh = ttk.Frame(lf_thresh)
        f_thresh.pack(fill=tk.X, padx=5, pady=5)
        
        def _toggle_manual_thresholds(*_):
            state = "normal" if self.mpcp_thresh_mode.get() == "manual" else "disabled"
            sb_shadow.config(state=state)
            sb_contrast.config(state=state)
            
        ttk.Radiobutton(f_thresh, text="Otsu Auto (Recommended)", variable=self.mpcp_thresh_mode, value="otsu", command=_toggle_manual_thresholds).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(f_thresh, text="Manual", variable=self.mpcp_thresh_mode, value="manual", command=_toggle_manual_thresholds).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(f_thresh, text=" | Shadow (>):").pack(side=tk.LEFT, padx=(5, 2))
        sb_shadow = ttk.Spinbox(
            f_thresh, textvariable=self.mpcp_shadow_val, from_=0, to=255, increment=1,
            width=4, justify="center", state="disabled"
        )
        sb_shadow.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(f_thresh, text="Contrast (>):").pack(side=tk.LEFT, padx=(10, 2))
        sb_contrast = ttk.Spinbox(
            f_thresh, textvariable=self.mpcp_contrast_val, from_=0, to=255, increment=1,
            width=4, justify="center", state="disabled"
        )
        sb_contrast.pack(side=tk.LEFT, padx=2)

        # ── 6. Run ────────────────────────────────────────────────────────
        self.btn_run_mpcp = ttk.Button(
            root, text="▶  START GENERATING PLY", command=self.do_multi_pcp)
        self.btn_run_mpcp.pack(fill=tk.X, padx=20, pady=12)

        # ── 7. Log ────────────────────────────────────────────────────────
        lf4 = ttk.LabelFrame(root, text="Processing Log")
        lf4.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        self.txt_log_mpcp = tk.Text(lf4, state="disabled", height=10,
                                    wrap="word", font=("Consolas", 9))
        sb2 = ttk.Scrollbar(lf4, orient="vertical",
                             command=self.txt_log_mpcp.yview)
        self.txt_log_mpcp.configure(yscrollcommand=sb2.set)
        sb2.pack(side=tk.RIGHT, fill=tk.Y)
        self.txt_log_mpcp.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # -- helpers for multiPCP tab --
    def _mpcp_on_mode(self):
        if self.mpcp_mode.get() == "files":
            self._mpcp_frame_folder.pack_forget()
            self._mpcp_frame_files.pack(fill=tk.X, padx=5, pady=2)
        else:
            self._mpcp_frame_files.pack_forget()
            self._mpcp_frame_folder.pack(fill=tk.X, padx=5, pady=2)

    def _mpcp_sel_files(self):
        files = filedialog.askopenfilenames(
            title="Select structured-light images (sorted order)",
            filetypes=[("Image files", "*.png *.bmp"), ("All", "*.*")]
        )
        if files:
            self.mpcp_selected_files = sorted(list(files))
            self._mpcp_lbl_files.config(
                text=f"{len(self.mpcp_selected_files)} file(s) selected")
            self.mpcp_log(f"Selected {len(self.mpcp_selected_files)} files.")

    def _mpcp_clamp(self, var, lo, hi):
        try:  v = int(var.get())
        except ValueError: v = lo
        var.set(str(max(lo, min(hi, v))))

    def _mpcp_step(self, var, lo, hi, delta):
        try:  v = int(var.get())
        except ValueError: v = lo
        var.set(str(max(lo, min(hi, v + delta))))

    def _mpcp_preset(self, n):
        self.mpcp_col_sets.set(str(n))
        self.mpcp_row_sets.set(str(n))
    def setup_scan_tab(self):
        # Main screen for Scanning, Calibration, and Point Cloud generation
        root = self.tab_scan
        
        # Large header label at the top of the screen
        ttk.Label(root, text="3D Scanner Workflow", font=("Arial", 16, "bold")).pack(pady=10)
        # IP address label (initially display Connecting...)
        self.ip_lbl = ttk.Label(root, text="Connecting...", foreground="blue")
        self.ip_lbl.pack()
        # Pull LAN IP to display for phone connection
        self.update_ip()
        
        # --- Camera Mode Selector ---
        lf_cam_mode = ttk.LabelFrame(root, text="📷  Camera Mode")
        lf_cam_mode.pack(fill=tk.X, padx=10, pady=(5, 8))

        f_cam_radio = ttk.Frame(lf_cam_mode); f_cam_radio.pack(fill=tk.X, padx=8, pady=(6, 2))
        ttk.Radiobutton(
            f_cam_radio, text="Web Frontend  (browser, ~8MP, no app needed)",
            variable=self.camera_mode, value="web",
            command=self._update_cam_mode_label
        ).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(
            f_cam_radio, text="Android Native App  (Camera2 API, full sensor — up to 50MP)",
            variable=self.camera_mode, value="android",
            command=self._update_cam_mode_label
        ).pack(side=tk.LEFT, padx=4)

        self.cam_mode_lbl = ttk.Label(
            lf_cam_mode,
            text="",
            foreground="#0066CC",
            font=("Arial", 9, "italic"),
            justify=tk.LEFT,
        )
        self.cam_mode_lbl.pack(anchor=tk.W, padx=10, pady=(2, 6))
        self._update_cam_mode_label()   # set initial label text

        # --- Frame STEP 1: Calibrate Capture ---
        lf1 = ttk.LabelFrame(root, text="1. Calibration Capture")
        lf1.pack(fill=tk.X, padx=10, pady=5)
        
        f1_top = ttk.Frame(lf1)
        f1_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f1_top, text="Number of Poses:").pack(side=tk.LEFT)
        # Spinner field to input number of poses with arrows (locked between 3-20 poses)
        ttk.Spinbox(f1_top, from_=3, to=20, textvariable=self.num_poses, width=5).pack(side=tk.LEFT, padx=5)
        
        # Button to start capturing Calibration chessboard photos (calls function)
        ttk.Button(lf1, text="Capture Calib Images", command=self.do_calib_capture).pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(lf1, text="Save Folder:").pack(anchor=tk.W, padx=5)
        # Input/view field for selected folder (bound to calib_capture_dir)
        ttk.Entry(lf1, textvariable=self.calib_capture_dir).pack(fill=tk.X, padx=5, pady=(0,5))
        
        # --- Frame STEP 2: Calib Process ---
        lf2 = ttk.LabelFrame(root, text="2. Calibration Processing")
        lf2.pack(fill=tk.X, padx=10, pady=5)
        
        # Button to calculate and analyze camera angles based on the saved images folder
        ttk.Button(lf2, text="Compute Calibration (Select Folder)", command=self.do_calib_compute).pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(lf2, text="Result File (.mat):").pack(anchor=tk.W, padx=5)
        # Input/view field for location of .mat file
        ttk.Entry(lf2, textvariable=self.calib_file).pack(fill=tk.X, padx=5, pady=(0,5))
        
        # --- Frame STEP 3: Scan Capture ---
        lf3 = ttk.LabelFrame(root, text="3. Scan Capture")
        lf3.pack(fill=tk.X, padx=10, pady=5)
        
        f3 = ttk.Frame(lf3); f3.pack(fill=tk.X)
        ttk.Label(f3, text="Object Name:").pack(side=tk.LEFT, padx=5)
        # Object name input field. For scanning multiple items without overwriting
        ttk.Entry(f3, textvariable=self.scan_name).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Button to start projecting horizontal/vertical patterns for 3D coordinates
        ttk.Button(lf3, text="Capture Scan Images", command=self.do_scan_capture).pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(lf3, text="Scan Folder:").pack(anchor=tk.W, padx=5)
        ttk.Entry(lf3, textvariable=self.scan_capture_dir).pack(fill=tk.X, padx=5, pady=(0,5))
        
        # --- Frame STEP 4: Application Logs ---
        lf4 = ttk.LabelFrame(root, text="4. Application Logs")
        lf4.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Text box to display Log replacing the black console window
        self.txt_log_main = tk.Text(lf4, state='disabled', height=10)
        self.txt_log_main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def setup_processing_tab(self):
        # Tab 2: Clear noise, remove background walls
        main_frame = self.tab_proc
        
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        
        root = ttk.Frame(canvas)
        
        root.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        frame_id = canvas.create_window((0, 0), window=root, anchor="nw")
        
        def on_canvas_configure(e):
            canvas.itemconfig(frame_id, width=e.width)
            
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def _on_mousewheel(event):
            try:
                if self.notebook.select() == str(self.tab_proc):
                    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except Exception:
                pass
                
        canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")
        ttk.Label(root, text="Step 2: Cleanup & Process", font=("Arial", 14, "bold")).pack(pady=10)
        ttk.Label(root, text="Pipeline: Load -> Remove Background -> Remove Outliers -> Save", foreground="blue").pack()

        # ── Mode selector (Single File vs Folder Batch) ──────────────────────────
        lf_mode = ttk.LabelFrame(root, text="Input / Output Mode")
        lf_mode.pack(fill=tk.X, padx=10, pady=5)

        f_radio = ttk.Frame(lf_mode); f_radio.pack(fill=tk.X, padx=5, pady=5)
        ttk.Radiobutton(f_radio, text="Single File  (one .ply in → one .ply out)",
                        variable=self.proc_mode, value="file").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(f_radio, text="Folder Batch (all .ply in folder → output folder)",
                        variable=self.proc_mode, value="folder").pack(side=tk.LEFT, padx=10)

        # ── Single-file rows ─────────────────────────────────────────────────────
        lf_single = ttk.LabelFrame(root, text="Single File")
        lf_single.pack(fill=tk.X, padx=10, pady=2)

        f_sf_in = ttk.Frame(lf_single); f_sf_in.pack(fill=tk.X, padx=5, pady=4)
        ttk.Button(f_sf_in, text="Select Input .PLY",
                   command=lambda: self.sel_file_load(self.proc_input_file, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_sf_in, textvariable=self.proc_input_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        f_sf_out = ttk.Frame(lf_single); f_sf_out.pack(fill=tk.X, padx=5, pady=4)
        ttk.Button(f_sf_out, text="Select Output .PLY",
                   command=lambda: self.sel_file_save(self.proc_output_file, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_sf_out, textvariable=self.proc_output_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # ── Folder-batch rows ────────────────────────────────────────────────────
        lf_batch = ttk.LabelFrame(root, text="Folder Batch")
        lf_batch.pack(fill=tk.X, padx=10, pady=2)

        f_in = ttk.Frame(lf_batch); f_in.pack(fill=tk.X, padx=5, pady=4)
        ttk.Button(f_in, text="Select Input Folder",
                   command=lambda: self.sel_dir(self.proc_input_dir)).pack(side=tk.LEFT)
        ttk.Entry(f_in, textvariable=self.proc_input_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        f_out = ttk.Frame(lf_batch); f_out.pack(fill=tk.X, padx=5, pady=4)
        ttk.Button(f_out, text="Select Output Folder",
                   command=lambda: self.sel_dir(self.proc_output_dir)).pack(side=tk.LEFT)
        ttk.Entry(f_out, textvariable=self.proc_output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 1. Background Remove parameters
        lf_bg = ttk.LabelFrame(root, text="1. Background Removal (Plane Segmentation)")
        lf_bg.pack(fill=tk.X, padx=10, pady=5)
        
        # 📌 Add Checkbox to toggle intelligent background removal (Plane Segmentation)
        f_enable_bg = ttk.Frame(lf_bg)
        f_enable_bg.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(f_enable_bg, text="Enable Background Removal", variable=self.enable_bg_removal).pack(side=tk.LEFT)
        
        f_dist = ttk.Frame(lf_bg); f_dist.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_dist, text="Distance Threshold (default 50.0):").pack(side=tk.LEFT)
        ttk.Entry(f_dist, textvariable=self.bg_dist_thresh, width=10).pack(side=tk.LEFT, padx=5)
        
        f_rn = ttk.Frame(lf_bg); f_rn.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_rn, text="RANSAC n (3) & Iterations (1000):").pack(side=tk.LEFT)
        ttk.Entry(f_rn, textvariable=self.bg_ransac_n, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Entry(f_rn, textvariable=self.bg_iterations, width=8).pack(side=tk.LEFT, padx=5)
        
        # Text hint helping user understand
        bg_desc = ("Distance Thresh: Max distance a point can be from the wall plane to be considered 'wall'.\n"
                   "RANSAC n: Points sampled per iteration. Iterations: How many times to try fitting the plane.")
        ttk.Label(lf_bg, text=bg_desc, foreground="#555", justify=tk.LEFT, wraplength=550).pack(padx=5, pady=5)

        # 2. Statistical Outlier Removal group
        lf_out = ttk.LabelFrame(root, text="2. Statistical Outlier Removal")
        lf_out.pack(fill=tk.X, padx=10, pady=5)
        
        # 📌 Add Checkbox to toggle Statistical Noise Removal process
        f_enable_out = ttk.Frame(lf_out)
        f_enable_out.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(f_enable_out, text="Enable Statistical Outlier Removal", variable=self.enable_outlier_removal).pack(side=tk.LEFT)
        
        f_nb = ttk.Frame(lf_out); f_nb.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_nb, text="nb_neighbors (20):").pack(side=tk.LEFT)
        ttk.Entry(f_nb, textvariable=self.proc_nb_neighbors, width=10).pack(side=tk.LEFT, padx=5)
        
        f_std = ttk.Frame(lf_out); f_std.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_std, text="std_ratio (2.0):").pack(side=tk.LEFT)
        ttk.Entry(f_std, textvariable=self.proc_std_ratio, width=10).pack(side=tk.LEFT, padx=5)
        
        out_desc = ("nb_neighbors: Points to analyze around each point. Higher = smoother/safer but slower.\n"
                    "std_ratio: Threshold. Lower (0.5-1.0) = Aggressive removal. Higher (2.0+) = Conservative.")
        ttk.Label(lf_out, text=out_desc, foreground="#555", justify=tk.LEFT, wraplength=550).pack(padx=5, pady=5)

        # 3. Radius Outlier Removal group
        lf_rad = ttk.LabelFrame(root, text="3. Radius Outlier Removal")
        lf_rad.pack(fill=tk.X, padx=10, pady=5)
        
        f_enable_rad = ttk.Frame(lf_rad); f_enable_rad.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(f_enable_rad, text="Enable Radius Outlier Removal", variable=self.enable_radius_outlier).pack(side=tk.LEFT)
        
        f_rnb = ttk.Frame(lf_rad); f_rnb.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_rnb, text="nb_points (100):").pack(side=tk.LEFT)
        ttk.Entry(f_rnb, textvariable=self.proc_radius_nb, width=10).pack(side=tk.LEFT, padx=5)
        
        f_r = ttk.Frame(lf_rad); f_r.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_r, text="radius (5.0):").pack(side=tk.LEFT)
        ttk.Entry(f_r, textvariable=self.proc_radius_r, width=10).pack(side=tk.LEFT, padx=5)
        
        rad_desc = "Removes points that have fewer than 'nb_points' within a given 'radius'."
        ttk.Label(lf_rad, text=rad_desc, foreground="#555", justify=tk.LEFT, wraplength=550).pack(padx=5, pady=5)

        # 4. Keep only the Largest Cluster
        lf_clus = ttk.LabelFrame(root, text="4. Keep Largest Cluster (DBSCAN)")
        lf_clus.pack(fill=tk.X, padx=10, pady=5)
        
        f_enable_clus = ttk.Frame(lf_clus); f_enable_clus.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(f_enable_clus, text="Enable Largest Cluster Filter", variable=self.enable_cluster).pack(side=tk.LEFT)
        
        f_eps = ttk.Frame(lf_clus); f_eps.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_eps, text="eps radius (5.0):").pack(side=tk.LEFT)
        ttk.Entry(f_eps, textvariable=self.proc_cluster_eps, width=10).pack(side=tk.LEFT, padx=5)
        
        f_min = ttk.Frame(lf_clus); f_min.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_min, text="min_points (200):").pack(side=tk.LEFT)
        ttk.Entry(f_min, textvariable=self.proc_cluster_min, width=10).pack(side=tk.LEFT, padx=5)
        
        clus_desc = "Groups points closer than 'eps radius'. Keeps only the largest group. Removes floating fragments."
        ttk.Label(lf_clus, text=clus_desc, foreground="#555", justify=tk.LEFT, wraplength=550).pack(padx=5, pady=5)

        # Button to start Batch processing all at once
        ttk.Button(root, text="Run Processing Pipeline", command=self.do_batch_processing).pack(fill=tk.X, padx=20, pady=20)
    
    def setup_merge_tab(self):
        # Tab 3: Align models then merge into one single form
        main_frame = self.tab_merge
        
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        
        root = ttk.Frame(canvas)
        
        root.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        frame_id = canvas.create_window((0, 0), window=root, anchor="nw")
        
        def on_canvas_configure(e):
            canvas.itemconfig(frame_id, width=e.width)
            
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        def _on_mousewheel(event):
            try:
                if self.notebook.select() == str(self.tab_merge):
                    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except Exception:
                pass
                
        canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")
        
        ttk.Label(root, text="Step 3: 360 Degree Merge (Multi-view Alignment)", font=("Arial", 14, "bold")).pack(pady=10)
        
        lf_files = ttk.LabelFrame(root, text="Files")
        lf_files.pack(fill=tk.X, padx=10, pady=5)
        
        # Throw all raw scan files (in the same folder)
        f_in = ttk.Frame(lf_files); f_in.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in, text="Select Input Folder (All PLYs)", command=lambda: self.sel_dir(self.merge_input_dir)).pack(side=tk.LEFT)
        ttk.Entry(f_in, textvariable=self.merge_input_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Destination name of the processed file
        f_out = ttk.Frame(lf_files); f_out.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_out, text="Select Output File (.ply)", command=lambda: self.sel_file_save(self.merge_output_file, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_out, textvariable=self.merge_output_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        lf_param = ttk.LabelFrame(root, text="Parameters")
        lf_param.pack(fill=tk.X, padx=10, pady=5)
        
        f_vx = ttk.Frame(lf_param); f_vx.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_vx, text="Voxel Size (mm) [Default 3]:").pack(side=tk.LEFT)
        ttk.Entry(f_vx, textvariable=self.merge_voxel, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_vx, text="(Size of downsampling grid. Smaller = distincter but slower/noisier. Larger = coarse alignment.)", foreground="#555").pack(side=tk.LEFT)

        f_icp = ttk.Frame(lf_param); f_icp.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_icp, text="ICP Dist Ratio [Default 1.5]:").pack(side=tk.LEFT)
        ttk.Entry(f_icp, textvariable=self.merge_icp_dist, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_icp, text="(Multiplier for RANSAC alignment search radius. Larger = looser matching.)", foreground="#555").pack(side=tk.LEFT)

        f_onb = ttk.Frame(lf_param); f_onb.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_onb, text="Outlier Neighbors [Default 20]:").pack(side=tk.LEFT)
        ttk.Entry(f_onb, textvariable=self.merge_outlier_nb, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_onb, text="(How many nearby points needed to not be considered floating dust.)", foreground="#555").pack(side=tk.LEFT)

        f_ost = ttk.Frame(lf_param); f_ost.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_ost, text="Outlier StdDev [Default 2.0]:").pack(side=tk.LEFT)
        ttk.Entry(f_ost, textvariable=self.merge_outlier_std, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_ost, text="(Aggressiveness of noise trimming. Lower = cuts more edge points.)", foreground="#555").pack(side=tk.LEFT)

        f_sb = ttk.Frame(lf_param); f_sb.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_sb, text="Sampling Number Before [Default 1]:").pack(side=tk.LEFT)
        ttk.Entry(f_sb, textvariable=self.merge_sample_before, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_sb, text="(1 = All pts, 2 = Keep 1/2. Reduces points before matching)", foreground="#555").pack(side=tk.LEFT)

        f_sa = ttk.Frame(lf_param); f_sa.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_sa, text="Sampling Number After [Default 1]:").pack(side=tk.LEFT)
        ttk.Entry(f_sa, textvariable=self.merge_sample_after, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_sa, text="(1 = All pts, 2 = Keep 1/2. Reduces final merged points)", foreground="#555").pack(side=tk.LEFT)

        f_fvx = ttk.Frame(lf_param); f_fvx.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_fvx, text="Final Voxel Size (mm) [0 = Disable]:").pack(side=tk.LEFT)
        ttk.Entry(f_fvx, textvariable=self.merge_final_voxel, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_fvx, text="(Merges overlapping points perfectly. Default 0.5. Set 0 to keep true 100% cloud)", foreground="#555").pack(side=tk.LEFT)

        # --- Accumulative Merge Mode ---
        lf_accum = ttk.LabelFrame(lf_param, text="Registration Target")
        lf_accum.pack(fill=tk.X, padx=5, pady=(6, 2))

        f_accum = ttk.Frame(lf_accum); f_accum.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(
            f_accum,
            text="Accumulative merge mode",
            variable=self.merge_accum_mode
        ).pack(side=tk.LEFT)

        accum_desc = (
            "OFF (default): Each scan is aligned against only the immediately preceding scan\n"
            "                  e.g.  step 2 aligns  scan[2]  vs  scan[1]\n"
            "ON:              Each scan is aligned against the FULL accumulated cloud so far\n"
            "                  e.g.  step 2 aligns  scan[2]  vs  scan[0]+scan[1]\n"
            "→ Accumulative gives RANSAC/ICP much more overlap to work with (more robust),\n"
            "   but each step is slightly slower because the target grows larger."
        )
        ttk.Label(lf_accum, text=accum_desc, foreground="#555", justify=tk.LEFT, wraplength=650).pack(padx=5, pady=(0, 4))

        # ICP Fine Pass toggle (inside the same Registration Target sub-panel)
        f_fine = ttk.Frame(lf_accum); f_fine.pack(fill=tk.X, padx=5, pady=(0, 3))
        ttk.Checkbutton(
            f_fine,
            text="ICP Fine Pass (recommended ON)",
            variable=self.merge_icp_fine_pass
        ).pack(side=tk.LEFT)
        ttk.Label(
            f_fine,
            text="  Run a 2nd tighter ICP at 0.4× voxel after coarse ICP for extra sub-voxel precision.",
            foreground="#555", font=("Arial", 8, "italic")
        ).pack(side=tk.LEFT, padx=4)

        # --- Step Preview Checkbox ---
        lf_preview = ttk.LabelFrame(root, text="Step-by-Step 3D Preview")
        lf_preview.pack(fill=tk.X, padx=10, pady=5)
        
        f_prev = ttk.Frame(lf_preview); f_prev.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(
            f_prev,
            text="Show 3D model preview after each merge step",
            variable=self.merge_show_preview
        ).pack(side=tk.LEFT)
        
        prev_desc = (
            "When checked, an Open3D 3D viewer window will pop up after EACH step showing\n"
            "the previous scans (OLD colour) vs the newly added scan (NEW colour).\n"
            "⚠ The merge process PAUSES until you close each preview window."
        )
        ttk.Label(lf_preview, text=prev_desc, foreground="#555", justify=tk.LEFT, wraplength=650).pack(padx=5, pady=(0, 3))

        # --- Colour pickers + shading toggle ---
        f_colors = ttk.Frame(lf_preview); f_colors.pack(fill=tk.X, padx=5, pady=(2, 5))

        # Helper: open a colour-chooser dialog and update the stored RGB list
        def _pick_color(current_list, btn_widget, label_text):
            import tkinter.colorchooser as cc
            # Convert 0-1 floats → #RRGGBB for the initial colour
            init_hex = "#{:02x}{:02x}{:02x}".format(
                int(current_list[0]*255), int(current_list[1]*255), int(current_list[2]*255))
            result = cc.askcolor(color=init_hex, title=f"Choose {label_text} colour")
            if result and result[0]:  # result = ((r,g,b), '#hex') or (None, None)
                r, g, b = result[0]
                current_list[0] = r / 255.0
                current_list[1] = g / 255.0
                current_list[2] = b / 255.0
                btn_widget.config(bg=result[1], activebackground=result[1])

        ttk.Label(f_colors, text="Preview colours:").pack(side=tk.LEFT, padx=(0, 6))

        # OLD cloud colour button
        old_hex = "#{:02x}{:02x}{:02x}".format(
            int(self.merge_prev_color[0]*255),
            int(self.merge_prev_color[1]*255),
            int(self.merge_prev_color[2]*255))
        self._btn_old_color = tk.Button(
            f_colors, text="OLD (previous)", bg=old_hex, width=14,
            relief="raised", borderwidth=2)
        self._btn_old_color.config(
            command=lambda: _pick_color(self.merge_prev_color, self._btn_old_color, "OLD"))
        self._btn_old_color.pack(side=tk.LEFT, padx=4)

        # NEW cloud colour button
        new_hex = "#{:02x}{:02x}{:02x}".format(
            int(self.merge_new_color[0]*255),
            int(self.merge_new_color[1]*255),
            int(self.merge_new_color[2]*255))
        self._btn_new_color = tk.Button(
            f_colors, text="NEW (added)", bg=new_hex, width=14,
            relief="raised", borderwidth=2)
        self._btn_new_color.config(
            command=lambda: _pick_color(self.merge_new_color, self._btn_new_color, "NEW"))
        self._btn_new_color.pack(side=tk.LEFT, padx=4)

        # Depth shading toggle
        ttk.Checkbutton(
            f_colors,
            text="Enable depth shading (normals)",
            variable=self.merge_preview_shading
        ).pack(side=tk.LEFT, padx=(12, 4))

        shade_tip = ttk.Label(
            f_colors,
            text="(Estimates surface normals so the viewer shows realistic 3-D shading/shadow)",
            foreground="#888", font=("Arial", 8, "italic"))
        shade_tip.pack(side=tk.LEFT, padx=2)

        ttk.Button(root, text="Merge 360 Point Clouds", command=self.do_merge_360).pack(fill=tk.X, padx=20, pady=20)

    def setup_360_meshing_tab(self):
        # Tab 5 (unified): Meshing & Reconstruction — scrollable canvas
        main_frame = self.tab_mesh360

        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        root = ttk.Frame(canvas)
        root.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        frame_id = canvas.create_window((0, 0), window=root, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(frame_id, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            try:
                if self.notebook.select() == str(self.tab_mesh360):
                    canvas.yview_scroll(int(-1*(event.delta/120)), "units")
            except Exception:
                pass
        canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")

        ttk.Label(root, text="Meshing & Reconstruction  (PLY → STL)", font=("Arial", 14, "bold")).pack(pady=(8, 2))

        ttk.Label(root,
            text="Convert a point cloud (.PLY) into a 3D mesh (.STL).\n"
                 "Select your normal orientation method, reconstruction algorithm, and optional post-processing below.",
            foreground="#444", font=("Arial", 9, "italic"), justify=tk.CENTER).pack(pady=(0, 8))

        # ── 1. Files ──────────────────────────────────────────────────────────
        lf_files = ttk.LabelFrame(root, text="1. Files")
        lf_files.pack(fill=tk.X, padx=10, pady=5)

        f_in = ttk.Frame(lf_files); f_in.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in, text="Select Input .PLY",
                   command=lambda: self.sel_file_load(self.m360_input_ply, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_in, textvariable=self.m360_input_ply).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        f_out = ttk.Frame(lf_files); f_out.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_out, text="Select Output .STL",
                   command=lambda: self.sel_file_save(self.m360_output_stl, "STL")).pack(side=tk.LEFT)
        ttk.Entry(f_out, textvariable=self.m360_output_stl).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # ── 2. Normal Vector Orientation ──────────────────────────────────────
        lf_norm = ttk.LabelFrame(root, text="2. Normal Vector Orientation  (choose one)")
        lf_norm.pack(fill=tk.X, padx=10, pady=5)

        norm_intro = (
            "Normals define which direction each point 'faces'. Getting them right prevents inside-out surfaces.\n"
            "Radial/Tangent estimate normals from the cloud geometry; Centroid forces all normals away from the "
            "cloud's geometric center."
        )
        ttk.Label(lf_norm, text=norm_intro, foreground="#555", justify=tk.LEFT, wraplength=650).pack(padx=8, pady=(4, 2))

        f_norm_radios = ttk.Frame(lf_norm); f_norm_radios.pack(fill=tk.X, padx=8, pady=4)
        ttk.Radiobutton(f_norm_radios,
            text="Radial  — orients all normals outward from the cloud centre (best for 360° scans)",
            variable=self.unified_normal_mode, value="radial",
            command=self._update_unified_norm_ui).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Radiobutton(f_norm_radios,
            text="Tangent  — graph-consistency orient (better for complex/open shapes)",
            variable=self.unified_normal_mode, value="tangent",
            command=self._update_unified_norm_ui).grid(row=1, column=0, sticky="w", pady=2)
        ttk.Radiobutton(f_norm_radios,
            text="Centroid  — geometric-centre orient + optional neighbourhood consistency pass",
            variable=self.unified_normal_mode, value="centroid",
            command=self._update_unified_norm_ui).grid(row=2, column=0, sticky="w", pady=2)

        # Normal estimation params — shown for Radial/Tangent
        self._frm_norm_est = ttk.Frame(lf_norm)
        f_nr = ttk.Frame(self._frm_norm_est); f_nr.pack(fill=tk.X, padx=12, pady=2)
        ttk.Label(f_nr, text="Normal Search Radius  [Default 0.1]:", width=32).pack(side=tk.LEFT)
        ttk.Entry(f_nr, textvariable=self.m360_normal_radius, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_nr, text="(Increase if you see bubbles / inverted surfaces)", foreground="#555").pack(side=tk.LEFT)
        f_nn = ttk.Frame(self._frm_norm_est); f_nn.pack(fill=tk.X, padx=12, pady=2)
        ttk.Label(f_nn, text="Normal Max Neighbors  [Default 30]:", width=32).pack(side=tk.LEFT)
        ttk.Entry(f_nn, textvariable=self.m360_normal_max_nn, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_nn, text="(Max neighbours used when estimating each normal)", foreground="#555").pack(side=tk.LEFT)

        # Centroid params — shown for Centroid
        self._frm_centroid = ttk.Frame(lf_norm)
        ttk.Label(self._frm_centroid,
            text="    Forces all normals to face away from the geometric centre of the whole cloud.\n"
                 "    ✔ Best for closed objects scanned from all sides.",
            foreground="#555", justify=tk.LEFT).pack(anchor=tk.W, padx=12, pady=(2, 4))
        f_cp = ttk.Frame(self._frm_centroid); f_cp.pack(fill=tk.X, padx=12, pady=2)
        ttk.Checkbutton(f_cp,
            text="Consistency Pass  (propagates outward direction through neighbourhood graph after centroid orient)",
            variable=self.s_consistency_pass).pack(side=tk.LEFT)
        f_cpk = ttk.Frame(self._frm_centroid); f_cpk.pack(fill=tk.X, padx=12, pady=2)
        ttk.Label(f_cpk, text="    Neighbours (k)  [Default 30]:", width=28).pack(side=tk.LEFT)
        ttk.Entry(f_cpk, textvariable=self.s_consistency_k, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_cpk, text="(20–50 typical. Higher k = more influence per point, slower)",
                  foreground="#555").pack(side=tk.LEFT)
        ttk.Label(self._frm_centroid,
            text="    ⚠ On very noisy clouds the consistency pass may re-flip some correct normals — use with care.",
            foreground="#886600", justify=tk.LEFT).pack(anchor=tk.W, padx=12, pady=(0, 4))

        self._update_unified_norm_ui()   # set initial visibility

        # ── 3. Reconstruction Method ──────────────────────────────────────────
        lf_recon = ttk.LabelFrame(root, text="3. Reconstruction Method  (choose one)")
        lf_recon.pack(fill=tk.X, padx=10, pady=5)

        f_recon_r = ttk.Frame(lf_recon); f_recon_r.pack(fill=tk.X, padx=8, pady=4)
        ttk.Radiobutton(f_recon_r,
            text="Poisson Surface Reconstruction  (watertight closed mesh — recommended for 360° scans)",
            variable=self.unified_recon_method, value="poisson",
            command=self._update_unified_recon_ui).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Radiobutton(f_recon_r,
            text="Ball-Pivoting Reconstruction  (open surface mesh — good for partial/organic shapes; requires Centroid orientation)",
            variable=self.unified_recon_method, value="ball_pivot",
            command=self._update_unified_recon_ui).grid(row=1, column=0, sticky="w", pady=2)

        # Poisson params frame
        self._frm_poisson = ttk.Frame(lf_recon)
        for lbl, var, hint in [
            ("Poisson Depth  [Default 10]:",    self.m360_depth,    "(Octree depth — higher = more detail but slower; >12 may freeze)"),
            ("Target Width  [Default 0.0]:",    self.m360_width,    "(Finest octree cell size — leave 0.0 to use Depth instead)"),
            ("Scale Ratio  [Default 1.1]:",     self.m360_scale,    "(Ratio of reconstruction bounding box to sample bounding box)"),
            ("Threads  [Default -1]:",           self.m360_threads,  "(-1 = auto / all CPU cores)"),
            ("Density Trim  [0.0=Watertight]:", self.m360_trim,     "(>0.0 trims low-density bubbles on the outer surface)"),
        ]:
            fr = ttk.Frame(self._frm_poisson); fr.pack(fill=tk.X, padx=12, pady=2)
            ttk.Label(fr, text=lbl, width=32).pack(side=tk.LEFT)
            ttk.Entry(fr, textvariable=var, width=10).pack(side=tk.LEFT, padx=5)
            ttk.Label(fr, text=hint, foreground="#555").pack(side=tk.LEFT)
        f_lin = ttk.Frame(self._frm_poisson); f_lin.pack(fill=tk.X, padx=12, pady=2)
        ttk.Checkbutton(f_lin, text="Linear Fit Interpolation", variable=self.m360_linear_fit).pack(side=tk.LEFT)
        ttk.Label(f_lin, text="(Use linear fitting instead of default cubic interpolation)", foreground="#555").pack(side=tk.LEFT)

        # Ball-Pivot params frame
        self._frm_ballpivot = ttk.Frame(lf_recon)
        ttk.Label(self._frm_ballpivot,
            text="    Ball-Pivoting rolls a virtual ball across the cloud and stitches triangles wherever it touches 3 points.\n"
                 "    It works best on dense, clean clouds and may leave holes where the cloud is sparse.",
            foreground="#555", justify=tk.LEFT).pack(anchor=tk.W, padx=12, pady=(2, 4))
        f_bp = ttk.Frame(self._frm_ballpivot); f_bp.pack(fill=tk.X, padx=12, pady=2)
        ttk.Label(f_bp, text="Ball Radii (comma-separated mm)  [e.g. 1, 2, 4]:", width=40).pack(side=tk.LEFT)
        ttk.Entry(f_bp, textvariable=self.s_radii, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_bp, text="(Multiple radii fill gaps at different scales)", foreground="#555").pack(side=tk.LEFT)
        ttk.Label(self._frm_ballpivot,
            text="    ⚠ Ball-Pivoting requires Centroid orientation. Selecting it will auto-switch the orientation above.",
            foreground="#886600", justify=tk.LEFT).pack(anchor=tk.W, padx=12, pady=(0, 4))

        self._update_unified_recon_ui()   # set initial visibility

        # ── 4. MeshLab Post-Processing ────────────────────────────────────────
        lf_ml = ttk.LabelFrame(root, text="4. MeshLab Post-Processing  (optional — requires: pip install pymeshlab)")
        lf_ml.pack(fill=tk.X, padx=10, pady=5)

        f_ml_en = ttk.Frame(lf_ml); f_ml_en.pack(fill=tk.X, padx=8, pady=5)
        ttk.Checkbutton(f_ml_en, text="Enable MeshLab post-processing",
                        variable=self.s_use_meshlab).pack(side=tk.LEFT)
        ttk.Label(f_ml_en,
            text="  Applies smoothing, hole-filling, and simplification AFTER reconstruction.",
            foreground="#555").pack(side=tk.LEFT)

        # Smoothing
        f_smt = ttk.Frame(lf_ml); f_smt.pack(fill=tk.X, padx=12, pady=2)
        ttk.Label(f_smt, text="Smoothing Algorithm:", width=22).pack(side=tk.LEFT)
        ttk.Radiobutton(f_smt, text="Taubin  (recommended — preserves shape)",
                        variable=self.s_ml_smooth_type, value="taubin").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(f_smt, text="Laplacian  (stronger — may shrink model)",
                        variable=self.s_ml_smooth_type, value="laplacian").pack(side=tk.LEFT, padx=5)
        f_smi = ttk.Frame(lf_ml); f_smi.pack(fill=tk.X, padx=12, pady=2)
        ttk.Label(f_smi, text="Smooth Iterations  [Default 10]:", width=32).pack(side=tk.LEFT)
        ttk.Entry(f_smi, textvariable=self.s_ml_smooth_iters, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_smi, text="(Higher = smoother surface, more time)", foreground="#555").pack(side=tk.LEFT)

        # Close holes
        f_ch = ttk.Frame(lf_ml); f_ch.pack(fill=tk.X, padx=12, pady=2)
        ttk.Checkbutton(f_ch, text="Close Holes", variable=self.s_ml_close_holes).pack(side=tk.LEFT)
        ttk.Label(f_ch, text="   Max Hole Size (edges):", width=24).pack(side=tk.LEFT)
        ttk.Entry(f_ch, textvariable=self.s_ml_close_max_size, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_ch, text="(Fills openings smaller than this edge count. Default: 30)",
                  foreground="#555").pack(side=tk.LEFT)

        # Simplify
        f_simp = ttk.Frame(lf_ml); f_simp.pack(fill=tk.X, padx=12, pady=2)
        ttk.Checkbutton(f_simp, text="Simplify Mesh (Quadric Edge Collapse)",
                        variable=self.s_ml_simplify).pack(side=tk.LEFT)
        ttk.Label(f_simp, text="   Target Faces:", width=16).pack(side=tk.LEFT)
        ttk.Entry(f_simp, textvariable=self.s_ml_target_faces, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_simp, text="(Default: 50 000 — reduce if STL is too large for slicer)",
                  foreground="#555").pack(side=tk.LEFT)

        # ── 5. Save Normals PLY ───────────────────────────────────────────────
        lf_sn = ttk.LabelFrame(root, text="5. Save Normals Point Cloud  (optional debug output)")
        lf_sn.pack(fill=tk.X, padx=10, pady=5)

        f_sncb = ttk.Frame(lf_sn); f_sncb.pack(fill=tk.X, padx=8, pady=4)
        ttk.Checkbutton(f_sncb,
            text="Save point cloud with embedded normals BEFORE meshing step  (as .PLY)",
            variable=self.m360_save_normals).pack(side=tk.LEFT)
        f_snp = ttk.Frame(lf_sn); f_snp.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(f_snp, text="Select Output .PLY",
                   command=lambda: self.sel_file_save(self.m360_normals_out, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_snp, textvariable=self.m360_normals_out).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Label(lf_sn,
            text="Open the saved .PLY in CloudCompare or MeshLab to verify normals are pointing outward correctly.",
            foreground="#555", justify=tk.LEFT, wraplength=650).pack(padx=8, pady=(0, 5))

        # ── Run Button ────────────────────────────────────────────────────────
        ttk.Button(root, text="▶  Run Meshing & Reconstruction",
                   command=self.do_unified_meshing).pack(fill=tk.X, padx=20, pady=20)

    def setup_turntable_tab(self):
        # Tab 5 Automatic Arduino motor control (Turntable)
        root = self.tab_turntable
        ttk.Label(root, text="Step 5: Auto-Scan with Turntable (Arduino)", font=("Arial", 14, "bold")).pack(pady=10)
        
        # 1. Port input box
        lf_conn = ttk.LabelFrame(root, text="1. Arduino Connection")
        lf_conn.pack(fill=tk.X, padx=10, pady=5)
        
        f_p = ttk.Frame(lf_conn); f_p.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_p, text="Port:").pack(side=tk.LEFT)
        self.cb_port = ttk.Combobox(f_p, textvariable=self.tt_port, width=15)
        self.cb_port.pack(side=tk.LEFT, padx=5)
        ttk.Button(f_p, text="Refresh", command=self.refresh_ports).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_p, text="Connect", command=self.connect_arduino).pack(side=tk.LEFT, padx=5)
        
        # 2. Set rotation distance
        lf_set = ttk.LabelFrame(root, text="2. Scan Settings")
        lf_set.pack(fill=tk.X, padx=10, pady=5)
        
        f_deg = ttk.Frame(lf_set); f_deg.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_deg, text="Degrees per Turn (e.g., 30):").pack(side=tk.LEFT)
        ttk.Entry(f_deg, textvariable=self.tt_degrees, width=10).pack(side=tk.LEFT, padx=5)
        
        f_cnt = ttk.Frame(lf_set); f_cnt.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_cnt, text="Number of Turns (e.g., 12):").pack(side=tk.LEFT)
        ttk.Entry(f_cnt, textvariable=self.tt_turns, width=10).pack(side=tk.LEFT, padx=5)
        
        # Update total display number every time a number is typed (e.g. 30 x 12 = 360 degrees!)
        self.lbl_total = ttk.Label(lf_set, text="Total: 360 degrees", foreground="blue")
        self.lbl_total.pack(padx=5, pady=5)
        self.tt_degrees.trace_add("write", self.update_tt_totals)
        self.tt_turns.trace_add("write", self.update_tt_totals)
        
        # 3. Save destination control box 
        lf_out = ttk.LabelFrame(root, text="3. Output")
        lf_out.pack(fill=tk.X, padx=10, pady=5)
        
        f_name = ttk.Frame(lf_out); f_name.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_name, text="Base Object Name:").pack(side=tk.LEFT)
        ttk.Entry(f_name, textvariable=self.tt_base_name).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        f_dir = ttk.Frame(lf_out); f_dir.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_dir, text="Select Save Folder", command=lambda: self.sel_dir(self.tt_save_dir)).pack(side=tk.LEFT)
        ttk.Entry(f_dir, textvariable=self.tt_save_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 4. Button to start running the automated machine
        ttk.Label(root, textvariable=self.tt_status, font=("Arial", 12)).pack(pady=10)
        ttk.Button(root, text="START AUTO SCAN", command=self.do_auto_scan_sequence, state="normal").pack(fill=tk.X, padx=20, pady=10)

    # ── Unified Meshing tab: UI update helpers ─────────────────────────────

    def _update_unified_norm_ui(self):
        """Show/hide the normal-estimation or centroid param frames based on selection."""
        mode = self.unified_normal_mode.get()
        if mode in ("radial", "tangent"):
            self._frm_centroid.pack_forget()
            self._frm_norm_est.pack(fill=tk.X, padx=5, pady=2)
        else:  # centroid
            self._frm_norm_est.pack_forget()
            self._frm_centroid.pack(fill=tk.X, padx=5, pady=2)

    def _update_unified_recon_ui(self):
        """Show/hide the Poisson or Ball-Pivot param frames; auto-enforce Centroid for ball_pivot."""
        method = self.unified_recon_method.get()
        if method == "poisson":
            self._frm_ballpivot.pack_forget()
            self._frm_poisson.pack(fill=tk.X, padx=5, pady=2)
        else:  # ball_pivot — requires centroid orientation
            self._frm_poisson.pack_forget()
            self._frm_ballpivot.pack(fill=tk.X, padx=5, pady=2)
            if self.unified_normal_mode.get() != "centroid":
                self.unified_normal_mode.set("centroid")
                self._update_unified_norm_ui()

    # ── Unified Meshing tab: run function ──────────────────────────────────

    def do_unified_meshing(self):
        """Run meshing from the unified Tab 5. Routes to mesh_360 or reconstruct_stl
        based on the selected normal orientation and reconstruction method."""
        in_file  = self.m360_input_ply.get()
        out_file = self.m360_output_stl.get()
        normal_mode  = self.unified_normal_mode.get()    # radial | tangent | centroid
        recon_method = self.unified_recon_method.get()   # poisson | ball_pivot

        # Validate files
        if not os.path.isfile(in_file):
            messagebox.showerror("Error", "Input .PLY not found.")
            return
        if not out_file:
            messagebox.showerror("Error", "Please select an output .STL file.")
            return

        # Save-normals path
        save_normals_path = None
        if self.m360_save_normals.get():
            save_normals_path = self.m360_normals_out.get()
            if not save_normals_path:
                messagebox.showerror("Error", "Please select an output path for the normals .PLY.")
                return

        popup = self._make_progress_popup("Meshing & Reconstruction…")
        log   = popup["log_cb"]
        stop  = popup["stop_event"]

        log(f"Input:  {in_file}")
        log(f"Output: {out_file}")
        log(f"Normal mode: {normal_mode}   Recon: {recon_method}")

        # ── Path A: Radial/Tangent + Poisson → use mesh_360 (advanced Poisson path) ──
        if normal_mode in ("radial", "tangent") and recon_method == "poisson":
            depth     = self.m360_depth.get()
            trim      = self.m360_trim.get()
            p_width   = self.m360_width.get()
            p_scale   = self.m360_scale.get()
            p_linear  = self.m360_linear_fit.get()
            p_threads = self.m360_threads.get()
            n_rad     = self.m360_normal_radius.get()
            n_max     = self.m360_normal_max_nn.get()

            log(f"Depth={depth}  Trim={trim}  Width={p_width}  Scale={p_scale}  Threads={p_threads}")
            log(f"Normal radius={n_rad}  max_nn={n_max}")

            def run_a():
                try:
                    log("Estimating normals and running Poisson reconstruction…")
                    self.processor.mesh_360(
                        input_path=in_file, output_path=out_file,
                        depth=depth, density_trim=trim, orientation_mode=normal_mode,
                        width=p_width, scale=p_scale, linear_fit=p_linear, n_threads=p_threads,
                        normal_radius=n_rad, normal_max_nn=n_max,
                        save_normals_path=save_normals_path
                    )
                    if stop.is_set():
                        self._close_progress_popup(popup); return
                    log("Meshing complete!")
                    self._close_progress_popup(popup, success=True,
                        message=f"Mesh saved to:\n{out_file}")
                except Exception as e:
                    log(f"ERROR: {e}")
                    self._close_progress_popup(popup, success=False, message=str(e))

            threading.Thread(target=run_a, daemon=True).start()

        # ── Path B: Centroid orient or Ball-Pivoting → use reconstruct_stl ──────────
        else:
            mode_str = "watertight" if recon_method == "poisson" else "surface"
            params = {}
            if mode_str == "watertight":
                params["depth"] = self.m360_depth.get()
            else:
                params["radii"] = self.s_radii.get()

            use_centroid    = (normal_mode == "centroid")
            use_consistency = self.s_consistency_pass.get() if use_centroid else False
            consistency_k   = self.s_consistency_k.get()

            meshlab_params = None
            if self.s_use_meshlab.get():
                meshlab_params = {
                    "enabled":        True,
                    "smooth_type":    self.s_ml_smooth_type.get(),
                    "smooth_iters":   self.s_ml_smooth_iters.get(),
                    "close_holes":    self.s_ml_close_holes.get(),
                    "close_max_size": self.s_ml_close_max_size.get(),
                    "simplify":       self.s_ml_simplify.get(),
                    "target_faces":   self.s_ml_target_faces.get(),
                }

            log(f"Mode: {mode_str}  Centroid orient: {use_centroid}  Consistency: {use_consistency}")
            if meshlab_params:
                log("MeshLab post-processing: enabled")

            def run_b():
                try:
                    log("Running reconstruction…")
                    self.processor.reconstruct_stl(
                        in_file, out_file, mode_str, params,
                        centroid_orient=use_centroid,
                        consistency_pass=use_consistency,
                        consistency_k=consistency_k,
                        meshlab_params=meshlab_params,
                        save_normals_path=save_normals_path
                    )
                    if stop.is_set():
                        self._close_progress_popup(popup); return
                    log(f"STL saved → {out_file}")
                    self._close_progress_popup(popup, success=True,
                        message=f"STL saved to:\n{out_file}")
                    # If centroid mode + save-normals: open the Centroid Inspector popup
                    if use_centroid and save_normals_path:
                        self.root.after(200, lambda: self._show_centroid_inspector(
                            in_file=in_file,
                            out_file=out_file,
                            mode_str=mode_str,
                            params=params,
                            use_consistency=use_consistency,
                            consistency_k=consistency_k,
                            meshlab_params=meshlab_params,
                            save_normals_path=save_normals_path,
                        ))
                except Exception as e:
                    log(f"ERROR: {e}")
                    self._close_progress_popup(popup, success=False, message=str(e))

            threading.Thread(target=run_b, daemon=True).start()


    # ── Centroid Inspector popup ────────────────────────────────────────────

    def _show_centroid_inspector(self, in_file, out_file, mode_str, params,
                                  use_consistency, consistency_k,
                                  meshlab_params, save_normals_path):
        """Interactive popup: shows the input point cloud (white) + centroid (red).
        The user can drag X/Y/Z sliders to reposition the centroid and then
        click Recalculate & Reconstruct to re-run the whole pipeline with the
        custom centroid position."""
        import numpy as np
        import open3d as o3d
        import threading
        import tkinter as tk
        from tkinter import ttk

        # ── 1. Load point cloud & compute default AABB centroid ──────────────
        try:
            pcd = o3d.io.read_point_cloud(in_file)
            pts = np.asarray(pcd.points)
        except Exception as e:
            import tkinter.messagebox as mb
            mb.showerror("Centroid Inspector", f"Could not load point cloud:\n{e}")
            return

        if len(pts) == 0:
            return

        mn = pts.min(axis=0)
        mx = pts.max(axis=0)
        auto_center = (mn + mx) / 2.0

        # Down-sample for display speed (keep ≤ 8 000 pts in the scatter)
        MAX_DISPLAY = 8000
        if len(pts) > MAX_DISPLAY:
            step = max(1, len(pts) // MAX_DISPLAY)
            disp_pts = pts[::step]
        else:
            disp_pts = pts

        # ── 2. Build Toplevel window ─────────────────────────────────────────
        win = tk.Toplevel(self.root)
        win.title("Centroid Inspector  —  adjust before final reconstruction")
        win.geometry("1100x620")
        win.resizable(True, True)
        win.grab_set()   # modal-like (blocks interaction with main window)

        # Title bar
        hdr = tk.Frame(win, bg="#1a1a2e", pady=8)
        hdr.pack(fill=tk.X)
        tk.Label(hdr,
                 text="🔴  Centroid Inspector",
                 font=("Arial", 14, "bold"),
                 fg="white", bg="#1a1a2e").pack(side=tk.LEFT, padx=16)
        tk.Label(hdr,
                 text="White = point cloud     Red sphere = centroid used for normal orientation",
                 font=("Arial", 9), fg="#aaa", bg="#1a1a2e").pack(side=tk.LEFT, padx=8)

        # Main horizontal split
        body = tk.Frame(win, bg="#f4f4f4")
        body.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # ── 3. Left: matplotlib 3-D scatter ──────────────────────────────────
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        left_frame = tk.Frame(body, bg="#1e1e1e", relief="flat")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        fig = plt.Figure(figsize=(6.5, 5.5), facecolor="#1e1e1e")
        ax  = fig.add_subplot(111, projection="3d", facecolor="#1e1e1e")
        ax.tick_params(colors="#888", labelsize=7)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xlabel("X", color="#aaa", fontsize=8)
        ax.set_ylabel("Y", color="#aaa", fontsize=8)
        ax.set_zlabel("Z", color="#aaa", fontsize=8)

        # Draw cloud
        ax.scatter(disp_pts[:, 0], disp_pts[:, 1], disp_pts[:, 2],
                   s=0.8, c="white", alpha=0.45, linewidths=0, label="Point cloud")

        # Draw centroid (will be updated by sliders)
        centroid_scatter = ax.scatter(
            [auto_center[0]], [auto_center[1]], [auto_center[2]],
            s=180, c="red", marker="o", zorder=10, label="Centroid")

        # Draw crosshair lines through centroid
        pad = (mx - mn) * 0.05
        line_x, = ax.plot([mn[0]-pad[0], mx[0]+pad[0]],
                          [auto_center[1], auto_center[1]],
                          [auto_center[2], auto_center[2]],
                          color="red", linewidth=0.8, alpha=0.6)
        line_y, = ax.plot([auto_center[0], auto_center[0]],
                          [mn[1]-pad[1], mx[1]+pad[1]],
                          [auto_center[2], auto_center[2]],
                          color="red", linewidth=0.8, alpha=0.6)
        line_z, = ax.plot([auto_center[0], auto_center[0]],
                          [auto_center[1], auto_center[1]],
                          [mn[2]-pad[2], mx[2]+pad[2]],
                          color="red", linewidth=0.8, alpha=0.6)

        ax.legend(loc="upper left", fontsize=7, facecolor="#333", labelcolor="white")
        fig.tight_layout(pad=0.5)

        canvas_widget = FigureCanvasTkAgg(fig, master=left_frame)
        canvas_widget.draw()
        canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── 4. Right: controls panel ─────────────────────────────────────────
        right_frame = tk.Frame(body, bg="#f0f0f0", width=320, relief="flat")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=0)
        right_frame.pack_propagate(False)

        tk.Label(right_frame, text="Adjust Centroid Position",
                 font=("Arial", 12, "bold"), bg="#f0f0f0", fg="#222").pack(pady=(16, 4))
        tk.Label(right_frame,
                 text="Move sliders to reposition the red centroid.\n"
                      "Drag to rotate the 3-D view on the left.",
                 font=("Arial", 8), bg="#f0f0f0", fg="#666",
                 justify=tk.CENTER, wraplength=290).pack(pady=(0, 10))

        ttk.Separator(right_frame, orient="horizontal").pack(fill=tk.X, padx=16, pady=4)

        # Slider range: extend bounds by 20% on each side
        slack = (mx - mn) * 0.20 + 1.0   # +1 so zero-extent axes still work
        s_min = mn - slack
        s_max = mx + slack

        cx_var = tk.DoubleVar(value=float(auto_center[0]))
        cy_var = tk.DoubleVar(value=float(auto_center[1]))
        cz_var = tk.DoubleVar(value=float(auto_center[2]))

        coord_label = tk.StringVar(value=(
            f"Centroid:  X={auto_center[0]:.3f}  "
            f"Y={auto_center[1]:.3f}  Z={auto_center[2]:.3f}"
        ))

        def _update_view(*_):
            """Redraw the centroid marker and crosshairs whenever a slider moves."""
            cx, cy, cz = cx_var.get(), cy_var.get(), cz_var.get()
            # Update scatter (must use _offsets3d for mpl 3-D)
            centroid_scatter._offsets3d = ([cx], [cy], [cz])
            # Update crosshair lines
            line_x.set_data([mn[0]-pad[0], mx[0]+pad[0]], [cy, cy])
            line_x.set_3d_properties([cz, cz])
            line_y.set_data([cx, cx], [mn[1]-pad[1], mx[1]+pad[1]])
            line_y.set_3d_properties([cz, cz])
            line_z.set_data([cx, cx], [cy, cy])
            line_z.set_3d_properties([mn[2]-pad[2], mx[2]+pad[2]])
            canvas_widget.draw_idle()
            coord_label.set(
                f"Centroid:  X={cx:.3f}  Y={cy:.3f}  Z={cz:.3f}"
            )

        cx_var.trace_add("write", _update_view)
        cy_var.trace_add("write", _update_view)
        cz_var.trace_add("write", _update_view)

        def _make_axis_row(parent, label, var, lo, hi):
            """Create one labelled slider+spinbox row."""
            row = tk.Frame(parent, bg="#f0f0f0")
            row.pack(fill=tk.X, padx=16, pady=6)
            tk.Label(row, text=label, width=3, font=("Arial", 10, "bold"),
                     bg="#f0f0f0", fg="#333").pack(side=tk.LEFT)
            sl = ttk.Scale(row, from_=lo, to=hi, orient=tk.HORIZONTAL,
                           variable=var, length=160)
            sl.pack(side=tk.LEFT, padx=(4, 6))
            sp = ttk.Spinbox(row, from_=lo, to=hi, increment=0.1,
                             textvariable=var, width=9,
                             format="%.3f")
            sp.pack(side=tk.LEFT)

        _make_axis_row(right_frame, " X", cx_var, float(s_min[0]), float(s_max[0]))
        _make_axis_row(right_frame, " Y", cy_var, float(s_min[1]), float(s_max[1]))
        _make_axis_row(right_frame, " Z", cz_var, float(s_min[2]), float(s_max[2]))

        ttk.Separator(right_frame, orient="horizontal").pack(fill=tk.X, padx=16, pady=8)

        # Coordinate readout
        tk.Label(right_frame, textvariable=coord_label,
                 font=("Consolas", 8), bg="#f0f0f0", fg="#0055aa",
                 wraplength=290).pack(pady=2)

        # Reset button
        def _reset():
            cx_var.set(float(auto_center[0]))
            cy_var.set(float(auto_center[1]))
            cz_var.set(float(auto_center[2]))

        ttk.Button(right_frame, text="\u21ba  Reset to Auto (AABB center)",
                   command=_reset).pack(fill=tk.X, padx=16, pady=(8, 2))

        ttk.Separator(right_frame, orient="horizontal").pack(fill=tk.X, padx=16, pady=10)

        tk.Label(right_frame,
                 text="Click below to re-run normal\norientation + reconstruction\nwith the adjusted centroid:",
                 font=("Arial", 9), bg="#f0f0f0", fg="#444",
                 justify=tk.CENTER).pack(pady=(0, 6))

        # ── Recalculate button ────────────────────────────────────────────────
        def _recalculate():
            custom = [cx_var.get(), cy_var.get(), cz_var.get()]
            win.destroy()    # close inspector first

            popup2 = self._make_progress_popup("Recalculating with custom centroid…")
            log2   = popup2["log_cb"]
            stop2  = popup2["stop_event"]

            log2(f"Custom centroid: X={custom[0]:.3f}  Y={custom[1]:.3f}  Z={custom[2]:.3f}")
            log2(f"Output: {out_file}")

            def _run():
                try:
                    log2("Re-running reconstruction with custom centroid…")
                    self.processor.reconstruct_stl(
                        in_file, out_file, mode_str, params,
                        centroid_orient=True,
                        consistency_pass=use_consistency,
                        consistency_k=consistency_k,
                        meshlab_params=meshlab_params,
                        save_normals_path=save_normals_path,
                        custom_center=custom,
                    )
                    if stop2.is_set():
                        self._close_progress_popup(popup2); return
                    log2(f"STL saved \u2192 {out_file}")
                    self._close_progress_popup(popup2, success=True,
                        message=f"Reconstruction complete!\nSTL saved to:\n{out_file}")
                except Exception as e:
                    log2(f"ERROR: {e}")
                    self._close_progress_popup(popup2, success=False, message=str(e))

            threading.Thread(target=_run, daemon=True).start()

        recon_btn = tk.Button(
            right_frame,
            text="\u25b6  Recalculate & Reconstruct",
            font=("Arial", 11, "bold"),
            bg="#0d6efd", fg="white",
            activebackground="#0b5ed7", activeforeground="white",
            relief="flat", pady=10, cursor="hand2",
            command=_recalculate)
        recon_btn.pack(fill=tk.X, padx=16, pady=(0, 4))

        # Cancel button
        cancel_btn = tk.Button(
            right_frame,
            text="\u2715  Cancel  (keep current STL)",
            font=("Arial", 9),
            bg="#e0e0e0", fg="#444",
            activebackground="#ccc",
            relief="flat", pady=6, cursor="hand2",
            command=win.destroy)
        cancel_btn.pack(fill=tk.X, padx=16, pady=(0, 16))






    def setup_calib_check_tab(self):
        # Tab 8: Visualize Calibration 3D space
        root = self.tab_calib_check
        ttk.Label(root, text="Step 8: Calib Check (3D Visualization)", font=("Arial", 14, "bold")).pack(pady=10)
        
        lf_files = ttk.LabelFrame(root, text="1. Calibration File (.mat)")
        lf_files.pack(fill=tk.X, padx=10, pady=5)
        
        f_in = ttk.Frame(lf_files)
        f_in.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in, text="Browse .mat", command=lambda: self.sel_file_load(self.chk_calib_file, "MAT")).pack(side=tk.LEFT)
        ttk.Entry(f_in, textvariable=self.chk_calib_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Button(root, text="Show 3D Visualization", command=self.do_show_calib_3d).pack(fill=tk.X, padx=20, pady=20)

    def setup_ply_inspect_tab(self):
        """Tab 9: PLY Inspector — check ASCII vs Binary, convert to binary."""
        main_frame = self.tab_ply_inspect

        canvas   = tk.Canvas(main_frame, highlightthickness=0)
        scrollb  = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        root     = ttk.Frame(canvas)
        root.bind("<Configure>",
                  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        fid = canvas.create_window((0, 0), window=root, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(fid, width=e.width))
        canvas.configure(yscrollcommand=scrollb.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollb.pack(side="right", fill="y")

        def _mwheel(ev):
            try:
                if self.notebook.select() == str(self.tab_ply_inspect):
                    canvas.yview_scroll(int(-1 * (ev.delta / 120)), "units")
            except Exception:
                pass
        canvas.bind_all("<MouseWheel>", _mwheel, add="+")

        ttk.Label(root, text="PLY Inspector & Binary Converter",
                  font=("Arial", 14, "bold")).pack(pady=10)
        ttk.Label(root,
                  text="Inspect PLY files to check their format (ASCII or Binary),\n"
                       "then optionally convert ASCII files to compact Binary_little_endian format.",
                  foreground="#555", justify=tk.CENTER).pack(pady=(0, 6))

        # ── 1. Input source ──────────────────────────────────────────────────
        lf_src = ttk.LabelFrame(root, text="1. Input Source")
        lf_src.pack(fill=tk.X, padx=10, pady=6)

        self.pinsp_mode = tk.StringVar(value="folder")  # 'folder' | 'file'
        f_radio = ttk.Frame(lf_src); f_radio.pack(fill=tk.X, padx=5, pady=4)
        ttk.Radiobutton(f_radio, text="Folder  (scan all .ply inside)",
                        variable=self.pinsp_mode, value="folder",
                        command=self._pinsp_toggle_mode).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(f_radio, text="Single File",
                        variable=self.pinsp_mode, value="file",
                        command=self._pinsp_toggle_mode).pack(side=tk.LEFT, padx=6)

        # Folder row
        self._pinsp_f_folder = ttk.Frame(lf_src)
        self._pinsp_f_folder.pack(fill=tk.X, padx=5, pady=2)
        self.pinsp_folder = tk.StringVar()
        ttk.Button(self._pinsp_f_folder, text="Select Folder",
                   command=lambda: self.sel_dir(self.pinsp_folder)).pack(side=tk.LEFT)
        ttk.Entry(self._pinsp_f_folder, textvariable=self.pinsp_folder).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # File row (hidden by default)
        self._pinsp_f_file = ttk.Frame(lf_src)
        self.pinsp_file = tk.StringVar()
        ttk.Button(self._pinsp_f_file, text="Select .PLY File",
                   command=lambda: self.sel_file_load(self.pinsp_file, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(self._pinsp_f_file, textvariable=self.pinsp_file).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ttk.Button(lf_src, text="🔍  Inspect PLY Files",
                   command=self.do_ply_inspect).pack(fill=tk.X, padx=10, pady=8)

        # ── 2. Results table ─────────────────────────────────────────────────
        lf_res = ttk.LabelFrame(root, text="2. Inspection Results")
        lf_res.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # Header row
        hdr = ttk.Frame(lf_res); hdr.pack(fill=tk.X, padx=4, pady=(4, 0))
        ttk.Label(hdr, text="Format",  width=10, font=("Arial", 9, "bold"),
                  foreground="#333").pack(side=tk.LEFT, padx=2)
        ttk.Label(hdr, text="Size",    width=10, font=("Arial", 9, "bold"),
                  foreground="#333").pack(side=tk.LEFT, padx=2)
        ttk.Label(hdr, text="Points",  width=10, font=("Arial", 9, "bold"),
                  foreground="#333").pack(side=tk.LEFT, padx=2)
        ttk.Label(hdr, text="File Path", font=("Arial", 9, "bold"),
                  foreground="#333").pack(side=tk.LEFT, padx=2)
        ttk.Separator(lf_res, orient="horizontal").pack(fill=tk.X, padx=4, pady=2)

        # Scrollable results area
        res_canvas  = tk.Canvas(lf_res, highlightthickness=0, height=180)
        res_scroll  = ttk.Scrollbar(lf_res, orient="vertical", command=res_canvas.yview)
        self._pinsp_results_frame = ttk.Frame(res_canvas)
        self._pinsp_results_frame.bind(
            "<Configure>",
            lambda e: res_canvas.configure(scrollregion=res_canvas.bbox("all")))
        res_canvas.create_window((0, 0), window=self._pinsp_results_frame, anchor="nw")
        res_canvas.configure(yscrollcommand=res_scroll.set)
        res_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        res_canvas.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Summary label
        self.pinsp_summary = tk.StringVar(value="No files inspected yet.")
        ttk.Label(lf_res, textvariable=self.pinsp_summary,
                  foreground="#0066CC", font=("Arial", 9, "italic")).pack(pady=(2, 4))

        # ── 3. Convert to Binary ─────────────────────────────────────────────
        lf_conv = ttk.LabelFrame(root, text="3. Convert ASCII → Binary")
        lf_conv.pack(fill=tk.X, padx=10, pady=6)

        conv_desc = (
            "Converts ASCII PLY files to Binary_little_endian format.\n"
            "Binary PLY files load ~10× faster and are 30-50% smaller on disk.\n"
            "The original file is overwritten in-place (a .bak backup is kept alongside it)."
        )
        ttk.Label(lf_conv, text=conv_desc, foreground="#555",
                  justify=tk.LEFT, wraplength=640).pack(padx=8, pady=(4, 2))

        # Output mode
        self.pinsp_conv_mode = tk.StringVar(value="inplace")  # 'inplace' | 'folder'
        f_cmode = ttk.Frame(lf_conv); f_cmode.pack(fill=tk.X, padx=8, pady=4)
        ttk.Radiobutton(f_cmode, text="Overwrite in-place  (keep .bak backup)",
                        variable=self.pinsp_conv_mode, value="inplace",
                        command=self._pinsp_toggle_conv).pack(side=tk.LEFT, padx=6)
        ttk.Radiobutton(f_cmode, text="Save to output folder",
                        variable=self.pinsp_conv_mode, value="folder",
                        command=self._pinsp_toggle_conv).pack(side=tk.LEFT, padx=6)

        self._pinsp_f_outdir = ttk.Frame(lf_conv)
        self.pinsp_outdir = tk.StringVar()
        ttk.Button(self._pinsp_f_outdir, text="Select Output Folder",
                   command=lambda: self.sel_dir(self.pinsp_outdir)).pack(side=tk.LEFT)
        ttk.Entry(self._pinsp_f_outdir, textvariable=self.pinsp_outdir).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Scope selector
        self.pinsp_conv_scope = tk.StringVar(value="ascii_only")
        f_scope = ttk.Frame(lf_conv); f_scope.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(f_scope, text="Convert:").pack(side=tk.LEFT, padx=(0, 6))
        ttk.Radiobutton(f_scope, text="ASCII files only (from results above)",
                        variable=self.pinsp_conv_scope, value="ascii_only").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(f_scope, text="All PLY files in source",
                        variable=self.pinsp_conv_scope, value="all").pack(side=tk.LEFT, padx=4)

        ttk.Button(lf_conv, text="⚙  Convert to Binary",
                   command=self.do_ply_convert).pack(fill=tk.X, padx=10, pady=8)

        # Store inspected results for conversion use
        self._pinsp_last_results = []   # list of dicts: {path, fmt, size, points}

    def setup_manual_merge_tab(self):
        root = self.tab_manual_merge
        ttk.Label(root, text="Step 10: Manual Plane Merge", font=("Arial", 14, "bold")).pack(pady=10)
        
        explanation = (
            "Align two box-shaped point clouds by picking 3 planes on each.\n"
            "This works by mathematically finding the corner intersection of 3 orthogonal planes.\n"
            "During the process, a 3D window will pop up 3 times per file. "
            "Hold Shift + Left Click to pick exactly 3 points per plane, then close the window."
        )
        ttk.Label(root, text=explanation, justify=tk.CENTER, foreground="#333", font=("Arial", 9, "italic")).pack(pady=(0, 10))
        
        lf_files = ttk.LabelFrame(root, text="Files")
        lf_files.pack(fill=tk.X, padx=10, pady=5)
        
        f_in1 = ttk.Frame(lf_files); f_in1.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in1, text="Select File 1 (.ply)", command=lambda: self.sel_file_load(self.mm_input1, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_in1, textvariable=self.mm_input1).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        f_in2 = ttk.Frame(lf_files); f_in2.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in2, text="Select File 2 (.ply)", command=lambda: self.sel_file_load(self.mm_input2, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_in2, textvariable=self.mm_input2).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        f_out = ttk.Frame(lf_files); f_out.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_out, text="Select Output (.ply)", command=lambda: self.sel_file_save(self.mm_output, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_out, textvariable=self.mm_output).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        lf_params = ttk.LabelFrame(root, text="Merge Settings")
        lf_params.pack(fill=tk.X, padx=10, pady=5)
        
        f_mode = ttk.Frame(lf_params); f_mode.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_mode, text="Match Mode:").pack(side=tk.LEFT, padx=(0, 5))
        cb_mode = ttk.Combobox(f_mode, textvariable=self.mm_match_mode, values=["3", "2"], state="readonly", width=5)
        cb_mode.pack(side=tk.LEFT)
        ttk.Label(f_mode, text="(3 = Exact Corner, 2 = Edge Align + ICP Slide)", foreground="#555").pack(side=tk.LEFT, padx=5)
        
        f_icp = ttk.Frame(lf_params); f_icp.pack(fill=tk.X, padx=5, pady=5)
        ttk.Checkbutton(f_icp, text="Run ICP Refinement after plane alignment", variable=self.mm_enable_icp).pack(side=tk.LEFT)
        
        ttk.Button(root, text="▶ START MANUAL PLANE MERGE", command=self.do_manual_plane_merge).pack(fill=tk.X, padx=20, pady=20)


    # ── PLY Inspector helpers ─────────────────────────────────────────────────

    def _pinsp_toggle_mode(self):
        if self.pinsp_mode.get() == "folder":
            self._pinsp_f_file.pack_forget()
            self._pinsp_f_folder.pack(fill=tk.X, padx=5, pady=2)
        else:
            self._pinsp_f_folder.pack_forget()
            self._pinsp_f_file.pack(fill=tk.X, padx=5, pady=2)

    def _pinsp_toggle_conv(self):
        if self.pinsp_conv_mode.get() == "folder":
            self._pinsp_f_outdir.pack(fill=tk.X, padx=8, pady=(0, 4))
        else:
            self._pinsp_f_outdir.pack_forget()

    @staticmethod
    def _ply_detect(path):
        """Return (format_str, num_points) by reading the PLY header only.
        format_str is one of: 'ascii', 'binary_little_endian', 'binary_big_endian', 'unknown'
        """
        fmt    = "unknown"
        npts   = 0
        try:
            with open(path, "rb") as f:
                for _ in range(30):       # header is never more than 30 lines
                    raw = f.readline()
                    if not raw:
                        break
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line.startswith("format "):
                        parts = line.split()
                        fmt = parts[1] if len(parts) >= 2 else "unknown"
                    elif line.startswith("element vertex"):
                        parts = line.split()
                        npts = int(parts[2]) if len(parts) >= 3 else 0
                    elif line == "end_header":
                        break
        except Exception:
            pass
        return fmt, npts

    def do_ply_inspect(self):
        """Scan source for .ply files, detect format, and populate results table."""
        import glob, os

        mode = self.pinsp_mode.get()
        if mode == "folder":
            src = self.pinsp_folder.get().strip()
            if not src or not os.path.isdir(src):
                messagebox.showerror("Error", "Please select a valid folder.")
                return
            files = sorted(glob.glob(os.path.join(src, "*.ply")))
            if not files:
                messagebox.showinfo("Info", "No .ply files found in the selected folder.")
                return
        else:
            src = self.pinsp_file.get().strip()
            if not src or not os.path.isfile(src):
                messagebox.showerror("Error", "Please select a valid .ply file.")
                return
            files = [src]

        # Clear previous results
        for w in self._pinsp_results_frame.winfo_children():
            w.destroy()
        self._pinsp_last_results = []

        popup = self._make_progress_popup(
            f"Inspecting {len(files)} PLY file(s)…", total_steps=len(files))
        log  = popup["log_cb"]
        step = popup["step_cb"]
        stop = popup["stop_event"]

        def run():
            results = []
            for idx, path in enumerate(files, 1):
                if stop.is_set():
                    log("Stopped by user.")
                    self._close_progress_popup(popup)
                    return
                fmt, npts = self._ply_detect(path)
                size_bytes = os.path.getsize(path)
                size_str = (f"{size_bytes/1024/1024:.2f} MB" if size_bytes >= 1_048_576
                            else f"{size_bytes/1024:.1f} KB")
                results.append({"path": path, "fmt": fmt,
                                 "size": size_str, "points": npts})
                log(f"[{idx}/{len(files)}] {os.path.basename(path)}  →  {fmt}  |  {npts:,} pts  |  {size_str}")
                step(idx, len(files))

            self._pinsp_last_results = results

            # Build result rows on main thread
            def build_rows():
                for r in results:
                    row  = ttk.Frame(self._pinsp_results_frame)
                    row.pack(fill=tk.X, padx=2, pady=1)

                    if r["fmt"] == "ascii":
                        fg, badge = "#C0392B", "ASCII  ⚠"
                    elif "binary" in r["fmt"]:
                        fg, badge = "#27AE60", "BINARY ✓"
                    else:
                        fg, badge = "#888888", "UNKNOWN"

                    tk.Label(row, text=badge, width=12, fg=fg,
                             font=("Consolas", 9, "bold"),
                             bg="#f0f0f0", relief="groove").pack(side=tk.LEFT, padx=2)
                    tk.Label(row, text=r["size"], width=10,
                             font=("Consolas", 9)).pack(side=tk.LEFT, padx=2)
                    tk.Label(row, text=f"{r['points']:,}", width=12,
                             font=("Consolas", 9)).pack(side=tk.LEFT, padx=2)
                    tk.Label(row, text=r["path"],
                             font=("Consolas", 9), anchor="w").pack(
                                 side=tk.LEFT, fill=tk.X, expand=True, padx=2)

                ascii_n  = sum(1 for r in results if r["fmt"] == "ascii")
                binary_n = sum(1 for r in results if "binary" in r["fmt"])
                self.pinsp_summary.set(
                    f"{len(results)} file(s) scanned  —  "
                    f"{binary_n} Binary ✓  |  {ascii_n} ASCII ⚠"
                )
            self.root.after(0, build_rows)

            log(f"Done — {len(results)} file(s) inspected.")
            self._close_progress_popup(popup)

        threading.Thread(target=run, daemon=True).start()

    def do_ply_convert(self):
        """Convert ASCII PLY files to binary_little_endian using open3d."""
        import os, shutil

        results = self._pinsp_last_results
        scope   = self.pinsp_conv_scope.get()
        conv_mode = self.pinsp_conv_mode.get()

        if not results:
            messagebox.showerror("Error",
                "No inspection results found.\nPlease run 'Inspect PLY Files' first.")
            return

        if scope == "ascii_only":
            targets = [r for r in results if r["fmt"] == "ascii"]
        else:
            targets = list(results)

        if not targets:
            messagebox.showinfo("Nothing to do",
                "No ASCII PLY files found in the inspection results.")
            return

        if conv_mode == "folder":
            out_dir = self.pinsp_outdir.get().strip()
            if not out_dir:
                messagebox.showerror("Error",
                    "Please select an output folder for the converted files.")
                return
            os.makedirs(out_dir, exist_ok=True)
        else:
            out_dir = None   # in-place

        popup = self._make_progress_popup(
            f"Converting {len(targets)} PLY file(s) to Binary…",
            total_steps=len(targets))
        log  = popup["log_cb"]
        step = popup["step_cb"]
        stop = popup["stop_event"]

        def run():
            try:
                import open3d as o3d
            except ImportError:
                self._close_progress_popup(popup, success=False,
                    message="open3d is required for conversion.\npip install open3d")
                return

            ok_count = 0
            for idx, r in enumerate(targets, 1):
                if stop.is_set():
                    log("Stopped by user.")
                    self._close_progress_popup(popup)
                    return

                src_path = r["path"]
                if out_dir:
                    dst_path = os.path.join(out_dir, os.path.basename(src_path))
                else:
                    dst_path = src_path  # overwrite in-place

                try:
                    log(f"[{idx}/{len(targets)}] Loading: {os.path.basename(src_path)}")
                    pcd = o3d.io.read_point_cloud(src_path)

                    if not out_dir:
                        # Backup original
                        bak = src_path + ".bak"
                        shutil.copy2(src_path, bak)
                        log(f"  Backup saved: {os.path.basename(bak)}")

                    o3d.io.write_point_cloud(
                        dst_path, pcd,
                        write_ascii=False,           # force binary
                        compressed=False,
                        print_progress=False)
                    old_sz = r["size"]
                    new_sz_b = os.path.getsize(dst_path)
                    new_sz = (f"{new_sz_b/1024/1024:.2f} MB"
                              if new_sz_b >= 1_048_576 else f"{new_sz_b/1024:.1f} KB")
                    log(f"  ✓ Done  {old_sz} → {new_sz}  |  {dst_path}")
                    ok_count += 1
                except Exception as e:
                    log(f"  ✗ Error: {e}")

                step(idx, len(targets))

            self._close_progress_popup(popup, success=True,
                message=f"Conversion complete!\n{ok_count}/{len(targets)} file(s) converted to binary.")

        threading.Thread(target=run, daemon=True).start()

    # ==========================================
    # Button command functions section (Actions and Helper Actions)
    # ==========================================


    def update_stl_params(self, event=None):
        # Function to toggle feature menus in Tab 6 depending on mode (Watertight/Surface)
        for widget in self.f_stl_params.winfo_children():
            widget.destroy() # Clear out all old items first
            
        mode = self.s_mode.get()
        if mode == "watertight":
            # If it is solid mode, there will be a Depth input field
            ttk.Label(self.f_stl_params, text="Poisson Depth (default 10):").pack(anchor=tk.W)
            ttk.Entry(self.f_stl_params, textvariable=self.s_depth).pack(fill=tk.X)
            ttk.Label(self.f_stl_params, text="Creates a closed (watertight) mesh. Higher depth = more detail but slower.", foreground="#555").pack(anchor=tk.W)
        else:
            # If it is surface mode, there will only be Ball Radii
            ttk.Label(self.f_stl_params, text="Ball Radii Multipliers (default '1, 2, 4'):").pack(anchor=tk.W)
            ttk.Entry(self.f_stl_params, textvariable=self.s_radii).pack(fill=tk.X)
            ttk.Label(self.f_stl_params, text="Multiples of average point distance. Connects dots without filling large holes.", foreground="#555").pack(anchor=tk.W)

    def sel_file_load(self, var, ftype):
        # Function to open a window to select a file (Standard Dialog box)
        if ftype == "PLY": ext = "*.ply"
        elif ftype == "MAT": ext = "*.mat"
        else: ext = "*.*"
        
        f = filedialog.askopenfilename(filetypes=[(ftype, ext)])
        if f: 
            var.set(f)
            # Fill Output filename (autofill output path so user doesn't have to type it manually if empty)
            if ftype == "PLY":
                # For Tab 6 STL mode
                if var == self.s_input_ply and not self.s_output_stl.get():
                    self.s_output_stl.set(f.replace(".ply", ".stl"))
                # For 360 Mesh mode
                if var == self.m360_input_ply and not self.m360_output_stl.get():
                    self.m360_output_stl.set(f.replace(".ply", ".stl"))

    def sel_file_save(self, var, ftype):
        # Function to call 'Save As' Dialog box
        ext = "*.ply" if ftype == "PLY" else "*.stl"
        f = filedialog.asksaveasfilename(filetypes=[(ftype, ext)], defaultextension=ext.replace("*", ""))
        if f: var.set(f)

    def sel_dir(self, var):
        # Function to call folder selection Dialog window 
        d = filedialog.askdirectory()
        if d: var.set(d)

    def update_ip(self):
        # Function to find local IP to show to mobile device for connection
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80))
            self._pc_ip = s.getsockname()[0]; s.close() # Dig up IP
            self.ip_lbl.config(text=f"Connect Phone to: http://{self._pc_ip}:5000") # Display on screen
            self._update_cam_mode_label()  # Refresh mode label with real IP
        except:
            self._pc_ip = None

    def _update_cam_mode_label(self):
        """Update the hint label inside the Camera Mode frame to show instructions for the selected mode."""
        mode = self.camera_mode.get()
        ip = getattr(self, "_pc_ip", None)
        url = f"http://{ip}:5000" if ip else "http://<PC-IP>:5000"
        if mode == "web":
            msg = (
                f"Open a browser on your phone and navigate to  {url}\n"
                "Images are captured via the browser MediaStream API (~8MP, PNG)."
            )
        else:
            msg = (
                f"📱  Open the 'SL Camera' app on your Android phone.\n"
                f"➜  In  Settings  type:  {url}   then tap  Save & Close.\n"
                "Images are captured via Camera2 API at full native resolution (PNG, lossless).\n"
                "⚠  After first switch to Android mode, redo Calibration (Step 1 + 2) once."
            )
        try:
            self.cam_mode_lbl.config(text=msg)
        except Exception:
            pass   # Called before widget exists (first __init__ pass) — safe to ignore


    def refresh_ports(self):
        # Pull COM 1 COM 2 into the Dropdown for the Turntable
        ports = self.arduino.get_ports()
        self.cb_port['values'] = ports
        if ports: self.cb_port.current(0) # If refreshed and appears, select the first one by default
    
    def connect_arduino(self):
        # Receive Connect Arduino button trigger
        p = self.tt_port.get()
        if not p: messagebox.showerror("Error", "Select a port"); return
        
        ok, msg = self.arduino.connect(p) # Check if port is connected
        if ok: messagebox.showinfo("Connected", "Arduino Connected!")
        else: messagebox.showerror("Error", f"Failed: {msg}")

    def update_tt_totals(self, *args):
        # When Degree or Turns is changed, dynamically calculate total degrees on screen e.g. 15*22=...
        try:
            d = self.tt_degrees.get()
            t = self.tt_turns.get()
            total = d * t
            self.lbl_total.config(text=f"Total: {total} degrees ({t} scans)")
        except: pass

    def mpcp_log(self, message):
        # Helper to neatly write logs to the text box in Tab 2
        self.root.after(0, self._append_mpcp_log, message)
        
    def _append_mpcp_log(self, message):
        self.txt_log_mpcp.config(state='normal')
        self.txt_log_mpcp.insert(tk.END, message + "\n")
        self.txt_log_mpcp.see(tk.END)
        self.txt_log_mpcp.config(state='disabled')
        
    def sys_log(self, message):
        # Helper to write logs to the main Application Logs in Tab 1
        self.root.after(0, self._append_sys_log, message)
        
    def _append_sys_log(self, message):
        try:
            self.txt_log_main.config(state='normal')
            self.txt_log_main.insert(tk.END, message + "\n")
            self.txt_log_main.see(tk.END)
            self.txt_log_main.config(state='disabled')
        except:
            pass # Failsafe just in case it's called before GUI builds

    # ==========================================
    # Progress Popup Helpers
    # ==========================================

    def _make_progress_popup(self, title, total_steps=None):
        """Create a modal progress popup with log area and Stop button.

        Returns a dict:
          'top'        – tk.Toplevel window
          'log_cb'     – callable(msg): appends timestamped line to popup log
          'step_cb'    – callable(current, total): advances determinate bar
          'stop_event' – threading.Event; set when user clicks Stop
        """
        import datetime
        stop_event = threading.Event()

        top = tk.Toplevel(self.root)
        top.title(title)
        top.resizable(True, True)
        top.protocol("WM_DELETE_WINDOW", lambda: None)   # disable × close
        top.grab_set()   # modal – block parent window

        # Centre over the parent
        top.update_idletasks()
        pw, ph = self.root.winfo_width(), self.root.winfo_height()
        px, py = self.root.winfo_x(), self.root.winfo_y()
        tw, th = 580, 440
        top.geometry(f"{tw}x{th}+{px + max(0,(pw-tw)//2)}+{py + max(0,(ph-th)//2)}")

        # Header
        ttk.Label(top, text=title,
                  font=("Arial", 12, "bold"), wraplength=540).pack(pady=(14, 4))

        status_var = tk.StringVar(value="Starting…")
        ttk.Label(top, textvariable=status_var,
                  foreground="#0066CC", font=("Arial", 9)).pack()

        # Progress bar
        if total_steps and total_steps > 0:
            pb = ttk.Progressbar(top, maximum=total_steps,
                                 mode="determinate", length=540)
        else:
            pb = ttk.Progressbar(top, mode="indeterminate", length=540)
        pb.pack(padx=20, pady=8)
        if total_steps is None:
            pb.start(10)   # animate

        # Log area
        lf_log = ttk.LabelFrame(top, text="Progress Log")
        lf_log.pack(fill=tk.BOTH, expand=True, padx=14, pady=4)

        txt = tk.Text(lf_log, state="disabled", height=9,
                      wrap="word", font=("Consolas", 9), bg="#f5f5f5")
        sb_log = ttk.Scrollbar(lf_log, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=sb_log.set)
        sb_log.pack(side=tk.RIGHT, fill=tk.Y)
        txt.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Stop button
        def _on_stop():
            stop_event.set()
            status_var.set("Stopping… please wait")
            btn_stop.config(state="disabled", text="Stopping…")

        btn_stop = tk.Button(
            top, text="■   STOP & CANCEL",
            command=_on_stop,
            bg="#C0392B", fg="white",
            font=("Arial", 10, "bold"),
            relief="raised", padx=14, pady=6,
            cursor="hand2"
        )
        btn_stop.pack(pady=(4, 14))

        # Callbacks ---------------------------------------------------------
        def log_cb(msg):
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            line = f"[{ts}] {msg}\n"
            def _do():
                try:
                    txt.config(state="normal")
                    txt.insert(tk.END, line)
                    txt.see(tk.END)
                    txt.config(state="disabled")
                    status_var.set(msg[:90])
                except Exception:
                    pass
            self.root.after(0, _do)

        def step_cb(current, total=None):
            def _do():
                try:
                    if total_steps and total_steps > 0:
                        pb.config(value=current)
                    t = total or total_steps
                    if t:
                        status_var.set(f"Step {current} / {t}")
                except Exception:
                    pass
            self.root.after(0, _do)

        return {
            "top":        top,
            "log_cb":     log_cb,
            "step_cb":    step_cb,
            "stop_event": stop_event,
            "status_var": status_var,
            "pb":         pb,
            "btn_stop":   btn_stop,
        }

    def _close_progress_popup(self, popup, success=None, message=None):
        """Destroy the popup on the main thread, optionally showing a result dialog."""
        def _do():
            try:
                popup["pb"].stop()
            except Exception:
                pass
            try:
                top = popup["top"]
                if top.winfo_exists():
                    top.grab_release()
                    top.destroy()
            except Exception:
                pass
            if message:
                if success:
                    messagebox.showinfo("Done", message)
                else:
                    messagebox.showerror("Error", message)
        self.root.after(0, _do)

    # --- Execution Functions (Threading sections running in parallel to prevent GUI freezing) ---

    def do_calib_capture(self):
        # Receive first step button command: Capture Calibration photos
        d = self.calib_capture_dir.get()
        n = self.num_poses.get()

        popup = self._make_progress_popup(f"Capturing Calibration Images ({n} poses)…")
        log = popup["log_cb"]
        stop = popup["stop_event"]

        def run():
            try:
                log(f"Saving to: {d}")
                log(f"Projecting {n} calibration poses via phone…")
                self.sys.capture_calibration(d, n)
                if stop.is_set():
                    self._close_progress_popup(popup)
                    return
                log("Capture complete!")
                self._close_progress_popup(popup, success=True,
                    message=f"Calibration images saved to:\n{d}")
            except Exception as e:
                log(f"ERROR: {e}")
                self._close_progress_popup(popup, success=False, message=str(e))

        threading.Thread(target=run, daemon=True).start()

    def do_calib_compute(self):
        # Sub-calibration calculation step — ask for folder first (must be on main thread)
        initial = self.calib_capture_dir.get()
        if not os.path.exists(initial):
            initial = os.getcwd()
        in_dir = filedialog.askdirectory(title="Select Calibration Images Folder", initialdir=initial)
        if not in_dir:
            return
        self.calib_capture_dir.set(in_dir)
        out_file = os.path.join(in_dir, "calib.mat")

        popup = self._make_progress_popup("Computing Calibration…")
        log = popup["log_cb"]
        stop = popup["stop_event"]

        def run():
            try:
                log(f"Analyzing images in: {in_dir}")
                errors, available_poses = self.sys.analyze_calibration(in_dir)

                if stop.is_set():
                    self._close_progress_popup(popup)
                    return

                # Build pose-error message for user dialog
                msg = "Calibration Analysis (reprojection error in px):\n\n"
                for pose, (ce, pe) in errors.items():
                    msg += f"  {pose}:  Cam={ce:.2f}  Proj={pe:.2f}\n"
                msg += "\nEnter poses to KEEP (e.g. '1,3,4'  or  'all'):"

                log("Analysis complete — waiting for pose selection…")

                # Pose-selection dialog must run on main thread
                pose_result = [None]
                pose_ready  = threading.Event()

                def _ask():
                    pose_result[0] = simpledialog.askstring(
                        "Select Poses", msg, parent=popup["top"])
                    pose_ready.set()

                self.root.after(0, _ask)
                pose_ready.wait(timeout=300)

                if stop.is_set() or not pose_result[0]:
                    log("Pose selection cancelled.")
                    self._close_progress_popup(popup)
                    return

                user_input = pose_result[0].strip()
                selected_poses = []
                if user_input.lower() == "all":
                    selected_poses = available_poses
                else:
                    for idx in [x.strip() for x in user_input.split(",")]:
                        name = idx if idx.startswith("pose_") else f"pose_{idx}"
                        if name in available_poses:
                            selected_poses.append(name)

                log(f"Selected poses: {', '.join(selected_poses)}")
                log("Running final calibration… this may take a minute.")

                self.sys.calibrate_final(in_dir, selected_poses, out_file)
                self.root.after(0, lambda: self.calib_file.set(out_file))
                log(f"Saved → {out_file}")
                self._close_progress_popup(popup, success=True,
                    message=f"Calibration saved to:\n{out_file}")

            except Exception as e:
                log(f"ERROR: {e}")
                self._close_progress_popup(popup, success=False, message=str(e))

        threading.Thread(target=run, daemon=True).start()

    def do_scan_capture(self):
        # Command Scan capture decoding horizontal and vertical patterns
        base = os.path.join(DEFAULT_ROOT, "scans")
        name = self.scan_name.get()
        path = os.path.join(base, name)
        self.scan_capture_dir.set(path)

        popup = self._make_progress_popup(f"Capturing Scan: '{name}'")
        log = popup["log_cb"]
        stop = popup["stop_event"]

        def run():
            try:
                log(f"Projecting Gray-code patterns for: {name}")
                log(f"Saving to: {path}")
                self.sys.capture_scan(path)
                if stop.is_set():
                    self._close_progress_popup(popup)
                    return
                log("Scan capture complete!")
                self._close_progress_popup(popup, success=True,
                    message=f"Scan images saved to:\n{path}")
            except Exception as e:
                log(f"ERROR: {e}")
                self._close_progress_popup(popup, success=False, message=str(e))

        threading.Thread(target=run, daemon=True).start()



    def do_multi_pcp(self):
        calib  = self.mpcp_calib_file.get().strip()
        mode   = self.mpcp_mode.get()       # 'single' | 'batch' | 'files'
        is_bat = self.mpcp_batch.get()

        # Determine effective folder mode
        folder_mode = "batch" if (mode == "single" and is_bat) else mode
        if mode == "single" and not is_bat:
            folder_mode = "single"

        # Validate calibration
        if not calib:
            messagebox.showerror("Error", "Please select a calibration file.")
            return
        if not os.path.exists(calib):
            messagebox.showerror("Error", "Calibration file not found.")
            return

        # Validate pattern-set count
        try:
            n_col = int(self.mpcp_col_sets.get())
            n_row = int(self.mpcp_row_sets.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid pattern count values.")
            return
        if not (1 <= n_col <= 11) or not (1 <= n_row <= 11):
            messagebox.showerror("Error", "Pattern count must be between 1 and 11.")
            return

        # Validate Row Mode / Epipolar
        row_mode = self.mpcp_row_mode.get()
        ep_tol = 2.0
        if row_mode == 1:
            try:
                ep_tol = float(self.mpcp_epipolar_tol.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid epipolar tolerance value.")
                return

        # Validate Thresholds
        thresh_mode = self.mpcp_thresh_mode.get()
        s_val, c_val = 40, 10
        if thresh_mode == "manual":
            try:
                s_val = int(self.mpcp_shadow_val.get())
                c_val = int(self.mpcp_contrast_val.get())
            except ValueError:
                messagebox.showerror("Error", "Invalid threshold values.")
                return

        # Validate input source
        if mode == "files":
            if not self.mpcp_selected_files:
                messagebox.showerror("Error", "Please select image files first.")
                return
            target = ""  # not used in file-list mode
        else:
            target = self.mpcp_input_path.get().strip()
            if not target:
                messagebox.showerror("Error", "Please select an input folder.")
                return
            if not os.path.isdir(target):
                messagebox.showerror("Error", "Input folder not found.")
                return

        self.btn_run_mpcp.config(state="disabled")

        popup = self._make_progress_popup(
            f"Generating Point Cloud  [col={n_col}  row={n_row}]")
        log = popup["log_cb"]
        stop = popup["stop_event"]

        def combined_log(msg):
            log(msg)
            self.mpcp_log(msg)

        combined_log(f"=== Starting Processing [col-sets={n_col}  row-sets={n_row}] ===")

        def run():
            try:
                if stop.is_set():
                    return
                if mode == "files":
                    out_path_holder = [None]
                    save_ev = threading.Event()
                    def _ask_save():
                        p = filedialog.asksaveasfilename(
                            title="Save PLY as", defaultextension=".ply",
                            filetypes=[("PLY files", "*.ply")])
                        out_path_holder[0] = p
                        save_ev.set()
                    self.root.after(0, _ask_save)
                    save_ev.wait(timeout=120)
                    out_path = out_path_holder[0]
                    if not out_path:
                        combined_log("Save cancelled.")
                        self._close_progress_popup(popup)
                        return
                    self.processor.process_multi_ply(
                        calib, "", "files",
                        log_callback=combined_log,
                        n_sets_col=n_col, n_sets_row=n_row,
                        row_mode=row_mode, epipolar_tol=ep_tol,
                        thresh_mode=thresh_mode, shadow_val=s_val, contrast_val=c_val,
                        file_list=self.mpcp_selected_files,
                        out_path_override=out_path)
                else:
                    self.processor.process_multi_ply(
                        calib, target, folder_mode,
                        log_callback=combined_log,
                        n_sets_col=n_col, n_sets_row=n_row,
                        row_mode=row_mode, epipolar_tol=ep_tol,
                        thresh_mode=thresh_mode, shadow_val=s_val, contrast_val=c_val)

                combined_log("Processing complete!")
                self._close_progress_popup(popup, success=True, message="Point cloud generation complete!")

            except Exception as e:
                combined_log(f"CRITICAL ERROR: {e}")
                self._close_progress_popup(popup, success=False, message=str(e))
            finally:
                self.root.after(0, lambda: self.btn_run_mpcp.config(state="normal"))

        threading.Thread(target=run, daemon=True).start()

    def do_batch_processing(self):
        # Run Tab 3 — supports both Single-File mode and Folder-Batch mode
        mode = self.proc_mode.get()

        if not any([self.enable_bg_removal.get(), self.enable_outlier_removal.get(),
                    self.enable_radius_outlier.get(), self.enable_cluster.get()]):
            messagebox.showwarning("Warning", "Please enable at least one cleaning step!")
            return

        # ── Validate paths ───────────────────────────────────────────────────
        if mode == "file":
            in_file = self.proc_input_file.get()
            out_file = self.proc_output_file.get()
            if not in_file or not out_file:
                messagebox.showerror("Error", "Please select both an Input .PLY file and an Output .PLY file.")
                return
            if not os.path.isfile(in_file):
                messagebox.showerror("Error", f"Input file not found:\n{in_file}")
                return
        else:
            in_dir  = self.proc_input_dir.get()
            out_dir = self.proc_output_dir.get()
            if not in_dir or not out_dir:
                messagebox.showerror("Error", "Please select input and output folders.")
                return

        popup = self._make_progress_popup("Cleanup & Processing Pipeline…")
        log   = popup["log_cb"]
        step  = popup["step_cb"]
        stop  = popup["stop_event"]

        # ── Inner helper: process one file ───────────────────────────────────
        def process_one(path, final_output_path):
            filename = os.path.basename(path)
            current_data = path

            if self.enable_outlier_removal.get():
                log(f"[{filename}] Statistical outlier removal…")
                try:
                    current_data = self.processor.remove_outliers(
                        input_data=current_data, output_path=None,
                        nb_neighbors=self.proc_nb_neighbors.get(),
                        std_ratio=self.proc_std_ratio.get(), return_obj=True)
                except Exception as e:
                    log(f"[StatOutlier] Error: {e}"); return False

            if stop.is_set(): return False

            if self.enable_cluster.get():
                log(f"[{filename}] Largest-cluster filter…")
                try:
                    current_data = self.processor.keep_largest_cluster(
                        input_data=current_data, output_path=None,
                        eps=self.proc_cluster_eps.get(),
                        min_points=self.proc_cluster_min.get(), return_obj=True)
                except Exception as e:
                    log(f"[Cluster] Error: {e}"); return False

            if stop.is_set(): return False

            if self.enable_radius_outlier.get():
                log(f"[{filename}] Radius outlier removal…")
                try:
                    current_data = self.processor.remove_radius_outlier(
                        input_data=current_data, output_path=None,
                        nb_points=self.proc_radius_nb.get(),
                        radius=self.proc_radius_r.get(), return_obj=True)
                except Exception as e:
                    log(f"[RadOutlier] Error: {e}"); return False

            if stop.is_set(): return False

            if self.enable_bg_removal.get():
                log(f"[{filename}] Background removal…")
                try:
                    current_data = self.processor.remove_background(
                        input_data=current_data, output_path=None,
                        distance_threshold=self.bg_dist_thresh.get(),
                        ransac_n=self.bg_ransac_n.get(),
                        num_iterations=self.bg_iterations.get(), return_obj=True)
                except Exception as e:
                    log(f"[BG] Error: {e}"); return False

            import open3d as o3d, shutil
            if not isinstance(current_data, str):
                os.makedirs(os.path.dirname(final_output_path) or ".", exist_ok=True)
                o3d.io.write_point_cloud(final_output_path, current_data)
                log(f"Saved → {final_output_path}")
            else:
                os.makedirs(os.path.dirname(final_output_path) or ".", exist_ok=True)
                shutil.copy(path, final_output_path)
                log(f"Copied → {final_output_path}")
            return True

        # ── Threading ────────────────────────────────────────────────────────
        if mode == "file":
            def run():
                try:
                    log(f"Processing: {os.path.basename(in_file)}")
                    ok = process_one(in_file, out_file)
                    if stop.is_set():
                        self._close_progress_popup(popup)
                        return
                    if ok:
                        self._close_progress_popup(popup, success=True,
                            message=f"Saved to:\n{out_file}")
                    else:
                        self._close_progress_popup(popup, success=False,
                            message="Processing failed — check log.")
                except Exception as e:
                    log(f"ERROR: {e}")
                    self._close_progress_popup(popup, success=False, message=str(e))
            threading.Thread(target=run, daemon=True).start()
        else:
            def run():
                import glob
                try:
                    ply_files = glob.glob(os.path.join(in_dir, "*.ply"))
                    if not ply_files:
                        self._close_progress_popup(popup, success=False,
                            message="No .ply files found in input folder.")
                        return
                    os.makedirs(out_dir, exist_ok=True)
                    total = len(ply_files)
                    success = 0
                    for idx, path in enumerate(ply_files, 1):
                        if stop.is_set():
                            log("Stopped by user.")
                            self._close_progress_popup(popup)
                            return
                        log(f"File {idx}/{total}: {os.path.basename(path)}")
                        step(idx, total)
                        out_path = os.path.join(out_dir, os.path.basename(path))
                        if process_one(path, out_path):
                            success += 1
                    self._close_progress_popup(popup, success=True,
                        message=f"Batch complete: {success}/{total} files processed.")
                except Exception as e:
                    log(f"ERROR: {e}")
                    self._close_progress_popup(popup, success=False, message=str(e))
            threading.Thread(target=run, daemon=True).start()


    def do_merge_360(self):
        # Run Tab 4 merge 360 model
        in_dir = self.merge_input_dir.get()
        out_file = self.merge_output_file.get()
        vx = self.merge_voxel.get()
        icp_dist = self.merge_icp_dist.get()
        outlier_nb = self.merge_outlier_nb.get()
        outlier_std = self.merge_outlier_std.get()
        sample_before = self.merge_sample_before.get()
        sample_after = self.merge_sample_after.get()
        final_voxel = self.merge_final_voxel.get()
        show_preview = self.merge_show_preview.get()
        accum_mode = self.merge_accum_mode.get()
        icp_fine_pass = self.merge_icp_fine_pass.get()
        
        if not in_dir or not out_file:
            messagebox.showerror("Error", "Select Input Folder and Output File.")
            return

        popup = self._make_progress_popup("Merging 360° Point Clouds…")
        log   = popup["log_cb"]
        step  = popup["step_cb"]
        stop  = popup["stop_event"]

        log(f"Input folder: {in_dir}")
        log(f"Output file:  {out_file}")
        log(f"Voxel={vx}  ICP-dist={icp_dist}  accum={accum_mode}  fine={icp_fine_pass}")

        # --- Step preview callback ---
        # Called from the merge thread after each step with:
        #   step_index   : which step just finished (1-based)
        #   total_steps  : total number of merge steps
        #   prev_cloud   : the accumulated cloud BEFORE this step (old scans)
        #   new_cloud    : only the newly added scan (transformed into world frame)
        # The Open3D Visualizer call here is BLOCKING — the merge pauses
        # until the user closes the 3D window. This only runs if the checkbox is ticked.
        def step_preview_callback(step_index, total_steps, prev_cloud, new_cloud):
            import open3d as o3d
            import copy

            # Snapshot of the user-chosen colours (so they're stable for this popup)
            prev_rgb = list(self.merge_prev_color)   # e.g. [0.8, 0.2, 0.2]
            new_rgb  = list(self.merge_new_color)    # e.g. [0.2, 0.9, 0.3]
            use_shading = self.merge_preview_shading.get()

            # ── Apply flat colours ─────────────────────────────────────────
            old_vis = copy.deepcopy(prev_cloud)
            new_vis = copy.deepcopy(new_cloud)
            old_vis.paint_uniform_color(prev_rgb)
            new_vis.paint_uniform_color(new_rgb)

            geoms = [old_vis, new_vis]

            # ── Normal-based depth shading ─────────────────────────────────
            # Estimating normals on each sub-cloud lets Open3D's renderer apply
            # per-point Phong shading, giving the flat colour mass a 3-D look
            # with highlights and shadows — without losing the colour distinction.
            if use_shading:
                for g in geoms:
                    if not g.has_normals():
                        # Use a moderate radius so normals are smooth but fast
                        g.estimate_normals(
                            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                                radius=5.0, max_nn=30))
                        g.orient_normals_consistent_tangent_plane(20)

            # ── Open the interactive viewer ────────────────────────────────
            scan_labels = "+".join(str(k) for k in range(step_index + 1))
            window_title = (f"Step {step_index}/{total_steps}  |  "
                            f"Scans: {scan_labels}  "
                            f"[OLD={prev_rgb}  NEW={new_rgb}]  "
                            f"(close to continue)")
            print(f"[Preview] Opening 3D viewer: {window_title}")

            # Use the full Visualizer so we get proper lighting/shading when normals exist
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_title, width=960, height=720)
            for g in geoms:
                vis.add_geometry(g)

            # Render options: enable normal-based shading if requested
            opt = vis.get_render_option()
            opt.point_size = 2.0
            if use_shading:
                opt.light_on = True   # Phong point-cloud shading
            else:
                opt.light_on = False  # Pure flat colour — no shading

            import time
            while True:
                if not vis.poll_events():
                    break
                vis.update_renderer()
                if stop.is_set():
                    break
                time.sleep(0.01)
                
            vis.destroy_window()
            print(f"[Preview] Window closed, continuing to next step...")

        # Only attach the callback when the checkbox is ticked
        callback = step_preview_callback if show_preview else None

        # Wrap the step_preview_callback to also log into popup
        original_callback = callback
        def wrapped_callback(step_index, total_steps, prev_cloud, new_cloud):
            log(f"Step {step_index}/{total_steps} — opening 3D preview…")
            step(step_index, total_steps)
            if original_callback:
                original_callback(step_index, total_steps, prev_cloud, new_cloud)

        effective_callback = wrapped_callback if show_preview else None

        def run():
            try:
                self.processor.merge_pro_360(
                    in_dir, out_file,
                    vx, icp_dist,
                    outlier_nb, outlier_std,
                    sample_before, sample_after,
                    final_voxel,
                    step_callback=effective_callback,
                    accum_mode=accum_mode,
                    icp_fine_pass=icp_fine_pass,
                    stop_check=stop.is_set
                )
                if stop.is_set():
                    self._close_progress_popup(popup)
                    return
                log("Merge complete!")
                self._close_progress_popup(popup, success=True,
                    message=f"Merged cloud saved to:\n{out_file}")
            except Exception as e:
                log(f"ERROR: {e}")
                self._close_progress_popup(popup, success=False, message=str(e))

        threading.Thread(target=run, daemon=True).start()

    def do_360_meshing(self):
        # Run Tab 5 Normal Mesh
        in_file  = self.m360_input_ply.get()
        out_file = self.m360_output_stl.get()
        mode     = self.m360_mode.get()
        depth    = self.m360_depth.get()
        trim     = self.m360_trim.get()

        # Advanced Poisson settings from UI
        p_width   = self.m360_width.get()
        p_scale   = self.m360_scale.get()
        p_linear  = self.m360_linear_fit.get()
        p_threads = self.m360_threads.get()

        # Normal Estimation parameters
        n_rad = self.m360_normal_radius.get()
        n_max = self.m360_normal_max_nn.get()

        # Save-normals PLY option
        save_normals_path = None
        if self.m360_save_normals.get():
            save_normals_path = self.m360_normals_out.get()
            if not save_normals_path:
                messagebox.showerror("Error", "Please select an output path for the normals .PLY file.")
                return

        if not os.path.isfile(in_file):
            messagebox.showerror("Error", "Input .PLY not found.")
            return

        popup = self._make_progress_popup("360° Meshing (Poisson Reconstruction)…")
        log   = popup["log_cb"]
        stop  = popup["stop_event"]

        log(f"Input:  {in_file}")
        log(f"Output: {out_file}")
        log(f"Depth={depth}  Trim={trim}  Mode={mode}  Threads={p_threads}")
        log(f"Normals: radius={n_rad}  max_nn={n_max}")
        if save_normals_path:
            log(f"Save normals PLY: {save_normals_path}")

        def run_thread():
            try:
                log("Estimating normals and running Poisson reconstruction…")
                self.processor.mesh_360(
                    input_path=in_file, output_path=out_file,
                    depth=depth, density_trim=trim, orientation_mode=mode,
                    width=p_width, scale=p_scale, linear_fit=p_linear, n_threads=p_threads,
                    normal_radius=n_rad, normal_max_nn=n_max,
                    save_normals_path=save_normals_path
                )
                if stop.is_set():
                    self._close_progress_popup(popup)
                    return
                log("360 Meshing complete!")
                self._close_progress_popup(popup, success=True,
                    message=f"360 Mesh saved to:\n{out_file}")
            except Exception as e:
                log(f"ERROR: {e}")
                self._close_progress_popup(popup, success=False, message=str(e))

        threading.Thread(target=run_thread, daemon=True).start()


    def do_stl_recon(self):
        # Run Tab 7 STL Reconstruction with optional centroid normal orient + MeshLab post-processing
        i = self.s_input_ply.get()
        o = self.s_output_stl.get()
        m = self.s_mode.get()

        params = {}
        if m == "watertight": params["depth"] = self.s_depth.get()
        else: params["radii"] = self.s_radii.get()

        if not i or not o:
            messagebox.showerror("Error", "Select files first.")
            return

        # Collect the centroid normal orientation flag
        use_centroid = self.s_centroid_orient.get()
        # Consistency pass: propagate outward direction through neighborhood after centroid orient
        use_consistency = self.s_consistency_pass.get()
        consistency_k   = self.s_consistency_k.get()

        # Build the MeshLab params dict (only passed if enabled)
        meshlab_params = None
        if self.s_use_meshlab.get():
            meshlab_params = {
                "enabled": True,
                "smooth_type":    self.s_ml_smooth_type.get(),
                "smooth_iters":   self.s_ml_smooth_iters.get(),
                "close_holes":    self.s_ml_close_holes.get(),
                "close_max_size": self.s_ml_close_max_size.get(),
                "simplify":       self.s_ml_simplify.get(),
                "target_faces":   self.s_ml_target_faces.get(),
            }

        # Save-normals PLY option
        save_normals_path = None
        if self.s_save_normals.get():
            save_normals_path = self.s_normals_out.get()
            if not save_normals_path:
                messagebox.showerror("Error", "Please select an output path for the normals .PLY file.")
                return

        popup = self._make_progress_popup("STL Reconstruction…")
        log   = popup["log_cb"]
        stop  = popup["stop_event"]

        log(f"Input:  {i}")
        log(f"Output: {o}  Mode: {m}")
        log(f"Centroid orient: {use_centroid}  Consistency: {use_consistency}")
        if meshlab_params:
            log("MeshLab post-processing: enabled")

        def run():
            try:
                log("Running reconstruction…")
                self.processor.reconstruct_stl(
                    i, o, m, params,
                    centroid_orient=use_centroid,
                    consistency_pass=use_consistency,
                    consistency_k=consistency_k,
                    meshlab_params=meshlab_params,
                    save_normals_path=save_normals_path
                )
                if stop.is_set():
                    self._close_progress_popup(popup)
                    return
                log(f"STL saved → {o}")
                self._close_progress_popup(popup, success=True,
                    message=f"STL saved to:\n{o}")
            except Exception as e:
                log(f"ERROR: {e}")
                self._close_progress_popup(popup, success=False, message=str(e))

        threading.Thread(target=run, daemon=True).start()

    def do_auto_scan_sequence(self):
        # Run Tab 6 Turntable auto-scan
        if not self.arduino.ser:
            if not messagebox.askyesno("Confirm",
                    "Arduino not connected. Continue anyway (Simulation)?"):
                return

        deg       = self.tt_degrees.get()
        turns     = self.tt_turns.get()
        base_name = self.tt_base_name.get()
        root_dir  = self.tt_save_dir.get()

        if not base_name or not root_dir:
            messagebox.showerror("Error", "Check Output settings"); return

        main_folder = os.path.join(root_dir, f"{base_name}_{int(deg)}deg_AUTO")
        os.makedirs(main_folder, exist_ok=True)

        popup = self._make_progress_popup(
            f"Auto-Scan 360°  ({turns} turns × {deg}°)", total_steps=turns)
        log  = popup["log_cb"]
        step = popup["step_cb"]
        stop = popup["stop_event"]

        log(f"Base name:   {base_name}")
        log(f"Save folder: {main_folder}")
        log(f"Total turns: {turns}  ({deg}° each)")

        def run_thread():
            start_time = time.time()
            for i in range(turns):
                if stop.is_set():
                    log("Stopped by user.")
                    self._close_progress_popup(popup)
                    return

                elapsed  = time.time() - start_time
                avg_time = (elapsed / i) if i > 0 else 0
                rem_time = avg_time * (turns - i)

                log(f"Scan {i+1}/{turns}  |  elapsed {int(elapsed)}s  |  est. left {int(rem_time)}s")
                step(i + 1, turns)

                current_angle = i * deg
                sub_name = f"{base_name}_{int(current_angle)}deg_scan"
                sub_path = os.path.join(main_folder, sub_name)

                try:
                    self.sys.capture_scan(sub_path, silent=True)
                except Exception as e:
                    log(f"Scan error: {e}")
                    self._close_progress_popup(popup, success=False,
                        message=f"Scan failed at step {i+1}:\n{e}")
                    return

                if i < turns - 1:
                    if stop.is_set():
                        log("Stopped by user after capture.")
                        self._close_progress_popup(popup)
                        return
                    log(f"Rotating {deg}°…")
                    if self.arduino.ser:
                        self.arduino.rotate(deg)
                        done = self.arduino.wait_for_done(timeout=10)
                        if not done:
                            log("Warning: Arduino move timeout.")
                        time.sleep(0.5)
                    else:
                        time.sleep(2)

            total_time = time.time() - start_time
            log(f"All {turns} scans complete!  Total time: {int(total_time)}s")
            self._close_progress_popup(popup, success=True,
                message=f"Auto Scan Complete!\nTotal Time: {int(total_time)}s\nLocation: {main_folder}")

        threading.Thread(target=run_thread, daemon=True).start()

    def do_show_calib_3d(self):
        calib_file = self.chk_calib_file.get()
        if not os.path.exists(calib_file):
            messagebox.showerror("Error", f"Calibration file not found at: {calib_file}")
            return
            
        try:
            from scipy.spatial.transform import Rotation as R_sci
            
            data = scipy.io.loadmat(calib_file)
            if 'R' not in data or 'T' not in data:
                messagebox.showerror("Error", "Selected file doesn't contain complete Stereo Calibration matrices (R and T flags).")
                return
                
            R = data['R']
            T = data['T']
            
            # Camera origin is [0, 0, 0]
            cam_center = np.zeros(3)
            
            # Projector origin from geometric stereo transformation
            R_inv = R.T
            proj_center = (-R_inv @ T).flatten()
            
            # Calculate metrics for display
            dx, dy, dz = proj_center[0], proj_center[1], proj_center[2]
            distance = np.linalg.norm(proj_center)
            
            # Convert Rotation matrix to Euler angles (degrees)
            # 'xyz' means rotation around x, then y, then z.
            rot = R_sci.from_matrix(R_inv)
            euler_angles = rot.as_euler('xyz', degrees=True)
            rx, ry, rz = euler_angles[0], euler_angles[1], euler_angles[2]
            
            # Define axis lines for visualization
            axis_length = max(distance * 0.5, 50.0) # Base length on distance or a minimum
            
            # Camera axes
            cam_x = np.array([axis_length, 0, 0])
            cam_y = np.array([0, axis_length, 0])
            cam_z = np.array([0, 0, axis_length])
            
            # Projector axes (rotated by R_inv)
            proj_x = R_inv @ np.array([[axis_length], [0], [0]])
            proj_x = proj_center + proj_x.flatten()
            
            proj_y = R_inv @ np.array([[0], [axis_length], [0]])
            proj_y = proj_center + proj_y.flatten()
            
            proj_z = R_inv @ np.array([[0], [0], [axis_length]])
            proj_z = proj_center + proj_z.flatten()
            
            # Initialize Matplotlib Figure
            fig = plt.figure(figsize=(12, 7)) # Wider figure to fit side text
            
            # Create a 3D subplot that takes up the left side
            ax = fig.add_axes([0.05, 0.1, 0.6, 0.8], projection='3d')
            
            # Plot connection line
            ax.plot([cam_center[0], proj_center[0]], 
                    [cam_center[1], proj_center[1]], 
                    [cam_center[2], proj_center[2]], 'k--', label=f'Baseline ({distance:.1f}mm)')
                    
            # Plot Camera
            ax.scatter(*cam_center, c='b', marker='s', s=100, label='Camera (Origin)')
            ax.text(*cam_center, "  Camera", color='blue')
            ax.plot([cam_center[0], cam_x[0]], [cam_center[1], cam_x[1]], [cam_center[2], cam_x[2]], 'r-')
            ax.plot([cam_center[0], cam_y[0]], [cam_center[1], cam_y[1]], [cam_center[2], cam_y[2]], 'g-')
            ax.plot([cam_center[0], cam_z[0]], [cam_center[1], cam_z[1]], [cam_center[2], cam_z[2]], 'b-')
            
            # Plot Projector
            ax.scatter(*proj_center, c='r', marker='o', s=100, label='Projector')
            ax.text(*proj_center, "  Projector", color='red')
            ax.plot([proj_center[0], proj_x[0]], [proj_center[1], proj_x[1]], [proj_center[2], proj_x[2]], 'r-')
            ax.plot([proj_center[0], proj_y[0]], [proj_center[1], proj_y[1]], [proj_center[2], proj_y[2]], 'g-')
            ax.plot([proj_center[0], proj_z[0]], [proj_center[1], proj_z[1]], [proj_center[2], proj_z[2]], 'b-')
            
            # Equalize aspect ratio logic roughly for generic matplotlib 3D plots
            all_pts = np.vstack([cam_center, proj_center, cam_x, cam_y, cam_z, proj_x, proj_y, proj_z])
            max_range = np.array([all_pts[:,0].max()-all_pts[:,0].min(), 
                                  all_pts[:,1].max()-all_pts[:,1].min(), 
                                  all_pts[:,2].max()-all_pts[:,2].min()]).max() / 2.0
            
            # Find bounds
            mid_x = (all_pts[:,0].max()+all_pts[:,0].min()) * 0.5
            mid_y = (all_pts[:,1].max()+all_pts[:,1].min()) * 0.5
            mid_z = (all_pts[:,2].max()+all_pts[:,2].min()) * 0.5
            
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
            
            ax.set_xlabel('X axis (mm)')
            ax.set_ylabel('Y axis (mm)')
            ax.set_zlabel('Z axis (mm)')
            ax.set_title("Stereo Calibration Spatial Graph")
            ax.legend()
            
            # Add explanatory text on the right side of the figure
            info_text = (
                "--- 3D System Properties ---\n\n"
                "Axis Meanings:\n"
                "• Red Line (X-Axis): Horizontal width (Left/Right)\n"
                "• Green Line (Y-Axis): Vertical height (Up/Down)\n"
                "• Blue Line (Z-Axis): Depth (Forward/Backward)\n\n"
                "The Camera is the Origin [0, 0, 0].\n"
                "All coordinates are relative to the Camera lens.\n\n"
                "--- Projector Position (Translation) ---\n"
                f"Absolute Distance:\n  {distance:.2f} mm\n\n"
                f"Offset from Camera (XYZ):\n"
                f"• X offset: {dx:+.2f} mm\n"
                f"• Y offset: {dy:+.2f} mm\n"
                f"• Z offset: {dz:+.2f} mm\n\n"
                "--- Projector Angle (Rotation) ---\n"
                f"Euler Angles (XYZ):\n"
                f"• Pitch (X-rotation): {rx:+.2f}°\n"
                f"• Yaw   (Y-rotation): {ry:+.2f}°\n"
                f"• Roll  (Z-rotation): {rz:+.2f}°\n"
            )
            
            fig.text(0.70, 0.5, info_text, fontsize=11, family='monospace',
                     va='center', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=1'))

            plt.show()
            
        except ImportError:
            messagebox.showerror("Graph Error", "Package 'scipy' is required for Euler angle conversion. Ensure it is fully installed.")
        except Exception as e:
            messagebox.showerror("Graph Build Error", str(e))

    def do_manual_plane_merge(self):
        f1 = self.mm_input1.get().strip()
        f2 = self.mm_input2.get().strip()
        out = self.mm_output.get().strip()
        
        if not f1 or not f2 or not out:
            messagebox.showerror("Error", "Please select both input files and an output file.")
            return
            
        if not os.path.isfile(f1) or not os.path.isfile(f2):
            messagebox.showerror("Error", "Input files must exist.")
            return

        popup = self._make_progress_popup("Manual Plane Merge (Interactive)")
        log = popup["log_cb"]
        stop = popup["stop_event"]
        
        def run():
            try:
                log(f"Starting manual plane merge.\nFile 1: {f1}\nFile 2: {f2}")
                log("Please follow the pop-up instructions. Close the 3D window after each picking step.")
                
                # We need to run UI/Visualization on the main thread, but we are inside a background thread.
                # However, Open3D's VisualizerWithEditing works fine if called from a background thread 
                # as long as it's the only GUI running its own event loop. Let's try it.
                self.processor.manual_plane_merge(
                    f1, f2, out, 
                    log_callback=log, 
                    stop_check=stop.is_set,
                    enable_icp=self.mm_enable_icp.get(),
                    match_mode=self.mm_match_mode.get()
                )
                
                if stop.is_set():
                    self._close_progress_popup(popup)
                    return
                
                self._close_progress_popup(popup, success=True, message=f"Manual plane merge successful!\nSaved to: {out}")
            except Exception as e:
                log(f"ERROR: {str(e)}")
                import traceback
                log(traceback.format_exc())
                self._close_progress_popup(popup, success=False, message=str(e))
                
        threading.Thread(target=run, daemon=True).start()
