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
        self.mpcp_calib_file = tk.StringVar(value=os.path.join(DEFAULT_ROOT, "calib", "calib.mat"))
        self.mpcp_input_path = tk.StringVar()
        self.mpcp_mode = tk.StringVar(value="single") # 'single' or 'batch'
        
        # --- State Variables (Combined Processing - Tab 3) ---
        # Unified Processing menu
        self.proc_input_dir = tk.StringVar()  # Input folder
        self.proc_output_dir = tk.StringVar() # Output folder
        
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
        self.proc_nb_neighbors = tk.IntVar(value=20)   # Number of neighbors for distance calculation
        self.proc_std_ratio = tk.DoubleVar(value=2.0)  # Standard deviation ratio for outlier threshold
        
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

        # STL Reconstruction (Standard 3D modeling parameters)
        self.s_input_ply = tk.StringVar()
        self.s_output_stl = tk.StringVar()
        self.s_mode = tk.StringVar(value="watertight")
        self.s_depth = tk.IntVar(value=10)
        self.s_radii = tk.StringVar(value="1, 2, 4")

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

        # --- TABS (Setting up program tab sheets) ---
        self.notebook = ttk.Notebook(root) # Create horizontal tab menu
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create frames for each of the 7 tabs
        self.tab_scan = ttk.Frame(self.notebook)
        self.tab_multiPCP = ttk.Frame(self.notebook)
        self.tab_proc = ttk.Frame(self.notebook)
        self.tab_merge = ttk.Frame(self.notebook)
        self.tab_mesh360 = ttk.Frame(self.notebook) 
        self.tab_turntable = ttk.Frame(self.notebook) 
        self.tab_recon = ttk.Frame(self.notebook)
        self.tab_calib_check = ttk.Frame(self.notebook)
        
        
        # Add frames to the menu with headings 1-7
        self.notebook.add(self.tab_scan, text="1. Scan & Generate")
        self.notebook.add(self.tab_multiPCP, text="2. Multi .ply process")
        self.notebook.add(self.tab_proc, text="3. Cleanup & Process")
        self.notebook.add(self.tab_merge, text="4. Merge 360")
        self.notebook.add(self.tab_mesh360, text="5. 360 Meshing")
        self.notebook.add(self.tab_turntable, text="6. Auto-Scan 360")
        self.notebook.add(self.tab_recon, text="7. STL Reconstruction")
        self.notebook.add(self.tab_calib_check, text="8. Calib Check")
        
        
        # Initialize UI components for each tab
        self.setup_scan_tab()
        self.setup_multiPCP_tab()
        self.setup_processing_tab()
        self.setup_merge_tab()
        self.setup_360_meshing_tab()
        self.setup_turntable_tab()
        self.setup_stl_tab()
        self.setup_calib_check_tab()

    # ==========================================
    # GUI Layout Functions for Each Tab
    # ==========================================
    def setup_multiPCP_tab(self):
        # screen 2: Multi .ply process (Batch Point Cloud Generator)
        root = self.tab_multiPCP
        ttk.Label(root, text="Batch Point Cloud Generator", font=("Arial", 14, "bold")).pack(pady=10)
        
        # 1. Calibration Section
        lf1 = ttk.LabelFrame(root, text="1. Calibration File (.mat)")
        lf1.pack(fill=tk.X, padx=10, pady=10)
        
        f1 = ttk.Frame(lf1)
        f1.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f1, text="Browse .mat", command=lambda: self.sel_file_load(self.mpcp_calib_file, "MAT")).pack(side=tk.LEFT)
        ttk.Entry(f1, textvariable=self.mpcp_calib_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 2. Input Section
        lf2 = ttk.LabelFrame(root, text="2. Input Target")
        lf2.pack(fill=tk.X, padx=10, pady=5)
        
        f_radio = ttk.Frame(lf2)
        f_radio.pack(fill=tk.X, padx=5, pady=5)
        ttk.Radiobutton(f_radio, text="Single Scan Folder", variable=self.mpcp_mode, value="single").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(f_radio, text="Batch (Parent Folder of Scans)", variable=self.mpcp_mode, value="batch").pack(side=tk.LEFT, padx=10)
        
        f2 = ttk.Frame(lf2)
        f2.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f2, text="Select Folder", command=lambda: self.sel_dir(self.mpcp_input_path)).pack(side=tk.LEFT)
        ttk.Entry(f2, textvariable=self.mpcp_input_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 3. Execution
        self.btn_run_mpcp = ttk.Button(root, text="START GENERATING PLY", command=self.do_multi_pcp)
        self.btn_run_mpcp.pack(fill=tk.X, padx=20, pady=15)

        # 4. Logs
        lf3 = ttk.LabelFrame(root, text="Processing Logs")
        lf3.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.txt_log_mpcp = tk.Text(lf3, state='disabled', height=10)
        self.txt_log_mpcp.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
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
        root = self.tab_proc
        ttk.Label(root, text="Step 2: Cleanup & Process (Batch)", font=("Arial", 14, "bold")).pack(pady=10)
        ttk.Label(root, text="Pipeline: Load -> Remove Background -> Remove Outliers -> Save", foreground="blue").pack()

        # Frame for all source files (Allows multi-file loading - Batch process)
        lf_files = ttk.LabelFrame(root, text="Files")
        lf_files.pack(fill=tk.X, padx=10, pady=5)
        
        # Source folder
        f_in = ttk.Frame(lf_files); f_in.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in, text="Select Input Folder", command=lambda: self.sel_dir(self.proc_input_dir)).pack(side=tk.LEFT)
        ttk.Entry(f_in, textvariable=self.proc_input_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Destination folder
        f_out = ttk.Frame(lf_files); f_out.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_out, text="Select Output Folder", command=lambda: self.sel_dir(self.proc_output_dir)).pack(side=tk.LEFT)
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
        root = self.tab_merge
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

        ttk.Button(root, text="Merge 360 Point Clouds", command=self.do_merge_360).pack(fill=tk.X, padx=20, pady=20)

    def setup_360_meshing_tab(self):
        # Tab 4: Mesh stitching surface coating exclusively for 360 degree 3D models
        root = self.tab_mesh360
        ttk.Label(root, text="Step 4: 360 Meshing (Poisson + Normal Re-orientation)", font=("Arial", 14, "bold")).pack(pady=5)
        
        # Explanation Text Block
        explanation = (
            "This tab uses 'Screened Poisson Surface Reconstruction' to wrap a watertight 3D mesh \n"
            "over your point cloud. Unique to 360 Meshing, it calculates the 'Normal' direction of every \n"
            "point and attempts to flip them all outwards so that the model doesn't render inside-out.\n"
            "If you see bubbling artifacts, it means the Normal vectors were calculated incorrectly. \n"
            "Adjusting the Normal Estimation Search Radius can help fix these bubbles."
        )
        ttk.Label(root, text=explanation, justify=tk.CENTER, foreground="#333", font=("Arial", 9, "italic")).pack(pady=(0, 10))
        
        lf_files = ttk.LabelFrame(root, text="Files")
        lf_files.pack(fill=tk.X, padx=10, pady=5)
        
        f_in = ttk.Frame(lf_files); f_in.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in, text="Select Input .PLY", command=lambda: self.sel_file_load(self.m360_input_ply, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_in, textvariable=self.m360_input_ply).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        f_out = ttk.Frame(lf_files); f_out.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_out, text="Select Output .STL", command=lambda: self.sel_file_save(self.m360_output_stl, "STL")).pack(side=tk.LEFT)
        ttk.Entry(f_out, textvariable=self.m360_output_stl).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        lf_param = ttk.LabelFrame(root, text="Parameters")
        lf_param.pack(fill=tk.X, padx=10, pady=5)
        
        # Select mesh stitching direction (Radial, Tangent)
        f_m = ttk.Frame(lf_param); f_m.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_m, text="Orientation Mode:").pack(side=tk.LEFT)
        # Dropdown Combobox for user selection
        ttk.Combobox(f_m, textvariable=self.m360_mode, values=["radial", "tangent"], state="readonly", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_m, text="(Radial = Outwards from center | Tangent = Graph consistency)", foreground="#555").pack(side=tk.LEFT)

        f_nr = ttk.Frame(lf_param); f_nr.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_nr, text="Normal Search Radius [Default 0.1]:").pack(side=tk.LEFT)
        ttk.Entry(f_nr, textvariable=self.m360_normal_radius, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_nr, text="(Increase if you see bubbles/inverted surfaces)", foreground="#555").pack(side=tk.LEFT)

        f_nn = ttk.Frame(lf_param); f_nn.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_nn, text="Normal Max Neighbors [Default 30]:").pack(side=tk.LEFT)
        ttk.Entry(f_nn, textvariable=self.m360_normal_max_nn, width=10).pack(side=tk.LEFT, padx=5)

        f_d = ttk.Frame(lf_param); f_d.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_d, text="Poisson Depth [Default 10]:").pack(side=tk.LEFT)
        ttk.Entry(f_d, textvariable=self.m360_depth, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_d, text="(Max octree depth. Higher = more detail but slower. >12 may freeze PC.)", foreground="#555").pack(side=tk.LEFT)
        
        f_w = ttk.Frame(lf_param); f_w.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_w, text="Target Width [Default 0.0]:").pack(side=tk.LEFT)
        ttk.Entry(f_w, textvariable=self.m360_width, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_w, text="(Finest level octree cell size. Leave 0.0 to rely on Depth instead.)", foreground="#555").pack(side=tk.LEFT)
        
        f_s = ttk.Frame(lf_param); f_s.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_s, text="Scale Ratio [Default 1.1]:").pack(side=tk.LEFT)
        ttk.Entry(f_s, textvariable=self.m360_scale, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_s, text="(Ratio of reconstruction bounding box to sample bounding box.)", foreground="#555").pack(side=tk.LEFT)

        f_lf = ttk.Frame(lf_param); f_lf.pack(fill=tk.X, padx=5, pady=2)
        ttk.Checkbutton(f_lf, text="Linear Fit Interpolation", variable=self.m360_linear_fit).pack(side=tk.LEFT)
        ttk.Label(f_lf, text="(Toggle on to use linear fitting instead of default cubic interpolation.)", foreground="#555").pack(side=tk.LEFT)

        f_th = ttk.Frame(lf_param); f_th.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_th, text="Number Threads [Default -1]:").pack(side=tk.LEFT)
        ttk.Entry(f_th, textvariable=self.m360_threads, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_th, text="(-1 = Auto/All CPU Cores)", foreground="#555").pack(side=tk.LEFT)

        f_t = ttk.Frame(lf_param); f_t.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_t, text="Density Trim [0.0 = Watertight]:").pack(side=tk.LEFT)
        ttk.Entry(f_t, textvariable=self.m360_trim, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_t, text="(>0.0 trims external overlapping bubbles)", foreground="#555").pack(side=tk.LEFT)
        
        ttk.Button(root, text="Run 360 Meshing", command=self.do_360_meshing).pack(fill=tk.X, padx=20, pady=20)

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

    def setup_stl_tab(self):
        # Tab 6 (Final): Normal 3D meshing for flat point clouds 
        root = self.tab_recon
        ttk.Label(root, text="STL Reconstruction", font=("Arial", 14, "bold")).pack(pady=10)
        
        lf_files = ttk.LabelFrame(root, text="Files")
        lf_files.pack(fill=tk.X, padx=10, pady=5)
        
        f_in = ttk.Frame(lf_files); f_in.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in, text="Select Input .PLY", command=lambda: self.sel_file_load(self.s_input_ply, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_in, textvariable=self.s_input_ply).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        f_out = ttk.Frame(lf_files); f_out.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_out, text="Select Output .STL", command=lambda: self.sel_file_save(self.s_output_stl, "STL")).pack(side=tk.LEFT)
        ttk.Entry(f_out, textvariable=self.s_output_stl).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        lf_mode = ttk.LabelFrame(root, text="Method & Parameters")
        lf_mode.pack(fill=tk.X, padx=10, pady=5)
        
        f_m = ttk.Frame(lf_mode); f_m.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_m, text="Reconstruction Mode:").pack(side=tk.LEFT)
        cb = ttk.Combobox(f_m, textvariable=self.s_mode, values=["watertight", "surface"], state="readonly")
        cb.pack(side=tk.LEFT, padx=5)
        # Changing mode will hide/show additional feature frames
        cb.bind("<<ComboboxSelected>>", self.update_stl_params)
        
        # Frame to show hidden variables (Hide/Show when Combobox mode changes)
        self.f_stl_params = ttk.Frame(lf_mode)
        self.f_stl_params.pack(fill=tk.X, padx=5, pady=5)
        self.update_stl_params() # Refresh once
        
        ttk.Button(root, text="Run STL Reconstruction", command=self.do_stl_recon).pack(fill=tk.X, padx=20, pady=20)

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
            ip = s.getsockname()[0]; s.close() # Dig up IP
            self.ip_lbl.config(text=f"Connect Phone to: http://{ip}:5000") # Display on screen
        except: pass

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

    # --- Execution Functions (Threading sections running in parallel to prevent GUI freezing) ---

    def do_calib_capture(self):
        # Receive first step button command: Capture Calibration photos 
        d = self.calib_capture_dir.get()
        n = self.num_poses.get()
        # Start projecting structured light onto phone screen via Thread, keeping app responsive
        threading.Thread(target=self.sys.capture_calibration, args=(d, n), daemon=True).start()

    def do_calib_compute(self):
        # Sub-calibration calculation step
        initial = self.calib_capture_dir.get()
        if not os.path.exists(initial): initial = os.getcwd()
        
        in_dir = filedialog.askdirectory(title="Select Calibration Images Folder", initialdir=initial)
        if not in_dir: return
        
        self.calib_capture_dir.set(in_dir)
        out_file = os.path.join(in_dir, "calib.mat")
        
        threading.Thread(target=self.run_calib_analysis, args=(in_dir, out_file), daemon=True).start()

    def run_calib_analysis(self, in_dir, out_file):
        try:
            self.sys_log(f"Analyzing {in_dir}...")
            # Pull Error Analysis return values
            errors, available_poses = self.sys.analyze_calibration(in_dir)
            # Pop up window for user decision on the main thread
            self.root.after(0, self.prompt_pose_selection, errors, available_poses, in_dir, out_file)
        except Exception as e:
            err_msg = str(e)
            self.sys_log(f"Calib Analysis Error: {err_msg}")
            self.root.after(0, lambda: messagebox.showerror("Calib Error", err_msg))

    def prompt_pose_selection(self, errors, available_poses, in_dir, out_file):
        # Show error limits, ask to discard any bad images? 
        msg = "Calibration Analysis (Error in px):\n\n"
        for pose, (ce, pe) in errors.items():
            msg += f"{pose}: Cam={ce:.2f}, Proj={pe:.2f}\n"
        msg += "\nEnter poses to KEEP (e.g., '1,3,4' OR 'all' for all):"
        
        self.sys_log("Displayed pose selection prompt to user.")
        user_input = simpledialog.askstring("Select Poses", msg, parent=self.root)
        if not user_input: 
            self.sys_log("Pose selection cancelled.")
            return
        
        selected_poses = []
        user_input = user_input.strip()
        
        if user_input.lower() == 'all':
            selected_poses = available_poses
        else:
            selected_indices = [x.strip() for x in user_input.split(',')]
            for idx in selected_indices:
                name = f"pose_{idx}"
                if idx.startswith("pose_"): name = idx
                if name in available_poses: selected_poses.append(name)
        
        self.sys_log(f"Selected poses: {', '.join(selected_poses)}")
        self.sys_log("Starting final calibration calculation. This may take a minute...")
        # Continue running the Calibration process
        threading.Thread(target=self.run_calib_final, args=(in_dir, selected_poses, out_file), daemon=True).start()

    def run_calib_final(self, in_dir, selected_poses, out_file):
        try:
            self.sys.calibrate_final(in_dir, selected_poses, out_file)
            self.sys_log(f"Calibration successfully saved to {out_file}")
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Calibration Saved to:\n{out_file}"))
            self.root.after(0, lambda: self.calib_file.set(out_file)) # Set the selected file into the input field
        except Exception as e:
            err_msg = str(e)
            self.sys_log(f"Calibration Final Error: {err_msg}")
            self.root.after(0, lambda: messagebox.showerror("Calib Final Error", err_msg))

    def do_scan_capture(self):
        # Command Scan capture decoding horizontal and vertical patterns
        base = os.path.join(DEFAULT_ROOT, "scans")
        name = self.scan_name.get()
        path = os.path.join(base, name)
        self.scan_capture_dir.set(path)
        
        self.sys_log(f"Starting Scan Capture for target: {name}")
        threading.Thread(target=self.sys.capture_scan, args=(path,), daemon=True).start()



    def do_multi_pcp(self):
        calib = self.mpcp_calib_file.get()
        target = self.mpcp_input_path.get()
        mode = self.mpcp_mode.get()
        
        if not calib or not target:
            messagebox.showerror("Error", "Please select both a calibration file and an input folder.")
            return
            
        if not os.path.exists(calib):
            messagebox.showerror("Error", "Calibration file not found.")
            return

        self.btn_run_mpcp.config(state='disabled')
        self.mpcp_log(f"=== Starting {mode.upper()} Processing ===")
        
        def run():
            try:
                self.processor.process_multi_ply(calib, target, mode, log_callback=self.mpcp_log)
                self.root.after(0, lambda: messagebox.showinfo("Done", "Processing Completed!"))
            except Exception as e:
                self.mpcp_log(f"CRITICAL ERROR: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, lambda: self.btn_run_mpcp.config(state='normal'))
                
        threading.Thread(target=run, daemon=True).start()

    def do_batch_processing(self):
        # Run Tab 3 (Batch remove noise + background)
        in_dir = self.proc_input_dir.get()
        out_dir = self.proc_output_dir.get()
        
        # Check if any folder path input fields are empty
        if not in_dir or not out_dir:
            messagebox.showerror("Error", "Please select input and output folders.")
            return
            
        # Check that the user selects at least one cleaning process
        if not any([self.enable_bg_removal.get(), self.enable_outlier_removal.get(), 
                    self.enable_radius_outlier.get(), self.enable_cluster.get()]):
            messagebox.showwarning("Warning", "Please select at least one cleaning process!")
            return

        def run():
            # Pull all point cloud files with .ply extension in the Source folder
            import glob
            ply_files = glob.glob(os.path.join(in_dir, "*.ply"))
            if not ply_files:
                self.root.after(0, lambda: messagebox.showerror("Error", "No .ply files found in input directory."))
                return
                
            os.makedirs(out_dir, exist_ok=True)
            
            # Assign task to batch process files sequentially in queue
            for path in ply_files:
                filename = os.path.basename(path)
                final_output_path = os.path.join(out_dir, filename)
                
                # Store Point Cloud data to be processed (initially the original file)
                current_data = path 
                
                # 📌 1. Intelligent Background Wall Reduction Process (Plane Segmentation)
                if self.enable_bg_removal.get():
                    bg_dist = self.bg_dist_thresh.get()
                    bg_rn = self.bg_ransac_n.get()
                    bg_iters = self.bg_iterations.get()
                    try:
                        current_data = self.processor.remove_background(
                            input_data=current_data,
                            output_path=None, # Save only at the end
                            distance_threshold=bg_dist,
                            ransac_n=bg_rn,
                            num_iterations=bg_iters,
                            return_obj=True # Must return the model to continue to the next step
                        )
                    except Exception as e:
                        print(f"Error BG removal on {filename}: {e}")
                        continue
                        
                # 📌 2. Keep Largest Cluster
                if self.enable_cluster.get():
                    eps = self.proc_cluster_eps.get()
                    min_pts = self.proc_cluster_min.get()
                    try:
                        current_data = self.processor.keep_largest_cluster(
                            input_data=current_data,
                            output_path=None,
                            eps=eps,
                            min_points=min_pts,
                            return_obj=True
                        )
                    except Exception as e:
                         print(f"Error Largest Cluster on {filename}: {e}")
                         continue

                # 📌 3. Radius Outlier Removal
                if self.enable_radius_outlier.get():
                    nb = self.proc_radius_nb.get()
                    rad = self.proc_radius_r.get()
                    try:
                         current_data = self.processor.remove_radius_outlier(
                             input_data=current_data,
                             output_path=None,
                             nb_points=nb,
                             radius=rad,
                             return_obj=True
                         )
                    except Exception as e:
                         print(f"Error Radius Outlier on {filename}: {e}")
                         continue

                # 📌 4. Statistical Outlier Removal Process
                if self.enable_outlier_removal.get():
                    nb = self.proc_nb_neighbors.get()
                    sr = self.proc_std_ratio.get()
                    try:
                        current_data = self.processor.remove_outliers(
                            input_data=current_data, 
                            output_path=None, 
                            nb_neighbors=nb, 
                            std_ratio=sr,
                            return_obj=True
                        )
                    except Exception as e:
                        print(f"Error Outlier removal on {filename}: {e}")
                        continue
                
                # If no Error and passes pipeline, save to file
                if not isinstance(current_data, str): 
                    import open3d as o3d
                    o3d.io.write_point_cloud(final_output_path, current_data)
                    print(f"[Done] Saved to {final_output_path}")
                else:
                    import shutil
                    shutil.copy(path, final_output_path)
                    print(f"[Copied] Saved untouched to {final_output_path}")
                        
            # Pop up clear notification when all tasks finish at once 
            self.root.after(0, lambda: messagebox.showinfo("Done", "Batch processing finished!"))
            
        threading.Thread(target=run, daemon=True).start()

    def do_merge_360(self):
        # Run Tab 3 merge 360 model
        vx = self.merge_voxel.get()
        
        if not in_dir or not out_file:
            messagebox.showerror("Error", "Select Input Folder and Output File.")
            return

        def run():
            try:
                self.processor.merge_pro_360(in_dir, out_file, vx)
                self.root.after(0, lambda: messagebox.showinfo("Merge Done", f"Saved merged cloud to:\n{out_file}"))
            except Exception as e:
                 err_msg = str(e)
                 print(err_msg)
                 self.root.after(0, lambda: messagebox.showerror("Error", err_msg))
        
        threading.Thread(target=run, daemon=True).start()

    def do_360_meshing(self):
        # Run Tab 4 Normal Mesh 
        in_file = self.m360_input_ply.get()
        out_file = self.m360_output_stl.get()
        mode = self.m360_mode.get()
        depth = self.m360_depth.get()
        trim = self.m360_trim.get()
        
        # Retrieve the advanced Poisson settings from UI
        p_width = self.m360_width.get()
        p_scale = self.m360_scale.get()
        p_linear = self.m360_linear_fit.get()
        p_threads = self.m360_threads.get()
        
        # Retrieve Normal Estimation parameters
        n_rad = self.m360_normal_radius.get()
        n_max = self.m360_normal_max_nn.get()
        
        if not os.path.isfile(in_file):
            messagebox.showerror("Error", "Input .PLY not found.")
            return
            
        def run_thread():
            try:
                self.sys_log(f"Starting 360 Meshing (Poisson):\nDepth: {depth}, Trim: {trim}, Mode: {mode}, Threads: {p_threads}\nNormals (Rad: {n_rad}, MaxNN: {n_max})")
                self.processor.mesh_360(
                    input_path=in_file, output_path=out_file, 
                    depth=depth, density_trim=trim, orientation_mode=mode,
                    width=p_width, scale=p_scale, linear_fit=p_linear, n_threads=p_threads,
                    normal_radius=n_rad, normal_max_nn=n_max
                )
                self.sys_log("360 Meshing complete.")
                self.root.after(0, lambda: messagebox.showinfo("Done", f"360 Mesh Saved to:\n{out_file}"))
            except Exception as e:
                err_msg = str(e)
                print(f"Error: {e}")
                self.root.after(0, lambda: messagebox.showerror("Error", err_msg))
                # Don't popup error if thread dying... print is safer.
        
        threading.Thread(target=run_thread, daemon=True).start()

    def do_stl_recon(self):
        # Run Tab 6 basic operations
        i = self.s_input_ply.get()
        o = self.s_output_stl.get()
        m = self.s_mode.get()
        
        params = {}
        if m == "watertight": params["depth"] = self.s_depth.get()
        else: params["radii"] = self.s_radii.get()
        
        if not i or not o: messagebox.showerror("Error", "Select files first."); return
        
        def run():
            try:
                self.processor.reconstruct_stl(i, o, m, params)
                self.root.after(0, lambda: messagebox.showinfo("Done", f"STL Saved to:\n{o}"))
            except Exception as e:
                err_msg = str(e)
                self.root.after(0, lambda: messagebox.showerror("Error", err_msg))
        
        threading.Thread(target=run, daemon=True).start()

    def do_auto_scan_sequence(self):
        # Run Tab 5 Turntable auto-scan 
        
        # Check Arduino dependency
        if not self.arduino.ser:
            # If Arduino is not connected, ask if want to continue in Simulation mode
            if not messagebox.askyesno("Confirm", "Arduino not connected (in software). Continue anyway (Simulation)?"):
                return
        
        deg = self.tt_degrees.get()
        turns = self.tt_turns.get()
        base_name = self.tt_base_name.get()
        root_dir = self.tt_save_dir.get()
        
        if not base_name or not root_dir:
            messagebox.showerror("Error", "Check Output settings"); return
            
        # Create Main Folder (Run folder for 360 object)
        main_folder = os.path.join(root_dir, f"{base_name}_{int(deg)}deg_AUTO")
        os.makedirs(main_folder, exist_ok=True)
        
        # New Popup Progress (Secondary window to notify progress during run)
        top = tk.Toplevel(self.root)
        top.title("Auto Scan Progress")
        top.geometry("400x300")
        
        lbl_info = ttk.Label(top, text="Starting...", font=("Arial", 12))
        lbl_info.pack(pady=20)
        
        lbl_time = ttk.Label(top, text="Time: 0s")
        lbl_time.pack(pady=5)
        
        pb = ttk.Progressbar(top, maximum=turns, mode='determinate')
        pb.pack(fill=tk.X, padx=20, pady=20)
        
        # Thread Logic Auto process execution
        def run_thread():
            start_time = time.time()
            
            for i in range(turns): # How many turns to cycle through
                # Update UI (Update UI state displayed on screen)
                elapsed = time.time() - start_time
                avg_time = (elapsed / i) if i > 0 else 0
                rem_time = avg_time * (turns - i)
                
                msg = f"Scanning {i+1}/{turns}\nElapsed: {int(elapsed)}s\nEst. Left: {int(rem_time)}s"
                
                self.root.after(0, lambda: lbl_info.config(text=msg))
                self.root.after(0, lambda: lbl_time.config(text=f"Time: {int(elapsed)}s"))
                self.root.after(0, lambda m=i: pb.config(value=m))
                
                # 1. CAPTURE Take burst photos
                current_angle = i * deg
                sub_name = f"{base_name}_{int(current_angle)}deg_scan" # Pose sub-name
                sub_path = os.path.join(main_folder, sub_name)
                
                print(f"[Auto] Capturing to {sub_path}")
                
                try:
                    # Input hidden command silent=True to skip popup alerts during projection, keeping it smooth
                    self.sys.capture_scan(sub_path, silent=True)
                except Exception as e:
                    print(f"Scan Error: {e}")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Scan failed: {e}"))
                    return

                # 2. MOVE rotate the turntable to prepare for the next shot
                if i < turns - 1: # If not the final loop, command motor to move
                    msg_move = f"Rotating {deg} degrees..."
                    self.root.after(0, lambda: lbl_info.config(text=msg_move))
                    
                    if self.arduino.ser:
                        self.arduino.rotate(deg)
                        # Wait for 'DONE' from Arduino with a 10s timeout, otherwise turntable might be stuck
                        done = self.arduino.wait_for_done(timeout=10) 
                        if not done:
                            print("Warning: Arduino move timeout or no DONE received.")
                        time.sleep(0.5) # Pause slightly to prevent object vibration
                    else:
                        time.sleep(2) # Running simulation as a side test

            # Finish (Wrap up)
            total_time = time.time() - start_time
            done_msg = f"Auto Scan Complete!\nTotal Time: {int(total_time)}s\nLocation: {main_folder}"
            self.root.after(0, lambda: messagebox.showinfo("Done", done_msg))
            self.root.after(0, top.destroy) # Close the ProgressBar window

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
