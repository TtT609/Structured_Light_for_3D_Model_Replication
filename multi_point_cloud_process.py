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

def gray_decode(folder, n_cols=1920, n_rows=1080):
    files = sorted(glob.glob(os.path.join(folder, "*.bmp")))
    if not files:
        files = sorted(glob.glob(os.path.join(folder, "*.png")))
        
    if len(files) < 4:
        raise ValueError(f"Not enough images in {folder} to decode.")

    img_white = cv2.imread(files[0], 0).astype(np.float32)
    img_black = cv2.imread(files[1], 0).astype(np.float32)
    
    height, width = img_white.shape
    
    mask_shadow = img_white > 40
    mask_contrast = (img_white - img_black) > 10
    valid_mask = mask_shadow & mask_contrast

    n_col_bits = int(np.ceil(np.log2(n_cols)))
    n_row_bits = int(np.ceil(np.log2(n_rows)))
    
    current_idx = 2
    
    def decode_sequence(n_bits):
        nonlocal current_idx
        gray_val = np.zeros((height, width), dtype=np.int32)
        
        for b in range(n_bits):
            if current_idx >= len(files): break
            p_path = files[current_idx]; current_idx += 1
            i_path = files[current_idx]; current_idx += 1
            
            img_p = cv2.imread(p_path, 0).astype(np.float32)
            img_i = cv2.imread(i_path, 0).astype(np.float32)
            
            bit = np.zeros((height, width), dtype=np.int32)
            bit[img_p > img_i] = 1
            gray_val = np.bitwise_or(gray_val, np.left_shift(bit, (n_bits - 1 - b)))

        mask = np.right_shift(gray_val, 1)
        while np.any(mask > 0):
            gray_val = np.bitwise_xor(gray_val, mask)
            mask = np.right_shift(mask, 1)
            
        return gray_val

    col_map = decode_sequence(n_col_bits)
    row_map = decode_sequence(n_row_bits)
    
    return col_map, row_map, valid_mask, cv2.imread(files[0])

def reconstruct_point_cloud(col_map, row_map, mask, texture, calib):
    Nc = calib["Nc"]
    Oc = calib["Oc"]
    wPlaneCol = calib["wPlaneCol"]
    
    if wPlaneCol.shape[0] == 4: wPlaneCol = wPlaneCol.T
    
    h, w = col_map.shape
    col_flat = col_map.flatten()
    mask_flat = mask.flatten()
    tex_flat = texture.reshape(-1, 3)
    
    valid_indices = np.where(mask_flat)[0]
    
    if Nc.shape[1] == h * w:
        rays = Nc[:, valid_indices]
    else:
        K = calib["cam_K"]
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        y_v, x_v = np.unravel_index(valid_indices, (h, w))
        x_n = (x_v - cx) / fx
        y_n = (y_v - cy) / fy
        z_n = np.ones_like(x_n)
        
        rays = np.stack((x_n, y_n, z_n))
        norms = np.linalg.norm(rays, axis=0)
        rays /= norms
        
    proj_cols = col_flat[valid_indices]
    proj_cols = np.clip(proj_cols, 0, wPlaneCol.shape[0] - 1)
    
    planes = wPlaneCol[proj_cols, :]
    N = planes[:, 0:3].T
    d = planes[:, 3]
    
    denom = np.sum(N * rays, axis=0)
    numer = np.dot(N.T, Oc).flatten() + d
    
    valid_intersect = np.abs(denom) > 1e-6
    t = -numer[valid_intersect] / denom[valid_intersect]
    
    rays_valid = rays[:, valid_intersect]
    P = Oc + rays_valid * t
    C = tex_flat[valid_indices[valid_intersect]]
    
    return P.T, C

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
# GUI APPLICATION
# ==========================================
class BatchCloudApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Batch Point Cloud Generator")
        self.root.geometry("600x550")
        
        # Variables
        self.calib_file = tk.StringVar()
        self.input_path = tk.StringVar()
        self.mode = tk.StringVar(value="single") # 'single' or 'batch'
        
        self.setup_ui()

    def setup_ui(self):
        # 1. Calibration Section
        lf1 = ttk.LabelFrame(self.root, text="1. Calibration File (.mat)")
        lf1.pack(fill=tk.X, padx=10, pady=10)
        
        f1 = ttk.Frame(lf1)
        f1.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f1, text="Browse .mat", command=self.sel_calib).pack(side=tk.LEFT)
        ttk.Entry(f1, textvariable=self.calib_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 2. Input Section
        lf2 = ttk.LabelFrame(self.root, text="2. Input Target")
        lf2.pack(fill=tk.X, padx=10, pady=5)
        
        f_radio = ttk.Frame(lf2)
        f_radio.pack(fill=tk.X, padx=5, pady=5)
        ttk.Radiobutton(f_radio, text="Single Scan Folder", variable=self.mode, value="single").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(f_radio, text="Batch (Parent Folder of Scans)", variable=self.mode, value="batch").pack(side=tk.LEFT, padx=10)
        
        f2 = ttk.Frame(lf2)
        f2.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f2, text="Select Folder", command=self.sel_input).pack(side=tk.LEFT)
        ttk.Entry(f2, textvariable=self.input_path).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 3. Execution
        self.btn_run = ttk.Button(self.root, text="START GENERATING PLY", command=self.start_processing)
        self.btn_run.pack(fill=tk.X, padx=20, pady=15)

        # 4. Logs
        lf3 = ttk.LabelFrame(self.root, text="Processing Logs")
        lf3.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.txt_log = tk.Text(lf3, state='disabled', height=10)
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def log(self, message):
        self.root.after(0, self._append_log, message)
        
    def _append_log(self, message):
        self.txt_log.config(state='normal')
        self.txt_log.insert(tk.END, message + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.config(state='disabled')

    def sel_calib(self):
        f = filedialog.askopenfilename(filetypes=[("MAT Files", "*.mat")])
        if f: self.calib_file.set(f)

    def sel_input(self):
        d = filedialog.askdirectory()
        if d: self.input_path.set(d)

    def process_single(self, scan_dir, calib_data):
        ply_name = os.path.basename(scan_dir) + ".ply"
        out_path = os.path.join(scan_dir, ply_name)
        
        self.log(f"-> Decoding images in {os.path.basename(scan_dir)}...")
        c_map, r_map, mask, texture = gray_decode(scan_dir)
        
        self.log("-> Reconstructing 3D points...")
        points, colors = reconstruct_point_cloud(c_map, r_map, mask, texture, calib_data)
        
        self.log(f"-> Saving {len(points)} points...")
        save_ply(points, colors, out_path)
        self.log(f"✔ Saved: {ply_name}\n")

    def start_processing(self):
        calib = self.calib_file.get()
        target = self.input_path.get()
        mode = self.mode.get()
        
        if not calib or not target:
            messagebox.showerror("Error", "Please select both a calibration file and an input folder.")
            return
            
        if not os.path.exists(calib):
            messagebox.showerror("Error", "Calibration file not found.")
            return

        self.btn_run.config(state='disabled')
        self.log(f"=== Starting {mode.upper()} Processing ===")
        
        threading.Thread(target=self._process_thread, args=(calib, target, mode), daemon=True).start()

    def _process_thread(self, calib_path, target_path, mode):
        try:
            self.log("Loading Calibration Data...")
            calib_data = load_calibration(calib_path)
            
            if mode == "single":
                self.process_single(target_path, calib_data)
            else:
                # Batch mode: Find all subfolders
                subfolders = [f.path for f in os.scandir(target_path) if f.is_dir()]
                self.log(f"Found {len(subfolders)} subfolders to process.")
                
                success_count = 0
                for folder in subfolders:
                    # Quick check to ensure it's a valid scan folder (has images)
                    if glob.glob(os.path.join(folder, "*.bmp")) or glob.glob(os.path.join(folder, "*.png")):
                        try:
                            self.process_single(folder, calib_data)
                            success_count += 1
                        except Exception as e:
                            self.log(f"❌ Error in {os.path.basename(folder)}: {e}\n")
                    else:
                        self.log(f"Skipping {os.path.basename(folder)} (No images found).")
                        
                self.log(f"=== Batch Complete: Successfully processed {success_count}/{len(subfolders)} folders ===")

            self.root.after(0, lambda: messagebox.showinfo("Done", "Processing Completed!"))

        except Exception as e:
            self.log(f"CRITICAL ERROR: {e}")
            self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.root.after(0, lambda: self.btn_run.config(state='normal'))

if __name__ == "__main__":
    root = tk.Tk()
    app = BatchCloudApp(root)
    root.mainloop()