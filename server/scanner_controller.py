# Adapted from Douglas Lanman's "Structured Light for 3D Scanning" (2009)

import numpy as np
import cv2
import scipy.io
import os
import sys
import time
import threading
import uuid
import glob
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, filedialog
import open3d as o3d
import copy
import serial
import serial.tools.list_ports



# ==========================================
# CONFIGURATION
# ==========================================
# --- Folder Structure ---
DEFAULT_ROOT = os.path.join(os.getcwd(), f"{datetime.now().strftime('%d_%m_%Y')}_3Dscan")

# --- Hardware Settings ---
SCREEN_OFFSET_X = 1920  # Primary monitor width (Projector starts here)
SCREEN_WIDTH = 1920     # Projector Width
SCREEN_HEIGHT = 1080     # Projector Height
PROJ_VALUE = 200        # Brightness (0-255)
D_SAMPLE_PROJ = 1       # Downsampling factor

# --- Checkerboard Settings ---
CHECKER_ROWS = 7        # Inner corners
CHECKER_COLS = 7
SQUARE_SIZE = 35.0      # mm

# ==========================================
# FLASK SERVER
# ==========================================
app = Flask(__name__)
CORS(app) # อนุญาตให้เครื่องอื่น (เช่น มือถือ) เชื่อมต่อเข้ามาได้

# Silence Flask Logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

SERVER_STATE = {
    "command": "idle",                              # คำสั่งปัจจุบัน (เช่น idle, capture)
    "command_id": "",                               # ไอดีของคำสั่งเพื่อไม่ให้ทำซ้ำ
    "last_image_path": None,                        # path ที่ต้องการให้บันทึกรูป
    "upload_received_event": threading.Event(),     # ตัวส่งสัญญาณว่า "ได้รับรูปแล้วนะ"
    "last_seen": 0,                                 # เวลาล่าสุดที่มือถือติดต่อมา
    "connected": False                              # สถานะการเชื่อมต่อ
}

@app.route('/poll_command', methods=['GET'])
def poll_command():
    current_time = time.time()
    if current_time - SERVER_STATE["last_seen"] > 5.0:
        print(f"[System] 📱 Phone Connected! (IP: {request.remote_addr})")
        SERVER_STATE["connected"] = True
    SERVER_STATE["last_seen"] = current_time

    return jsonify({
        "action": SERVER_STATE["command"],
        "id": SERVER_STATE["command_id"]
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    print(f"[System] 📥 Receiving image...")
    if 'file' not in request.files: return "No file", 400
    file = request.files['file']
    if file.filename == '': return "No filename", 400
    
    if SERVER_STATE["last_image_path"]:
        os.makedirs(os.path.dirname(SERVER_STATE["last_image_path"]), exist_ok=True)
        file.save(SERVER_STATE["last_image_path"])
        print(f"[System] ✅ Saved: {os.path.basename(SERVER_STATE['last_image_path'])}")
        SERVER_STATE["upload_received_event"].set()
    return "Success", 200

def monitor_disconnect():
    while True:
        time.sleep(2)
        if SERVER_STATE["connected"]:
            if time.time() - SERVER_STATE["last_seen"] > 5.0:
                print("[System] ❌ Phone Disconnected")
                SERVER_STATE["connected"] = False

def run_flask():
    threading.Thread(target=monitor_disconnect, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

# ==========================================
# STRUCTURED LIGHT SYSTEM
# ==========================================
# ==========================================
# ARDUINO CONTROLLER
# ==========================================
class ArduinoController:
    def __init__(self):
        self.ser = None

    def get_ports(self):
        return [p.device for p in serial.tools.list_ports.comports()]

    def connect(self, port, baudrate=115200):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=2)
            time.sleep(2)  # Wait for reset
            return True, "Connected"
        except Exception as e:
            return False, str(e)

    def disconnect(self):
        if self.ser:
            self.ser.close()
            self.ser = None

    def rotate(self, degrees):
        if not self.ser: return False
        try:
            cmd = f"{degrees}\n"
            self.ser.write(cmd.encode())
            return True
        except:
            return False

    def wait_for_done(self, timeout=30):
        if not self.ser: return False
        start = time.time()
        buffer = ""
        while time.time() - start < timeout:
            if self.ser.in_waiting:
                try:
                    line = self.ser.read_until().decode().strip()
                    if "DONE" in line: return True
                except: pass
            time.sleep(0.1)
        return False

class SLSystem:
    def __init__(self):
        self.window_name = "Projector"
        
    def init_projector(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, SCREEN_OFFSET_X, 0)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        black = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)
        cv2.imshow(self.window_name, black)
        cv2.waitKey(50)

    def close_projector(self):
        cv2.destroyWindow(self.window_name)

    def generate_patterns(self):
        width, height = SCREEN_WIDTH // D_SAMPLE_PROJ, SCREEN_HEIGHT // D_SAMPLE_PROJ
        n_cols = int(np.ceil(np.log2(width)))
        n_rows = int(np.ceil(np.log2(height)))
        
        def get_gray_1d(n):
            if n == 1: return ['0', '1']
            prev = get_gray_1d(n - 1)
            return ['0' + s for s in prev] + ['1' + s for s in prev[::-1]]

        col_gray = get_gray_1d(n_cols)
        row_gray = get_gray_1d(n_rows)
        
        P = [[], []] 
        for b in range(n_cols):
            pat = np.zeros((height, width), dtype=np.uint8)
            for c in range(width):
                if c < len(col_gray) and col_gray[c][b] == '1': pat[:, c] = 1
            P[0].append(pat)

        for b in range(n_rows):
            pat = np.zeros((height, width), dtype=np.uint8)
            for r in range(height):
                if r < len(row_gray) and row_gray[r][b] == '1': pat[r, :] = 1
            P[1].append(pat)
        return P

    def trigger_capture(self, save_path):
        SERVER_STATE["upload_received_event"].clear()
        SERVER_STATE["last_image_path"] = save_path
        SERVER_STATE["command_id"] = str(uuid.uuid4())
        SERVER_STATE["command"] = "capture"
        
        if not SERVER_STATE["upload_received_event"].wait(timeout=20):
            print(f"[Error] Timeout capturing {save_path}")
            return False
        SERVER_STATE["command"] = "idle"
        return True

    # ---------------------------------------------------------
    # 1. CAPTURE CALIBRATION
    # ---------------------------------------------------------
    def capture_calibration(self, save_dir, num_poses=5):
        try:
            num_poses = int(num_poses)
        except:
            num_poses = 5
            
        self.init_projector()
        P = self.generate_patterns()
        
        # Prepare patterns
        patterns = []
        white = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8) * PROJ_VALUE
        black = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)
        
        patterns.append(("01.png", white))
        patterns.append(("02.png", black))
        
        idx = 3
        for j in range(2): 
            for pat in P[j]:
                pat_img = (pat * PROJ_VALUE).astype(np.uint8)
                inv_img = ((1-pat) * PROJ_VALUE).astype(np.uint8)
                if D_SAMPLE_PROJ > 1:
                    pat_img = cv2.resize(pat_img, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    inv_img = cv2.resize(inv_img, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST)
                patterns.append((f"{idx:02d}.png", pat_img)); idx+=1
                patterns.append((f"{idx:02d}.png", inv_img)); idx+=1

        os.makedirs(save_dir, exist_ok=True)
        messagebox.showinfo("Step 1", f"Starting Calibration Capture ({num_poses} poses).\nImages will be saved to:\n{save_dir}")
        
        for pose in range(1, num_poses + 1):
            pose_dir = os.path.join(save_dir, f"pose_{pose}")
            os.makedirs(pose_dir, exist_ok=True)
            
            cv2.imshow(self.window_name, white)
            cv2.waitKey(50)
            messagebox.showinfo("Calibration", f"Pose {pose}/{num_poses}.\nMove board then click OK.")
            
            for fname, img in patterns:
                cv2.imshow(self.window_name, img)
                cv2.waitKey(250)
                if not self.trigger_capture(os.path.join(pose_dir, fname)):
                    messagebox.showerror("Error", "Capture timeout.")
                    self.close_projector()
                    return

        self.close_projector()
        messagebox.showinfo("Step 1 Done", "Calibration Capture Complete.")

    # ---------------------------------------------------------
    # 2. PROCESS CALIBRATION
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # 2. PROCESS CALIBRATION (Refactored for Thread Safety)
    # ---------------------------------------------------------
    def analyze_calibration(self, input_dir):
        """Step 1: Analyzes poses and returns errors + list of available poses"""
        # 1. ค้นหาโฟลเดอร์ท่าทางต่างๆ (Poses) ใน input_dir
        available_poses = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
        if len(available_poses) < 3: # ต้องมีอย่างน้อย 3 ท่าทางเพื่อให้การ Calibrate มีความแม่นยำ
            raise ValueError(f"Need at least 3 pose folders in {input_dir}")

        print(f"[Calib] analyzing {len(available_poses)} poses...")
        
        # 2. Analyze Errors # 2. วิเคราะห์ค่าความคลาดเคลื่อน (Reprojection Error) ของแต่ละท่าทาง
        errors = self.compute_reprojection_errors(input_dir, available_poses)
        return errors, available_poses

    def load_calib_data(self, base_dir, pose_list):
        # สร้างพิกัด 3D สมมติ (Object Points) ของกระดานหมากรุก (เช่น 0,0,0, 1,0,0, ...)
        objp = np.zeros((CHECKER_ROWS * CHECKER_COLS, 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKER_ROWS, 0:CHECKER_COLS].T.reshape(-1, 2)
        objp *= SQUARE_SIZE # คูณด้วยขนาดจริงของช่องสี่เหลี่ยม (เช่น 30 มม.)
        
        obj_pts, cam_pts, proj_pts, valid_poses = [], [], [], []
        img_shape = None
        
        for pose in pose_list:
            path = os.path.join(base_dir, pose)
            img = cv2.imread(os.path.join(path, "01.png")) # อ่านภาพแรกเพื่อหาจุดบนกระดาน
            if img is None: continue
            if img_shape is None: img_shape = (img.shape[1], img.shape[0])
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # แปลงเป็นขาวดำและหาจุดตัดของกระดานหมากรุก
            blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Gaussian Blur
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Adaptive Histogram Equalization (CLAHE)
            enhanced = clahe.apply(blurred) 
            ret, corners = cv2.findChessboardCorners(enhanced, (CHECKER_ROWS, CHECKER_COLS), None)
            
            if ret:
                # ปรับแต่งพิกัดจุดตัดให้มีความละเอียดระดับ sub-pixel (แม่นยำกว่า 1 พิกเซล)
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                
                # Decode
                files = sorted(glob.glob(os.path.join(path, "*.png")))
                
                n_col_bits = int(np.ceil(np.log2(SCREEN_WIDTH)))
                n_row_bits = int(np.ceil(np.log2(SCREEN_HEIGHT)))
                
                # ตรวจสอบว่ามีจำนวนภาพแสงโครงสร้างครบถ้วนหรือไม่
                if len(files) - 2 < 2 * (n_col_bits + n_row_bits):
                    print(f"Skipping {pose}: Not enough images.")
                    continue
                
                col_val = np.zeros(len(corners2)); row_val = np.zeros(len(corners2))
                base_idx = 2
                
                # ฟังก์ชันย่อยสำหรับถอดรหัสลำดับภาพบิต (Binary/Gray Code)
                def decode_seq(n_bits, idx_start): # it compares a Positive image to its Inverse.
                    val = np.zeros(len(corners2))
                    idx = idx_start
                    bin_code = None
                    for b in range(n_bits): 
                        img_p = cv2.imread(files[idx], 0)
                        img_i = cv2.imread(files[idx+1], 0)
                        idx += 2
                        x = corners2[:,0,0]
                        y = corners2[:,0,1]
                        
                        vp = img_p[y.astype(int), x.astype(int)]
                        vi = img_i[y.astype(int), x.astype(int)]
                        
                        bit = (vp > vi).astype(int)
                        if b == 0: 
                            bin_code = bit
                        else: 
                            bin_code = np.bitwise_xor(bin_code, bit)
                        
                        val += bin_code * (2**(n_bits - 1 - b))
                    return val, idx # val = final decoded decimal positions,    idx = index use for
                
                col_val, base_idx = decode_seq(n_col_bits, base_idx)
                row_val, base_idx = decode_seq(n_row_bits, base_idx)
                
                proj_pts_pose = np.column_stack((col_val, row_val)).astype(np.float32).reshape(-1, 1, 2)
                # column_stack : It takes the two 1D arrays (col_val and row_val) and zips them together into pairs. If a corner was at Column 50 and Row 100, it becomes a coordinate [50, 100].
                # reshape(-1, 1, 2): This is a specific requirement for OpenCV's calibration functions (like cv2.calibrateCamera or cv2.stereoCalibrate). It formats the data as a list of points, where each point is a 1 x 2 array (the U, V coordinates of the projector).
                obj_pts.append(objp)
                cam_pts.append(corners2)
                proj_pts.append(proj_pts_pose)
                valid_poses.append(pose)
                
        return obj_pts, cam_pts, proj_pts, img_shape, valid_poses

    def compute_reprojection_errors(self, base_dir, pose_list):
        obj_pts, cam_pts, proj_pts, shape, poses = self.load_calib_data(base_dir, pose_list)
        # Quick Calib
        rc, mc, dc, rvc, tvc = cv2.calibrateCamera(obj_pts, cam_pts, shape, None, None) # use list of point to solve in and ex param
        rp, mp, dp, rvp, tvp = cv2.calibrateCamera(obj_pts, proj_pts, (SCREEN_WIDTH, SCREEN_HEIGHT), None, None)
        
        errors = {}
        for i, p in enumerate(poses):
            p2_c, _ = cv2.projectPoints(obj_pts[i], rvc[i], tvc[i], mc, dc)
            err_c = cv2.norm(cam_pts[i], p2_c, cv2.NORM_L2)/len(p2_c)
            
            p2_p, _ = cv2.projectPoints(obj_pts[i], rvp[i], tvp[i], mp, dp)
            err_p = cv2.norm(proj_pts[i], p2_p, cv2.NORM_L2)/len(p2_p)
            errors[p] = (err_c, err_p)
        return errors

    def calibrate_final(self, base_dir, selected_poses, output_file):
        obj_pts, cam_pts, proj_pts, shape, _ = self.load_calib_data(base_dir, selected_poses)
        
        # Stereo Calib
        print("Calibrating Camera...")
        rc, mc, dc, _, _ = cv2.calibrateCamera(obj_pts, cam_pts, shape, None, None) # use list of point to solve in and ex param
        print("Calibrating Projector...")
        rp, mp, dp, _, _ = cv2.calibrateCamera(obj_pts, proj_pts, (SCREEN_WIDTH, SCREEN_HEIGHT), None, None)
        
        print("Stereo Calibration...")
        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            obj_pts, cam_pts, proj_pts, mc, dc, mp, dp, shape, flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        # 1. Camera Center (Oc)
        Oc = np.zeros((3, 1))
        
        # 2. Camera Rays (Nc)
        # CRITICAL FIX: shape from load_calib_data is (Width, Height). 
        # We must unpack it as w, h (NOT h, w) so X and Y don't get flipped!
        w, h = shape 
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        fx, fy, cx, cy = K1[0,0], K1[1,1], K1[0,2], K1[1,2]
        
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        z_norm = np.ones_like(x_norm)
        
        rays = np.stack((x_norm, y_norm, z_norm), axis=2)
        norms = np.linalg.norm(rays, axis=2, keepdims=True)
        rays /= norms
        Nc = rays.reshape(-1, 3).T
        
        # 3. Projector Planes (wPlaneCol and wPlaneRow)
        wPlaneCol = np.zeros((SCREEN_WIDTH, 4))
        wPlaneRow = np.zeros((SCREEN_HEIGHT, 4))
        
        fx_p, fy_p = K2[0,0], K2[1,1]
        cx_p, cy_p = K2[0,2], K2[1,2]
        
        R_inv = R.T
        C_p_cam = -R_inv @ T
        
        def get_plane_from_proj_line(u_p, v_p_start, v_p_end, is_col=True):
            if is_col:
                p1_n = np.array([(u_p - cx_p)/fx_p, (v_p_start - cy_p)/fy_p, 1]).reshape(3,1)
                p2_n = np.array([(u_p - cx_p)/fx_p, (v_p_end - cy_p)/fy_p, 1]).reshape(3,1)
            else:
                p1_n = np.array([(v_p_start - cx_p)/fx_p, (u_p - cy_p)/fy_p, 1]).reshape(3,1)
                p2_n = np.array([(v_p_end - cx_p)/fx_p, (u_p - cy_p)/fy_p, 1]).reshape(3,1)
                
            r1 = R_inv @ p1_n
            r2 = R_inv @ p2_n
            
            normal = np.cross(r1.flatten(), r2.flatten())
            normal /= np.linalg.norm(normal)
            d = -np.dot(normal, C_p_cam.flatten())
            
            return np.array([normal[0], normal[1], normal[2], d])

        for c in range(SCREEN_WIDTH):
            wPlaneCol[c, :] = get_plane_from_proj_line(c, 0, SCREEN_HEIGHT, is_col=True)
            
        for r in range(SCREEN_HEIGHT):
            wPlaneRow[r, :] = get_plane_from_proj_line(r, 0, SCREEN_WIDTH, is_col=False)
            
        scipy.io.savemat(output_file, {
            "Nc": Nc,
            "Oc": Oc,
            "wPlaneCol": wPlaneCol.T,
            "wPlaneRow": wPlaneRow.T,
            "cam_K": K1,
            "proj_K": K2,
            "R": R,
            "T": T
        })
        messagebox.showinfo("Success", f"Calibration Saved to:\n{output_file}\nError: {ret:.4f}")

    # ---------------------------------------------------------
    # 3. CAPTURE SCAN 
    # ---------------------------------------------------------
    def capture_scan(self, save_dir):
        # เริ่มต้นตั้งค่าโปรเจกเตอร์ (เช่น เปิดหน้าต่าง Fullscreen)
        self.init_projector()
        P = self.generate_patterns()# สร้างชุดลวดลาย (Gray Code Patterns) ทั้งแนวตั้งและแนวนอนเก็บไว้ในตัวแปร P
        
        patterns = []# เตรียมลิสต์สำหรับเก็บภาพที่จะฉาย และภาพพื้นฐาน (ขาว/ดำ)
        white = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8) * PROJ_VALUE # สร้างภาพ "ดำล้วน" (เพื่อใช้ดูระดับ Noise หรือ Ambient light)
        black = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)
        
        patterns.append(("01.bmp", white))
        patterns.append(("02.bmp", black))
        idx = 3
        for j in range(2): 
            for pat in P[j]:
                pat_img = (pat * PROJ_VALUE).astype(np.uint8)
                inv_img = ((1-pat) * PROJ_VALUE).astype(np.uint8)
                if D_SAMPLE_PROJ > 1:
                    pat_img = cv2.resize(pat_img, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    inv_img = cv2.resize(inv_img, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST)
                patterns.append((f"{idx:02d}.bmp", pat_img)); idx+=1
                patterns.append((f"{idx:02d}.bmp", inv_img)); idx+=1
        
        os.makedirs(save_dir, exist_ok=True)
        cv2.imshow(self.window_name, white)
        cv2.waitKey(100)
        messagebox.showinfo("Step 3", f"Ready to scan.\nImages saved to: {save_dir}")
        
        for fname, img in patterns:
            cv2.imshow(self.window_name, img)
            cv2.waitKey(200)
            if not self.trigger_capture(os.path.join(save_dir, fname)):
                messagebox.showerror("Error", "Timeout"); self.close_projector(); return
        
        self.close_projector()
        messagebox.showinfo("Step 3 Done", "Scan Capture Complete.")

    # ---------------------------------------------------------
    # 4. GENERATE CLOUD (Exact 1:1 Clone of process_cloud.py)
    # ---------------------------------------------------------
    def generate_cloud(self, scan_dir, calib_file):
        # ตรวจสอบว่ามีไฟล์ Calibration อยู่จริงไหม
        if not os.path.exists(calib_file):
            raise FileNotFoundError(f"Calibration file not found at {calib_file}")

        print(f"[Process] Processing {scan_dir} using {calib_file}...")
        
        # 1. โหลดไฟล์ .mat (ที่ได้จากขั้นตอน Calibrate) เข้ามาในโปรแกรม
        data = scipy.io.loadmat(calib_file)
        if 'Oc' not in data: # เช็คว่าไฟล์สมบูรณ์ไหม (ต้องมีค่า Oc หรือจุดศูนย์กลางกล้อง)
            raise ValueError("Calibration file missing 'Oc'.")
        # เก็บค่าต่างๆ ลงใน Dictionary เพื่อให้เรียกใช้ง่ายๆ
        calib_data = {
            "Nc": data["Nc"],               # รังสีของแสงจากกล้อง
            "Oc": data["Oc"],               # จุดศูนย์กลางกล้อง (0,0,0)
            "wPlaneCol": data["wPlaneCol"], # ระนาบแสงของโปรเจกเตอร์แนวตั้ง
            "wPlaneRow": data["wPlaneRow"], # ระนาบแสงของโปรเจกเตอร์แนวนอน
            "cam_K": data["cam_K"]          # ค่าคงที่ภายในกล้อง
        }

        # Embed gray_decode exactly as it is in standalone
        def gray_decode(folder, n_cols=1920, n_rows=1080):
            # ค้นหาไฟล์ภาพ .bmp หรือ .png ในโฟลเดอร์ที่สแกน
            files = sorted(glob.glob(os.path.join(folder, "*.bmp")))
            if not files:
                files = sorted(glob.glob(os.path.join(folder, "*.png")))
                
            if len(files) < 4:
                raise ValueError("Not enough images in folder to decode.")
            # อ่านภาพสีขาว (01) และสีดำ (02) มาหาค่าความสว่าง
            img_white = cv2.imread(files[0], 0).astype(np.float32)
            img_black = cv2.imread(files[1], 0).astype(np.float32)
            
            height, width = img_white.shape
            # สร้าง Mask เพื่อคัดจุดที่ไม่ชัดเจนออก (จุดที่มืดเกินไป หรือคอนทราสต์ต่ำเกินไป)
            """mask_shadow = img_white > 40
            mask_contrast = (img_white - img_black) > 10      #old code
            valid_mask = mask_shadow & mask_contrast"""
            # Calculate Contrast and Noise Floor
            contrast = img_white - img_black
            noise_floor = np.percentile(img_black, 95) # Higher than 95% of black pixels

            # Define 'Dynamic Range' of the current scene
            dynamic_range = np.max(contrast)

            # Apply relative thresholds
            mask_shadow = img_white > (noise_floor * 1.5)
            mask_contrast = contrast > (dynamic_range * 0.05) # 5% of max contrast

            valid_mask = mask_shadow & mask_contrast

            # คำนวณจำนวนบิตที่ต้องใช้ตามความละเอียดหน้าจอ
            n_col_bits = int(np.ceil(np.log2(n_cols)))
            n_row_bits = int(np.ceil(np.log2(n_rows)))
            
            current_idx = 2 # เริ่มอ่านภาพลวดลายตั้งแต่ใบที่ 3 เป็นต้นไป
            
            def decode_sequence(n_bits):
                nonlocal current_idx
                gray_val = np.zeros((height, width), dtype=np.int32)
                
                for b in range(n_bits):
                    if current_idx >= len(files): break
                    # อ่านภาพคู่ปกติ (p) และภาพตรงข้าม (i)
                    p_path = files[current_idx]; current_idx += 1
                    i_path = files[current_idx]; current_idx += 1
                    img_p = cv2.imread(p_path, 0).astype(np.float32)
                    img_i = cv2.imread(i_path, 0).astype(np.float32)

                    # ถ้าภาพปกติสว่างกว่าภาพตรงข้าม ให้บิตนั้นเป็น 1
                    bit = np.zeros((height, width), dtype=np.int32)
                    bit[img_p > img_i] = 1

                    # เอาบิตที่ได้ไปต่อกัน (Bit Shifting) เพื่อสร้างตัวเลข Gray Code
                    gray_val = np.bitwise_or(gray_val, np.left_shift(bit, (n_bits - 1 - b)))
                # แปลง Gray Code กลับเป็นเลขฐานสอง (Binary) เพื่อให้ได้พิกัดจริง
                mask = np.right_shift(gray_val, 1)
                while np.any(mask > 0):
                    gray_val = np.bitwise_xor(gray_val, mask)
                    mask = np.right_shift(mask, 1)
                    
                return gray_val

            print("Decoding Columns...")
            col_map = decode_sequence(n_col_bits)  # ถอดรหัสพิกัดแนวตั้ง
            print("Decoding Rows...")
            row_map = decode_sequence(n_row_bits)  # ถอดรหัสพิกัดแนวนอน
            
            return col_map, row_map, valid_mask, cv2.imread(files[0]) # ส่งคืนพิกัดและภาพ Texture
            
        # Embed reconstruct_point_cloud exactly as it is in standalone
        def reconstruct_point_cloud(col_map, row_map, mask, texture, calib): 
            print("Reconstructing 3D points...")
            
            Nc = calib["Nc"] # รังสีแสง
            Oc = calib["Oc"] # จุดศูนย์กลางกล้อง
            wPlaneCol = calib["wPlaneCol"] # ระนาบโปรเจกเตอร์
            
            if wPlaneCol.shape[0] == 4: wPlaneCol = wPlaneCol.T
            
            h, w = col_map.shape
            
            col_flat = col_map.flatten()
            mask_flat = mask.flatten()
            tex_flat = texture.reshape(-1, 3) # เตรียมสีของพิกเซล
            # เลือกเฉพาะจุดที่ผ่านการคัดกรอง (Mask) มาคำนวณ
            valid_indices = np.where(mask_flat)[0]
            print(f"Processing {len(valid_indices)} valid pixels...")
            # คำนวณหาทิศทางของแสง (Rays) สำหรับทุกพิกเซล
            if Nc.shape[1] == h * w:
                rays = Nc[:, valid_indices]
            else:
                # ถ้าไม่มี Nc สำเร็จรูป ให้สร้างขึ้นใหม่จากค่า K (Intrinsic)
                K = calib["cam_K"]
                fx, fy = K[0,0], K[1,1]
                cx, cy = K[0,2], K[1,2]
                y_v, x_v = np.unravel_index(valid_indices, (h, w))
                x_n = (x_v - cx) / fx
                y_n = (y_v - cy) / fy
                z_n = np.ones_like(x_n)
                rays = np.stack((x_n, y_n, z_n))# ทำให้เป็น Unit Vector
                norms = np.linalg.norm(rays, axis=0)
                rays /= norms
            # ดึงสมการระนาบของโปรเจกเตอร์ที่ตรงกับพิกัดที่ถอดรหัสได้
            proj_cols = col_flat[valid_indices]
            proj_cols = np.clip(proj_cols, 0, wPlaneCol.shape[0] - 1)
            
            planes = wPlaneCol[proj_cols, :]
            
            N = planes[:, 0:3].T    # เวกเตอร์แนวฉากของระนาบ
            d = planes[:, 3]        # ระยะห่างจากจุดกำเนิด
            
            # คำนวณหาจุดตัด (Intersection) โดยใช้สูตร:
            # $t = -(N^T \cdot O_c + d) / (N^T \cdot \text{ray})$
            denom = np.sum(N * rays, axis=0)
            numer = np.dot(N.T, Oc).flatten() + d
            
            valid_intersect = np.abs(denom) > 1e-6 # ป้องกันการหารด้วยศูนย์
            t = -numer[valid_intersect] / denom[valid_intersect]
            
            rays_valid = rays[:, valid_intersect]
            P = Oc + rays_valid * t # คำนวณพิกัด 3D: $P = O_c + t \cdot \text{ray}$
            # เก็บค่าสี (BGR) จากภาพถ่ายจริง
            C = tex_flat[valid_indices[valid_intersect]]
            
            return P.T, C
            
        # 2. รันฟังก์ชันถอดรหัสและการสร้าง 3D ตามลำดับ
        c_map, r_map, mask_out, texture_out = gray_decode(scan_dir)
        points, colors = reconstruct_point_cloud(c_map, r_map, mask_out, texture_out, calib_data)
        
        # 3. สร้างและบันทึกไฟล์ .ply (รูปแบบมาตรฐานของ Point Cloud)
        ply_name = os.path.basename(scan_dir) + ".ply"
        out_path = os.path.join(scan_dir, ply_name)
        
        print(f"Saving {len(points)} points to {out_path}...")
        with open(out_path, 'w') as f:
            # เขียน Header ของไฟล์ PLY
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            # เขียนข้อมูลพิกัด (x, y, z) และสี (r, g, b) ทีละจุด
            for i in range(len(points)):
                p = points[i]
                c = colors[i]
                # สลับลำดับสีจาก BGR (OpenCV) เป็น RGB (PLY)
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[2]} {c[1]} {c[0]}\n")
                
        print(f"[Success] Generated {out_path}")


# ==========================================
# PROCESSING LOGIC (Open3D)
# ==========================================
class ProcessingLogic:
    @staticmethod
    def _load_pcd(input_data):
        if isinstance(input_data, str):
            if not os.path.exists(input_data):
                raise FileNotFoundError(f"Input file not found: {input_data}")
            return o3d.io.read_point_cloud(input_data)
        return input_data

    @staticmethod
    def remove_background(input_data, output_path=None, distance_threshold=50, ransac_n=3, num_iterations=1000, return_obj=False):
        print(f"[BG Remove] Processing...")
        pcd = ProcessingLogic._load_pcd(input_data)
        
        if not pcd.has_points():
            raise ValueError("Point cloud is empty.")

        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                 ransac_n=ransac_n,
                                                 num_iterations=num_iterations)
        
        # Keep outliers (the object)
        object_cloud = pcd.select_by_index(inliers, invert=True)
        
        print(f"[BG Remove] Original: {len(pcd.points)}, Remaining: {len(object_cloud.points)} pts")
        
        if output_path:
            o3d.io.write_point_cloud(output_path, object_cloud)
            print(f"[BG Remove] Saved to {output_path}")
            
        return object_cloud if return_obj else None

    @staticmethod
    def remove_outliers(input_data, output_path=None, nb_neighbors=20, std_ratio=2.0, return_obj=False):
        print(f"[Outlier] Processing...")
        pcd = ProcessingLogic._load_pcd(input_data)
        
        if not pcd.has_points():
            raise ValueError("Point cloud is empty.")

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        inlier_cloud = pcd.select_by_index(ind)
        
        print(f"[Outlier] Keeping: {len(inlier_cloud.points)} pts")
        
        if output_path:
            o3d.io.write_point_cloud(output_path, inlier_cloud)
            print(f"[Outlier] Saved to {output_path}")

        return inlier_cloud if return_obj else None

    @staticmethod
    def preprocess_point_cloud(pcd, voxel_size):
        # Downsample
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # Normals
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            
        # FPFH Features
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    @staticmethod
    def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        
        # RANSAC based on feature matching
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
        return result

    @staticmethod
    def merge_pro_360(input_folder, output_path, voxel_size=0.02):
        print(f"[Merge 360] Loading clouds from {input_folder}...")
        
        ply_files = sorted(glob.glob(os.path.join(input_folder, "*.ply")))
        if len(ply_files) < 2:
            raise ValueError("Need at least 2 .ply files to merge.")
            
        pcds = []
        for path in ply_files:
            pcd = o3d.io.read_point_cloud(path)
            # Ensure originals have no normals or we re-estimate them later properly?
            # New360 just reads them.
            pcds.append(pcd)
            
        print(f"[Merge 360] Loaded {len(pcds)} clouds. Running Sequential Registration (New360 Logic)...")
        
        # Accumulator (Start with Frame 0)
        merged_cloud = copy.deepcopy(pcds[0])
        
        # Current Global Transform (Frame i -> Frame 0)
        # T_accum = T_{i-1 -> 0}
        max_accum_T = np.identity(4)
        
        for i in range(1, len(pcds)):
            print(f"[Merge 360] Aligning Scan {i} -> Scan {i-1}...")
            source = pcds[i]
            target = pcds[i-1]
            
            # 1. Preprocess
            source_down, source_fpfh = ProcessingLogic.preprocess_point_cloud(source, voxel_size)
            target_down, target_fpfh = ProcessingLogic.preprocess_point_cloud(target, voxel_size)
            
            # 2. Global RANSAC (Source -> Target)
            ransac_result = ProcessingLogic.execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh, voxel_size)
            
            # 3. Local ICP Refinement
            icp_result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, voxel_size, ransac_result.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            
            T_local = icp_result.transformation # T_{i -> i-1}
            
            # 4. Update Cumulative Transform
            # T_{i -> 0} = T_{i-1 -> 0} @ T_{i -> i-1}
            max_accum_T = np.dot(max_accum_T, T_local)
            
            # 5. Transform and Merge
            pcd_temp = copy.deepcopy(source)
            pcd_temp.transform(max_accum_T)
            merged_cloud += pcd_temp
            
        print("[Merge 360] Post-processing (Downsample + Outlier removal)...")
        pcd_combined_down = merged_cloud.voxel_down_sample(voxel_size=voxel_size)
        
        # Outlier Removal
        cl, ind = pcd_combined_down.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd_final = pcd_combined_down.select_by_index(ind)
        
        # Estimate Normals
        pcd_final.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        
        o3d.io.write_point_cloud(output_path, pcd_final)
        print(f"[Merge 360] Saved merged cloud to {output_path}")

    @staticmethod
    def reconstruct_stl(input_path, output_path, mode="watertight", params=None):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        print(f"[Recon] Loading {input_path}...")
        pcd = o3d.io.read_point_cloud(input_path)
        
        if not pcd.has_points():
            raise ValueError("Point cloud is empty.")
            
        # Estimate Normals if needed
        if not pcd.has_normals():
            print("[Recon] Estimating normals...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
            pcd.orient_normals_consistent_tangent_plane(100)
            
        mesh = None
        if mode == "watertight":
            depth = int(params.get("depth", 10))
            if depth > 16:
                raise ValueError(f"Depth {depth} is too high! Maximum recommended is 12-14. >16 will freeze your PC.")
            
            print(f"[Recon] Poisson Reconstruction (depth={depth})...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, linear_fit=False)
            
            # Trim low density
            densities = np.asarray(densities)
            mask = densities < np.quantile(densities, 0.02)
            mesh.remove_vertices_by_mask(mask)
            
        elif mode == "surface":
            # Ball Pivoting
            # Params: radii list
            radii_str = params.get("radii", "1,2,4")
            try:
                # If user provides explicit numbers "10.5, 20.0"
                # OR we use multipliers of avg distance? 
                # The user prompt implied just "select", let's be smart.
                # Only if they type "auto" we calculate? 
                # Let's assume input is comma separated float values for Radii
                # BUT typical usage is multipliers of avg_dist.
                # Let's calculate avg_dist first.
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                
                # Parse layout: "1, 2, 4" means 1*avg, 2*avg...
                multipliers = [float(x) for x in radii_str.split(',')]
                radii = [avg_dist * m for m in multipliers]
                print(f"[Recon] Ball Pivoting (radii={radii})...")
                
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii))
            except Exception as e:
                raise ValueError(f"Invalid radii parameters: {e}")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if len(mesh.vertices) == 0:
            raise ValueError("Generated mesh is empty.")

        print("[Recon] Computing normals and saving...")
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"[Recon] Saved STL to {output_path}")

        print(f"[Recon] Saved STL to {output_path}")

    @staticmethod
    def mesh_360(input_path, output_path, depth=10, density_trim=0.01, orientation_mode="tangent"):
        """
        Specialized meshing for 360 scans.
        orientation_mode: "tangent" (Graph-based consistency) or "radial" (Spokes from center).
        density_trim: 0.0 = Watertight (Blob), >0.0 = Trims artifacts.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        print(f"[360 Mesh] Loading {input_path}...")
        pcd = o3d.io.read_point_cloud(input_path)
        
        if not pcd.has_points():
            raise ValueError("Point cloud is empty.")
            
        # 1. Estimate Normals
        print("[360 Mesh] Estimating normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # 2. Re-orient Normals
        print(f"[360 Mesh] Re-orienting normals (Mode: {orientation_mode})...")
        
        if orientation_mode == "radial":
            # Radial: Normals point towards/away from center.
            # Usually for 360 scan, "outside" is away from center.
            center = pcd.get_center()
            pcd.orient_normals_towards_camera_location(center) # Points INWARDS
            
            # Flip to point OUTWARDS
            pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) * -1.0)
            print("[360 Mesh] Radial orientation applied (Outwards).")
            
        else: # "tangent"
            try:
                pcd.orient_normals_consistent_tangent_plane(100)
                print("[360 Mesh] Consistent tangent plane orientation applied.")
            except Exception as e:
                print(f"[360 Mesh] Warning: Tangent plane failed ({e}). Fallback to radial.")
                center = pcd.get_center()
                pcd.orient_normals_towards_camera_location(center)
                pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) * -1.0)

        # 3. Poisson Reconstruction
        print(f"[360 Mesh] Poisson Reconstruction (depth={depth})...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, linear_fit=False)
            
        # 4. Trim (Optional)
        if density_trim > 0.0:
            print(f"[360 Mesh] Trimming low density vertices (threshold={density_trim})...")
            densities = np.asarray(densities)
            threshold = np.quantile(densities, density_trim)
            mask = densities < threshold
            mesh.remove_vertices_by_mask(mask)
        else:
            print("[360 Mesh] Density trim is 0.0 -> Keeping watertight result.")
        
        # 5. Cleanup
        mesh.compute_vertex_normals()
        
        # 6. Save
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"[360 Mesh] Saved to {output_path}")

# ==========================================
# GUI
# ==========================================
class ScannerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Project 3D Scanner Suite")
        self.root.geometry("800x700") # Wider for tabs
        self.sys = SLSystem()
        self.processor = ProcessingLogic()
        self.arduino = ArduinoController()
        
        # --- State Variables (Scanner) ---
        self.calib_capture_dir = tk.StringVar(value=os.path.join(DEFAULT_ROOT, "calib"))
        self.calib_file = tk.StringVar(value=os.path.join(DEFAULT_ROOT, "calib", "calib.mat"))
        self.num_poses = tk.IntVar(value=5)
        self.scan_name = tk.StringVar(value="object_01")
        self.scan_capture_dir = tk.StringVar(value=os.path.join(DEFAULT_ROOT, "scans", "object_01"))
        
        # --- State Variables (Combined Processing) ---
        self.proc_input_dir = tk.StringVar()
        self.proc_output_dir = tk.StringVar()
        
        # BG Params
        self.bg_dist_thresh = tk.DoubleVar(value=50.0)
        self.bg_ransac_n = tk.IntVar(value=3)
        self.bg_iterations = tk.IntVar(value=1000)
        
        # Outlier Params
        self.proc_nb_neighbors = tk.IntVar(value=20)
        self.proc_std_ratio = tk.DoubleVar(value=2.0)
        
        self.s_input_ply = tk.StringVar()
        self.s_output_stl = tk.StringVar()
        self.s_mode = tk.StringVar(value="watertight")
        self.s_depth = tk.IntVar(value=10)
        self.s_radii = tk.StringVar(value="1, 2, 4")


        # 360 Merge Params
        self.merge_input_dir = tk.StringVar()
        self.merge_output_file = tk.StringVar()
        self.merge_voxel = tk.DoubleVar(value=0.02)
        
        # 360 Meshing Params
        self.m360_input_ply = tk.StringVar()
        self.m360_output_stl = tk.StringVar()
        self.m360_depth = tk.IntVar(value=10)
        self.m360_trim = tk.DoubleVar(value=0.0) # Default to 0.0 (Watertight)
        self.m360_mode = tk.StringVar(value="radial") # Default to Radial (Better for single objects)

        # --- State Variables (Turntable) ---
        self.tt_port = tk.StringVar()
        self.tt_baud = tk.StringVar(value="115200")
        self.tt_degrees = tk.DoubleVar(value=30.0)
        self.tt_turns = tk.IntVar(value=12)
        self.tt_status = tk.StringVar(value="Status: Idle")
        self.tt_base_name = tk.StringVar(value="Object_360")
        self.tt_save_dir = tk.StringVar(value=os.path.join(DEFAULT_ROOT, "scans_360"))

        # --- TABS ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.tab_scan = ttk.Frame(self.notebook)
        self.tab_proc = ttk.Frame(self.notebook)
        self.tab_merge = ttk.Frame(self.notebook)
        self.tab_mesh360 = ttk.Frame(self.notebook) # New Tab
        self.tab_turntable = ttk.Frame(self.notebook) # NEW Turntable Tab
        self.tab_recon = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_scan, text="1. Scan & Generate")
        self.notebook.add(self.tab_proc, text="2. Cleanup & Process")
        self.notebook.add(self.tab_merge, text="3. Merge 360")
        self.notebook.add(self.tab_mesh360, text="4. 360 Meshing")
        self.notebook.add(self.tab_turntable, text="5. Auto-Scan 360")
        self.notebook.add(self.tab_recon, text="6. STL Reconstruction")
        
        self.setup_scan_tab()
        self.setup_processing_tab()
        self.setup_merge_tab()
        self.setup_360_meshing_tab()
        self.setup_turntable_tab()
        self.setup_stl_tab()

    def setup_scan_tab(self):
        root = self.tab_scan
        
        ttk.Label(root, text="3D Scanner Workflow", font=("Arial", 16, "bold")).pack(pady=10)
        self.ip_lbl = ttk.Label(root, text="Connecting...", foreground="blue")
        self.ip_lbl.pack()
        self.update_ip()
        
        # STEP 1: Calib Capture
        lf1 = ttk.LabelFrame(root, text="1. Calibration Capture")
        lf1.pack(fill=tk.X, padx=10, pady=5)
        
        f1_top = ttk.Frame(lf1)
        f1_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f1_top, text="Number of Poses:").pack(side=tk.LEFT)
        ttk.Spinbox(f1_top, from_=3, to=20, textvariable=self.num_poses, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(lf1, text="Capture Calib Images", command=self.do_calib_capture).pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(lf1, text="Save Folder:").pack(anchor=tk.W, padx=5)
        ttk.Entry(lf1, textvariable=self.calib_capture_dir).pack(fill=tk.X, padx=5, pady=(0,5))
        
        # STEP 2: Calib Process
        lf2 = ttk.LabelFrame(root, text="2. Calibration Processing")
        lf2.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(lf2, text="Compute Calibration (Select Folder)", command=self.do_calib_compute).pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(lf2, text="Result File (.mat):").pack(anchor=tk.W, padx=5)
        ttk.Entry(lf2, textvariable=self.calib_file).pack(fill=tk.X, padx=5, pady=(0,5))
        
        # STEP 3: Scan Capture
        lf3 = ttk.LabelFrame(root, text="3. Scan Capture")
        lf3.pack(fill=tk.X, padx=10, pady=5)
        
        f3 = ttk.Frame(lf3); f3.pack(fill=tk.X)
        ttk.Label(f3, text="Object Name:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(f3, textvariable=self.scan_name).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        ttk.Button(lf3, text="Capture Scan Images", command=self.do_scan_capture).pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(lf3, text="Scan Folder:").pack(anchor=tk.W, padx=5)
        ttk.Entry(lf3, textvariable=self.scan_capture_dir).pack(fill=tk.X, padx=5, pady=(0,5))
        
        # STEP 4: Cloud Gen
        lf4 = ttk.LabelFrame(root, text="4. Point Cloud Generation")
        lf4.pack(fill=tk.X, padx=10, pady=5)
        
        f4_top = ttk.Frame(lf4)
        f4_top.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f4_top, text="Calib File (.mat):").pack(side=tk.LEFT)
        ttk.Entry(lf4, textvariable=self.calib_file).pack(fill=tk.X, padx=5)
        ttk.Button(lf4, text="Select .mat File", command=self.select_calib_file).pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Button(lf4, text="Generate .PLY (Select Scan Folder)", command=self.do_cloud_gen).pack(fill=tk.X, padx=5, pady=5)

    def setup_processing_tab(self):
        root = self.tab_proc
        ttk.Label(root, text="Step 2: Cleanup & Process (Batch)", font=("Arial", 14, "bold")).pack(pady=10)
        
        ttk.Label(root, text="Pipeline: Load -> Remove Background -> Remove Outliers -> Save", foreground="blue").pack()

        # Folders
        lf_files = ttk.LabelFrame(root, text="Files")
        lf_files.pack(fill=tk.X, padx=10, pady=5)
        
        # Input
        f_in = ttk.Frame(lf_files); f_in.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in, text="Select Input Folder", command=lambda: self.sel_dir(self.proc_input_dir)).pack(side=tk.LEFT)
        ttk.Entry(f_in, textvariable=self.proc_input_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Output
        f_out = ttk.Frame(lf_files); f_out.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_out, text="Select Output Folder", command=lambda: self.sel_dir(self.proc_output_dir)).pack(side=tk.LEFT)
        ttk.Entry(f_out, textvariable=self.proc_output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # 1. BG Remove Params
        lf_bg = ttk.LabelFrame(root, text="1. Background Removal (Plane Segmentation)")
        lf_bg.pack(fill=tk.X, padx=10, pady=5)
        
        f_dist = ttk.Frame(lf_bg); f_dist.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_dist, text="Distance Threshold (default 50.0):").pack(side=tk.LEFT)
        ttk.Entry(f_dist, textvariable=self.bg_dist_thresh, width=10).pack(side=tk.LEFT, padx=5)
        
        f_rn = ttk.Frame(lf_bg); f_rn.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_rn, text="RANSAC n (3) & Iterations (1000):").pack(side=tk.LEFT)
        ttk.Entry(f_rn, textvariable=self.bg_ransac_n, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Entry(f_rn, textvariable=self.bg_iterations, width=8).pack(side=tk.LEFT, padx=5)
        
        # Explanation for BG
        bg_desc = ("Distance Thresh: Max distance a point can be from the wall plane to be considered 'wall'.\n"
                   "RANSAC n: Points sampled per iteration. Iterations: How many times to try fitting the plane.")
        ttk.Label(lf_bg, text=bg_desc, foreground="#555", justify=tk.LEFT, wraplength=550).pack(padx=5, pady=5)

        # 2. Outlier Params
        lf_out = ttk.LabelFrame(root, text="2. Statistical Outlier Removal")
        lf_out.pack(fill=tk.X, padx=10, pady=5)
        
        f_nb = ttk.Frame(lf_out); f_nb.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_nb, text="nb_neighbors (20):").pack(side=tk.LEFT)
        ttk.Entry(f_nb, textvariable=self.proc_nb_neighbors, width=10).pack(side=tk.LEFT, padx=5)
        
        f_std = ttk.Frame(lf_out); f_std.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(f_std, text="std_ratio (2.0):").pack(side=tk.LEFT)
        ttk.Entry(f_std, textvariable=self.proc_std_ratio, width=10).pack(side=tk.LEFT, padx=5)
        
        # Explanation for Outlier
        out_desc = ("nb_neighbors: Points to analyze around each point. Higher = smoother/safer but slower.\n"
                    "std_ratio: Threshold. Lower (0.5-1.0) = Aggressive removal. Higher (2.0+) = Conservative.")
        ttk.Label(lf_out, text=out_desc, foreground="#555", justify=tk.LEFT, wraplength=550).pack(padx=5, pady=5)

        # Run
        ttk.Button(root, text="Run Processing Pipeline", command=self.do_batch_processing).pack(fill=tk.X, padx=20, pady=20)
    
    def do_batch_processing(self):
        in_dir = self.proc_input_dir.get()
        out_dir = self.proc_output_dir.get()
        
        # BG Params
        bg_dist = self.bg_dist_thresh.get()
        bg_rn = self.bg_ransac_n.get()
        bg_iters = self.bg_iterations.get()
        
        # Outlier Params
        nb = self.proc_nb_neighbors.get()
        std = self.proc_std_ratio.get()
        
        if not in_dir or not out_dir:
            messagebox.showerror("Error", "Select both Input and Output folders.")
            return

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        ply_files = glob.glob(os.path.join(in_dir, "*.ply"))
        if not ply_files:
            messagebox.showerror("Error", "No .ply files found in input folder.")
            return

        def run():
            count = 0
            errors = 0
            total = len(ply_files)
            
            for fpath in ply_files:
                fname = os.path.basename(fpath)
                out_path = os.path.join(out_dir, fname.replace(".ply", "_processed.ply"))
                
                print(f"[Task] Processing {fname}...")
                try:
                    # 1. Load & Remove BG
                    pcd = self.processor.remove_background(fpath, distance_threshold=bg_dist, ransac_n=bg_rn, num_iterations=bg_iters, return_obj=True)
                    
                    # 2. Remove Outliers
                    pcd = self.processor.remove_outliers(pcd, nb_neighbors=nb, std_ratio=std, return_obj=True)
                    
                    # 3. Save
                    o3d.io.write_point_cloud(out_path, pcd)
                    print(f"[Task] Saved {out_path}")
                    count += 1
                except Exception as e:
                    print(f"[Task] Error processing {fname}: {e}")
                    errors += 1
            
            msg = f"Pipeline Completed.\nProcessed: {count}/{total}\nErrors: {errors}\nSaved to: {out_dir}"
            self.root.after(0, lambda: messagebox.showinfo("Processing Done", msg))
        
        threading.Thread(target=run, daemon=True).start()

        threading.Thread(target=run, daemon=True).start()

    def setup_merge_tab(self):
        root = self.tab_merge
        ttk.Label(root, text="Step 3: 360 Degree Merge (Multi-view Alignment)", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Files
        lf_files = ttk.LabelFrame(root, text="Files")
        lf_files.pack(fill=tk.X, padx=10, pady=5)
        
        # Input Folder
        f_in = ttk.Frame(lf_files); f_in.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in, text="Select Input Folder (All PLYs)", command=lambda: self.sel_dir(self.merge_input_dir)).pack(side=tk.LEFT)
        ttk.Entry(f_in, textvariable=self.merge_input_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Output File
        f_out = ttk.Frame(lf_files); f_out.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_out, text="Select Output File (.ply)", command=lambda: self.sel_file_save(self.merge_output_file, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_out, textvariable=self.merge_output_file).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Params
        lf_param = ttk.LabelFrame(root, text="Parameters")
        lf_param.pack(fill=tk.X, padx=10, pady=5)
        
        f_vx = ttk.Frame(lf_param); f_vx.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_vx, text="Voxel Size (m) [Default 0.02]:").pack(side=tk.LEFT)
        ttk.Entry(f_vx, textvariable=self.merge_voxel, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_vx, text="(Size of downsampling grid. Smaller = distincter but slower/noisier. Larger = coarse alignment.)", foreground="#555").pack(side=tk.LEFT)

        # Run
        ttk.Button(root, text="Merge 360 Point Clouds", command=self.do_merge_360).pack(fill=tk.X, padx=20, pady=20)

    def do_merge_360(self):
        in_dir = self.merge_input_dir.get()
        out_file = self.merge_output_file.get()
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

    def setup_360_meshing_tab(self):
        root = self.tab_mesh360
        ttk.Label(root, text="Step 4: 360 Meshing (Poisson + Normal Re-orientation)", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Files
        lf_files = ttk.LabelFrame(root, text="Files")
        lf_files.pack(fill=tk.X, padx=10, pady=5)
        
        # Input
        f_in = ttk.Frame(lf_files); f_in.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in, text="Select Input .PLY", command=lambda: self.sel_file_load(self.m360_input_ply, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_in, textvariable=self.m360_input_ply).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Output
        f_out = ttk.Frame(lf_files); f_out.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_out, text="Select Output .STL", command=lambda: self.sel_file_save(self.m360_output_stl, "STL")).pack(side=tk.LEFT)
        ttk.Entry(f_out, textvariable=self.m360_output_stl).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Parameters
        lf_param = ttk.LabelFrame(root, text="Parameters")
        lf_param.pack(fill=tk.X, padx=10, pady=5)
        
        # Orientation Mode
        f_m = ttk.Frame(lf_param); f_m.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_m, text="Orientation Mode:").pack(side=tk.LEFT)
        ttk.Combobox(f_m, textvariable=self.m360_mode, values=["radial", "tangent"], state="readonly", width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_m, text="(Radial = Outwards from center | Tangent = Graph consistency)", foreground="#555").pack(side=tk.LEFT)

        f_d = ttk.Frame(lf_param); f_d.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_d, text="Poisson Depth (Default 10):").pack(side=tk.LEFT)
        ttk.Entry(f_d, textvariable=self.m360_depth, width=10).pack(side=tk.LEFT, padx=5)
        
        f_t = ttk.Frame(lf_param); f_t.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_t, text="Density Trim (0.0 = Watertight):").pack(side=tk.LEFT)
        ttk.Entry(f_t, textvariable=self.m360_trim, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Label(f_t, text="(0.0 fills EVERYTHING. >0.0 cuts bubbles)", foreground="#555").pack(side=tk.LEFT)
        
        # Run
        ttk.Button(root, text="Run 360 Meshing", command=self.do_360_meshing).pack(fill=tk.X, padx=20, pady=20)

    def do_360_meshing(self):
        i = self.m360_input_ply.get()
        o = self.m360_output_stl.get()
        d = self.m360_depth.get()
        t = self.m360_trim.get()
        m = self.m360_mode.get()
        
        if not i or not o: messagebox.showerror("Error", "Select files first."); return
        
        def run():
            try:
                self.processor.mesh_360(i, o, d, t, m)
                self.root.after(0, lambda: messagebox.showinfo("Done", f"360 Mesh Saved to:\n{o}"))
            except Exception as e:
                err_msg = str(e)
                print(err_msg)
                # self.root.after(0, lambda: messagebox.showerror("Error", err_msg)) 
                # Don't popup error if thread dying... print is safer.
                print(f"Error: {e}")
        
        threading.Thread(target=run, daemon=True).start()

    def setup_turntable_tab(self):
        root = self.tab_turntable
        ttk.Label(root, text="Step 5: Auto-Scan with Turntable (Arduino)", font=("Arial", 14, "bold")).pack(pady=10)
        
        # 1. Connection
        lf_conn = ttk.LabelFrame(root, text="1. Arduino Connection")
        lf_conn.pack(fill=tk.X, padx=10, pady=5)
        
        f_p = ttk.Frame(lf_conn); f_p.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_p, text="Port:").pack(side=tk.LEFT)
        self.cb_port = ttk.Combobox(f_p, textvariable=self.tt_port, width=15)
        self.cb_port.pack(side=tk.LEFT, padx=5)
        ttk.Button(f_p, text="Refresh", command=self.refresh_ports).pack(side=tk.LEFT, padx=2)
        ttk.Button(f_p, text="Connect", command=self.connect_arduino).pack(side=tk.LEFT, padx=5)
        
        # 2. Settings
        lf_set = ttk.LabelFrame(root, text="2. Scan Settings")
        lf_set.pack(fill=tk.X, padx=10, pady=5)
        
        f_deg = ttk.Frame(lf_set); f_deg.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_deg, text="Degrees per Turn (e.g., 30):").pack(side=tk.LEFT)
        ttk.Entry(f_deg, textvariable=self.tt_degrees, width=10).pack(side=tk.LEFT, padx=5)
        
        f_cnt = ttk.Frame(lf_set); f_cnt.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_cnt, text="Number of Turns (e.g., 12):").pack(side=tk.LEFT)
        ttk.Entry(f_cnt, textvariable=self.tt_turns, width=10).pack(side=tk.LEFT, padx=5)
        
        # Display Totals
        self.lbl_total = ttk.Label(lf_set, text="Total: 360 degrees", foreground="blue")
        self.lbl_total.pack(padx=5, pady=5)
        # Update listener
        self.tt_degrees.trace_add("write", self.update_tt_totals)
        self.tt_turns.trace_add("write", self.update_tt_totals)
        
        # 3. Output
        lf_out = ttk.LabelFrame(root, text="3. Output")
        lf_out.pack(fill=tk.X, padx=10, pady=5)
        
        f_name = ttk.Frame(lf_out); f_name.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_name, text="Base Object Name:").pack(side=tk.LEFT)
        ttk.Entry(f_name, textvariable=self.tt_base_name).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        f_dir = ttk.Frame(lf_out); f_dir.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_dir, text="Select Save Folder", command=lambda: self.sel_dir(self.tt_save_dir)).pack(side=tk.LEFT)
        ttk.Entry(f_dir, textvariable=self.tt_save_dir).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # 4. Status & Run
        ttk.Label(root, textvariable=self.tt_status, font=("Arial", 12)).pack(pady=10)
        ttk.Button(root, text="START AUTO SCAN", command=self.do_auto_scan_sequence, state="normal").pack(fill=tk.X, padx=20, pady=10)

    def refresh_ports(self):
        ports = self.arduino.get_ports()
        self.cb_port['values'] = ports
        if ports: self.cb_port.current(0)
    
    def connect_arduino(self):
        p = self.tt_port.get()
        if not p: messagebox.showerror("Error", "Select a port"); return
        
        ok, msg = self.arduino.connect(p)
        if ok: messagebox.showinfo("Connected", "Arduino Connected!")
        else: messagebox.showerror("Error", f"Failed: {msg}")

    def update_tt_totals(self, *args):
        try:
            d = self.tt_degrees.get()
            t = self.tt_turns.get()
            total = d * t
            self.lbl_total.config(text=f"Total: {total} degrees ({t} scans)")
        except: pass

    def do_auto_scan_sequence(self):
        # 1. Validate
        if not self.arduino.ser:
            # For testing without arduino, maybe allow? But user asked for arduino works.
            # Let's check but allow bypass if we want to sim... No, stick to requirements.
            # "work with arduino... untill met the condition"
            if not messagebox.askyesno("Confirm", "Arduino not connected (in software). Continue anyway (Simulation)?"):
                return
        
        deg = self.tt_degrees.get()
        turns = self.tt_turns.get()
        base_name = self.tt_base_name.get()
        root_dir = self.tt_save_dir.get()
        
        if not base_name or not root_dir:
            messagebox.showerror("Error", "Check Output settings"); return
            
        # Create Main Folder
        main_folder = os.path.join(root_dir, f"{base_name}_{int(deg)}deg_AUTO")
        os.makedirs(main_folder, exist_ok=True)
        
        # Popup Progress
        top = tk.Toplevel(self.root)
        top.title("Auto Scan Progress")
        top.geometry("400x300")
        
        lbl_info = ttk.Label(top, text="Starting...", font=("Arial", 12))
        lbl_info.pack(pady=20)
        
        lbl_time = ttk.Label(top, text="Time: 0s")
        lbl_time.pack(pady=5)
        
        pb = ttk.Progressbar(top, maximum=turns, mode='determinate')
        pb.pack(fill=tk.X, padx=20, pady=20)
        
        # Thread Logic
        def run_thread():
            start_time = time.time()
            
            for i in range(turns):
                # Update UI
                elapsed = time.time() - start_time
                avg_time = (elapsed / i) if i > 0 else 0
                rem_time = avg_time * (turns - i)
                
                msg = f"Scanning {i+1}/{turns}\nElapsed: {int(elapsed)}s\nEst. Left: {int(rem_time)}s"
                
                # Safe UI update
                self.root.after(0, lambda: lbl_info.config(text=msg))
                self.root.after(0, lambda: lbl_time.config(text=f"Time: {int(elapsed)}s"))
                self.root.after(0, lambda: pb.config(value=i))
                
                # 1. CAPTURE
                # Subfolder: ObjectName_Anglexx
                current_angle = i * deg
                sub_name = f"{base_name}_{int(current_angle)}deg_scan"
                sub_path = os.path.join(main_folder, sub_name)
                
                print(f"[Auto] Capturing to {sub_path}")
                
                try:
                    self.sys.capture_scan(sub_path)
                except Exception as e:
                    print(f"Scan Error: {e}")
                    self.root.after(0, lambda: messagebox.showerror("Error", f"Scan failed: {e}"))
                    return

                # 2. MOVE
                if i < turns - 1: # Don't move after last scan? Or maybe return to 0?
                    # User: "turn 30 degree... after round table move it will scan again"
                    
                    msg_move = f"Rotating {deg} degrees..."
                    self.root.after(0, lambda: lbl_info.config(text=msg_move))
                    
                    if self.arduino.ser:
                        self.arduino.rotate(deg)
                        # Wait for DONE
                        done = self.arduino.wait_for_done(timeout=10) # 10s timeout per move
                        if not done:
                            print("Warning: Arduino move timeout or no DONE received.")
                    else:
                        time.sleep(2) # Sim delay

            # Finish
            total_time = time.time() - start_time
            done_msg = f"Auto Scan Complete!\nTotal Time: {int(total_time)}s\nLocation: {main_folder}"
            self.root.after(0, lambda: messagebox.showinfo("Done", done_msg))
            self.root.after(0, top.destroy)

        threading.Thread(target=run_thread, daemon=True).start()

    def setup_stl_tab(self):
        root = self.tab_recon
        ttk.Label(root, text="STL Reconstruction", font=("Arial", 14, "bold")).pack(pady=10)
        
        # File Selection
        lf_files = ttk.LabelFrame(root, text="Files")
        lf_files.pack(fill=tk.X, padx=10, pady=5)
        
        # Input
        f_in = ttk.Frame(lf_files); f_in.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_in, text="Select Input .PLY", command=lambda: self.sel_file_load(self.s_input_ply, "PLY")).pack(side=tk.LEFT)
        ttk.Entry(f_in, textvariable=self.s_input_ply).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Output
        f_out = ttk.Frame(lf_files); f_out.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(f_out, text="Select Output .STL", command=lambda: self.sel_file_save(self.s_output_stl, "STL")).pack(side=tk.LEFT)
        ttk.Entry(f_out, textvariable=self.s_output_stl).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Mode
        lf_mode = ttk.LabelFrame(root, text="Method & Parameters")
        lf_mode.pack(fill=tk.X, padx=10, pady=5)
        
        f_m = ttk.Frame(lf_mode); f_m.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(f_m, text="Reconstruction Mode:").pack(side=tk.LEFT)
        cb = ttk.Combobox(f_m, textvariable=self.s_mode, values=["watertight", "surface"], state="readonly")
        cb.pack(side=tk.LEFT, padx=5)
        cb.bind("<<ComboboxSelected>>", self.update_stl_params)
        
        # Dynamic Frame
        self.f_stl_params = ttk.Frame(lf_mode)
        self.f_stl_params.pack(fill=tk.X, padx=5, pady=5)
        self.update_stl_params() # Init
        
        # Run
        ttk.Button(root, text="Run STL Reconstruction", command=self.do_stl_recon).pack(fill=tk.X, padx=20, pady=20)

    def update_stl_params(self, event=None):
        # Clear frame
        for widget in self.f_stl_params.winfo_children():
            widget.destroy()
            
        mode = self.s_mode.get()
        if mode == "watertight":
            ttk.Label(self.f_stl_params, text="Poisson Depth (default 10):").pack(anchor=tk.W)
            ttk.Entry(self.f_stl_params, textvariable=self.s_depth).pack(fill=tk.X)
            ttk.Label(self.f_stl_params, text="Creates a closed (watertight) mesh. Higher depth = more detail but slower.", foreground="#555").pack(anchor=tk.W)
        else:
            ttk.Label(self.f_stl_params, text="Ball Radii Multipliers (default '1, 2, 4'):").pack(anchor=tk.W)
            ttk.Entry(self.f_stl_params, textvariable=self.s_radii).pack(fill=tk.X)
            ttk.Label(self.f_stl_params, text="Multiples of average point distance. Connects dots without filling large holes.", foreground="#555").pack(anchor=tk.W)

    # --- HELPER ACTIONS ---
    def sel_file_load(self, var, ftype):
        ext = "*.ply" if ftype == "PLY" else "*.*"
        f = filedialog.askopenfilename(filetypes=[(ftype, ext)])
        if f: 
            var.set(f)
            # Auto set output if empty
    # --- HELPER ACTIONS ---
    def sel_file_load(self, var, ftype):
        ext = "*.ply" if ftype == "PLY" else "*.*"
        f = filedialog.askopenfilename(filetypes=[(ftype, ext)])
        if f: 
            var.set(f)
            # Auto set output if empty
            if ftype == "PLY":
                # If we are in STL tab
                if var == self.s_input_ply and not self.s_output_stl.get():
                    self.s_output_stl.set(f.replace(".ply", ".stl"))
                # If we are in 360 Mesh tab
                if var == self.m360_input_ply and not self.m360_output_stl.get():
                    self.m360_output_stl.set(f.replace(".ply", ".stl"))

    def sel_file_save(self, var, ftype):
        ext = "*.ply" if ftype == "PLY" else "*.stl"
        f = filedialog.asksaveasfilename(filetypes=[(ftype, ext)], defaultextension=ext.replace("*", ""))
        if f: var.set(f)

    def sel_dir(self, var):
        d = filedialog.askdirectory()
        if d: var.set(d)



    def do_stl_recon(self):
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

    def update_ip(self):
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]; s.close()
            self.ip_lbl.config(text=f"Connect Phone to: http://{ip}:5000")
        except: pass

    # --- ACTIONS ---
    def do_calib_capture(self):
        # Update dir based on entry
        d = self.calib_capture_dir.get()
        n = self.num_poses.get() # Get value from UI
        threading.Thread(target=self.sys.capture_calibration, args=(d, n), daemon=True).start()

    def do_calib_compute(self):
        # Ask for folder, default to current value of Step 1
        initial = self.calib_capture_dir.get()
        if not os.path.exists(initial): initial = os.getcwd()
        
        in_dir = filedialog.askdirectory(title="Select Calibration Images Folder", initialdir=initial)
        if not in_dir: return
        
        # Update Entry
        self.calib_capture_dir.set(in_dir)
        
        # Output file default
        out_file = os.path.join(in_dir, "calib.mat")
        
        # Start Thread for Step 1
        threading.Thread(target=self.run_calib_analysis, args=(in_dir, out_file), daemon=True).start()

    def run_calib_analysis(self, in_dir, out_file):
        try:
            errors, available_poses = self.sys.analyze_calibration(in_dir)
            # Schedule UI on main thread
            self.root.after(0, self.prompt_pose_selection, errors, available_poses, in_dir, out_file)
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Calib Error", err_msg))

    def prompt_pose_selection(self, errors, available_poses, in_dir, out_file):
        # Runs on Main Thread
        msg = "Calibration Analysis (Error in px):\n\n"
        for pose, (ce, pe) in errors.items():
            msg += f"{pose}: Cam={ce:.2f}, Proj={pe:.2f}\n"
        msg += "\nEnter poses to KEEP (e.g., '1,3,4' OR 'all' for all):"
        
        user_input = simpledialog.askstring("Select Poses", msg, parent=self.root)
        if not user_input: return
        
        selected_poses = []
        user_input = user_input.strip()
        
        if user_input.lower() == 'all':
            selected_poses = available_poses
        else:
            selected_indices = [x.strip() for x in user_input.split(',')]
            for idx in selected_indices:
                # Handle both "1" and "pose_1" input styles if needed, but assuming ID input
                name = f"pose_{idx}"
                # If user typed full name "pose_1", accommodate
                if idx.startswith("pose_"): name = idx
                
                if name in available_poses: selected_poses.append(name)
        
        # Continue to Step 2 in thread
        threading.Thread(target=self.run_calib_final, args=(in_dir, selected_poses, out_file), daemon=True).start()

    def run_calib_final(self, in_dir, selected_poses, out_file):
        try:
            self.sys.calibrate_final(in_dir, selected_poses, out_file)
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Calibration Saved to:\n{out_file}"))
            # Update Step 2 result for Steps 3 & 4 (Thread safe variable set? Tkinter Vars are NOT thread safe technically but often work. 
            # Better to schedule it.)
            self.root.after(0, lambda: self.calib_file.set(out_file))
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda: messagebox.showerror("Calib Final Error", err_msg))

    def do_scan_capture(self):
        # Update scan dir based on name
        base = os.path.join(DEFAULT_ROOT, "scans")
        name = self.scan_name.get()
        path = os.path.join(base, name)
        self.scan_capture_dir.set(path)
        
        threading.Thread(target=self.sys.capture_scan, args=(path,), daemon=True).start()

    def select_calib_file(self):
        initial = self.calib_file.get()
        if not initial or not os.path.exists(os.path.dirname(initial)): initial = os.getcwd()
        f = filedialog.askopenfilename(title="Select Calibration .mat", initialdir=os.path.dirname(initial), filetypes=[("MAT Files", "*.mat")])
        if f: self.calib_file.set(f)

    def do_cloud_gen(self):
        # Ask for folder, default to Step 3 result
        initial = self.scan_capture_dir.get()
        if not os.path.exists(initial): initial = os.getcwd()
        
        scan_dir = filedialog.askdirectory(title="Select Scan Images Folder", initialdir=initial)
        if not scan_dir: return
        
        calib_path = self.calib_file.get()
        
        threading.Thread(target=self.run_cloud_gen, args=(scan_dir, calib_path), daemon=True).start()

    def run_cloud_gen(self, scan_dir, calib_path):
        try:
            self.sys.generate_cloud(scan_dir, calib_path)
            self.root.after(0, lambda: messagebox.showinfo("Done", f"Cloud generation finished for\n{os.path.basename(scan_dir)}"))
        except Exception as e:
             err_msg = str(e)
             self.root.after(0, lambda: messagebox.showerror("Error", err_msg))

def main():
    print("----------------------------------------------------------------")
    print("   Project 3D Scanner Controller v2.1 (With 360 Radial Fix)")
    print("----------------------------------------------------------------")
    threading.Thread(target=run_flask, daemon=True).start()
    root = tk.Tk()
    ScannerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
