import os
import cv2
import glob
import time
import uuid
import numpy as np
import scipy.io
from tkinter import messagebox

# Import Configuration variables and server state from other files
from config import SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_OFFSET_X, PROJ_VALUE, D_SAMPLE_PROJ, CHECKER_ROWS, CHECKER_COLS, SQUARE_SIZE
from server import SERVER_STATE

class SLSystem:
    # Class for managing the Structured Light Scanning System
    # Responsible for projecting light, controlling patterns, and analyzing 3D distance
    def __init__(self):
        # ---------------------------------------------------------
        # Name of the projector window that will be opened fullscreen
        self.window_name = "Projector"
        
    def init_projector(self):
        # ---------------------------------------------------------
        # Function to initialize the projector display system
        # Create an OpenCV window (resizable)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # Move this window to the second monitor (based on the set OFFSET, e.g., 1920)
        cv2.moveWindow(self.window_name, SCREEN_OFFSET_X, 0)
        # Force the projector window to fullscreen, removing menu bars
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        
        # Create a completely black image matching the projector's resolution
        black = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)
        # Show the black image (to clear the screen and prepare)
        cv2.imshow(self.window_name, black)
        # Wait 50 milliseconds for the system to draw the image
        cv2.waitKey(50)

    def close_projector(self):
        # ---------------------------------------------------------
        # Function to close the projector window
        cv2.destroyWindow(self.window_name)

    def generate_patterns(self):
        # ---------------------------------------------------------
        # Function to generate a set of black-and-white stripe patterns (Gray Code) to project
        
        # Calculate the output resolution (if downsampling, image size will be smaller)
        width, height = SCREEN_WIDTH // D_SAMPLE_PROJ, SCREEN_HEIGHT // D_SAMPLE_PROJ
        
        # Calculate number of bit images needed for the X-axis (vertical)
        n_cols = int(np.ceil(np.log2(width)))
        # Calculate number of bit images needed for the Y-axis (horizontal)
        n_rows = int(np.ceil(np.log2(height)))
        
        def get_gray_1d(n):
            # Algorithm function to arrange binary numbers into Gray Code
            # Example: 00, 01, 11, 10
            if n == 1: return ['0', '1'] # If only one bit is needed (first level)
            prev = get_gray_1d(n - 1)  # Recursively call itself to create the previous base
            # Prepend 0 to original base, combined with prepending 1 to reversed original base
            return ['0' + s for s in prev] + ['1' + s for s in prev[::-1]]

        # Generate Gray Code frequency patterns for vertical and horizontal axes
        col_gray = get_gray_1d(n_cols)
        row_gray = get_gray_1d(n_rows)
        
        P = [[], []] # Prepare a 2D list (index [0] for vertical, [1] for horizontal)
        
        # Create vertical pattern images bit by bit
        for b in range(n_cols):
            pat = np.zeros((height, width), dtype=np.uint8) # Black background
            for c in range(width): # Loop through the width
                # If the code's bit is '1', fill the entire column with white
                if c < len(col_gray) and col_gray[c][b] == '1': pat[:, c] = 1
            P[0].append(pat) # Store in the vertical list

        # Create horizontal pattern images bit by bit
        for b in range(n_rows):
            pat = np.zeros((height, width), dtype=np.uint8) # Black background
            for r in range(height): # Loop through the height
                # If the code's bit is '1', make the horizontal line a bright stripe
                if r < len(row_gray) and row_gray[r][b] == '1': pat[r, :] = 1
            P[1].append(pat) # Store in the horizontal list
            
        return P # Return all generated pattern sets

    def trigger_capture(self, save_path):
        # ---------------------------------------------------------
        # Function to command image capture via the server and wait to receive the picture
        
        # Clear the wait Event status (set to not yet received picture)
        SERVER_STATE["upload_received_event"].clear()
        
        # Tell the backend that the next picture must be saved to this path
        SERVER_STATE["last_image_path"] = save_path
        # Create a unique queue order ID
        SERVER_STATE["command_id"] = str(uuid.uuid4())
        # Specify that the server must now release the 'capture' command
        SERVER_STATE["command"] = "capture"
        
        # Program waits for 'picture received' signal from upload function (max 20 secs)
        if not SERVER_STATE["upload_received_event"].wait(timeout=20):
            print(f"[Error] Timeout capturing {save_path}") # If it takes too long, return error
            return False
            
        # Once the picture is received, return to the 'idle' state
        SERVER_STATE["command"] = "idle"
        return True # Report success

    # ---------------------------------------------------------
    # 1. CAPTURE CALIBRATION (Step 1: Accumulate images to Calibrate camera and projector)
    # ---------------------------------------------------------
    def capture_calibration(self, save_dir, num_poses=5):
        # Receive the desired number of Poses
        try:
            num_poses = int(num_poses)
        except:
            num_poses = 5
            
        # Open the projector window
        self.init_projector()
        # Calculate and load the patterns to project into memory
        P = self.generate_patterns()
        
        # Prepare to store basic images needed for every pose
        patterns = []
        # White light (full brightness) (to see the board fully)
        white = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8) * PROJ_VALUE
        # Pitch black light 
        black = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)
        
        patterns.append(("01.png", white)) # First image
        patterns.append(("02.png", black)) # Second image
        
        idx = 3 # Stripe images start at index 3
        # Expand patterns into image pairs (Normal light and its Inverse)
        for j in range(2): 
            for pat in P[j]:
                pat_img = (pat * PROJ_VALUE).astype(np.uint8) # Light pattern based on code
                inv_img = ((1-pat) * PROJ_VALUE).astype(np.uint8) # Inverse of the pattern
                
                # If patterns were downsampled, scale back to full screen before projecting
                if D_SAMPLE_PROJ > 1:
                    pat_img = cv2.resize(pat_img, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    inv_img = cv2.resize(inv_img, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    
                # Store in list and set filename for capture
                patterns.append((f"{idx:02d}.png", pat_img)); idx+=1
                patterns.append((f"{idx:02d}.png", inv_img)); idx+=1

        # Create a folder to store this image set
        os.makedirs(save_dir, exist_ok=True)
        # Notify user that camera preparation is starting
        messagebox.showinfo("Step 1", f"Starting Calibration Capture ({num_poses} poses).\nImages will be saved to:\n{save_dir}")
        
        # Loop to shoot and project light pose by pose
        for pose in range(1, num_poses + 1):
            pose_dir = os.path.join(save_dir, f"pose_{pose}")
            os.makedirs(pose_dir, exist_ok=True) # Folder for this pose
            
            # Turn screen fully white so the user can arrange the board clearly
            cv2.imshow(self.window_name, white)
            cv2.waitKey(50)
            
            # Notify user: 'Adjust the chessboard angle, then click OK'
            messagebox.showinfo("Calibration", f"Pose {pose}/{num_poses}.\nMove board then click OK.")
            
            # After OK, start looping projection picture by picture 
            for fname, img in patterns:
                cv2.imshow(self.window_name, img) # Project the pattern onto the projector
                cv2.waitKey(250) # Delay to allow screen light to enter the camera sensor fully
                
                # Command shutter and save
                if not self.trigger_capture(os.path.join(pose_dir, fname)):
                    messagebox.showerror("Error", "Capture timeout.") # If phone battery drops or network disconnects, report error
                    self.close_projector()
                    return

        # Close projector after 100% completion
        self.close_projector()
        messagebox.showinfo("Step 1 Done", "Calibration Capture Complete.")

    # ---------------------------------------------------------
    # 2. PROCESS CALIBRATION (Step 2: Analyze camera disparity/margins)
    # ---------------------------------------------------------
    def analyze_calibration(self, input_dir):
        # Function to analyze poses and overlap to estimate initial errors
        
        # Find all pose folders in the storage directory
        available_poses = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
        
        # Need at least 3 poses for the 3D stereo equations to be accurate
        if len(available_poses) < 3: 
            raise ValueError(f"Need at least 3 pose folders in {input_dir}")

        print(f"[Calib] analyzing {len(available_poses)} poses...")
        
        # Analyze Reprojection Error for each captured pose 
        errors = self.compute_reprojection_errors(input_dir, available_poses)
        
        return errors, available_poses # Return pose list and distortion levels to user for filtering

    def load_calib_data(self, base_dir, pose_list):
        # Core function to decode light and read coordinates from Calibration data
        # Finds 'matching points' (Camera <-> Real World <-> Projector screen)
        
        # Create exact 3D coordinates for the chessboard as world dimensions 
        objp = np.zeros((CHECKER_ROWS * CHECKER_COLS, 3), np.float32)
        objp[:, :2] = np.mgrid[0:CHECKER_ROWS, 0:CHECKER_COLS].T.reshape(-1, 2)
        objp *= SQUARE_SIZE # Multiply by actual chessboard square size
        
        # Arrays for summarized totals
        obj_pts = []  # Grid points in world (Object Points)
        cam_pts = []  # Points found on photo (Camera Points)
        proj_pts = [] # Projector screen points from light decode
        valid_poses = [] # Keep only poses with complete/usable images
        img_shape = None # Camera image size
        
        # Process each pose sequentially
        for pose in pose_list:
            path = os.path.join(base_dir, pose)
            
            # Open full white image (01.png) to find chessboard grid easily
            img = cv2.imread(os.path.join(path, "01.png")) 
            if img is None: continue # Skip if file broken
            if img_shape is None: img_shape = (img.shape[1], img.shape[0]) # Summarize image dimensions
            
            # Enhance white image for clarity
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convert to grayscale
            blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Blur to remove noise (Gaussian Blur)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # Pull shadow contrast (Adaptive Histogram Equalization)
            enhanced = clahe.apply(blurred) 
            
            # Find the chessboard corners in the image
            ret, corners = cv2.findChessboardCorners(enhanced, (CHECKER_ROWS, CHECKER_COLS), None)
            
            if ret: # If chessboard corners found successfully
                # Refine corners to sub-pixel resolution
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                
                # Draw chessboard corners and save preview image for user
                preview_dir = os.path.join(base_dir, "corners_preview")
                os.makedirs(preview_dir, exist_ok=True)
                preview_img = img.copy()
                cv2.drawChessboardCorners(preview_img, (CHECKER_ROWS, CHECKER_COLS), corners2, ret)
                cv2.imwrite(os.path.join(preview_dir, f"{pose}.png"), preview_img)
                
                # Start decoding structured light data
                files = sorted(glob.glob(os.path.join(path, "*.png")))
                
                # Check max bits for display
                n_col_bits = int(np.ceil(np.log2(SCREEN_WIDTH)))
                n_row_bits = int(np.ceil(np.log2(SCREEN_HEIGHT)))
                
                # Verify if all structured light images are present
                if len(files) - 2 < 2 * (n_col_bits + n_row_bits):
                    print(f"Skipping {pose}: Not enough images.")
                    continue
                
                base_idx = 2 # Start reading image pairs from index 3 (Index = 2)
                
                # Sub-function to absorb bit image sequences into Projector X,Y planes
                def decode_seq(n_bits, idx_start): 
                    # Compares positive image with its inverse to determine bit 1 or 0 
                    val = np.zeros(len(corners2)) # List to store data for all target pixels
                    idx = idx_start
                    bin_code = None
                    for b in range(n_bits):  # Read image pairs bit by bit
                        img_p = cv2.imread(files[idx], 0)     # Real image (Positive)
                        img_i = cv2.imread(files[idx+1], 0)   # Inverse image
                        idx += 2
                        
                        x = corners2[:,0,0] # Sub-pixel width coords
                        y = corners2[:,0,1] # Sub-pixel height coords
                        
                        # Extract shading values from positive and inverse
                        vp = img_p[y.astype(int), x.astype(int)]
                        vi = img_i[y.astype(int), x.astype(int)]
                        
                        # If an area is brighter than Inverse, it is a '1' bit
                        bit = (vp > vi).astype(int)
                        
                        # Encode based on Gray Binary rules
                        if b == 0: 
                            bin_code = bit
                        else: 
                            # Apply XOR to find genuine bit
                            bin_code = np.bitwise_xor(bin_code, bit)
                        
                        # Shift bits to calculate screen pixel coordinates
                        val += bin_code * (2**(n_bits - 1 - b))
                    
                    # Return decimal screen value and next index
                    return val, idx 
                
                # Execute and intersect screen coords with grid elements
                col_val, base_idx = decode_seq(n_col_bits, base_idx) # Vertical axis
                row_val, base_idx = decode_seq(n_row_bits, base_idx) # Horizontal axis
                
                # Combine columns and rows
                # Format data to fit OpenCV algorithm requirements 
                proj_pts_pose = np.column_stack((col_val, row_val)).astype(np.float32).reshape(-1, 1, 2)
                
                # Store into a single set 
                obj_pts.append(objp)
                cam_pts.append(corners2)
                proj_pts.append(proj_pts_pose)
                valid_poses.append(pose)
                
        # Return values to caller
        return obj_pts, cam_pts, proj_pts, img_shape, valid_poses

    def compute_reprojection_errors(self, base_dir, pose_list):
        # Predict initial errors via Quick Calib
        obj_pts, cam_pts, proj_pts, shape, poses = self.load_calib_data(base_dir, pose_list)
        
        # Quick camera calibration 
        rc, mc, dc, rvc, tvc = cv2.calibrateCamera(obj_pts, cam_pts, shape, None, None) 
        # Quick projector calibration (assuming it acts as a camera)
        rp, mp, dp, rvp, tvp = cv2.calibrateCamera(obj_pts, proj_pts, (SCREEN_WIDTH, SCREEN_HEIGHT), None, None)
        
        errors = {}
        for i, p in enumerate(poses):
            # Simulate back projection and measure deviation
            p2_c, _ = cv2.projectPoints(obj_pts[i], rvc[i], tvc[i], mc, dc)
            err_c = cv2.norm(cam_pts[i], p2_c, cv2.NORM_L2)/len(p2_c)
            
            p2_p, _ = cv2.projectPoints(obj_pts[i], rvp[i], tvp[i], mp, dp)
            err_p = cv2.norm(proj_pts[i], p2_p, cv2.NORM_L2)/len(p2_p)
            
            # Store Camera/Projector Errors sequentially by pose
            errors[p] = (err_c, err_p)
        return errors

    def calibrate_final(self, base_dir, selected_poses, output_file):
        # Full Stereo Calibration mapping parameters (Final step)
        obj_pts, cam_pts, proj_pts, shape, _ = self.load_calib_data(base_dir, selected_poses)
        
        # Start standalone calibrations
        print("Calibrating Camera...")
        rc, mc, dc, _, _ = cv2.calibrateCamera(obj_pts, cam_pts, shape, None, None) 
        print("Calibrating Projector...")
        rp, mp, dp, _, _ = cv2.calibrateCamera(obj_pts, proj_pts, (SCREEN_WIDTH, SCREEN_HEIGHT), None, None)
        
        # Stereo Calibrate, bonding the relationship between camera and projector 
        print("Stereo Calibration...")
        ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            obj_pts, cam_pts, proj_pts, mc, dc, mp, dp, shape, flags=cv2.CALIB_FIX_INTRINSIC
        )
        
        # Process geometry using main camera as origin (0,0,0)
        
        # 1. Camera Center is origin (Oc) set to 0,0,0
        Oc = np.zeros((3, 1))
        
        # 2. Camera Rays (Nc) passing through lens per pixel
        w, h = shape  # Bug fix for width/height screen flip edge case 
        
        u, v = np.meshgrid(np.arange(w), np.arange(h)) # Grid of every screen pixel
        fx, fy, cx, cy = K1[0,0], K1[1,1], K1[0,2], K1[1,2] # Retrieve Intrinsic parameters
        
        # Format 2D image axes to negating lens curvature 
        x_norm = (u - cx) / fx
        y_norm = (v - cy) / fy
        z_norm = np.ones_like(x_norm) # Z-axis points straight with constant depth 1
        
        # Consolidate into 3D Ray (Unit Vector)
        rays = np.stack((x_norm, y_norm, z_norm), axis=2)
        norms = np.linalg.norm(rays, axis=2, keepdims=True)
        rays /= norms # Normalize to unit length 1 to prevent variable overflow 
        Nc = rays.reshape(-1, 3).T
        
        # 3. Projector Planes (Create 4-coordinate equations for the projector's light planes: vertical wPlaneCol and horizontal wPlaneRow)
        wPlaneCol = np.zeros((SCREEN_WIDTH, 4))
        wPlaneRow = np.zeros((SCREEN_HEIGHT, 4))
        
        # Intrinsic parameters of projector
        fx_p, fy_p = K2[0,0], K2[1,1]
        cx_p, cy_p = K2[0,2], K2[1,2]
        
        R_inv = R.T # Transposed rotation for mirror reflection
        C_p_cam = -R_inv @ T # Projector origin point on camera axis
        
        # Sub-function to shape light plane coordinates perpendicular to Oc 
        def get_plane_from_proj_line(u_p, v_p_start, v_p_end, is_col=True):
            if is_col: # If it's a vertical light line
                p1_n = np.array([(u_p - cx_p)/fx_p, (v_p_start - cy_p)/fy_p, 1]).reshape(3,1)
                p2_n = np.array([(u_p - cx_p)/fx_p, (v_p_end - cy_p)/fy_p, 1]).reshape(3,1)
            else: # If it's a horizontal watermark line
                p1_n = np.array([(v_p_start - cx_p)/fx_p, (u_p - cy_p)/fy_p, 1]).reshape(3,1)
                p2_n = np.array([(v_p_end - cx_p)/fx_p, (u_p - cy_p)/fy_p, 1]).reshape(3,1)
                
            r1 = R_inv @ p1_n
            r2 = R_inv @ p2_n
            
            # Cross Vector to find perpendicular direction 
            normal = np.cross(r1.flatten(), r2.flatten())
            normal /= np.linalg.norm(normal)
            d = -np.dot(normal, C_p_cam.flatten())
            
            return np.array([normal[0], normal[1], normal[2], d])

        # Extract horizontal screen to generate wall equations
        for c in range(SCREEN_WIDTH):
            wPlaneCol[c, :] = get_plane_from_proj_line(c, 0, SCREEN_HEIGHT, is_col=True)
            
        # Extract vertical light screen equations
        for r in range(SCREEN_HEIGHT):
            wPlaneRow[r, :] = get_plane_from_proj_line(r, 0, SCREEN_WIDTH, is_col=False)
            
        # Save the calculated 3D world parameters to Scipy .mat matrix file
        scipy.io.savemat(output_file, {
            "Nc": Nc,             # Rays of the mobile camera sensor
            "Oc": Oc,             # Camera coordinates
            "wPlaneCol": wPlaneCol.T,  # Vertical screen lines
            "wPlaneRow": wPlaneRow.T,  # Horizontal screen lines
            "cam_K": K1,          # Camera lens Intrinsic 
            "proj_K": K2,         # Projector lens Intrinsic 
            "R": R,               # World rotation matrix (R) 
            "T": T                # Translation vector (T)
        })
        # Finish and notify Error rate 
        messagebox.showinfo("Success", f"Calibration Saved to:\n{output_file}\nError: {ret:.4f}")

    # ---------------------------------------------------------
    # 3. CAPTURE SCAN (Step 3: Sweep scan real images onto object)
    # ---------------------------------------------------------
    def capture_scan(self, save_dir, silent=False):
        # Initiate automated scan by projecting patterned light onto target
        
        # Set up projector configurations
        self.init_projector()
        
        # Generate Gray Code Patterns into variable P
        P = self.generate_patterns()
        
        # Prepare list for projection images including white/black baselines
        patterns = []
        white = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8) * PROJ_VALUE # Full white image to capture texture layers 
        black = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8) # Full black image for ambient noise testing
        
        patterns.append(("01.bmp", white))
        patterns.append(("02.bmp", black))
        
        idx = 3 # Pattern images start at the 3rd image
        # Add striped images and Inverses into patterns list for projection
        for j in range(2): 
            for pat in P[j]:
                pat_img = (pat * PROJ_VALUE).astype(np.uint8)
                inv_img = ((1-pat) * PROJ_VALUE).astype(np.uint8)
                
                if D_SAMPLE_PROJ > 1:
                    pat_img = cv2.resize(pat_img, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    inv_img = cv2.resize(inv_img, (SCREEN_WIDTH, SCREEN_HEIGHT), interpolation=cv2.INTER_NEAREST)
                    
                patterns.append((f"{idx:02d}.bmp", pat_img)); idx+=1
                patterns.append((f"{idx:02d}.bmp", inv_img)); idx+=1
        
        # Create folder for lossless .bmp image storage
        os.makedirs(save_dir, exist_ok=True)
        cv2.imshow(self.window_name, white) # Turn on white screen briefly to survey target
        cv2.waitKey(100)
        
        # If not in silent mode, popup Ready alert
        if not silent: 
            messagebox.showinfo("Step 3", f"Ready to scan.\nImages saved to: {save_dir}")
            
        # Launch projector and shoot images sequentially
        for fname, img in patterns:
            cv2.imshow(self.window_name, img) # Project current pattern
            cv2.waitKey(200) # Wait slightly for phone exposure to prevent blurriness 
            
            # Command phone to snap photo, error if Timeout
            if not self.trigger_capture(os.path.join(save_dir, fname)):
                if not silent:
                    messagebox.showerror("Error", "Timeout"); 
                self.close_projector(); return
        
        # Finished projection, close projector screen
        self.close_projector()
        
        # Output report verifying successful projection
        if not silent:
            messagebox.showinfo("Step 3 Done", "Scan Capture Complete.")

    # ---------------------------------------------------------
    # 4. GENERATE CLOUD (Step 4: Process images into 3D Point Cloud)
    # ---------------------------------------------------------
    def generate_cloud(self, scan_dir, calib_file):
        # Function to decode 3D points using light axes and stored Calibration
        
        # Verify Calibration file exists, abort if none 
        if not os.path.exists(calib_file):
            raise FileNotFoundError(f"Calibration file not found at {calib_file}")

        print(f"[Process] Processing {scan_dir} using {calib_file}...")
        
        # Load the setup .mat Calibration data
        data = scipy.io.loadmat(calib_file)
        if 'Oc' not in data: # Check if file is valid (Requires Oc for computation origin)
            raise ValueError("Calibration file missing 'Oc'.")
            
        # Assemble variables into a Dictionary
        calib_data = {
            "Nc": data["Nc"],               # Light rays from camera
            "Oc": data["Oc"],               # Origin of camera
            "wPlaneCol": data["wPlaneCol"], # Vertical projector planes
            "wPlaneRow": data["wPlaneRow"], # Horizontal projector planes
            "cam_K": data["cam_K"]          # Camera intrinsics
        }

        # Embed gray_decode exactly as it is in standalone
        # Function to Decode Gray coded patterns from the object
        def gray_decode(folder, n_cols=1920, n_rows=1080):
            # Find all .bmp or .png images in scan directory
            files = sorted(glob.glob(os.path.join(folder, "*.bmp")))
            if not files:
                files = sorted(glob.glob(os.path.join(folder, "*.png")))
                
            # Must contain sufficient files (>4), otherwise Error 
            if len(files) < 4:
                raise ValueError("Not enough images in folder to decode.")
                
            # Load white(01) and black(02) as baseline for Contrast
            img_white = cv2.imread(files[0], 0).astype(np.float32)
            img_black = cv2.imread(files[1], 0).astype(np.float32)
            
            height, width = img_white.shape
            
            # Calculate and mask out blurry/dark pixels to avoid noise
            # Calculate Contrast and Noise Floor (Dark noise wiping calculation)
            contrast = img_white - img_black # Contrast shadow gap distance 
            noise_floor = np.percentile(img_black, 95) # Find 95th percentile noise floor of black pixels
            dynamic_range = np.max(contrast) # Max dynamic range contrast (highest clarity)

            # Apply adaptive dynamic Thresholding limits
            mask_shadow = img_white > (noise_floor * 1.5) # Brightness must surpass noise shadow thoroughly
            mask_contrast = contrast > (dynamic_range * 0.05) # Contrast target dynamically trimmed by 5% 

            # Fuse both masks. Dark areas are discarded from 3D computation.
            valid_mask = mask_shadow & mask_contrast

            # Number of bit steps from projector screen resolution 
            n_col_bits = int(np.ceil(np.log2(n_cols)))
            n_row_bits = int(np.ceil(np.log2(n_rows)))
            
            current_idx = 2 # Start accessing image number 3
            
            # Sub-function processing light bits bit by bit
            def decode_sequence(n_bits):
                nonlocal current_idx # Utilizing variables across scoping depths
                gray_val = np.zeros((height, width), dtype=np.int32) # Valid empty screen
                
                # Execute decoding per bit matching requirements
                for b in range(n_bits):
                    if current_idx >= len(files): break
                    
                    # Process regular (p) and Inverse (i) pairs for lightness
                    p_path = files[current_idx]; current_idx += 1
                    i_path = files[current_idx]; current_idx += 1
                    
                    img_p = cv2.imread(p_path, 0).astype(np.float32)
                    img_i = cv2.imread(i_path, 0).astype(np.float32)

                    # If foreground is brighter, grant bit '1' 
                    bit = np.zeros((height, width), dtype=np.int32)
                    bit[img_p > img_i] = 1

                    # Bit Shift variables crafting standard Gray Code structures
                    gray_val = np.bitwise_or(gray_val, np.left_shift(bit, (n_bits - 1 - b)))
                    
                # Afterwards transform Gray Code explicitly back to native Binary decimal coordinates
                mask = np.right_shift(gray_val, 1)
                while np.any(mask > 0): # Transform purely via complete XOR runs 
                    gray_val = np.bitwise_xor(gray_val, mask)
                    mask = np.right_shift(mask, 1)
                    
                return gray_val # Yield fully mapped valid pixel coordinates

            print("Decoding Columns...")
            col_map = decode_sequence(n_col_bits)  # Decode vertical columns specifically mapped on object
            print("Decoding Rows...")
            row_map = decode_sequence(n_row_bits)  # Decode horizontally mapped rows
            
            # Return coordinate maps alongside the raw Texture canvas for 3D model coloring
            return col_map, row_map, valid_mask, cv2.imread(files[0]) 
            
        # Embed reconstruct_point_cloud exactly as it is in standalone
        # Sub-function calculating 3D Ray-Intersection Triangulation using prior 2D data 
        def reconstruct_point_cloud(col_map, row_map, mask, texture, calib): 
            print("Reconstructing 3D points...")
            
            Nc = calib["Nc"] # Accurate 3D light rays modeled by the camera
            Oc = calib["Oc"] # Impacted precisely in the core sensor Oc
            wPlaneCol = calib["wPlaneCol"] # Vertical bounding planes overlapping targets
            
            if wPlaneCol.shape[0] == 4: wPlaneCol = wPlaneCol.T # Transpose assuring clean numeric mappings
            
            h, w = col_map.shape 
            
            # Flatten data matrix. It massively expedites calculating multidimensional variables.
            col_flat = col_map.flatten()
            mask_flat = mask.flatten()
            tex_flat = texture.reshape(-1, 3) # Formulate exact BGR coloring parameters per point
            
            # Execute utilizing merely points which triumphed the thresholding Mask test
            valid_indices = np.where(mask_flat)[0]
            print(f"Processing {len(valid_indices)} valid pixels...")
            
            # Resolve light rays tracing from origin to all valid pixels 
            if Nc.shape[1] == h * w:
                rays = Nc[:, valid_indices] # Extract ray vectors immediately
            else:
                # If prepackaged Nc is absent, construct rays directly utilizing Matrix K intrinsic values
                K = calib["cam_K"]
                fx, fy = K[0,0], K[1,1]
                cx, cy = K[0,2], K[1,2]
                
                # Re-construct arrays into 2D forms mapping layout spaces
                y_v, x_v = np.unravel_index(valid_indices, (h, w))
                x_n = (x_v - cx) / fx
                y_n = (y_v - cy) / fy
                z_n = np.ones_like(x_n)
                
                rays = np.stack((x_n, y_n, z_n)) # Consolidate 3 arrays
                norms = np.linalg.norm(rays, axis=0) # Normalize ray length yielding Unit dimensions
                rays /= norms
                
            # Retrieve Plane data corresponding closely to the previously deciphered grid data 
            proj_cols = col_flat[valid_indices]
            # Strictly prevent overflows, clip to boundary max limits
            proj_cols = np.clip(proj_cols, 0, wPlaneCol.shape[0] - 1) 
            
            # Capture required mapping planar equations
            planes = wPlaneCol[proj_cols, :]
            
            # N(x,y,z) and d for Plane equation : Ax+By+Cz+D = 0
            N = planes[:, 0:3].T    # Perpendicular Normal vector directing frontal planar walls
            d = planes[:, 3]        # Spacing metric marking Plane offsets from origin
            
            # Intersect Rays and Planes forming distinct 3D locational markers
            # Leveraging Intersection mathematics computing t scale:
            # $t = -(N^T \cdot O_c + d) / (N^T \cdot \text{ray})$ 
            denom = np.sum(N * rays, axis=0) # Equation denominator segment 
            numer = np.dot(N.T, Oc).flatten() + d # Equation numerator segment 
            
            # Mitigate against divided zero faults or parallel rays that do not converge at all 
            valid_intersect = np.abs(denom) > 1e-6 
            t = -numer[valid_intersect] / denom[valid_intersect] # Finalize solving values determining ray impact ranges t
            
            # Exploit ray structures calculating 3D Global Space destinations via extending them through length t! 
            rays_valid = rays[:, valid_intersect]
            # Calculated true 3D coordinates anchored to reality: $P = O_c + t \cdot \text{ray}$
            P = Oc + rays_valid * t 
            
            # Additionally acquire colored hue features mapping against particular matched locational points
            C = tex_flat[valid_indices[valid_intersect]]
            
            return P.T, C # Transmit 3D points and coloration items properly
            
        # -----------------------------------------------------------------------------------
        # Resume main processing pipeline 
        # Order 1: Process and execute the initial Gray Decode 
        c_map, r_map, mask_out, texture_out = gray_decode(scan_dir)
        
        # Order 2: Construct actual 3D points utilizing triangulations
        points, colors = reconstruct_point_cloud(c_map, r_map, mask_out, texture_out, calib_data)
        
        # Order 3: Frame, package and store inside a standard formatted .ply Point Cloud file incorporating Colors
        # Configure naming parameters
        ply_name = os.path.basename(scan_dir) + ".ply"
        out_path = os.path.join(scan_dir, ply_name)
        
        print(f"Saving {len(points)} points to {out_path}...")
        
        # Drive raw file saving outputs firmly 
        with open(out_path, 'w') as f:
            # Incorporate metadata Header tagging PLY characteristics using distinct readable ASCII strings
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
            
            # Engrave coordinates tightly adjacent corresponding colored layers individually on loop iterations (writing each specific locus)
            for i in range(len(points)):
                p = points[i] # 3 coordinates
                c = colors[i] # 3 colors
                
                # Hack: OpenCV saves format reversed B-G-R colorings, while PLY specifically requests R-G-B alignment
                # Therefore, we strictly must invert layout protocols upon write executions: (c[2]=R, c[1]=G, c[0]=B)
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[2]} {c[1]} {c[0]}\n")
                
        # Conclude and declare the accomplished saving process
        print(f"[Success] Generated {out_path}")
