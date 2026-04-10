import os
import glob
import copy
import numpy as np
import open3d as o3d
import cv2
import scipy.io

# ==========================================
# PROCESSING LOGIC (Open3D)
# ==========================================
class ProcessingLogic:
    # Class for processing 3D models (Point Cloud and Mesh) using the Open3D library
    @staticmethod
    def _load_pcd(input_data):
        # Internal function to check and load Point Cloud files
        if isinstance(input_data, str): # If the input data is a string (file path)
            if not os.path.exists(input_data): # If the file is not found at the given path
                raise FileNotFoundError(f"Input file not found: {input_data}") # Raise an error indicating the file was not found
            # Read and return the Point Cloud data from the file using Open3D
            return o3d.io.read_point_cloud(input_data)
            
        # If it's not a string (assuming it's already a Point Cloud object), return it as-is
        return input_data

    # --- Multi PLY Processing Functions ---
    @staticmethod
    def _gray_decode(source, n_cols=1920, n_rows=1080,
                     n_sets_col=11, n_sets_row=11,
                     thresh_mode='otsu', shadow_val=40, contrast_val=10):
        """
        Decode Gray-code structured-light images.

        Parameters
        ----------
        source      : str (folder path) OR list[str] (sorted file list)
        n_cols      : projector width  (pixels)
        n_rows      : projector height (pixels)
        n_sets_col  : how many FIRST column bit-planes to use (1-11, default 11)
        n_sets_row  : how many FIRST row    bit-planes to use (1-11, default 11)
        thresh_mode : 'otsu' or 'manual'
        shadow_val  : manual shadow threshold (0-255)
        contrast_val: manual contrast threshold (0-255)

        Using fewer patterns skips the finest stripes and gives a coarser but
        geometrically CORRECT result because the decoded values are scaled back
        to the full projector coordinate range automatically.
        """
        if isinstance(source, list):
            files = source
        else:
            files = sorted(glob.glob(os.path.join(source, "*.bmp")))
            if not files:
                files = sorted(glob.glob(os.path.join(source, "*.png")))

        if len(files) < 4:
            raise ValueError(f"Not enough images (got {len(files)}, need at least 4).")

        img_white = cv2.imread(files[0], 0).astype(np.float32)
        img_black = cv2.imread(files[1], 0).astype(np.float32)
        height, width = img_white.shape

        if thresh_mode == 'otsu':
            # Auto-calculate optimal threshold using Otsu's method
            # Must convert to uint8 for cv2.threshold
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

        max_col_bits = int(np.ceil(np.log2(n_cols)))  # 11 for 1920
        max_row_bits = int(np.ceil(np.log2(n_rows)))  # 11 for 1080

        n_use_col = max(1, min(int(n_sets_col), max_col_bits))
        n_use_row = max(1, min(int(n_sets_row), max_row_bits))

        current_idx = 2

        def decode_first_n(max_bits, n_use):
            """Read ALL max_bits pairs but only decode the first n_use.
            The decoded value is in [0, 2^n_use - 1]."""
            nonlocal current_idx
            gray_val = np.zeros((height, width), dtype=np.int32)
            for b in range(max_bits):
                if current_idx + 1 >= len(files):
                    current_idx += 2
                    continue
                if b < n_use:
                    img_p = cv2.imread(files[current_idx],     0).astype(np.float32)
                    img_i = cv2.imread(files[current_idx + 1], 0).astype(np.float32)
                    bit = np.zeros((height, width), dtype=np.int32)
                    bit[img_p > img_i] = 1
                    # bit 0 = MSB of n_use-bit number
                    gray_val = np.bitwise_or(gray_val,
                                             np.left_shift(bit, (n_use - 1 - b)))
                current_idx += 2  # always advance pointer
            # Gray -> binary
            mask = np.right_shift(gray_val, 1)
            while np.any(mask > 0):
                gray_val = np.bitwise_xor(gray_val, mask)
                mask = np.right_shift(mask, 1)
            return gray_val

        col_map = decode_first_n(max_col_bits, n_use_col)
        row_map = decode_first_n(max_row_bits, n_use_row)

        # CRITICAL: scale decoded values back to the full projector coordinate
        # range so that wPlaneCol/wPlaneRow lookups remain geometrically correct.
        # Example: 9 bits gives values 0-511 -> *4 -> 0-2044 (covers 1920 cols).
        col_scale = 1 << (max_col_bits - n_use_col)  # 2^(11-n_use_col)
        row_scale = 1 << (max_row_bits - n_use_row)
        col_map = col_map * col_scale
        row_map = row_map * row_scale

        return col_map, row_map, valid_mask, cv2.imread(files[0])

    @staticmethod
    def _reconstruct_point_cloud(col_map, row_map, mask, texture, calib,
                                 row_mode=1, epipolar_tol=2.0):
        # Calculate the intersection to find the 3D position (Triangulation)
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

    @staticmethod
    def _save_ply(points, colors, filename):
        # Save as a .ply file with colors
        with open(filename, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            
            for i in range(len(points)):
                p = points[i]
                c = colors[i]
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[2]} {c[1]} {c[0]}\n")

    @staticmethod
    def process_multi_ply(calib_path, target_path, mode, log_callback=None,
                          n_sets_col=11, n_sets_row=11,
                          row_mode=1, epipolar_tol=2.0,
                          thresh_mode='otsu', shadow_val=40, contrast_val=10,
                          file_list=None, out_path_override=None):
        """
        Process structured-light images → .ply point cloud.

        Parameters
        ----------
        calib_path       : path to .mat calibration file
        target_path      : folder to process (single scan folder or parent for batch)
        mode             : 'single' | 'batch' | 'files'
        log_callback     : optional callable(str) for UI log output
        col_start        : first column bit-plane to use (0-indexed, inclusive)
        col_end          : last  column bit-plane to use (0-indexed, inclusive)
        row_start        : first row    bit-plane to use (0-indexed, inclusive)
        row_end          : last  row    bit-plane to use (0-indexed, inclusive)
        file_list        : sorted list of image paths – used when mode == 'files'
        out_path_override: output .ply path – used when mode == 'files'
        """
        def log(msg):
            if log_callback: log_callback(msg)
            else: print(msg)

        decode_kw = dict(n_sets_col=n_sets_col, n_sets_row=n_sets_row)

        log("Loading Calibration Data...")
        data = scipy.io.loadmat(calib_path)
        calib_data = {
            "Nc": data["Nc"], "Oc": data["Oc"],
            "wPlaneCol": data["wPlaneCol"], "wPlaneRow": data["wPlaneRow"],
            "cam_K": data["cam_K"]
        }

        def _process_source(source, out_path, label):
            log(f"  -> Decoding {label}  "
                f"[col-sets={n_sets_col}  row-sets={n_sets_row}]...")
            c_map, r_map, mask, texture = ProcessingLogic._gray_decode(
                source, **decode_kw,
                thresh_mode=thresh_mode, shadow_val=shadow_val, contrast_val=contrast_val)
            log("  -> Reconstructing 3D points...")
            points, colors = ProcessingLogic._reconstruct_point_cloud(
                c_map, r_map, mask, texture, calib_data,
                row_mode=row_mode, epipolar_tol=epipolar_tol)
            log(f"  -> Saving {len(points)} points...")
            ProcessingLogic._save_ply(points, colors, out_path)
            log(f"  ✔ Saved: {os.path.basename(out_path)}\n")

        if mode == "files":
            if not file_list:
                raise ValueError("mode='files' requires a non-empty file_list.")
            if not out_path_override:
                raise ValueError("mode='files' requires out_path_override.")
            _process_source(file_list, out_path_override,
                            f"{len(file_list)} selected files")

        elif mode == "single":
            ply_name = os.path.basename(target_path) + ".ply"
            out_path = os.path.join(target_path, ply_name)
            _process_source(target_path, out_path,
                            f"folder '{os.path.basename(target_path)}'")

        else:  # batch
            subfolders = [f.path for f in os.scandir(target_path) if f.is_dir()]
            log(f"Found {len(subfolders)} subfolders to process.")

            success_count = 0
            for folder in subfolders:
                has_imgs = (glob.glob(os.path.join(folder, "*.bmp")) or
                            glob.glob(os.path.join(folder, "*.png")))
                if has_imgs:
                    try:
                        ply_name = os.path.basename(folder) + ".ply"
                        out_path = os.path.join(folder, ply_name)
                        _process_source(folder, out_path,
                                        f"folder '{os.path.basename(folder)}'")
                        success_count += 1
                    except Exception as e:
                        log(f"  ❌ Error in {os.path.basename(folder)}: {e}\n")
                else:
                    log(f"  Skipping {os.path.basename(folder)} (No images found).")

            log(f"=== Batch Complete: {success_count}/{len(subfolders)} succeeded ===")

    @staticmethod
    def remove_background(input_data, output_path=None, distance_threshold=50, ransac_n=3, num_iterations=1000, return_obj=False):
        # Function to remove the background/back wall (Background Remove) from the 3D model
        print(f"[BG Remove] Processing...")
        
        # Load the Point cloud file for processing
        pcd = ProcessingLogic._load_pcd(input_data)
        
        # If the loaded 3D shape has no coordinate points
        if not pcd.has_points():
            raise ValueError("Point cloud is empty.") # Raise an error

        # Use the Segment Plane technique (find the largest plane), assuming the large plane is the background wall
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                 ransac_n=ransac_n,
                                                 num_iterations=num_iterations)
        
        # Select to remove only inliers (points belonging to the plane/wall), keeping the rest (Outliers) which is the main Object (invert=True)
        object_cloud = pcd.select_by_index(inliers, invert=True)
        
        # Display the number of coordinate points before and after removal
        print(f"[BG Remove] Original: {len(pcd.points)}, Remaining: {len(object_cloud.points)} pts")
        
        # If an output path for saving the file is specified
        if output_path:
            o3d.io.write_point_cloud(output_path, object_cloud) # Save as a new file
            print(f"[BG Remove] Saved to {output_path}")
            
        return object_cloud if return_obj else None # Return the object data (unless None is requested)

    @staticmethod
    def remove_outliers(input_data, output_path=None, nb_neighbors=20, std_ratio=2.0, return_obj=False):
        # Function to remove distance noise or scattered dust (Statistical Outlier Removal)
        print(f"[Outlier] Processing...")
        pcd = ProcessingLogic._load_pcd(input_data) # Load file
        
        if not pcd.has_points():
            raise ValueError("Point cloud is empty.")

        # Use the command to remove abnormally distant points using statistics, filtering by the number of Neighbors and standard deviation ratio
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        
        # Filter to keep only the points that pass the criteria
        inlier_cloud = pcd.select_by_index(ind)
        
        print(f"[Outlier] Keeping: {len(inlier_cloud.points)} pts")
        
        # Save as a 3D file to the computer if a path exists
        if output_path:
            o3d.io.write_point_cloud(output_path, inlier_cloud)
            print(f"[Outlier] Saved to {output_path}")

        return inlier_cloud if return_obj else None

    @staticmethod
    def keep_largest_cluster(input_data, output_path=None, eps=5.0, min_points=200, return_obj=False):
        # Function to group (Clustering) and choose to keep only the largest group (small floating points will be discarded)
        print(f"[Cluster] Processing...")
        pcd = ProcessingLogic._load_pcd(input_data)
        
        if not pcd.has_points():
            raise ValueError("Point cloud is empty.")
            
        # DBSCAN to cluster nearby points (distance not exceeding eps and must group together at least min_points)
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
        
        if len(labels) == 0: # If nothing is found at all
            return pcd if return_obj else None
            
        # Count the number of points in each cluster
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        # Ignore negative cluster numbers (-1 is noise in DBSCAN)
        valid_clusters = unique_labels != -1
        unique_labels = unique_labels[valid_clusters]
        counts = counts[valid_clusters]
        
        if len(unique_labels) == 0: 
            return pcd if return_obj else None # If there is only noise, return it
            
        # Choose to keep only the cluster group with the highest number of points (likely our main model)
        largest_cluster_label = unique_labels[counts.argmax()]
        largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
        
        cleaned_pcd = pcd.select_by_index(largest_cluster_indices)
        print(f"[Cluster] Kept largest group: {len(cleaned_pcd.points)} pts")
        
        if output_path:
            o3d.io.write_point_cloud(output_path, cleaned_pcd)
            print(f"[Cluster] Saved to {output_path}")
            
        return cleaned_pcd if return_obj else None

    @staticmethod
    def remove_radius_outlier(input_data, output_path=None, nb_points=100, radius=5.0, return_obj=False):
        # Function to eliminate noise points using a circular radius (Radius Outlier Removal). If a point doesn't have enough neighbors around it, it will be removed
        print(f"[Radius Outlier] Processing...")
        pcd = ProcessingLogic._load_pcd(input_data)
        
        if not pcd.has_points():
            raise ValueError("Point cloud is empty.")
            
        # Check the radius. If within the radius there are not at least nb_points neighbors, it will be considered a noise point
        cl, ind = pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        
        inlier_cloud = pcd.select_by_index(ind)
        print(f"[Radius Outlier] Keeping: {len(inlier_cloud.points)} pts")
        
        if output_path:
            o3d.io.write_point_cloud(output_path, inlier_cloud)
            print(f"[Radius Outlier] Saved to {output_path}")
            
        return inlier_cloud if return_obj else None

    @staticmethod
    def preprocess_point_cloud(pcd, voxel_size):
        # Function to prepare model data (Downsample + Normals + FPFH Features) before merging
        
        # 1. Reduce model resolution (Downsample) into a Voxel grid to save calculation time
        pcd_down = pcd.voxel_down_sample(voxel_size)
        
        # 2. Calculate surface directions (Normals) to help in model matching
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
            
        # 3. Calculate FPFH (Fast Point Feature Histograms) specific coordinate features for robust RANSAC
        radius_feature = voxel_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
            
        return pcd_down, pcd_fpfh # Return the downsampled model and the feature model

    @staticmethod
    def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size, icp_dist_ratio=1.5):
        # Function to perform Global Registration (roughly align 2 models facing each other using RANSAC)
        # Set max distance for recognizing matching points dynamically based on voxel_size and the user-defined multiplier
        distance_threshold = voxel_size * icp_dist_ratio
        
        # Use RANSAC together with FPFH to guess the most matched points
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
    def merge_pro_360(input_folder, output_path, voxel_size=0.02, icp_dist_ratio=1.5, outlier_nb=20, outlier_std=2.0, sample_before=1, sample_after=1, final_voxel=0.5, step_callback=None):
        # Main function to sequence and merge 3D models obtained from a 360-degree scan (multiple angles) together
        # step_callback: optional function(step_index, total_steps, accumulated_cloud) called after each merge step
        #                When provided, the UI can use this to show a 3D preview popup for that step
        print(f"[Merge 360] Loading clouds from {input_folder}...")
        
        # Find all .ply files in the folder
        ply_files = glob.glob(os.path.join(input_folder, "*.ply"))
        
        # Sort files based on the degree number in the filename (e.g., 'doraemon_30deg_scan.ply' -> 30)
        def extract_degree(filepath):
            filename = os.path.basename(filepath)
            try:
                # Assuming format like "name_numberdeg_scan.ply" or similar containing "deg"
                # Find the part containing "deg"
                parts = filename.split('_')
                for part in parts:
                    if 'deg' in part:
                        # Strip "deg" and convert to integer
                        num_str = part.replace('deg', '')
                        return int(num_str)
                # Fallback if "deg" not found, try to find any number
                import re
                numbers = re.findall(r'\d+', filename)
                if numbers:
                    return int(numbers[-1])
            except:
                pass
            return 0 # Default if parsing fails

        ply_files = sorted(ply_files, key=extract_degree)
        print(f"[Merge 360] Sorted file order:")
        for idx, f in enumerate(ply_files):
            print(f"  [{idx}] {os.path.basename(f)}")
        
        if len(ply_files) < 2:
            raise ValueError("Need at least 2 .ply files to merge.") # Must have at least 2 models to be able to merge
            
        pcds = []
        for path in ply_files:
            # Load each model file into a loop to store as a List (pcds)
            pcd = o3d.io.read_point_cloud(path)
            if not pcd.has_points():
                raise ValueError(f"Loaded empty point cloud from: {path}")
            # Apply initial sampling if requested
            if sample_before > 1:
                pcd = pcd.uniform_down_sample(every_k_points=int(sample_before))
            pcds.append(pcd)
            print(f"  Loaded: {os.path.basename(path)} ({len(pcd.points)} points)")
            
        total_steps = len(pcds) - 1
        print(f"[Merge 360] Loaded {len(pcds)} clouds. Running Sequential Registration ({total_steps} steps)...")
        
        # Set the starting model to be the first model (Frame 0) as the base (Accumulator)
        merged_cloud = copy.deepcopy(pcds[0])
        
        # Keep a history of the accumulated transformation matrices of every frame (Current Global Transform)
        max_accum_T = np.identity(4) 
        
        # Loop to compare and connect models pair by pair (Model 1 to 0, Model 2 to 1,...) continuously 
        for i in range(1, len(pcds)):
            print(f"\n[Merge 360] === Step {i}/{total_steps}: Aligning Scan {i} -> Scan {i-1} ===")
            source = pcds[i]      # Latest model (moving towards target)
            target = pcds[i-1]    # Previous model (standing still)
            
            # 1. Preprocess prepare both data (Downsample + calculate Normals)
            source_down, source_fpfh = ProcessingLogic.preprocess_point_cloud(source, voxel_size)
            target_down, target_fpfh = ProcessingLogic.preprocess_point_cloud(target, voxel_size)
            print(f"  Preprocessed: source={len(source_down.points)} pts, target={len(target_down.points)} pts (voxel={voxel_size})")
            
            # 2. Let Open3D try to blindly guess the broad overlapping position first (Global RANSAC)
            # Pass down the icp_dist_ratio multiplier to control the strictness of the search
            ransac_result = ProcessingLogic.execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh, voxel_size, icp_dist_ratio)
            
            # --- Fitness check after RANSAC ---
            # Fitness near 0.0 = very few matched points = bad initial alignment guess
            print(f"  [RANSAC] Fitness: {ransac_result.fitness:.4f} | RMSE: {ransac_result.inlier_rmse:.6f}")
            if ransac_result.fitness < 0.05:
                print(f"  [WARNING] Step {i}: RANSAC fitness is very low ({ransac_result.fitness:.4f})! "
                      f"Alignment may be unreliable. Try lowering Voxel Size or increasing ICP Dist Ratio.")
            
            # 3. Let Open3D precisely adjust the overlap from the initial RANSAC guess (Local ICP Refinement Point-to-Plane)
            icp_result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, voxel_size, ransac_result.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            
            # --- Fitness check after ICP ---
            # If fitness is still very low here, this step's merge will be corrupted and affect all subsequent steps
            print(f"  [ICP]    Fitness: {icp_result.fitness:.4f} | RMSE: {icp_result.inlier_rmse:.6f}")
            if icp_result.fitness < 0.05:
                print(f"  [WARNING] Step {i}: ICP fitness is very low ({icp_result.fitness:.4f})! "
                      f"This step's alignment is likely incorrect and WILL corrupt all subsequent steps. "
                      f"Consider adjusting parameters or checking if all PLY files are valid.")
            
            # Extract the relationship matrix to shift the position between i and i-1 to store
            T_local = icp_result.transformation 
            
            # 4. Convert it to a relationship from i shifted down to compare with the absolute base model 0, so that all pieces are on the same stage
            max_accum_T = np.dot(max_accum_T, T_local)
            
            # 5. Command to transform and combine it with the base stage
            pcd_temp = copy.deepcopy(source) 
            pcd_temp.transform(max_accum_T) # Change the position of the latest model and overlap it
            merged_cloud += pcd_temp        # Combine together
            
            print(f"  [Merge 360] Step {i}/{total_steps} complete. Accumulated cloud: {len(merged_cloud.points)} points total.")
            
            # 6. If a step_callback is registered (e.g. from GUI checkbox), call it now with a snapshot
            #    of the current accumulated cloud. This allows the GUI to display a 3D preview popup.
            #    The merge process will pause here (blocking) until the callback returns.
            if step_callback is not None:
                # Provide a lightweight copy to avoid mutating the live accumulator
                preview_cloud = copy.deepcopy(merged_cloud)
                step_callback(i, total_steps, preview_cloud)
            
        print(f"\n[Merge 360] All {total_steps} steps complete. Running post-processing...")
        print(f"[Merge 360] Post-processing (Final Voxel: {final_voxel}, Outlier removal)...")
        # Take the entire large finished model and reduce its resolution one last time to prevent the computer from lagging (optional if user set Final Voxel to >0)
        pcd_combined_down = merged_cloud
        if final_voxel > 0:
            pcd_combined_down = merged_cloud.voxel_down_sample(voxel_size=final_voxel)
            print(f"  After final voxel down-sample: {len(pcd_combined_down.points)} points")
        
        # Apply after merge sampling if requested
        if sample_after > 1:
            pcd_combined_down = pcd_combined_down.uniform_down_sample(every_k_points=int(sample_after))
            print(f"  After uniform sample-after: {len(pcd_combined_down.points)} points")
        
        # Filter out bad points for the final time of merging (Outlier Removal)
        # using the UI-defined parameters for neighbor count and standard deviation aggressiveness
        cl, ind = pcd_combined_down.remove_statistical_outlier(nb_neighbors=outlier_nb, std_ratio=outlier_std)
        pcd_final = pcd_combined_down.select_by_index(ind)
        print(f"  After outlier removal: {len(pcd_final.points)} points (removed {len(pcd_combined_down.points) - len(pcd_final.points)})")
        
        # Calculate the latest surface Normals for the large model
        pcd_final.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        
        # Save the completely merged image file and export as PLY
        o3d.io.write_point_cloud(output_path, pcd_final)
        print(f"[Merge 360] Saved merged cloud to {output_path}")

    @staticmethod
    def reconstruct_stl(input_path, output_path, mode="watertight", params=None, centroid_orient=True, consistency_pass=False, consistency_k=30, meshlab_params=None, save_normals_path=None):
        # Function used to create a 3D wireframe or solid mesh (STL from Point Cloud), suitable for 3D printing tasks
        # centroid_orient:   When True, calculates the geometric center of all points and forces every
        #                    normal to point OUTWARD from that center. More reliable than graph-consistency.
        # consistency_pass:  When True (and centroid_orient is True), runs orient_normals_consistent_
        #                    tangent_plane(k) AFTER centroid orient to propagate the outward direction
        #                    via the neighborhood graph, fixing any remaining stray normals.
        # consistency_k:     Number of nearest neighbors for the consistency pass (default 30).
        # meshlab_params:    Optional dict with MeshLab post-processing options (requires pymeshlab).
        # save_normals_path: Optional .ply path — if set, saves the point cloud WITH normals embedded
        #                    immediately after orientation (before meshing). Useful for debugging.
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        print(f"[Recon] Loading {input_path}...")
        pcd = o3d.io.read_point_cloud(input_path) # Read the point cloud file
        
        if not pcd.has_points():
            raise ValueError("Point cloud is empty.")
            
        # --- Normal Estimation & Orientation ---
        if not pcd.has_normals():
            print("[Recon] Estimating normals...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))

        if centroid_orient:
            # CENTROID METHOD: 
            # 1. Compute the geometric center (mean of all point positions)
            # 2. Use Open3D's orient_normals_towards_camera_location() with the centroid as
            #    the "camera" — this makes ALL normals point TOWARD the centroid (inward)
            # 3. Multiply all normals by -1 to flip them OUTWARD from the centroid
            # This is highly reliable for closed objects (e.g., 360-degree scans)
            center = np.asarray(pcd.points).mean(axis=0)  # Centroid = average of all XYZ coordinates
            print(f"[Recon] Centroid normal orient: center = [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
            pcd.orient_normals_towards_camera_location(center)   # Orient inward toward centroid
            pcd.normals = o3d.utility.Vector3dVector(            # Flip to outward
                np.asarray(pcd.normals) * -1.0
            )
            print("[Recon] Centroid-based outward orientation applied.")
        else:
            # GRAPH METHOD: check and rotate all Normal lines to point in the same direction
            # using a minimum spanning tree on the normal directions (may fail on complex shapes)
            print("[Recon] Using tangent-plane graph consistency for normal orientation...")
            pcd.orient_normals_consistent_tangent_plane(100)
            print("[Recon] Graph orientation applied.")

        # Consistency pass: propagate the outward direction via neighborhood graph to fix stray normals
        # This runs orient_normals_consistent_tangent_plane(k) a second time AFTER centroid orient.
        # Because centroid orient already set the global outward direction, the graph pass now has a
        # reliable reference and will flip any remaining "stray" inward-facing normals to match.
        if consistency_pass:
            k = int(consistency_k) if consistency_k > 0 else 30
            print(f"[Recon] Consistency pass: orient_normals_consistent_tangent_plane(k={k})...")
            pcd.orient_normals_consistent_tangent_plane(k)
            print("[Recon] Consistency pass applied.")

        # Optionally save the point cloud with normals embedded, BEFORE meshing
        # Allows the user to open the result in CloudCompare / MeshLab and verify normals face outward
        if save_normals_path:
            print(f"[Recon] Saving normals point cloud to {save_normals_path}...")
            o3d.io.write_point_cloud(save_normals_path, pcd)
            print(f"[Recon] Normals point cloud saved ({len(pcd.points)} points).")

        if mode == "watertight":
            # Create a 3D wireframe mesh that closes leaks and is completely sealed (Poisson Surface Reconstruction)
            depth = int(params.get("depth", 10)) # Get depth/resolution value 
            if depth > 16:
                raise ValueError(f"Depth {depth} is too high! Maximum recommended is 12-14. >16 will freeze your PC.")
            
            print(f"[Recon] Poisson Reconstruction (depth={depth})...")
            # Create Mesh directly from Point using Poisson equation formula
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=depth, linear_fit=False)
            
            # Trim excess flesh or false coordinates that Open3D tries to stretch to falsely close holes
            densities = np.asarray(densities)
            mask = densities < np.quantile(densities, 0.02) # Trim away edges with low density
            mesh.remove_vertices_by_mask(mask)
            
        elif mode == "surface":
            # Another 3D building method is Ball Pivoting (rolling a ball to connect points). Cannot close holes, but keeps details on the surface better
            radii_str = params.get("radii", "1,2,4")
            try:
                # Calculate the average density distance between each surrounding point first, to see how large the majority of points are in this work
                distances = pcd.compute_nearest_neighbor_distance()
                avg_dist = np.mean(distances)
                
                # Take that obtained size and multiply by the coefficient levels in UI (e.g., 1, 2, 4). Convert to a list of ball size multipliers to use for connecting
                multipliers = [float(x) for x in radii_str.split(',')]
                radii = [avg_dist * m for m in multipliers]
                print(f"[Recon] Ball Pivoting (radii={radii})...")
                
                # Create Mesh using cumulative ball sizes of multiple numbers
                mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                    pcd, o3d.utility.DoubleVector(radii))
            except Exception as e:
                raise ValueError(f"Invalid radii parameters: {e}")
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # If weaving to create Mesh fails and there is no model to display
        if len(mesh.vertices) == 0:
            raise ValueError("Generated mesh is empty.")

        # Process and apply virtual surfaces before saving to file
        print("[Recon] Computing normals and saving...")
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(output_path, mesh) # Save .stl file (or other extensions that Open3D supports)
        print(f"[Recon] Saved STL to {output_path}")

        # --- MeshLab Post-Processing (optional, requires pymeshlab) ---
        # Runs AFTER the Open3D save so the intermediate result is always safe even if MeshLab fails
        if meshlab_params and meshlab_params.get("enabled"):
            print("[MeshLab] Starting post-processing...")
            try:
                import pymeshlab
            except ImportError:
                raise ImportError(
                    "pymeshlab is not installed. Run: pip install pymeshlab\n"
                    "The STL has already been saved using Open3D only (without MeshLab improvements)."
                )

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(output_path)  # Load the STL we just saved
            print(f"[MeshLab] Loaded mesh: {ms.current_mesh().vertex_number()} vertices, "
                  f"{ms.current_mesh().face_number()} faces")

            # 1. Smoothing
            smooth_type  = str(meshlab_params.get("smooth_type", "taubin"))
            smooth_iters = int(meshlab_params.get("smooth_iters", 10))
            if smooth_type == "laplacian":
                # Laplacian: moves each vertex toward the mean of its neighbours — stronger, may shrink model
                print(f"[MeshLab] Applying Laplacian smoothing ({smooth_iters} iterations)...")
                ms.apply_coord_laplacian_smoothing(stepsmoothnum=smooth_iters)
            else:
                # Taubin: alternates positive/negative lambda steps — preserves volume better
                print(f"[MeshLab] Applying Taubin smoothing ({smooth_iters} iterations)...")
                ms.apply_coord_taubin_smoothing(stepsmoothnum=smooth_iters)

            # 2. Close Holes (fill gaps smaller than max_size edges)
            if meshlab_params.get("close_holes"):
                max_size = int(meshlab_params.get("close_max_size", 30))
                print(f"[MeshLab] Closing holes (max hole size = {max_size} edges)...")
                ms.meshing_close_holes(maxholesize=max_size)

            # 3. Mesh Simplification (Quadric Edge Collapse Decimation)
            if meshlab_params.get("simplify"):
                target = int(meshlab_params.get("target_faces", 50000))
                print(f"[MeshLab] Simplifying mesh to {target} faces...")
                ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target)

            # Overwrite the output file with the MeshLab-improved mesh
            ms.save_current_mesh(output_path)
            final = ms.current_mesh()
            print(f"[MeshLab] Post-processing complete. Final: {final.vertex_number()} vertices, "
                  f"{final.face_number()} faces. Saved to {output_path}")


    @staticmethod
    def mesh_360(input_path, output_path, depth=10, density_trim=0.01, orientation_mode="tangent", width=0.0, scale=1.1, linear_fit=False, n_threads=-1, normal_radius=0.1, normal_max_nn=30, save_normals_path=None):
        # Function to create and refine the Mesh specifically for processing models from a 360-degree all-around scan
        # save_normals_path: optional file path (.ply) to save the point cloud AFTER normal estimation
        #                    and orientation, but BEFORE Poisson meshing. Useful for inspection/debugging.
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        print(f"[360 Mesh] Loading {input_path}...")
        pcd = o3d.io.read_point_cloud(input_path) 
        
        if not pcd.has_points():
            raise ValueError("Point cloud is empty.")
            
        # 1. Calculate the initial surface Normal direction for the point cloud
        print(f"[360 Mesh] Estimating normals (radius={normal_radius}, max_nn={normal_max_nn})...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
        
        # 2. Re-adjust the alignment setting of the surface Normal directions to prevent inside-out surface flipping symptoms
        print(f"[360 Mesh] Re-orienting normals (Mode: {orientation_mode})...")
        
        if orientation_mode == "radial":
            # Radial mode (star radius angle) will always point its direction towards the center axis. Suitable for rotation objects.
            center = pcd.get_center() # Find the center point of the model
            pcd.orient_normals_towards_camera_location(center) # Force all Normal tips to point towards the center
            
            # Once the pointing direction is inward, we alternate to multiply by a negative value to flip all surfaces to face outward instead
            pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) * -1.0)
            print("[360 Mesh] Radial orientation applied (Outwards).")
            
        else: # tangent normal case 
            try:
                # Try to orient them consistently relative to each other (Graph-based Consistency)
                pcd.orient_normals_consistent_tangent_plane(100)
                print("[360 Mesh] Consistent tangent plane orientation applied.")
            except Exception as e:
                # If it fails, fallback to doing Radial pose instead
                print(f"[360 Mesh] Warning: Tangent plane failed ({e}). Fallback to radial.")
                center = pcd.get_center()
                pcd.orient_normals_towards_camera_location(center)
                pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals) * -1.0)

        # 2b. Optionally save the point cloud with normals embedded, BEFORE meshing
        #     This lets you inspect the normal orientation result as a separate .ply file
        if save_normals_path:
            print(f"[360 Mesh] Saving normals point cloud to {save_normals_path}...")
            o3d.io.write_point_cloud(save_normals_path, pcd)
            print(f"[360 Mesh] Normals point cloud saved ({len(pcd.points)} points).")

        # 3. Form the Mesh body to fill the model using Screened Poisson Reconstruction
        print(f"[360 Mesh] Poisson Reconstruction (depth={depth}, width={width}, scale={scale}, linear={linear_fit}, threads={n_threads})...")
        # Pass the dynamic UI parameters directly into the Open3D algorithm
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit, n_threads=n_threads)
            
        # 4. Trim excess (Optional). If the value is > 0, it will delete the bulging meat lumps that the system blindly generated around hollow areas to a certain extent
        if density_trim > 0.0:
            print(f"[360 Mesh] Trimming low density vertices (threshold={density_trim})...")
            densities = np.asarray(densities)
            threshold = np.quantile(densities, density_trim) 
            mask = densities < threshold
            mesh.remove_vertices_by_mask(mask) # Clear unneeded coordinate vertices
        else:
            print("[360 Mesh] Density trim is 0.0 -> Keeping watertight result.")
        
        # 5. Flush the final surface processing before export
        mesh.compute_vertex_normals()
        
        # 6. Save the fully completed model file output as a 3D model (e.g., .stl) display on the computer and clear the calculations left behind
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"[360 Mesh] Saved to {output_path}")

