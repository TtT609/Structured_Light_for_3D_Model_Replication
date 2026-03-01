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
    def _gray_decode(folder, n_cols=1920, n_rows=1080):
        # Decode the black and white stripe images (Gray Code)
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

    @staticmethod
    def _reconstruct_point_cloud(col_map, row_map, mask, texture, calib):
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
    def process_multi_ply(calib_path, target_path, mode, log_callback=None):
        # Function to process and filter 3D data from raw images (Single or Batch)
        # log_callback is a function to send messages back to print on the UI
        def log(msg):
            if log_callback: log_callback(msg)
            else: print(msg)
            
        log("Loading Calibration Data...")
        data = scipy.io.loadmat(calib_path)
        calib_data = {
            "Nc": data["Nc"], "Oc": data["Oc"],
            "wPlaneCol": data["wPlaneCol"], "wPlaneRow": data["wPlaneRow"],
            "cam_K": data["cam_K"]
        }
        
        def process_single(scan_dir):
            ply_name = os.path.basename(scan_dir) + ".ply"
            out_path = os.path.join(scan_dir, ply_name)
            
            log(f"-> Decoding images in {os.path.basename(scan_dir)}...")
            c_map, r_map, mask, texture = ProcessingLogic._gray_decode(scan_dir)
            
            log("-> Reconstructing 3D points...")
            points, colors = ProcessingLogic._reconstruct_point_cloud(c_map, r_map, mask, texture, calib_data)
            
            log(f"-> Saving {len(points)} points...")
            ProcessingLogic._save_ply(points, colors, out_path)
            log(f"✔ Saved: {ply_name}\n")
            
        if mode == "single":
            process_single(target_path)
        else:
            subfolders = [f.path for f in os.scandir(target_path) if f.is_dir()]
            log(f"Found {len(subfolders)} subfolders to process.")
            
            success_count = 0
            for folder in subfolders:
                if glob.glob(os.path.join(folder, "*.bmp")) or glob.glob(os.path.join(folder, "*.png")):
                    try:
                        process_single(folder)
                        success_count += 1
                    except Exception as e:
                        log(f"❌ Error in {os.path.basename(folder)}: {e}\n")
                else:
                    log(f"Skipping {os.path.basename(folder)} (No images found).")
                    
            log(f"=== Batch Complete: Successfully processed {success_count}/{len(subfolders)} folders ===")

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
    def merge_pro_360(input_folder, output_path, voxel_size=0.02, icp_dist_ratio=1.5, outlier_nb=20, outlier_std=2.0):
        # Main function to sequence and merge 3D models obtained from a 360-degree scan (multiple angles) together
        print(f"[Merge 360] Loading clouds from {input_folder}...")
        
        # Find all .ply files in the folder and sort by name
        ply_files = sorted(glob.glob(os.path.join(input_folder, "*.ply")))
        if len(ply_files) < 2:
            raise ValueError("Need at least 2 .ply files to merge.") # Must have at least 2 models to be able to merge
            
        pcds = []
        for path in ply_files:
            # Load each model file into a loop to store as a List (pcds)
            pcd = o3d.io.read_point_cloud(path)
            pcds.append(pcd)
            
        print(f"[Merge 360] Loaded {len(pcds)} clouds. Running Sequential Registration (New360 Logic)...")
        
        # Set the starting model to be the first model (Frame 0) as the base (Accumulator)
        merged_cloud = copy.deepcopy(pcds[0])
        
        # Keep a history of the accumulated transformation matrices of every frame (Current Global Transform)
        max_accum_T = np.identity(4) 
        
        # Loop to compare and connect models pair by pair (Model 1 to 0, Model 2 to 1,...) continuously 
        for i in range(1, len(pcds)):
            print(f"[Merge 360] Aligning Scan {i} -> Scan {i-1}...")
            source = pcds[i]      # Latest model (moving towards target)
            target = pcds[i-1]    # Previous model (standing still)
            
            # 1. Preprocess prepare both data (Downsample + calculate Normals)
            source_down, source_fpfh = ProcessingLogic.preprocess_point_cloud(source, voxel_size)
            target_down, target_fpfh = ProcessingLogic.preprocess_point_cloud(target, voxel_size)
            
            # 2. Let Open3D try to blindly guess the broad overlapping position first (Global RANSAC)
            # Pass down the icp_dist_ratio multiplier to control the strictness of the search
            ransac_result = ProcessingLogic.execute_global_registration(
                source_down, target_down, source_fpfh, target_fpfh, voxel_size, icp_dist_ratio)
            
            # 3. Let Open3D precisely adjust the overlap from the original distance (Local ICP Refinement Point-to-Plane)
            icp_result = o3d.pipelines.registration.registration_icp(
                source_down, target_down, voxel_size, ransac_result.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            
            # Extract the relationship matrix to shift the position between i and i-1 to store
            T_local = icp_result.transformation 
            
            # 4. Convert it to a relationship from i shifted down to compare with the absolute base model 0, so that all pieces are on the same stage
            max_accum_T = np.dot(max_accum_T, T_local)
            
            # 5. Command to transform and combine it with the base stage
            pcd_temp = copy.deepcopy(source) 
            pcd_temp.transform(max_accum_T) # Change the position of the latest model and overlap it
            merged_cloud += pcd_temp        # Combine together
            
        print("[Merge 360] Post-processing (Downsample + Outlier removal)...")
        # Take the entire large finished model and reduce its resolution one last time to prevent the computer from lagging
        pcd_combined_down = merged_cloud.voxel_down_sample(voxel_size=voxel_size)
        
        # Filter out bad points for the final time of merging (Outlier Removal)
        # using the UI-defined parameters for neighbor count and standard deviation aggressiveness
        cl, ind = pcd_combined_down.remove_statistical_outlier(nb_neighbors=outlier_nb, std_ratio=outlier_std)
        pcd_final = pcd_combined_down.select_by_index(ind)
        
        # Calculate the latest surface Normals for the large model
        pcd_final.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
        
        # Save the completely merged image file and export as PLY
        o3d.io.write_point_cloud(output_path, pcd_final)
        print(f"[Merge 360] Saved merged cloud to {output_path}")

    @staticmethod
    def reconstruct_stl(input_path, output_path, mode="watertight", params=None):
        # Function used to create a 3D wireframe or solid mesh (STL from Point Cloud), suitable for 3D printing tasks
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        print(f"[Recon] Loading {input_path}...")
        pcd = o3d.io.read_point_cloud(input_path) # Read the point cloud file
        
        if not pcd.has_points():
            raise ValueError("Point cloud is empty.")
            
        # Create initial Normals (surface direction) immediately if the source file doesn't provide them, otherwise it cannot be shaded
        if not pcd.has_normals():
            print("[Recon] Estimating normals...")
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
            # Check and rotate all Normal lines to point in the same direction
            pcd.orient_normals_consistent_tangent_plane(100)
            
        mesh = None
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

    @staticmethod
    def mesh_360(input_path, output_path, depth=10, density_trim=0.01, orientation_mode="tangent", width=0.0, scale=1.1, linear_fit=False, n_threads=-1, normal_radius=0.1, normal_max_nn=30):
        # Function to create and refine the Mesh specifically for processing models from a 360-degree all-around scan
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

