import open3d as o3d
import copy
import numpy as np

def keep_largest_cluster(pcd, eps=5, min_points=200):
    print("Clustering points...")
    # eps: The density radius. Points closer than this are grouped together.
    # min_points: Minimum points needed to form a valid cluster.
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    
    if len(labels) == 0:
        return pcd

    # Count points in each cluster
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Ignore the noise label (-1)
    valid_clusters = unique_labels != -1
    unique_labels = unique_labels[valid_clusters]
    counts = counts[valid_clusters]
    
    if len(unique_labels) == 0:
        return pcd # Everything was noise

    # Find the cluster with the most points
    largest_cluster_label = unique_labels[counts.argmax()]
    
    # Select only the points belonging to the largest cluster
    largest_cluster_indices = np.where(labels == largest_cluster_label)[0]
    cleaned_pcd = pcd.select_by_index(largest_cluster_indices)
    
    return cleaned_pcd

def remove_outliers_keep_color1(file_path,out, nb_neighbors=20, std_ratio=2.0):
    print(f"Loading {file_path}...")
    pcd = o3d.io.read_point_cloud(file_path)

    # Check if input file actually has colors
    if not pcd.has_colors():
        print("Warning: The input file does not have color data.")

    # 1. Apply Statistical Outlier Removal
    print("Removing outliers...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors,
                                             std_ratio=std_ratio)

    # 2. Select the inliers (This preserves original colors)
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    print(f"Cleaned points: {len(inlier_cloud.points)}")
    print(f"Removed points: {len(outlier_cloud.points)}")

    # 3. Save the Cleaned Cloud (WITH original colors)
    # We save *now* before we mess with colors for visualization
    output_filename = out
    o3d.io.write_point_cloud(output_filename, inlier_cloud)
    print(f"Saved cleaned cloud with colors to: {output_filename}")

    # 4. Visualization
    # We create a deep copy so we don't affect the data we just saved
    vis_inlier = copy.deepcopy(inlier_cloud)
    vis_outlier = copy.deepcopy(outlier_cloud)

    # Paint them distinct colors just for the viewer window
    # Inliers = Grey, Outliers = Red
    vis_inlier.paint_uniform_color([0.8, 0.8, 0.8]) 
    vis_outlier.paint_uniform_color([1, 0, 0])
    
    print("Visualizing... (Red = Removed points)")
    o3d.visualization.draw_geometries([vis_inlier, vis_outlier], 
                                      window_name="Statistical Outlier Removal Result")
    
def remove_outliers_keep_color2(file_path,out, nb_neighbors=20, std_ratio=2.0):
    print(f"Loading {file_path}...")
    pcd = o3d.io.read_point_cloud(file_path)
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    print(f"Average point distance: {avg_dist}")
    pcd = keep_largest_cluster(pcd)
    # Check if input file actually has colors
    if not pcd.has_colors():
        print("Warning: The input file does not have color data.")

    # 1. Apply Statistical Outlier Removal
    print("Removing outliers...")
    cl, ind = pcd.remove_radius_outlier(nb_points=100, radius=5)

    # 2. Select the inliers (This preserves original colors)
    inlier_cloud = pcd.select_by_index(ind)
    outlier_cloud = pcd.select_by_index(ind, invert=True)
    print(f"Cleaned points: {len(inlier_cloud.points)}")
    print(f"Removed points: {len(outlier_cloud.points)}")

    # 3. Save the Cleaned Cloud (WITH original colors)
    # We save *now* before we mess with colors for visualization
    output_filename = out
    o3d.io.write_point_cloud(output_filename, inlier_cloud)
    print(f"Saved cleaned cloud with colors to: {output_filename}")

    # 4. Visualization
    # We create a deep copy so we don't affect the data we just saved
    vis_inlier = copy.deepcopy(inlier_cloud)
    vis_outlier = copy.deepcopy(outlier_cloud)

    # Paint them distinct colors just for the viewer window
    # Inliers = Grey, Outliers = Red
    vis_inlier.paint_uniform_color([0.8, 0.8, 0.8]) 
    vis_outlier.paint_uniform_color([1, 0, 0])
    
    print("Visualizing... (Red = Removed points)")
    o3d.visualization.draw_geometries([vis_inlier, vis_outlier], 
                                      window_name="Statistical Outlier Removal Result")

if __name__ == "__main__":
    # Replace with your colored .ply file
    input_file = r"C:\Users\Tvang\Downloads\APP_0.4\APP_0.4\24_02_2026_3Dscan\Bear\Object_360_bear9_90deg_scan.ply"
    out = r"C:\Users\Tvang\Downloads\APP_0.4\APP_0.4\24_02_2026_3Dscan\Bear - Copy\Object_360_bear9_90deg_scan.ply"
    remove_outliers_keep_color2(input_file,out)

"""
How to Tune the Parameters
The success of the filter depends entirely on two parameters in the remove_statistical_outlier function:

nb_neighbors (Default: 20)

This specifies how many neighbors are taken into account in order to calculate the average distance for a given point.

Higher value: More computationally expensive, but smoother results.

Lower value: Faster, but may be sensitive to local noise.

std_ratio (Default: 2.0)

This is the standard deviation multiplier. It sets the threshold for how far a point can be from the mean distance of its neighbors before being considered an outlier.

Lower value (e.g., 0.5 - 1.0): Aggressive filtering. You will remove more noise, but you might delete useful data (features).

Higher value (e.g., 3.0+): Conservative filtering. You keep more data, but some noise may remain.
"""