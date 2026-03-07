import scipy.io
import numpy as np
import os

def analyze_calibration_mat(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Load .mat file
    data = scipy.io.loadmat(file_path)
    
    print("="*60)
    print(f"CALIBRATION DATA ANALYSIS: {os.path.basename(file_path)}")
    print("="*60)

    # 1. Camera Intrinsic Matrix (K)
    if 'cam_K' in data:
        K = data['cam_K']
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        print(f"\n[1] CAMERA INTRINSICS (K)")
        print(f"    - Focal Length (px): fx = {fx:.2f}, fy = {fy:.2f}")
        print(f"    - Principal Point:   cx = {cx:.2f}, cy = {cy:.2f}")
        print(f"    - Aspect Ratio:      {fx/fy:.4f} (1.0 is ideal)")

    # 2. Lens Distortion (dc)
    if 'dc' in data:
        dc = data['dc'].flatten()
        print(f"\n[2] LENS DISTORTION (dc)")
        labels = ['k1 (Radial)', 'k2 (Radial)', 'p1 (Tangential)', 'p2 (Tangential)', 'k3 (Radial)']
        for label, val in zip(labels, dc):
            print(f"    - {label:16}: {val:.6f}")
        
        dist_strength = np.linalg.norm(dc[:2]) # Rough estimate of distortion strength
        print(f"    - Distortion Intensity: {dist_strength:.4f} (Lower is better)")

    # 3. Stereo Geometry (R and T)
    if 'R' in data and 'T' in data:
        R = data['R']
        T = data['T'].flatten()
        
        # Calculate Euclidean Distance (Baseline)
        baseline = np.linalg.norm(T)
        
        # Convert Rotation Matrix to Euler Angles (Degrees)
        sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
        singular = sy < 1e-6
        if not singular:
            rx = np.arctan2(R[2,1], R[2,2])
            ry = np.arctan2(-R[2,0], sy)
            rz = np.arctan2(R[1,0], R[0,0])
        else:
            rx = np.arctan2(-R[1,2], R[1,1])
            ry = np.arctan2(-R[2,0], sy)
            rz = 0
            
        print(f"\n[3] STEREO GEOMETRY (Camera-to-Projector)")
        print(f"    - Baseline Distance: {baseline:.2f} mm")
        print(f"    - Offset (X, Y, Z):  [{T[0]:.2f}, {T[1]:.2f}, {T[2]:.2f}] mm")
        print(f"    - Relative Rotation: Pitch={np.degrees(rx):.2f}°, Yaw={np.degrees(ry):.2f}°, Roll={np.degrees(rz):.2f}°")

    # 4. Projector Planes Data
    if 'wPlaneCol' in data:
        planes = data['wPlaneCol']
        print(f"\n[4] PROJECTOR PLANES")
        print(f"    - Total Columns:     {planes.shape[1] if planes.ndim > 1 else planes.shape[0]}")
        print(f"    - Plane Data Shape:  {planes.shape}")

    # 5. Efficiency & Comparison Metrics
    print(f"\n[5] EFFICIENCY METRICS")
    if 'ret' in data:
        print(f"    - Reprojection Error: {data['ret'][0][0]:.4f} px")
        if data['ret'][0][0] < 0.5:
            print("      Status: EXCELLENT (High precision scan expected)")
        elif data['ret'][0][0] < 1.0:
            print("      Status: GOOD (Standard quality)")
        else:
            print("      Status: POOR (Needs recalibration for better accuracy)")

    print("\n" + "="*60)

if __name__ == "__main__":
    # ระบุชื่อไฟล์ .mat ของคุณที่นี่
    mat_file_path = r"C:\Users\Tvang\Downloads\APP_0.4\APP_0.4\24_02_2026_3Dscan\24_02_2026_3Dscan\calib1\calib.mat"
    analyze_calibration_mat(mat_file_path)