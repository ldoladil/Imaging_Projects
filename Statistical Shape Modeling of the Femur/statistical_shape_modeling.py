# STATISTICAL SHAPE MODELING PIPELINE FOR FEMUR BONES

# 0: CREATING THE PROJECT ENVIRONMENT

# 0.1: IMPORTING THE LIBRARIES USEFUL FOR THE PROJECT

import re
from time import time
import numpy as np
from scipy.spatial import KDTree, procrustes
import pyvista as pv
import os
np.random.seed(42)

# 0.2: SETTING THE CORRECT WORKING DIRECTORY
 
current_path = os.getcwd()
print(current_path) # Make sure to be in the "data" folder

# 0.3: LOADING THE DATA "MANUALLY" FROM .IV FILES

def read_iv_file_manually(file_path):
    """
    Manually parses an ASCII Open Inventor (.iv) file to extract vertices.
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # 1. Find the coordinate block. 
    # Look for 'Coordinate3' followed eventually by 'point ['
    # This regex looks for the content inside point [ ... ]
    match = re.search(r'point\s*\[([^\]]+)\]', content, re.DOTALL)
    
    if not match:
        raise ValueError(f"Could not find 'point [...]' block in {file_path}. Is it binary?")

    # 2. Extract the text block containing numbers
    point_data_str = match.group(1)

    # 3. Clean up formatting (remove commas, newlines)
    # Replace commas with spaces to handle "1.0, 2.0, 3.0" format
    point_data_str = point_data_str.replace(',', ' ')
    
    # 4. Convert to numpy array
    try:
        # Parse all numbers from the string
        values = np.fromstring(point_data_str, sep=' ')
        
        # Reshape into (N, 3) for X, Y, Z coordinates
        points = values.reshape(-1, 3)
        return points
        
    except Exception as e:
        print(f"Error parsing points in {file_path}: {e}")
        return None

all_femurs_points = []
file_prefix = ["001", "002", "004", "005", "006", "007", "008", "009", "010",
    "011", "013", "014", "015", "016", "017", "018", "019", "020", 
    "021", "022"]
iv_files = [f"{prefix}.iv" for prefix in file_prefix]
count = 0

# 0.4: TEST FUNCTION TO CHECK SIZE AND DIMENSION OF EACH LOADED POINT CLOUD

for filename in iv_files:
    pts = read_iv_file_manually(filename)
    print(pts.shape[0], pts.shape[1])
    if pts is not None:
        count += 1
        all_femurs_points.append(pts)

print(count)
print(f"Loaded {len(all_femurs_points)} femurs.")
print(np.median([len(pts) for pts in all_femurs_points]))
print(np.min([len(pts) for pts in all_femurs_points]))
print(np.max([len(pts) for pts in all_femurs_points]))

# 0.5: VISUALIZATION FUNCTIONS FOR POINT CLOUDS

def _maybe_downsample(pts, n):
    if n is None or pts.shape[0] <= n:
        return pts
    idx = np.random.choice(pts.shape[0], n, replace=False)
    return pts[idx]

def visualize_point_clouds_overlaid(point_list, downsample, point_size=3):
    """
    Overlay all point clouds in one PyVista window. Pass either all_femurs_points
    or [t['aligned_mesh'] for t in transformations] to `point_list`.
    """
    plt = pv.Plotter()
    for i, pts in enumerate(point_list):
        pts_ds = _maybe_downsample(np.asarray(pts), downsample)
        cloud = pv.PolyData(pts_ds)
        color = tuple(np.random.rand(3))
        plt.add_points(cloud, color=color, point_size=point_size, render_points_as_spheres=True, name=f"pc{i}")
    plt.add_axes()
    plt.show()

def visualize_point_clouds_grid(point_list, downsample=2000, ncols=5, point_size=3):
    """
    Show each point cloud in its own subplot (grid). Good for quick per-sample inspection.
    """
    n = len(point_list)
    nrows = int(np.ceil(n / ncols))
    plotter = pv.Plotter(shape=(nrows, ncols), notebook=False)
    for i, pts in enumerate(point_list):
        r = i // ncols
        c = i % ncols
        plotter.subplot(r, c)
        pts_ds = _maybe_downsample(np.asarray(pts), downsample)
        plotter.add_points(pv.PolyData(pts_ds), point_size=point_size, render_points_as_spheres=True)
        plotter.add_text(f"#{i}", font_size=10)
        plotter.add_axes()
    # empty subplots cleanup
    for i in range(n, nrows * ncols):
        r = i // ncols
        c = i % ncols
        plotter.subplot(r, c)
        plotter.add_text("", font_size=1)
    plotter.show()

def quick_mpl_3d(point_list, downsample=2000, size=1):
    """
    Very lightweight 3D scatter (non-interactive or for notebooks) using matplotlib.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    for pts in point_list:
        pts_ds = _maybe_downsample(np.asarray(pts), downsample)
        ax.scatter(pts_ds[:, 0], pts_ds[:, 1], pts_ds[:, 2], s=size, alpha=0.6)
    plt.show()

# Example calls:

#visualize_point_clouds_overlaid([t['aligned_mesh'] for t in transformations], downsample=4000)
#visualize_point_clouds_grid([t['aligned_mesh'] for t in transformations], downsample=4000, ncols=5)
#quick_mpl_3d(all_femurs_points, downsample=1000)

# 1. ICP IMPLEMENTATION FOR THE SCALED RIGID REGISTRATION

# 1.1 : DOWNSAMPLING FUNCTION

# We perform a downsampling of the meshes in order to speed up the registration process. 
# This is especially useful for ICP, which could not be performed with the original point clouds due to their excessive size. 

def farthest_point_sampling(points, k):
    """
    Selects k points from the point cloud such that they are 
    maximally distant from each other.
    
    Input:
        points: (N, 3) numpy array
        k: int, number of points to select
    Output:
        selected_points: (k, 3) numpy array
        indices: list of indices selected
    """
    N, D = points.shape
    
    if N < k:
        raise ValueError(f"Input has fewer points ({N}) than requested k ({k}).")

    # Array to store the "distance to the nearest selected point" for every point
    # Initialize with Infinity
    min_distances = np.full(N, np.inf)
    
    selected_indices = []
    
    # 1. Pick the first point randomly
    current_idx = np.random.randint(0, N)
    selected_indices.append(current_idx)
    
    # Loop until we have k points
    for _ in range(k - 1):
        # Get the coordinates of the last added point
        last_added_point = points[current_idx]
        
        # 2. Compute Euclidean distance (squared) from the last added point to ALL points
        # shape (N,)
        dist_to_last = np.sum((points - last_added_point)**2, axis=1)
        
        # 3. Update the minimum distance for every point
        # A point keeps its old distance unless the new point is closer
        min_distances = np.minimum(min_distances, dist_to_last)
        
        # 4. Pick the point that is farthest from the current set
        # (The one with the largest 'minimum distance')
        current_idx = np.argmax(min_distances)
        selected_indices.append(current_idx)
        
    return points[selected_indices]

# 1.2 : COMPUTE SIMILARITY TRANSFORM FUNCTION

def compute_similarity_transform(source_points, target_points):
    """
    Calculates the Rotation (R), Translation (t), and Scale (s) that best 
    aligns source_points to target_points.
    
    Input:  Nx3 arrays of corresponding points
    Output: R (3x3), t (3x1), s (scalar)
    """
    # 1. Compute Centroids
    centroid_source = np.mean(source_points, axis=0)
    centroid_target = np.mean(target_points, axis=0)

    # 2. Center the points
    source_centered = source_points - centroid_source
    target_centered = target_points - centroid_target

    # 3. Compute Scale (s)
    # s = sqrt( var(target) / var(source) )
    var_source = np.sum(np.square(source_centered))
    var_target = np.sum(np.square(target_centered))
    s = np.sqrt(var_target / var_source)

    # 4. Compute Rotation (R) using SVD
    # H = (Source_Centered * s)^T * Target_Centered
    H = np.dot((source_centered * s).T, target_centered)
    
    U, D, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Handle reflection case (determinant is -1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 5. Compute Translation (t)
    t = centroid_target - s * np.dot(R, centroid_source)

    return R, t, s
    
# 1.3 : ICP FUNCTION (ITERATIVE CLOSEST POINT WITH SCALING) 

def icp_scaled(source_points, target_points, max_iterations=50, tolerance=1e-5):

    current_source = source_points.copy()
    prev_error = float('inf')
    
    # Find closest points using KDTree for efficiency
    tree = KDTree(target_points)
    
    for i in range(max_iterations):

        # 1. Retrieve point correspondences
        distances, indices = tree.query(current_source)
        matched_target_points = target_points[indices]
        
        # 2. Compute Transform for THIS STEP
        R_step, t_step, s_step = compute_similarity_transform(current_source, matched_target_points)
        
        # 3. Apply Transform
        current_source = s_step * np.dot(current_source, R_step.T) + t_step
        
        # 4. Check Error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    
    # Re-calculate the TOTAL transform from the original input to the final result
    final_R, final_t, final_s = compute_similarity_transform(source_points, current_source)
        
    return current_source, final_R, final_t, final_s

# 1.4 : EXECUTION OF THE SCALED RIGID REGISTRATION FOR ALL FEMURS

# The initial reference shape is chosen as the 10th femur (index 9), since it is of maximal res. 
# This will be our "template" for the ICP registration.
# Then the moving points (or source) will be each of the other femurs in the dataset. 

transformations = []

# 1.4.1 : ALL FEMURS DOWNSAMPLED TO 4K POINTS. 

template_raw = all_femurs_points[9]
template_low = farthest_point_sampling(template_raw, 1000)

femurs_downsampled = []

for i in range(len(all_femurs_points)):
    femurs_downsampled.append(farthest_point_sampling(all_femurs_points[i], 1000))

femurs_downsampled[9] = template_low  # Ensure the template is the downsampled version

# 1.4.2 : ICP EXECUTION

for i in range(len(femurs_downsampled)):
    print(f"Registering femur {i}...")
    
    # Run Scaled Registration
    aligned_points, R, t, s = icp_scaled(femurs_downsampled[i], template_low)
    
    # Store
    transform_data = {
        "id": i,
        "rotation": R,       
        "translation": t,    
        "scale": s,          
        "aligned_mesh": aligned_points
    }
    transformations.append(transform_data)

# 1.4.3 : ICP OUTCOME VISUALIZATION

for i in range(len(transformations)):
    print(f"Femur {i}: Scale = {transformations[i]['scale']:.4f}")
    print(f"Femur {i}: Cloud_size = {len(transformations[i]['aligned_mesh'])}")

visualize_point_clouds_grid([t['aligned_mesh'] for t in transformations], downsample=4000, ncols=5)

# 2. DEFORMABLE REGISTRATION USING CPD

from pycpd import DeformableRegistration
import time 

def deformable_registration_cpd(template_points, target_points, alpha=2.0, beta=2.0):
    """
    Deforms the template_points to match the target_points using CPD.
    
    Parameters:
    - alpha: Smoothness weight (higher = more rigid/smooth deformation)
    - beta:  Width of the Gaussian filter (interaction between points)
    """
    print(f"Starting CPD: Template ({len(template_points)} pts) -> Target ({len(target_points)} pts)")
    
    # Initialize the Deformable Registration object
    # X = Target (Fixed), Y = Source (Moving/Deforming)
    reg = DeformableRegistration(X=target_points, Y=template_points, alpha=alpha, beta=beta)
    
    # Run the registration
    # output_points is the template DEFORMED to look like the target
    output_points, _ = reg.register()
    
    return output_points

# --- MAIN LOOP ---
aligned_dataset = [] # Stores the SSM-ready data

# Use the aligned mesh from the reference subject (output of ICP) as the template
template = transformations[9]["aligned_mesh"]  # aligned atlas

# Deform the template to each other aligned subject (target = aligned_mesh from ICP)
for i in range(len(transformations)):
    start_time = time.time()

    if i!=9:
        target = transformations[i]["aligned_mesh"]
        print(f"Deforming template to Subject {i}...")
        # CPD will deform `template` (Y) to match `target` (X)
        deformed_shape = deformable_registration_cpd(template, target)
        aligned_dataset.append(deformed_shape)
    else:
        # For the template itself, just add it directly
        aligned_dataset.append(template)
        
    end_time = time.time()
    duration = end_time - start_time
    print(f"Finished Subject {i} in {duration:.2f} seconds.")
    print(f"New shape has {len(deformed_shape)} vertices.\n")

# Check: All meshes in aligned_dataset should now have the same length as the template
print(f"Final check: {[len(p) for p in aligned_dataset]}")