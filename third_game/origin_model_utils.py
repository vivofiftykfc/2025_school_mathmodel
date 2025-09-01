import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from scipy.io import loadmat
import os
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')


def level_set_segmentation():
    """
    Main function for level set image segmentation on BSDS500 dataset
    """
    # Step 1: Set paths for BSDS500 dataset
    bsds_root = 'D:/temp_code/2025_school_mathmodel/third_game/data/BSDS500'  # Adjust path accordingly
    image_dir = os.path.join(bsds_root, 'images', 'test')
    groundtruth_dir = os.path.join(bsds_root, 'groundTruth', 'test')

    # Get all JPG images
    image_files = glob.glob(os.path.join(image_dir, '*.jpg'))
    num_images = len(image_files)

    # Initialize arrays to store metrics
    all_accuracies = np.zeros(num_images)
    all_precisions = np.zeros(num_images)
    all_recalls = np.zeros(num_images)
    all_f1_scores = np.zeros(num_images)
    all_dices = np.zeros(num_images)

    # Main loop for processing each image
    for img_idx, img_path in enumerate(image_files):
        img_name = os.path.basename(img_path)
        print(f'Processing image {img_idx + 1}/{num_images}: {img_name}')

        # Read grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Could not load image: {img_name}")
            continue

        # Load ground truth
        gt_file_name = img_name.replace('.jpg', '.mat')
        gt_file_path = os.path.join(groundtruth_dir, gt_file_name)

        try:
            gt_data = loadmat(gt_file_path)
            if 'groundTruth' in gt_data and len(gt_data['groundTruth']) > 0:
                # Extract first segmentation
                gt_segmentation = gt_data['groundTruth'][0][0]['Segmentation'][0][0]
                gt_mask = gt_segmentation.astype(bool)

                # Resize GT to match image dimensions if necessary
                if img.shape != gt_mask.shape:
                    gt_mask = cv2.resize(gt_mask.astype(np.uint8),
                                         (img.shape[1], img.shape[0]),
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
            else:
                print(f"Ground truth not found for {img_name}. Skipping...")
                continue
        except Exception as e:
            print(f"Error loading ground truth for {img_name}: {e}")
            continue

        # Step 2: Set parameters (following Algorithm 1 from paper)
        timestep = 1.0  # Δt = 1
        mu = 0.1  # μ = 0.1
        lambda_param = 2.0  # λ = 2
        epsilon = 1.0  # ε = 1
        c0 = 2.0  # c0 = 2
        maxiter = 800  # maximum iterations
        sigma = 3.0  # σ = 3
        w = 3  # neighborhood window size
        tau = 24.0  # τ = 24
        gamma = 0.5  # γ = 0.5

        # Step 3: Smooth image with Gaussian filter
        I_sigma = gaussian_filter(img.astype(np.float64), sigma)

        # Step 3.5: Pre-fitting functions calculation (Eq. 7-8)
        print("Starting pre-fitting calculation...")
        rows, cols = img.shape
        f_med = np.zeros((rows, cols))
        cl = np.zeros((rows, cols))
        cs = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                # Define neighborhood Ω_x
                r_min = max(0, i - w)
                r_max = min(rows, i + w + 1)
                c_min = max(0, j - w)
                c_max = min(cols, j + w + 1)

                neighborhood = I_sigma[r_min:r_max, c_min:c_max]

                # Step 1: Calculate f_med(x) = median(I(y) | y ∈ Ω_x)
                f_med[i, j] = np.median(neighborhood)

                # Step 2: Define Ω_l and Ω_s
                larger_pixels = neighborhood[neighborhood > f_med[i, j]]
                smaller_pixels = neighborhood[neighborhood < f_med[i, j]]

                # Step 3: Calculate c_l and c_s
                if len(larger_pixels) > 0:
                    cl[i, j] = np.mean(larger_pixels)
                else:
                    cl[i, j] = f_med[i, j]

                if len(smaller_pixels) > 0:
                    cs[i, j] = np.mean(smaller_pixels)
                else:
                    cs[i, j] = f_med[i, j]

        print("Pre-fitting calculation finished.")

        # Step 4: Calculate edge indicator according to Eq. 10-11
        beta = 2 * np.std(I_sigma)  # β(I) = 2S(I_σ)

        # Calculate gradient magnitude
        grad_x = ndimage.sobel(I_sigma, axis=1)
        grad_y = ndimage.sobel(I_sigma, axis=0)
        grad_magnitude_squared = grad_x ** 2 + grad_y ** 2
        g = 1 - np.tanh(grad_magnitude_squared / beta)  # g_β

        # Step 5: Set initial phi
        phi = c0 * np.ones((rows, cols))
        img_mean = np.mean(I_sigma)
        initial_mask = I_sigma > img_mean
        phi[initial_mask] = -c0

        # Main iteration loop
        for k in range(maxiter):
            # Step 6: Apply Neumann boundary conditions
            phi = neumann_bound_cond(phi)

            # Step 7: Calculate differential of regularized term
            dist_reg_term = dist_reg_p3(phi)

            # Step 8-10: Update phi according to Eq. 16
            dirac_phi = dirac(phi, epsilon)
            phi_x, phi_y = np.gradient(phi)
            s = np.sqrt(phi_x ** 2 + phi_y ** 2)

            # Normalize gradients
            s_safe = s + 1e-10
            Nx = phi_x / s_safe
            Ny = phi_y / s_safe

            # 1. Length term (based on new g_β)
            g_x, g_y = np.gradient(g)
            div_g_normal = divergence(g * Nx, g * Ny)
            edge_term = dirac_phi * (g_x * Nx + g_y * Ny + div_g_normal)

            # 2. Area term (using adaptive sign function)
            adaptive_sign_func = gamma * np.arctan((I_sigma - (cl + cs) / 2) / tau)
            area_term = adaptive_sign_func * g * dirac_phi

            # 3. Combined evolution equation (Eq. 16)
            phi = phi + timestep * (mu * dist_reg_term + lambda_param * edge_term + area_term)

            # Display results (only for first image to avoid too many plots)
            if k % 50 == 0 and img_idx == 0:
                plt.figure(figsize=(12, 6))

                plt.subplot(1, 2, 1)
                plt.imshow(img, cmap='gray')
                plt.contour(phi, levels=[0], colors='red', linewidths=2)
                plt.title(f'Contour result, iteration={k}')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(-phi, cmap='viridis')
                plt.contour(phi, levels=[0], colors='red', linewidths=2)
                plt.title(f'Level set φ, iteration={k}')
                plt.axis('off')

                plt.tight_layout()
                plt.pause(0.01)

            # Debug output
            if k % 100 == 0:
                print(f"  Iter {k}: RegTerm={np.mean(np.abs(dist_reg_term)):.6f}, "
                      f"EdgeTerm={np.mean(np.abs(edge_term)):.6f}, "
                      f"AreaTerm={np.mean(np.abs(area_term)):.6f}")

        # Final segmentation and metrics calculation
        final_segmentation = phi <= 0

        # Ensure same size as ground truth
        if final_segmentation.shape != gt_mask.shape:
            final_segmentation = cv2.resize(final_segmentation.astype(np.uint8),
                                            (gt_mask.shape[1], gt_mask.shape[0]),
                                            interpolation=cv2.INTER_NEAREST).astype(bool)

        # Calculate performance metrics
        metrics = calculate_segmentation_metrics(final_segmentation, gt_mask)

        all_accuracies[img_idx] = metrics['Accuracy']
        all_precisions[img_idx] = metrics['Precision']
        all_recalls[img_idx] = metrics['Recall']
        all_f1_scores[img_idx] = metrics['F1_Score']
        all_dices[img_idx] = metrics['Dice']

        print(f"  Metrics for {img_name}: Acc={metrics['Accuracy']:.4f}, "
              f"Prec={metrics['Precision']:.4f}, Rec={metrics['Recall']:.4f}, "
              f"F1={metrics['F1_Score']:.4f}, Dice={metrics['Dice']:.4f}")

        # Optional: Display final result
        if img_idx < 5:  # Only show first 5 images
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap='gray')
            plt.contour(phi, levels=[0], colors='red', linewidths=2)
            plt.title(f'Final segmentation for {img_name}')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            # Create overlay
            overlay = img.copy()
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB)
            overlay[final_segmentation] = [255, 0, 0]  # Red overlay
            plt.imshow(overlay)
            plt.title(f'Segmentation Overlay (Dice: {metrics["Dice"]:.4f})')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

    # Display average metrics
    print(f"\n--- Average Metrics Across {num_images} Images ---")
    print(f"Average Accuracy:   {np.mean(all_accuracies):.4f}")
    print(f"Average Precision:  {np.mean(all_precisions):.4f}")
    print(f"Average Recall:     {np.mean(all_recalls):.4f}")
    print(f"Average F1-Score:   {np.mean(all_f1_scores):.4f}")
    print(f"Average Dice:       {np.mean(all_dices):.4f}")

    # Save results
    np.savez('bsds_segmentation_results.npz',
             accuracies=all_accuracies,
             precisions=all_precisions,
             recalls=all_recalls,
             f1_scores=all_f1_scores,
             dices=all_dices)


def dist_reg_p3(phi):
    """Distance regularization term with p3 function"""
    phi_x, phi_y = np.gradient(phi)
    s = np.sqrt(phi_x ** 2 + phi_y ** 2)

    # Eq. 15: d_p3(s)
    dp3 = np.zeros_like(s)

    # Case 1: s in [0, 1]
    idx1 = (s >= 0) & (s <= 1)
    s1 = s[idx1]
    dp3[idx1] = s1 + (1 / np.pi) * np.sin(np.pi * (s1 - 1)) - 1

    # Case 2: s in (1, inf)
    idx2 = s > 1
    s2 = s[idx2]
    dp3[idx2] = 1 - (2 / (1 + s2 ** 4))

    # Avoid division by zero
    s_safe = s + 1e-10
    dphi_x = dp3 * phi_x / s_safe
    dphi_y = dp3 * phi_y / s_safe

    return divergence(dphi_x, dphi_y) + 4 * ndimage.laplace(phi)


def divergence(nx, ny):
    """Calculate divergence of vector field"""
    nxx = np.gradient(nx, axis=1)
    nyy = np.gradient(ny, axis=0)
    return nxx + nyy


def dirac(x, sigma):
    """Dirac delta function approximation"""
    f = (1 / (2 * sigma)) * (1 + np.cos(np.pi * x / sigma))
    b = (x <= sigma) & (x >= -sigma)
    return f * b


def neumann_bound_cond(f):
    """Apply Neumann boundary conditions"""
    g = f.copy()
    nrow, ncol = f.shape

    # Corner conditions
    g[0, 0] = g[2, 2]
    g[0, ncol - 1] = g[2, ncol - 3]
    g[nrow - 1, 0] = g[nrow - 3, 2]
    g[nrow - 1, ncol - 1] = g[nrow - 3, ncol - 3]

    # Edge conditions
    g[0, 1:ncol - 1] = g[2, 1:ncol - 1]
    g[nrow - 1, 1:ncol - 1] = g[nrow - 3, 1:ncol - 1]
    g[1:nrow - 1, 0] = g[1:nrow - 1, 2]
    g[1:nrow - 1, ncol - 1] = g[1:nrow - 1, ncol - 3]

    return g


def calculate_segmentation_metrics(segmented_mask, ground_truth_mask):
    """Calculate common segmentation performance metrics"""
    # Ensure both masks are boolean
    seg_mask = segmented_mask.astype(bool)
    gt_mask = ground_truth_mask.astype(bool)

    # Flatten masks
    seg_flat = seg_mask.flatten()
    gt_flat = gt_mask.flatten()

    # Calculate confusion matrix components
    TP = np.sum(seg_flat & gt_flat)
    TN = np.sum(~seg_flat & ~gt_flat)
    FP = np.sum(seg_flat & ~gt_flat)
    FN = np.sum(~seg_flat & gt_flat)

    # Calculate metrics
    metrics = {}

    # Accuracy
    metrics['Accuracy'] = (TP + TN) / (TP + TN + FP + FN)

    # Precision
    metrics['Precision'] = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Recall
    metrics['Recall'] = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1-Score
    if metrics['Precision'] + metrics['Recall'] > 0:
        metrics['F1_Score'] = 2 * (metrics['Precision'] * metrics['Recall']) / (
                    metrics['Precision'] + metrics['Recall'])
    else:
        metrics['F1_Score'] = 0

    # Dice Similarity Coefficient
    metrics['Dice'] = (2 * TP) / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0

    return metrics


if __name__ == "__main__":
    level_set_segmentation()