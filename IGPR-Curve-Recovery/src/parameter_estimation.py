import numpy as np
from .reconstruction import reconstruct_curve
from .utils import compute_l1_error

def estimate_parameters(x_actual, y_actual, t_original, theta_candidates=None, X_candidates=None):
    """
    Recover the hidden parameters (theta, X, M) from the actual curve data.
    
    Parameters:
    x_actual, y_actual (numpy.ndarray): Actual coordinates from the data.
    t_original (numpy.ndarray): Original parameter t array for validation.
    theta_candidates (numpy.ndarray): Angles in degrees to search over.
    X_candidates (numpy.ndarray): X translations to search over.
    
    Returns:
    dict: Best parameters {'theta': theta, 'X': X, 'M': M, 'l1_error': l1_error}
    """
    if theta_candidates is None:
        theta_candidates = np.arange(0.5, 50.0, 0.5)
    if X_candidates is None:
        X_candidates = np.arange(0.0, 100.0, 0.5)
        
    best_theta = None
    best_X = None
    best_M = None
    lowest_score = float("inf")
    lowest_l1 = float("inf")
    
    for theta_deg in theta_candidates:
        theta = np.deg2rad(theta_deg)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        for X in X_candidates:
            # Recovered t under translation and rotation
            recovered_t = (x_actual - X) * cos_theta + (y_actual - 42) * sin_theta
            
            # Constraint: t must be positive for all actual data points
            if np.any(recovered_t <= 0):
                continue
                
            # Recovered perpendicular component A
            recovered_A = -(x_actual - X) * sin_theta + (y_actual - 42) * cos_theta
            
            # Solve for M from A = e^(M * t) * sin(0.3 * t)
            denominator = np.sin(0.3 * recovered_t)
            mask = np.abs(denominator) > 1e-6
            
            if np.sum(mask) < 100:
                continue
                
            recovered_t_valid = recovered_t[mask]
            recovered_A_valid = recovered_A[mask]
            denominator_valid = denominator[mask]
            
            ratio = recovered_A_valid / denominator_valid
            
            # Constraint: ratio must be positive for log
            positive_mask = ratio > 0
            if np.sum(positive_mask) < 100:
                continue
                
            ratio = ratio[positive_mask]
            recovered_t_valid = recovered_t_valid[positive_mask]
            
            # Calculate M for each point
            M_values = np.log(ratio) / np.abs(recovered_t_valid)
            M_mean = np.mean(M_values)
            
            # Physical constraint check on growth factor
            if not (-0.05 < M_mean < 0.05):
                continue
                
            # Variance of recovered M values is our score to minimize
            M_variance = np.var(M_values)
            score = M_variance
            
            # Test parameters by reconstructing curve and evaluating error
            x_pred, y_pred = reconstruct_curve(t_original, theta_deg, X, M_mean)
            l1_error = compute_l1_error(x_actual, y_actual, x_pred, y_pred)
            
            if score < lowest_score:
                lowest_score = score
                lowest_l1 = l1_error
                best_theta = theta_deg
                best_X = X
                best_M = M_mean
                
    return {
        "theta": best_theta,
        "X": best_X,
        "M": best_M,
        "l1_error": lowest_l1,
        "score": lowest_score
    }
