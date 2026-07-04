import numpy as np

def reconstruct_curve(t, theta_deg, X, M):
    """
    Reconstruct the curve coordinates (x_pred, y_pred) using parameters.
    
    Parameters:
    t (numpy.ndarray): Parameter array.
    theta_deg (float): Rotation angle in degrees.
    X (float): Translation parameter in x.
    M (float): Linear growth rate parameter in exponential envelope.
    
    Returns:
    tuple: (x_pred, y_pred) reconstructed coordinates.
    """
    theta = np.deg2rad(theta_deg)
    
    x_pred = (t * np.cos(theta)
             - np.exp(M * np.abs(t))
             * np.sin(0.3 * t)
             * np.sin(theta)
             + X)
             
    y_pred = (42
             + t * np.sin(theta)
             + np.exp(M * np.abs(t))
             * np.sin(0.3 * t)
             * np.cos(theta))
             
    return x_pred, y_pred
