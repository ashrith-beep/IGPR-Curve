import numpy as np
import pandas as pd

def load_data(file_path):
    """
    Load x and y actual coordinates from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    tuple: (x_actual, y_actual) as numpy arrays.
    """
    data = pd.read_csv(file_path)
    x_actual = data["x"].values
    y_actual = data["y"].values
    return x_actual, y_actual

def generate_t_original(n):
    """
    Generate the original parameter t array using piecewise linspace configurations.
    
    Parameters:
    n (int): Length of the actual coordinate arrays.
    
    Returns:
    numpy.ndarray: The parameter t array of length n.
    """
    t_original = np.zeros(n)
    
    # Segment 1
    end1 = min(500, n)
    t_original[:end1] = np.linspace(27, 33, end1)
    
    # Segment 2
    if n > 500:
        end2 = min(1000, n)
        t_original[500:end2] = np.linspace(33, 39, end2 - 500)
        
    # Segment 3
    if n > 1000:
        end3 = min(1500, n)
        t_original[1000:end3] = np.linspace(39, 45, end3 - 1000)
        
    # Segment 4
    if n > 1500:
        t_original[1500:] = np.linspace(45, 51, n - 1500)
        
    return t_original

def compute_l1_error(x_actual, y_actual, x_pred, y_pred):
    """
    Calculate the sum of absolute errors (L1 distance) between actual and predicted coordinates.
    
    Parameters:
    x_actual, y_actual (numpy.ndarray): Actual coordinates.
    x_pred, y_pred (numpy.ndarray): Predicted coordinates.
    
    Returns:
    float: Total L1 error.
    """
    return np.sum(np.abs(x_actual - x_pred)) + np.sum(np.abs(y_actual - y_pred))

def save_output_parameters(file_path, theta, X, M):
    """
    Save the recovered parameters to a text file.
    """
    with open(file_path, "w") as f:
        f.write(f"Theta (degrees) : {theta}\n")
        f.write(f"X (translation) : {X}\n")
        f.write(f"M (parameter)   : {M}\n")

def save_l1_error(file_path, l1_error):
    """
    Save the final L1 reconstruction error to a text file.
    """
    with open(file_path, "w") as f:
        f.write(f"L1 Error : {l1_error}\n")
