import os
import matplotlib.pyplot as plt
import numpy as np

# Use relative imports when run as part of the package or configure path for CLI execution
if __name__ == "__main__" and __package__ is None:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.utils import load_data, generate_t_original, save_output_parameters, save_l1_error
    from src.parameter_estimation import estimate_parameters
    from src.reconstruction import reconstruct_curve
else:
    from .utils import load_data, generate_t_original, save_output_parameters, save_l1_error
    from .parameter_estimation import estimate_parameters
    from .reconstruction import reconstruct_curve

def main():
    # Setup paths relative to this script
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(src_dir)
    
    data_path = os.path.join(project_dir, "data", "xy_data.csv")
    results_dir = os.path.join(project_dir, "results")
    
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")
        
    x_actual, y_actual = load_data(data_path)
    n = len(x_actual)
    print(f"Loaded {n} data points.")
    
    # Generate parameter t
    t_original = generate_t_original(n)
    
    print("Running parameter estimation (IGPR)...")
    results = estimate_parameters(x_actual, y_actual, t_original)
    
    theta = results["theta"]
    X = results["X"]
    M = results["M"]
    l1_error = results["l1_error"]
    
    print("\n========== FINAL IGPR RESULT ==========\n")
    print(f"Theta    : {theta:.2f} degrees")
    print(f"X        : {X:.2f}")
    print(f"M        : {M:.8f}")
    print(f"L1 Error : {l1_error:.4f}")
    print("\n=======================================\n")
    
    # Save output parameters and L1 error
    output_params_path = os.path.join(results_dir, "output_parameters.txt")
    l1_error_path = os.path.join(results_dir, "l1_error.txt")
    
    save_output_parameters(output_params_path, theta, X, M)
    save_l1_error(l1_error_path, l1_error)
    print(f"Saved parameters to {output_params_path}")
    print(f"Saved L1 error to {l1_error_path}")
    
    # Reconstruct final curve for plot
    x_final, y_final = reconstruct_curve(t_original, theta, X, M)
    
    # Plotting actual data vs. recovered curve
    plt.figure(figsize=(10, 7))
    plt.scatter(x_actual, y_actual, color='royalblue', alpha=0.6, label="Actual Data (Noisy / Scrambled)", s=10)
    plt.plot(x_final, y_final, color='crimson', linewidth=2.5, label="Recovered IGPR Curve")
    
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.title(f"IGPR Reconstruction (Theta={theta}°, X={X}, M={M:.5f})", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save figure
    plot_path = os.path.join(results_dir, "reconstructed_curve.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved reconstructed curve plot to {plot_path}")
    print("Done!")

if __name__ == "__main__":
    main()
