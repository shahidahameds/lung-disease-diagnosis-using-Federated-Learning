import os
import pandas as pd
import matplotlib.pyplot as plt
from config import RESULTS_DIR

def generate_convergence_plot():
    conv_file = os.path.join(RESULTS_DIR, "convergence.csv")
    if not os.path.exists(conv_file):
        print("Convergence data not found.")
        return
        
    df = pd.read_csv(conv_file)
    
    plt.figure(figsize=(10, 6))
    
    for config in df['Configuration'].unique():
        subset = df[df['Configuration'] == config]
        # Sort by round just in case
        subset = subset.sort_values(by='Round')
        plt.plot(subset['Round'], subset['Accuracy'], marker='o', label=config)
        
    plt.title('Accuracy Convergence across FL Configurations')
    plt.xlabel('Communication Round')
    plt.ylabel('Test Accuracy')
    plt.legend()
    plt.grid(True)
    
    out_path = os.path.join(RESULTS_DIR, "accuracy_convergence.png")
    plt.savefig(out_path)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    generate_convergence_plot()
