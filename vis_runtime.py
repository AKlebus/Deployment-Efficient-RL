import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

K = [5, 10, 15, 20, 25, 30]

runtime_discrete = [0.029647111892700195, 0.07951521873474121, 0.19600296020507812, 0.3171529769897461, 0.5108981132507324, 0.7251858711242676]

runtime_continuous = [3.64237380027771, 7.748786926269531, 12.19737720489502, 23.56052303314209, 35.67445397377014, 68.64874505996704] 


def plot_benchmarking(data):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

    sns.set(font_scale = 2)

    sns.lineplot(
        data=data, x="K", y="runtime", marker="o", markersize = 6)
    

    #ax.set_xticks(K)S
    ax.set_ylabel("Wall clock time (seconds)")
    ax.set_xlabel("K (Number of deployments)")
        
    fig.tight_layout()
    
    plt.show()
    

def make_data(runtime):
    d = {"K": K, "runtime": runtime}
    return pd.DataFrame(d)
    
if __name__ == "__main__":
    plot_benchmarking(make_data(runtime_continuous))
    plot_benchmarking(make_data(runtime_discrete))


