
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

ProbabilityFrozen = [0.5, 0.6, 0.7, 0.8, 0.9]


def plot_benchmarking(data):
    """Plot the steps and rewards from dataframes."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))

    sns.set(font_scale = 2)

    sns.lineplot(
        data=data, x="Frozen Probability", y="Success Rate", hue="Algorithm", marker="o", markersize = 6)

    ax.set_xticks(ProbabilityFrozen)
        
    fig.tight_layout()
    img_title = "Benchmarking Algorithm 1 and Q-Learning.png"
    
    fig.savefig(Path("./img/") / img_title, bbox_inches="tight")
    plt.show()


def plot_mapsize_comp(success_rate, map_sizes):
    
    maps = map_sizes * len(ProbabilityFrozen) 
    maps.sort()
    
    d = {"Frozen Probability": ProbabilityFrozen*len(map_sizes), "Success Rate": success_rate, "Map Size": maps}
    data = pd.DataFrame(data=d)
    
    sns.color_palette("Paired")

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 5))
    
    sns.set(font_scale = 2)

    sns.lineplot(
        data=data, x="Frozen Probability", y="Success Rate", hue="Map Size", marker="o", markersize = 6, palette = ["C0", "C1", "C2", "k"])

    ax.set_xticks(ProbabilityFrozen)
        
    fig.tight_layout()
    img_title = "success rate.png"
    
    fig.savefig(Path("./img/") / img_title, bbox_inches="tight")
    plt.show()



def make_plot_data(q_success_rate, a1_success_rate):
    d = {"Frozen Probability": ProbabilityFrozen*2, "Success Rate": q_success_rate + a1_success_rate, "Algorithm": ["Q-Learning"]*5 + ["Algorithm 1"]*5}
    return pd.DataFrame(data=d)


if __name__ == "__main__":
    #d = {"Frozen Probability": [0.5, 0.6, 0.7, 0.8, 0.9]*2, "Success Rate": [0.1775, 0.224, 0.2775, 0.5315, 0.812] + [0.71, 0.71, 0.8, 0.83, 0.91], "Algorithm": ["Q-Learning"]*5 + ["Algorithm 1"]*5}
    #data = pd.DataFrame(data = d)S
    
    q_success_rate = [0.0125, 0.0375, 0.0665, 0.127, 0.337]
    a1_success_rate =  [0.03, 0.12, 0.11, 0.19, 0.2]
    
    data = make_plot_data(q_success_rate, a1_success_rate)
    plot_benchmarking(data)
    
    plot_benchmarking(data)



"""
Map size 4:
different reward 100 runs:
Q Learning Success Rate:  [0.0225, 0.03, 0.0665, 0.184, 0.456]
Algorithm 1 Success Rate:  [0.95, 0.92, 0.91, 0.96, 0.98]


same reward 100 runs:
Q Learning Success Rate:  [0.1155, 0.191, 0.345, 0.4615, 0.683]
Algorithm 1 Success Rate:  [0.34, 0.36, 0.54, 0.57, 0.62]

Map size 5:
different reward 100 runs:
Q Learning Success Rate:  [0.001, 0.005, 0.0125, 0.0655, 0.1915]
Algorithm 1 Success Rate:  [0.8, 0.79, 0.77, 0.83, 0.92]

same reward 100 runs:
Q Learning Success Rate:  [0.0125, 0.0375, 0.0665, 0.127, 0.337]
Algorithm 1 Success Rate:  [0.03, 0.12, 0.11, 0.19, 0.2]

Success Rate:
[1.0, 0.96, 0.94, 0.96, 0.96, 0.82, 0.72, 0.8, 0.88, 0.94, 0.7, 0.6, 0.68, 0.74, 0.9, 0.64, 0.4, 0.52, 0.78, 0.96]


"""