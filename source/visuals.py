from typing import Optional

import numpy as np

import matplotlib.pyplot as plt

PARAMS = {
    'figure.figsize': (10, 5),
    'figure.constrained_layout.use': True,
    'figure.facecolor': 'white',
    'font.size': 10,
    'axes.labelsize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.titlesize': 16,
    'figure.max_open_warning': 50,
}

def plot_mtx(
    mtx, 
    title: str = '', 
    save_path: Optional[str] = None,
    show_plot: bool = True,
    dpi: int = 200, 
):
    plt.matshow(mtx)
    cbar = plt.colorbar()
    cbar.set_label("Values")
    plt.title(title)
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=dpi)
    if show_plot:
        plt.show()
    else:
        plt.close()

def plot_mtx_list(
    mtx_list, 
    titles=None, 
    overall_title: Optional[str] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    dpi: int = 200,
):
    """
    Plots multiple matrices in a single figure using subplots.
    
    Parameters:
    - mtx_list: List of 2D numpy arrays to be plotted.
    - titles: Optional list of titles for each subplot.
    - save_path: Optional path to save the figure.
    - show_plot: Whether to show the plot.
    - dpi: Resolution of the saved figure.
    """
    num_mtx = len(mtx_list)
    cols = min(num_mtx, 5)  # Limit to 3 columns for better readability
    rows = (num_mtx + cols - 1) // cols  # Compute rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)  # Flatten axes for easy indexing
    
    for i, (mtx, ax) in enumerate(zip(mtx_list, axes)):
        im = ax.matshow(mtx)
        ax.set_title(titles[i] if titles else f"Matrix {i+1}")
        plt.colorbar(im, ax=ax)
    
    for j in range(i + 1, len(axes)):  # Hide unused subplots
        fig.delaxes(axes[j])

    if overall_title:
        fig.suptitle(overall_title, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    if show_plot:
        plt.show()
    else:
        plt.close()
