from typing import Optional

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
