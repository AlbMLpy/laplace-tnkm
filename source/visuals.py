import os
import imageio
from typing import Optional

import numpy as np

import matplotlib as mpl
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

def plot_y_list( 
    y_list,
    x = None,
    leg_list=None,
    xlabel = 'Step',
    ylabel = 'Loss',
    title = '',
    x_scale='linear',
    y_scale='linear', # 'log'
    save_path: Optional[str] = None,
    show_plot: bool = True,
    dpi: int = 200,
    text: Optional[str] = None,  
    figsize: tuple[int] = (8, 4),
):
    if x is None:
        x = np.arange(len(y_list[0]))
    with mpl.rc_context(PARAMS):
        plt.figure(figsize=figsize)
        for i, y in enumerate(y_list): 
            plt.plot(x, y, marker='*', linestyle='-', label='' if leg_list is None else leg_list[i]) 
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale(x_scale)
        plt.yscale(y_scale)
        plt.title(title)
        if leg_list:
            plt.legend()
        plt.tight_layout()
        if text:
            plt.text(
                1.05, 0.5, text, transform=plt.gca().transAxes, 
                fontsize=9, color='black', va='center'
            )
        if save_path: 
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close()

def plot_prediction(
    y_gt, 
    y_mean, 
    y_std, 
    xlabel='',
    ylabel='',
    xscale='linear',
    yscale='linear',
    title='',
    save_path: Optional[str] = None,
    show_plot: bool = True,
    dpi: int = 200,
    ylim: Optional[tuple] = None
):
    y_lower = y_mean - y_std
    y_upper = y_mean + y_std
    with mpl.rc_context(PARAMS):
        plt.figure(figsize=(8, 4))
        plt.plot(y_gt, '*r', label='GT')
        plt.plot(y_mean, '+b', label='Mean')
        plt.fill_between(
            np.arange(len(y_mean)), 
            y_lower, 
            y_upper, 
            color='blue', 
            alpha=0.1, 
            label='±1 Std'
        )
        if ylim: plt.ylim(*ylim) 
        
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xscale(xscale)
        plt.yscale(yscale)
        plt.title(title)
        plt.legend(loc='upper right')
        plt.tight_layout()
        if save_path: plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show_plot: 
            plt.show()
        else: 
            plt.close()

def plot_dual_axis( 
    y1, 
    y2,
    x=None,
    leg_y1='',
    leg_y2='',
    xlabel='',
    y1_label='',
    y2_label='',
    x_scale='linear', # 'log'
    y1_scale='linear', # 'log'
    y2_scale='linear', # 'log'
    title='',
    save_path: Optional[str] = None,
    show_plot: bool = True,
    dpi: int = 200,
):
    if x is None:
        x = np.arange(len(y1))
    with mpl.rc_context(PARAMS):
        fig, ax1 = plt.subplots(figsize=(8, 4))
        # First Y-axis (Loss)
        ax1.plot(x, y1, marker='*', linestyle='-', color='b', label=leg_y1)
        ax1.set_xscale(x_scale)
        ax1.set_xlabel(xlabel)
        ax1.set_yscale(y1_scale)
        ax1.set_ylabel(y1_label, color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        # Second Y-axis (Gradient Norm)
        ax2 = ax1.twinx()
        ax2.plot(x, y2, marker='o', linestyle='--', color='r', label=leg_y2)
        ax2.set_yscale(y2_scale)
        ax2.set_ylabel(y2_label, color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        # Add legends
        ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 0.70))
        ax1.legend(loc="upper right", bbox_to_anchor=(1.0, 0.95))
        #plt.xticks(x)
        plt.title(title)
        plt.grid(False)
        plt.tight_layout()
        if save_path: 
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close()

def generate_prediction_gif(
    x_vals,
    y_gt,
    means_list,
    stds_list,
    save_gif_path,
    temp_dir='temp_gif_frames',
    xlabel='',
    ylabel='',
    title_prefix='Frame',
    duration=0.5,
    ylim=None,
):
    os.makedirs(temp_dir, exist_ok=True)
    frame_paths = []

    for i, (th, y_mean, y_std) in enumerate(zip(x_vals, means_list, stds_list)):
        title = title_prefix + r'$\hat{t}=$' + f'{th:.1e}'
        frame_path = os.path.join(temp_dir, f'frame_{i:03d}.png')
        plot_prediction(
            y_gt, y_mean, y_std,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            save_path=frame_path,
            show_plot=False,
            ylim=ylim
        )
        frame_paths.append(frame_path)

    # Create GIF
    images = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(save_gif_path, images, duration=duration, loop=0)

    # Optionally: clean up frames
    for path in frame_paths:
        os.remove(path)
    os.rmdir(temp_dir)
