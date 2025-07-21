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
    'axes.labelsize': 16,
    'axes.grid': True,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'lines.linestyle': '-',
    'legend.fontsize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'axes.titlesize': 18,
    'figure.max_open_warning': 50,
}

COLORS = dict(
    red='#e60000',
    green='#268c26', 
    blue='#005ce6', 
    orange='#ff7f0e', 
    purple='#9467bd'
)

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
        x_vals = [np.arange(len(y)) for y in y_list]
    elif isinstance(x[0], (int, float)):  # one shared x for all
        x_vals = [x] * len(y_list)
    else:
        x_vals = x
    with mpl.rc_context(PARAMS):
        plt.figure(figsize=figsize)
        for i, y in enumerate(y_list): 
            plt.plot(
                x_vals[i], y, marker='*', linestyle='-', 
                label='' if leg_list is None else leg_list[i]
            ) 
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

### Ablation Study ### 
def plot_xy_list(ax, x_list, y_list, legend_list, color_list, markers) -> list:
    line_list = []
    for x, y, legend, color, m in zip(x_list, y_list, legend_list, color_list, markers):
        line, = ax.plot(x, y, marker=m, label=legend, color=color)
        line_list.append(line)
    return line_list

def plot_multi_xy_list(
    multi_x_lists: list[list], 
    multi_y_lists: list[list],                               
    multi_legend_lists: list[list],           
    titles: list[str],   
    color_list: list[str],      
    marker_list: list[str], 
    xlabel='',
    ylabel='',
    x_scale='linear',
    y_scale='linear',
    save_path: Optional[str] = None,
    show_plot: bool = True,
    dpi: int = 500,
    figsize: tuple[int] = (12, 6),
    ncols: int = 2,
):
    num_subplots = len(multi_y_lists)
    nrows = (num_subplots + ncols - 1) // ncols
    with mpl.rc_context(PARAMS):
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
        axes = np.ravel(axes)
        for i, ax in enumerate(axes):
            handles = plot_xy_list(ax, multi_x_lists[i], multi_y_lists[i], 
                                   multi_legend_lists[i], color_list, marker_list)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xscale(x_scale)
            ax.set_yscale(y_scale)
            ax.set_title(titles[i])
        for j in range(num_subplots, len(axes)): fig.delaxes(axes[j]) # Hide unused axes
        # Add shared legend at bottom
        legend = fig.legend(handles=handles, labels=multi_legend_lists[0],
                            loc='lower center', bbox_to_anchor=(0.5, -0.1),
                            ncol=len(handles), frameon=True)
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.0)
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for legend
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if show_plot:
            plt.show()
        else:
            plt.close()

### Illustrative Example ###
def plot_x3_results_multiple(
    results,
    data, 
    figsize: Optional[tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show_plot: bool = True,
    y_limits: Optional[tuple] = None,
    dpi: int = 500,
):
    pmean_color, pmean_linewidth, pmean_label = COLORS['red'], 2.0, r'Predictive Mean $\mu$'
    pstd_color, pstd_alpha, pstd_label = pmean_color, 0.1, r'$\mu \pm 3 \sigma$'
    mean_color, mean_linewidth, mean_label = COLORS['blue'], 2.0, r'True $\mathbb{E}[y]$'
    train_marker, train_sz, train_color, train_label = '*', 40, mean_color, 'Train Data'
    test_marker, test_sz, test_color, test_label = '+', 40, COLORS['green'], 'Test Data'
    x_label, y_label, hm_pstd = 'x', 'y', 3

    with mpl.rc_context(PARAMS):
        num_models = len(results)
        rows, cols = 1, num_models 
        if figsize is None: figsize = (cols * 5, rows * 4)
        fig, axes = plt.subplots(rows, cols, figsize=figsize, sharex=False, sharey=False)
        axes = np.ravel(axes)
        x_train, x_test, y_train, y_test, y_test_true = data
        legend_handles, legend_labels = [], []
        for i, (model_name, model_results) in enumerate(results.items()):
            ax, pmean, pstd = axes[i], model_results['pred_mean'],  model_results['pred_std']
            pmean_line, = ax.plot(
                x_test, pmean, color=pmean_color, linewidth=pmean_linewidth, label=pmean_label,
            )
            pstd_area = ax.fill_between(
                x_test[:, 0], (pmean - hm_pstd*pstd), (pmean + hm_pstd*pstd), 
                color=pstd_color, alpha=pstd_alpha, label=pstd_label,
            )
            mean_line, = ax.plot(
                x_test, y_test_true, color=mean_color, linewidth=mean_linewidth, label=mean_label,
            )
            train_scatter = ax.scatter(
                x_train, y_train, marker=train_marker, s=train_sz, color=train_color, label=train_label,
            )
            test_scatter = ax.scatter(
                x_test, y_test, marker=test_marker, s=test_sz, color=test_color, label=test_label,
            )
            ax.set_title(model_name)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if y_limits: 
                ax.set_ylim(y_limits)
            if i == 0:
                legend_handles.extend([pmean_line, pstd_area, mean_line, test_scatter, train_scatter])
                legend_labels.extend([pmean_label, pstd_label, mean_label, test_label, train_label])
        for j in range(num_models, len(axes)):
            fig.delaxes(axes[j])
        legend = fig.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, -0.1),
            ncol=len(legend_handles),
            frameon=True,
        )
        legend.get_frame().set_edgecolor('black')
        legend.get_frame().set_linewidth(1.0)
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # leave space at bottom for legend
        if save_path: 
            plt.savefig(save_path, bbox_inches="tight", bbox_extra_artists=[legend], dpi=dpi)
        if show_plot:
            plt.show()
        else:
            plt.close()
