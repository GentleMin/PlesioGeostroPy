# -*- coding: utf-8 -*-

"""Miscellaneous plotting scripts
Jingtao Min @ ETH Zurich 2023
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def accessible_region_time_space():
    """Plot the measurements in the time-space diagram
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 6))
    # Set up time axis
    time_range = [-4.5e+9, 1e+3]
    ax.set_xlim(time_range)
    ax.set_xscale('symlog', linthresh=100)
    ax.set_xlabel("Time from present [yr]", fontsize=16)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    # Set up depth axis
    depth_range = [-1e+3, 6.371e+3]
    ax.set_ylim(depth_range)
    ax.invert_yaxis()
    # ax.set_yscale('symlog', linthresh=2.)
    ax.set_ylabel("Depth [km]", fontsize=16)
    # Plot grids
    ax.grid(which="both", linestyle='dotted', linewidth=1)
    # Plot zero axes
    ax.hlines([0,], *time_range, colors='k')
    ax.annotate("Earth surface", (time_range[0], 0), 
                xytext=(.1,-1.1), textcoords="offset fontsize", fontsize=12)
    ax.vlines([0,], *depth_range, colors='k')
    ax.annotate("Present", (0, depth_range[0]), 
                xytext=(.1,-1.1), textcoords="offset fontsize", fontsize=12)
    # Plot observational region for ground observatories
    ground_visual_depth = 40
    # patch_history = patches.Rectangle((-500, -ground_visual_depth), 300, 2*ground_visual_depth, 
    #     edgecolor=(0.8, 0.4, 0.6), facecolor=(0.8, 0.4, 0.6, 0.7), zorder=2, label="historical data")
    patch_ground = patches.Rectangle((-200, -ground_visual_depth), 200, 2*ground_visual_depth, 
        edgecolor=(0.8, 0.6, 0.4), facecolor=(0.8, 0.6, 0.4, 0.7), zorder=2, label="observatory data")
    patch_orsted = patches.Rectangle((-24, -850), 24, 220, 
        edgecolor=(0.4, 0.5, 0.8), facecolor=(0.4, 0.5, 0.8, 0.7), zorder=2, label="satellite data")
    patch_champ = patches.Rectangle((-23, -450), 10, 150, 
        edgecolor=(0.4, 0.5, 0.8), facecolor=(0.4, 0.5, 0.8, 0.7), zorder=2)
    patch_swarm = patches.Rectangle((-10, -530), 10, 230, 
        edgecolor=(0.4, 0.5, 0.8), facecolor=(0.4, 0.5, 0.8, 0.7), zorder=2)
    # ax.add_patch(patch_history)
    ax.add_patch(patch_ground)
    ax.add_patch(patch_orsted)
    ax.add_patch(patch_champ)
    ax.add_patch(patch_swarm)
    # ax.scatter(-10**(2.8+(9-2.8)*np.random.rand(20)), np.zeros(20), 50, color=(0.6, 0.4, 0.8, 0.7), 
    #     zorder=2, label="paleomagnetic data")
    # Plot desired region for data assimilation
    # Full version
    # patch_foc = patches.Rectangle((time_range[0], 2.9e+3), time_range[1] - time_range[0], 2.2e+3, 
    #     zorder=2, edgecolor=(0.4, 0.4, 0.4), facecolor=(0.4, 0.4, 0.4, 0.5))
    # short timescale version
    patch_foc = patches.Rectangle((-150, 2.9e+3), 300, 2.2e+3, 
        zorder=2, edgecolor=(0.4, 0.4, 0.4), facecolor=(0.4, 0.4, 0.4, 0.5))
    ax.add_patch(patch_foc)
    plt.legend()
    
    return fig, ax


def plot_ball_disc():
    """Plot a ball and a disk
    """
    fig = plt.figure(figsize=(12,6))
    # plot sphere
    u = np.linspace(0, 2*np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(x, y, z)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1, 1])
    # plot disc
    s = np.linspace(0, 1, 10)
    x = np.outer(s, np.cos(u))
    y = np.outer(s, np.sin(u))
    z = np.zeros_like(x)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot_surface(x, y, z)
    ax.set_box_aspect([1.0, 1.0, 1.0])
    ax.set_xlim3d([-1, 1])
    ax.set_ylim3d([-1, 1])
    ax.set_zlim3d([-1, 1])
    return fig, ax


def polar_mesh(shape="Cartesian", num=100):
    """Generate mesh in polar coordinates
    """
    x = np.linspace(-1, 1, num=num)
    y = np.linspace(-1, 1, num=num)
    X_mesh, Y_mesh = np.meshgrid(x, y)
    S_mesh = np.sqrt(X_mesh**2 + Y_mesh**2)
    P_mesh = np.arctan2(Y_mesh, X_mesh)
    return X_mesh, Y_mesh, S_mesh, P_mesh


def polar_singularity_scalar(sfunc=lambda s, p: np.cos(p)):
    """Visualize scalar
    """
    X_mesh, Y_mesh, S_mesh, P_mesh = polar_mesh()
    A_val = sfunc(S_mesh, P_mesh)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.pcolormesh(X_mesh, Y_mesh, np.real(A_val), shading="gouraud")
    ax.axis("equal")
    plt.colorbar(im, ax=ax)
    return fig, ax


def polar_singularity_vector(
    vfunc_s=lambda s, p: s*np.cos(p), 
    vfunc_p=lambda s, p: s*np.sin(p)):
    """Visualize vector
    """
    # Evaluate
    X_mesh, Y_mesh, S_mesh, P_mesh = polar_mesh()
    A_s = vfunc_s(S_mesh, P_mesh)
    A_p = vfunc_p(S_mesh, P_mesh)
    # Rotate
    A_x = np.cos(P_mesh)*A_s - np.sin(P_mesh)*A_p
    A_y = np.sin(P_mesh)*A_s + np.cos(P_mesh)*A_p
    vmin = min([np.real(A_x).min(), np.real(A_y).min()])
    vmax = max([np.real(A_x).max(), np.real(A_y).max()])
    # Plot
    plot_kw = {"shading": "gouraud", "vmin": vmin, "vmax": vmax}
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    ax = axes[0]
    ax.pcolormesh(X_mesh, Y_mesh, np.real(A_x), **plot_kw)
    ax.axis("equal")
    ax.set_title(r"$A_x$")
    ax = axes[1]
    im = ax.pcolormesh(X_mesh, Y_mesh, np.real(A_y), **plot_kw)
    ax.set_title(r"$A_y$")
    ax.axis("equal")
    # Add colorbar
    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig, ax


def polar_singularity_rank2tensor(
    tfunc_ss=lambda s, p: s*np.cos(p), 
    tfunc_pp=lambda s, p: s*np.sin(p), 
    tfunc_sp=lambda s, p: s*np.cos(p),
    tfunc_ps=lambda s, p: s*np.sin(p)):
    """Visualize vector
    """
    # Evaluate
    X_mesh, Y_mesh, S_mesh, P_mesh = polar_mesh()
    A_ss = tfunc_ss(S_mesh, P_mesh)
    A_pp = tfunc_pp(S_mesh, P_mesh)
    A_sp = tfunc_sp(S_mesh, P_mesh)
    A_ps = tfunc_ps(S_mesh, P_mesh)
    # Rotate
    sin_P = np.sin(P_mesh)
    cos_P = np.cos(P_mesh)
    A_xx = cos_P**2*A_ss + sin_P**2*A_pp - sin_P*cos_P*(A_sp + A_ps)
    A_yy = sin_P**2*A_ss + cos_P**2*A_pp + sin_P*cos_P*(A_sp + A_ps)
    A_xy = sin_P*cos_P*(A_ss - A_pp) + cos_P**2*A_sp - sin_P**2*A_ps
    A_yx = sin_P*cos_P*(A_ss - A_pp) + cos_P**2*A_ps - sin_P**2*A_sp
    vmin = min([np.real(A_xx).min(), np.real(A_yy).min(), 
                np.real(A_xy).min(), np.real(A_yx).min()])
    vmax = max([np.real(A_xx).max(), np.real(A_yy).max(), 
                np.real(A_xy).max(), np.real(A_yx).max()])
    # Plot
    plot_kw = {"shading": "gouraud", "vmin": vmin, "vmax": vmax}
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 10))
    ax = axes[0,0]
    ax.pcolormesh(X_mesh, Y_mesh, np.real(A_xx), **plot_kw)
    ax.axis("equal")
    ax.set_title(r"$A_{xx}$")
    ax = axes[0,1]
    ax.pcolormesh(X_mesh, Y_mesh, np.real(A_xy), **plot_kw)
    ax.set_title(r"$A_{xy}$")
    ax.axis("equal")
    ax = axes[1,0]
    ax.pcolormesh(X_mesh, Y_mesh, np.real(A_yx), **plot_kw)
    ax.axis("equal")
    ax.set_title(r"$A_{yx}$")
    ax = axes[1,1]
    im = ax.pcolormesh(X_mesh, Y_mesh, np.real(A_yy), **plot_kw)
    ax.set_title(r"$A_{yy}$")
    ax.axis("equal")
    # Add colorbar
    fig.subplots_adjust(right=0.83)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    return fig, ax
