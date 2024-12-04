import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_raw_membrane(coords, title="Raw Membrane Structure"):
    """
    Plots the membrane's raw lipid coordinates in 3D.

    Args:
        coords (np.array): Array of lipid coordinates (x, y, z).
        title (str): Title of the plot.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot of all lipid coordinates
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=10, color='blue', alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def plot_membrane(coords, leaflets):
    """
    Plots the membrane structure in 3D with different colors for each leaflet.

    Args:
        coords (np.array): Lipid coordinates (x, y, z).
        leaflets (list): Lipid IDs grouped by leaflet.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = ['blue', 'red']  # Colors for leaflets
    
    for idx, leaflet in enumerate(leaflets):
        leaflet_coords = coords[list(leaflet)]
        ax.scatter(leaflet_coords[:, 0], leaflet_coords[:, 1], leaflet_coords[:, 2], 
                   label=f"Leaflet {idx+1}", s=10, color=colors[idx % len(colors)])
    
    ax.set_title("Membrane Structure")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.legend()
    plt.show()
    
def plot_pairwise_distance_distribution(coords, box_dimensions, num_bins=50):
    """
    Plots the distribution of pairwise distances between lipids, considering periodic boundary conditions.

    Args:
        coords (np.array): Lipid coordinates (x, y, z).
        box_dimensions (list): Simulation box dimensions.
        num_bins (int): Number of bins for the histogram.
    """
    from MDAnalysis.lib.distances import apply_PBC

    # Compute pairwise distances with periodic boundary conditions
    n = len(coords)
    pairwise_distances = []
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = coords[i] - coords[j]
            diff_in_box = apply_PBC(diff, box_dimensions)
            distance = np.linalg.norm(diff_in_box)
            pairwise_distances.append(distance)
    
    # Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.hist(pairwise_distances, bins=num_bins, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Distribution of Pairwise Distances")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_z_coordinate_distribution(coords, num_bins=50):
    """
    Plots the distribution of the z-coordinates of lipids.

    Args:
        coords (np.array): Lipid coordinates (x, y, z).
        num_bins (int): Number of bins for the histogram.
    """
    z_coords = coords[:, 2]
    
    plt.figure(figsize=(10, 6))
    plt.hist(z_coords, bins=num_bins, color='green', alpha=0.7, edgecolor='black')
    plt.title("Distribution of z-Coordinates")
    plt.xlabel("z-Coordinate")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
