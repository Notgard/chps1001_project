import numpy as np
import networkx as nx
import MDAnalysis as mda
import matplotlib.pyplot as plt

from utils import *

def read_gromacs_file_with_mdanalysis(filename):
    """
    Reads a GROMACS .gro file and extracts phosphate atom coordinates and lipid IDs using MDAnalysis.
    
    Args:
        filename (str): Path to the .gro file.
    
    Returns:
        tuple: (phosphate coordinates, lipid IDs, box dimensions)
    """
    u = mda.Universe(filename)
    
    # Select phosphate atoms (adjust the selection for your specific lipid structure)
    phosphate_selection = u.select_atoms("name PO4")
    print(phosphate_selection)
    phosphate_coords = phosphate_selection.positions
    lipid_ids = phosphate_selection.residues.resids
    
    # Get simulation box dimensions
    box_dimensions = u.dimensions
    
    return phosphate_coords, lipid_ids, box_dimensions

def compute_periodic_distances_with_mdanalysis(coords, box_dimensions):
    """
    Computes pairwise distances with periodic boundary conditions (PBC) using MDAnalysis.
    
    Args:
        coords (np.array): Coordinates of points.
        box_dimensions (list): Simulation box dimensions.
    
    Returns:
        np.array: Distance matrix.
    """
    from MDAnalysis.lib.distances import apply_PBC

    n = len(coords)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # Apply periodic boundary conditions to compute the minimum image distance
            diff = coords[i] - coords[j]
            diff_in_box = apply_PBC(diff, box_dimensions)
            distance = np.linalg.norm(diff_in_box)
            distances[i, j] = distances[j, i] = distance

    return distances

def find_leaflets_by_graph(coords, ids, distances, distance_threshold):
    """
    Finds the leaflets by building a graph of connections based on a distance threshold.
    
    Args:
        coords (np.array): Coordinates of points.
        ids (list): Lipid IDs.
        distances (np.array): Distance matrix.
        distance_threshold (float): Connection threshold.
    
    Returns:
        tuple: (leaflets, graph)
    """
    # Construct the graph
    G = nx.Graph()
    for i in range(len(coords)):
        G.add_node(i, lipid_id=ids[i])
    
    # Add edges based on the distance threshold
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            print(distances[i, j], distance_threshold)
            if distances[i, j] < distance_threshold:
                G.add_edge(i, j)
    
    # Find connected components
    components = list(nx.connected_components(G))
    
    print("Number of components:", len(components))
    print(components)
    
    # Assign lipids to leaflets based on components
    leaflets = [[] for _ in range(len(components))]
    for idx, component in enumerate(components):
        leaflets[idx] = [G.nodes[n]['lipid_id'] for n in component]
    
    return leaflets, G

def analyse_membrane(filename):
    """
    Complete membrane analysis pipeline.
    
    Args:
        filename (str): Path to the GROMACS .gro file.
        distance_threshold (float): Connection threshold for lipid clustering.
    
    Returns:
        list: Lipid IDs per leaflet.
    """
    coords, ids, box_dimensions = read_gromacs_file_with_mdanalysis(filename)
    print(coords, ids, box_dimensions)
    
    plot_raw_membrane(coords, title="Raw Membrane Structure Before Leaflet Computation")
    
    # Step 3: Analyze distributions
    print("Analyzing pairwise distances...")
    plot_pairwise_distance_distribution(coords, box_dimensions)
    
    print("Analyzing z-coordinate distribution...")
    plot_z_coordinate_distribution(coords)
    
    # this threshold is simply gets the median of the z. For simple membrane structure, 
    # this cut off is the middle coordinates z axis cut off between the 2 leaflets
    distance_threshold = compute_distance_threshold(coords)
    print("Current distance threshold", distance_threshold)
    distances = compute_periodic_distances_with_mdanalysis(coords, box_dimensions)
    leaflets, graph = find_leaflets_by_graph(coords, ids, distances, distance_threshold)
    
    # Visualize the graph (optional)
    nx.draw(graph, with_labels=False, node_size=10)
    plt.title("Graph of Lipid Connections")
    plt.show()
    
    # Visualize the membrane structure in 3D
    plot_membrane(coords, leaflets)
    
    return leaflets

def save_results(leaflets, output_file):
    """
    Saves the leaflet analysis results to a text file.
    
    Args:
        leaflets (list): Lipid IDs grouped by leaflet.
        output_file (str): Output file path.
    """
    with open(output_file, 'w') as f:
        for i, leaflet in enumerate(leaflets, 1):
            f.write(f"Leaflet {i}: {len(leaflet)} lipids\n")
            f.write(", ".join(map(str, leaflet)) + "\n\n")

def compute_distance_threshold(coords):
    z_median = np.median(coords[:, 2])
    return z_median

# Example usage
if __name__ == "__main__":
    input_file = "./files/bilayer_chol/bilayer_chol.gro"
    output_file = "./output/leaflets_results.txt"
    
    leaflets = analyse_membrane(input_file)
    save_results(leaflets, output_file)
    
    print(f"Analysis complete. Results saved in {output_file}")
