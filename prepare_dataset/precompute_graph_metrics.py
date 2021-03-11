# General
import snap
import numpy as np
import scipy.sparse as sp
import json
import os
import multiprocessing

# Our methods
from . import config_prepare_dataset as config


"""
Use this script to precompute information about the underlying base graph.
"""


def get_shortest_path(node_id):
    NIdToDistH = snap.TIntH()
    _ = snap.GetShortPath(snap_graph, int(node_id), NIdToDistH)
    paths = np.zeros((max(node_ids) + 1))  # previously was n_nodes
    for dest_node in NIdToDistH:
        paths[dest_node] = NIdToDistH[dest_node]
    return paths


def calculate_stats():

    # create similarities folder
    if not os.path.exists(config.DATASET_DIR / "similarities"):
        os.makedirs(config.DATASET_DIR / "similarities")

    if config.CALCULATE_EGO_GRAPHS:
        print(f"Calculating ego graphs for {config.DATASET_DIR }...")
        if not (config.DATASET_DIR / "ego_graphs.txt").exists() or config.OVERRIDE:
            ego_graph_dict = {}
            for node in snap_graph.Nodes():
                node_id = int(node.GetId())
                nodes_vec = snap.TIntV()
                snap.GetNodesAtHop(snap_graph, node_id, 1, nodes_vec, False)
                ego_graph_dict[node_id] = list(nodes_vec)

            with open(str(config.DATASET_DIR / "ego_graphs.txt"), "w") as f:
                json.dump(ego_graph_dict, f)

    if config.CALCULATE_DEGREE_SEQUENCE:
        print(f"Calculating degree sequences for {config.DATASET_DIR}...")
        if not (config.DATASET_DIR / "degree_sequence.txt").exists() or config.OVERRIDE:
            # n_nodes = len(list(snap_graph.Nodes()))
            degrees = {}
            InDegV = snap.TIntPrV()
            snap.GetNodeInDegV(snap_graph, InDegV)
            OutDegV = snap.TIntPrV()
            snap.GetNodeOutDegV(snap_graph, OutDegV)
            for item1, item2 in zip(InDegV, OutDegV):
                degrees[item1.GetVal1()] = item1.GetVal2()
            with open(str(config.DATASET_DIR / "degree_sequence.txt"), "w") as f:
                json.dump(degrees, f)

    if config.CALCULATE_SHORTEST_PATHS:
        print(f"Calculating shortest paths for {config.DATASET_DIR}...")
        if (
            not (config.DATASET_DIR / "shortest_path_matrix.npz").exists()
            or config.OVERRIDE
        ):

            with multiprocessing.Pool(processes=config.N_PROCESSSES) as pool:
                chunk = 1000
                i = 0
                while i * chunk < len(node_ids):
                    print(f"Chunk #{i+1}/{int(len(node_ids)/chunk)+1}")
                    slc = slice(i * chunk, min((i + 1) * chunk, len(node_ids)))
                    new_paths = pool.map(get_shortest_path, node_ids[slc])
                    if "shortest_paths" not in locals():
                        shortest_paths = sp.csr_matrix(new_paths, dtype=np.int)
                    else:
                        shortest_paths = sp.vstack(
                            [shortest_paths, sp.csr_matrix(new_paths, dtype=np.int)]
                        )
                    i += 1
            print("Shortest paths calculation with multiprocessing finished...")
            print("Saving shortest paths...")
            sp.save_npz(
                str(config.DATASET_DIR / "shortest_path_matrix.npz"), shortest_paths
            )


if __name__ == "__main__":
    # get SNAP graph for the specified dataset
    # PUNGraph is python instance of snap's TUNGraph (undirected graph)
    snap_graph = snap.LoadEdgeList(
        snap.PUNGraph, str(config.DATASET_DIR / "edge_list.txt"), 0, 1
    )
    node_ids = np.sort([node.GetId() for node in snap_graph.Nodes()])

    # calculate graph metrics
    calculate_stats()
