from SubGNN import config as general_config

# Output directory ('density' as an example)
DATASET_DIR = general_config.PROJECT_ROOT / "bace"

# Random Seed
RANDOM_SEED = 42

# Parameters for generating synthetic subgraphs with specific properties
DESIRED_PROPERTY = "density"
BASE_GRAPH_TYPE = "barabasi_albert"
SUBGRAPH_TYPE = "bfs"
N_SUBGRAPHS = 250
N_CONNECTED_COMPONENTS = 1
N_SUBGRAPH_NODES = 20
FEATURES_TYPE = "one_hot"
N = 1000
P = 0.5
Q = 0
M = 5
N_BINS = 3
SUBGRAPH_GENERATOR = "complete"
MODIFY_GRAPH_FOR_PROPERTIES = True

DENSITY_EPSILON = 0.01
DENSITY_RANGE = [0.05, 0.25, 0.45]
CUT_RATIO_EPSILON = 0.001
CUT_RATIO_RANGE = [0.005, 0.0125, 0.02]
K_HOPS_RANGE = [0.12, 0.5, 1.0]
BA_P_RANGE = [0.1, 0.5, 0.9]
CC_RANGE = [1, 1, 1, 1, 5, 6, 7, 8, 9, 10]

MAX_TRIES = 100

# Parameters for training node embeddings for base graph
# CONV = "gin"
CONV = "gin"
# MINIBATCH = "GraphSaint"
MINIBATCH = "NeighborSampler"
POSSIBLE_BATCH_SIZES = [128, 256]
POSSIBLE_HIDDEN = [128]
POSSIBLE_OUTPUT = [64]
POSSIBLE_LR = [0.001, 0.0001]
POSSIBLE_WD = [5e-5]
POSSIBLE_DROPOUT = [0.4]
POSSIBLE_NB_SIZE = [-1]
POSSIBLE_NUM_HOPS = [1]
POSSIBLE_WALK_LENGTH = [32]
POSSIBLE_NUM_STEPS = [32]
EPOCHS = 100

# Flags for precomputing similarity metrics
CALCULATE_SHORTEST_PATHS = (
    True  # Calculate pairwise shortest paths between all nodes in the graph
)
CALCULATE_DEGREE_SEQUENCE = (
    True  # Create a dictionary containing degrees of the nodes in the graph
)
CALCULATE_EGO_GRAPHS = (
    True  # Calculate the 1-hop ego graph associated with each node in the graph
)
OVERRIDE = True  # Overwrite a similarity file even if it exists
N_PROCESSSES = 10  # Number of cores to use when precomputing similarity metrics
