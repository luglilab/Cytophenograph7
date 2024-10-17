import os
import numpy as np
import pandas as pd
import networkx as nx
import umap
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from sklearn.manifold import LocallyLinearEmbedding, TSNE, SpectralEmbedding
from sklearn.cluster import AffinityPropagation, SpectralClustering, KMeans
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances
from statsmodels.nonparametric.smoothers_lowess import lowess
import elpigraph
from copy import deepcopy
from pandas.api.types import is_string_dtype,is_numeric_dtype
import networkx as nx
import seaborn as sns
import pylab as plt
import matplotlib as mpl
import shapely.geometry as geom
from copy import deepcopy
import itertools
from scipy.spatial import distance,cKDTree,KDTree
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline,UnivariateSpline
from scipy.linalg import eigh, svd, qr, solve
from scipy.sparse import coo_matrix,diags
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import math
from decimal import *
from matplotlib.patches import Polygon
from LogClass import LoggerSetup
import warnings
warnings.filterwarnings("ignore")

class StreamTrajectory:
    def __init__(self, adata, output_folder, dimensionality_reduction):
        """
        Initialize the StreamTrajectory class.

        Parameters:
        adata: AnnData object containing the single-cell data and other related information.
        output_folder: Path to the folder where results will be saved.
        dimensionality_reduction: Method for dimensionality reduction.
        """
        self.adata = adata
        self.output_folder = output_folder
        self.dimensionality_reduction = dimensionality_reduction

    def set_output_folder(self):
        """
        Set the output folder for saving results. If the folder does not exist, create it.
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.adata.uns['output_folder'] = self.output_folder
        print(f'Saving results in: {self.output_folder}')

    def dimension_reduction(self, n_neighbors=50, nb_pct=None, n_components=3,
                            n_jobs=1, feature='var_genes', method='se', eigen_solver=None):
        """
        Perform dimensionality reduction on the data using specified methods.

        Parameters:
        n_neighbors: Number of neighbors to use for certain methods.
        nb_pct: Optional percentage of neighbors, overrides n_neighbors.
        n_components: Number of components for dimensionality reduction.
        n_jobs: Number of CPUs to use.
        feature: Which data feature to use ('var_genes', 'top_pcs', 'all').
        method: Dimensionality reduction method ('mlle', 'se', 'umap', 'pca').
        eigen_solver: Optional solver for spectral embedding methods.
        """
        valid_features = ['var_genes', 'top_pcs', 'all']
        valid_methods = ['mlle', 'se', 'umap', 'pca']

        if feature not in valid_features:
            raise ValueError(f"Unrecognized feature '{feature}'. Choose from {valid_features}.")
        if method not in valid_methods:
            raise ValueError(f"Unrecognized method '{method}'. Choose from {valid_methods}.")

        input_data = {
            'var_genes': self.adata.obsm['var_genes'],
            'top_pcs': self.adata.obsm['top_pcs'],
            'all': self.adata.X
        }[feature]

        print(f'Feature {feature} is being used...')
        print(f'{n_jobs} CPUs are being used...')

        if nb_pct is not None:
            n_neighbors = int(np.around(input_data.shape[0] * nb_pct))

        reducer = self._initialize_reducer(input_data, method, n_neighbors, n_components, n_jobs, eigen_solver)

        # Perform dimensionality reduction and store result
        self.adata.obsm['X_dr'] = reducer.fit_transform(input_data)

        # Save dimensionality reduction parameters
        self.adata.uns.setdefault('params', {})
        self.adata.uns['params']['dimension_reduction'] = {
            'feature': feature,
            'method': method,
            'n_components': n_components,
            'n_neighbors': n_neighbors
        }

    def _initialize_reducer(self, input_data, method, n_neighbors, n_components, n_jobs, eigen_solver):
        """
        Initialize the dimensionality reduction object based on the selected method.
        """
        if method == 'mlle':
            eigen_solver_type = 'dense' if input_data.shape[0] <= 2000 else 'arpack'
            return LocallyLinearEmbedding(
                n_neighbors=n_neighbors, n_components=n_components, n_jobs=n_jobs,
                method='modified', eigen_solver=eigen_solver_type, random_state=10)

        elif method == 'se':
            return SpectralEmbedding(
                n_neighbors=n_neighbors, n_components=n_components, n_jobs=n_jobs,
                eigen_solver=eigen_solver, random_state=10)

        elif method == 'umap':
            return umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=42)

        elif method == 'pca':
            return sklearnPCA(n_components=n_components, svd_solver='arpack', random_state=42)

    def add_branch_info(self, epg, dict_branches):
        """
        Add position, length, and color information to each branch in the principal graph.

        Parameters:
        epg: Elastic principal graph as a NetworkX object.
        dict_branches: Dictionary where keys are branch IDs and values are dictionaries with branch nodes.
        """
        dict_nodes_pos = nx.get_node_attributes(epg, 'pos')
        sns_palette = sns.color_palette("hls", len(dict_branches)).as_hex()

        for i, (br_key, br_value) in enumerate(dict_branches.items()):
            nodes = br_value['nodes']
            dict_branches[br_key]['id'] = (nodes[0], nodes[-1])  # Direction of nodes

            print(f"Processing branch: {br_key}")

            try:
                br_nodes_pos = np.array([dict_nodes_pos[n] for n in nodes])
                branch_length = np.sqrt(((br_nodes_pos[:-1, :] - br_nodes_pos[1:, :]) ** 2).sum(axis=1)).sum()
                dict_branches[br_key]['len'] = branch_length
            except KeyError as e:
                print(f"KeyError: Node {e} not found in dict_nodes_pos")
                dict_branches[br_key]['len'] = 0

            dict_branches[br_key]['color'] = sns_palette[i]

        return dict_branches

    def extract_branches(self, epg):
        """
        Extract branches from the elastic principal graph (EPG).

        Parameters:
        epg: Elastic principal graph (NetworkX object).

        Returns:
        dict_branches: Dictionary containing information about branches and their nodes.
        """
        epg_copy = epg.copy()
        degrees_of_nodes = epg.degree()
        dict_branches = {}
        clusters_to_merge = []

        while epg_copy.order() > 1:
            leaves = [n for n, d in epg_copy.degree() if d == 1]
            while leaves:
                leave = leaves.pop()
                nodes_to_merge = [leave]
                nodes_to_visit = list(epg_copy.nodes())
                self.dfs_from_leaf(epg_copy, leave, degrees_of_nodes, nodes_to_visit, nodes_to_merge)
                dict_branches[(nodes_to_merge[0], nodes_to_merge[-1])] = {'nodes': nodes_to_merge}

                epg_copy.remove_nodes_from(nodes_to_merge[:-1] if epg_copy.degree[nodes_to_merge[-1]] != 1 else nodes_to_merge)

        dict_branches = self.add_branch_info(epg, dict_branches)
        return dict_branches

    def dfs_from_leaf(self, epg_copy, node, degrees_of_nodes, nodes_to_visit, nodes_to_merge):
        """
        Perform a depth-first search (DFS) from a leaf node to identify the branch.
        """
        nodes_to_visit.remove(node)
        for neighbor in epg_copy.neighbors(node):
            if neighbor in nodes_to_visit:
                if degrees_of_nodes[neighbor] == 2:
                    nodes_to_merge.append(neighbor)
                    self.dfs_from_leaf(epg_copy, neighbor, degrees_of_nodes, nodes_to_visit, nodes_to_merge)
                else:
                    nodes_to_merge.append(neighbor)
                    return

    def construct_flat_tree(self, dict_branches):
        """
        Construct a flat tree from the extracted branches.

        Parameters:
        dict_branches: Dictionary containing information about branches and their nodes.
        """
        flat_tree = nx.Graph()
        flat_tree.add_edges_from(dict_branches.keys())

        root = list(flat_tree.nodes())[0]
        bfs_edges = list(nx.bfs_edges(flat_tree, root))
        nodes_ordered = [root] + [v for _, v in bfs_edges]
        dict_nodes_label = {node: f'S{i}' for i, node in enumerate(nodes_ordered)}

        nx.set_node_attributes(flat_tree, dict_nodes_label, 'label')

        # Set branch attributes
        for br_key, br_data in dict_branches.items():
            nx.set_edge_attributes(flat_tree, {br_key: br_data['nodes']}, 'nodes')
            nx.set_edge_attributes(flat_tree, {br_key: br_data['id']}, 'id')
            nx.set_edge_attributes(flat_tree, {br_key: br_data['color']}, 'color')
            nx.set_edge_attributes(flat_tree, {br_key: br_data['len']}, 'len')

        return flat_tree

    def project_cells_to_epg(self):
        """
        Project cells onto the elastic principal graph.
        """
        input_data = self.adata.obsm['X_dr']
        epg = self.adata.uns['epg']
        dict_nodes_pos = nx.get_node_attributes(epg, 'pos')
        nodes_pos = np.array([dict_nodes_pos[key] for key in sorted(dict_nodes_pos.keys())])
        nodes = np.array(list(sorted(dict_nodes_pos.keys())))

        indices = pairwise_distances_argmin_min(input_data, nodes_pos, axis=1, metric='euclidean')[0]
        x_node = nodes[indices]
        self.adata.obs['node'] = x_node

        # Further projection to flat tree branches
        self._project_to_flat_tree(input_data, x_node)

    def _project_to_flat_tree(self, input_data, x_node):
        """
        Helper function to project cells onto the flat tree structure.
        """
        flat_tree = self.adata.uns['flat_tree']
        dict_nodes_pos = nx.get_node_attributes(flat_tree, 'pos')

        branch_data = {'branch_id': [], 'branch_id_alias': [], 'branch_lam': [], 'branch_dist': []}

        for xp, node in zip(input_data, x_node):
            closest_branch = self._find_closest_branch(xp, node, flat_tree, dict_nodes_pos)
            branch_data['branch_id'].append(closest_branch['id'])
            branch_data['branch_id_alias'].append(closest_branch['alias'])
            branch_data['branch_lam'].append(closest_branch['lam'])
            branch_data['branch_dist'].append(closest_branch['dist'])

        for key, values in branch_data.items():
            self.adata.obs[key] = values

    def _find_closest_branch(self, xp, node, flat_tree, dict_nodes_pos):
        """
        Find the closest branch for a given point xp and node in the flat tree.
        """
        dict_branches = nx.get_edge_attributes(flat_tree, 'nodes')
        dict_branch_positions = {br_id: np.array([dict_nodes_pos[n] for n in nodes]) for br_id, nodes in dict_branches.items()}

        closest_branch, min_dist = None, float('inf')
        for br_id, br_pos in dict_branch_positions.items():
            projected = self.project_point_to_line_segment_matrix(br_pos, xp)
            if projected[2] < min_dist:
                min_dist = projected[2]
                closest_branch = {'id': br_id, 'alias': (flat_tree.nodes[br_id[0]]['label'], flat_tree.nodes[br_id[1]]['label']),
                                  'lam': projected[3], 'dist': min_dist}

        return closest_branch

    def calculate_pseudotime(self):
        """
        Calculate pseudotime for cells based on their projection on the flat tree.
        """
        flat_tree = self.adata.uns['flat_tree']
        dict_edge_len = nx.get_edge_attributes(flat_tree, 'len')

        # Clear previous pseudotime
        self.adata.obs = self.adata.obs.loc[:, ~self.adata.obs.columns.str.contains('_pseudotime')].copy()

        for root_node in flat_tree.nodes():
            df_pseudotime = pd.Series(index=self.adata.obs.index)
            bfs_edges = list(nx.bfs_edges(flat_tree, source=root_node))
            bfs_predecessors = dict(nx.bfs_predecessors(flat_tree, source=root_node))

            for edge in bfs_edges:
                pre_edges = self._get_predecessor_edges(edge[0], bfs_predecessors)
                len_pre_edges = sum(flat_tree.edges[pre]['len'] for pre in pre_edges)

                edge_indices = self.adata.obs[(self.adata.obs['branch_id'] == edge) |
                                              (self.adata.obs['branch_id'] == (edge[1], edge[0]))].index

                df_pseudotime.loc[edge_indices] = len_pre_edges + \
                                                  (flat_tree.edges[edge]['len'] - self.adata.obs.loc[edge_indices, 'branch_lam'])

            self.adata.obs[flat_tree.nodes[root_node]['label'] + '_pseudotime'] = df_pseudotime

    def _get_predecessor_edges(self, node, bfs_predecessors):
        """
        Get the list of predecessor edges for a given node.
        """
        pre_edges = []
        while node in bfs_predecessors:
            pre_edge = (bfs_predecessors[node], node)
            pre_edges.append(pre_edge)
            node = bfs_predecessors[node]
        return pre_edges

    def elastic_principal_graph(self, epg_n_nodes=50, incr_n_nodes=30, epg_lambda=0.02, epg_mu=0.1,
                                epg_trimmingradius=float('inf'), epg_finalenergy='Penalized', epg_alpha=0.02,
                                epg_beta=0.0, epg_n_processes=1, use_vis=False, **kwargs):
        """
        Learn an elastic principal graph based on input data.
        """
        """Elastic principal graph learning."""
        if 'seed_epg' in self.adata.uns_keys():
            epg = self.adata.uns['seed_epg']
            dict_nodes_pos = nx.get_node_attributes(epg, 'pos')
            init_nodes_pos = np.array(list(dict_nodes_pos.values()))
            init_edges = np.array(list(epg.edges()))

            if ((init_nodes_pos.shape[0] + incr_n_nodes) >= epg_n_nodes):
                print('epg_n_nodes is too small. It is corrected to the initial number of nodes plus incr_n_nodes')
                epg_n_nodes = init_nodes_pos.shape[0] + incr_n_nodes
        else:
            if use_vis:
                if 'X_vis' in self.adata.obsm_keys():
                    self.adata.obsm['X_dr'] = self.adata.obsm['X_vis']
                else:
                    raise ValueError("Please run `plot_visualization_2D()` first")
            init_nodes_pos = None
            init_edges = None

        input_data = self.adata.obsm['X_dr']
        pg_obj = elpigraph.computeElasticPrincipalTree(
            input_data, NumNodes=epg_n_nodes, Lambda=epg_lambda, Mu=epg_mu,
            TrimmingRadius=epg_trimmingradius, alpha=epg_alpha, beta=epg_beta,
            Do_PCA=False, CenterData=False, n_cores=epg_n_processes, **kwargs)

        epg_nodes_pos = pg_obj[0]['NodePositions']
        epg_edges = pg_obj[0]['Edges'][0]

        # Construct the graph
        epg = nx.Graph()
        epg.add_nodes_from(range(epg_nodes_pos.shape[0]))
        epg.add_edges_from([tuple(edge) for edge in epg_edges])

        # Set node positions
        dict_nodes_pos = {i: x for i, x in enumerate(epg_nodes_pos)}
        nx.set_node_attributes(epg, dict_nodes_pos, 'pos')

        # Extract branches and construct the flat tree
        dict_branches = self.extract_branches(epg)
        flat_tree = self.construct_flat_tree(dict_branches)

        # Update AnnData object with the learned graphs
        self.adata.uns['epg'] = deepcopy(epg)
        self.adata.uns['flat_tree'] = deepcopy(flat_tree)

        # Project cells onto the learned graph and calculate pseudotime
        self.project_cells_to_epg()
        self.calculate_pseudotime()

        print(f'Number of branches after learning elastic principal graph: {len(dict_branches)}')



