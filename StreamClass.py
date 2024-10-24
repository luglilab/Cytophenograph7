import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances, euclidean_distances
from statsmodels.nonparametric.smoothers_lowess import lowess
import elpigraph
import networkx as nx
import seaborn as sns
import pylab as plt
import matplotlib as mpl
import warnings
from pandas.api.types import is_string_dtype,is_numeric_dtype
from copy import deepcopy
import itertools
from scipy.spatial import distance,cKDTree
from scipy import interpolate
from scipy.signal import savgol_filter
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
import math
from matplotlib.patches import Polygon
warnings.filterwarnings("ignore")
from palette import palette28, palette102
# === StreamTrajectory Class ===

class StreamTrajectory:


# === Initialization and Setup Methods ===

# === StreamTrajectory Class ===

    # === Initialization and Setup Methods ===
    def __init__(self, adata, output_folder, tool, runtime,analysis_name):
        """
        Initialize the StreamTrajectory class.

        Parameters:
        self.adata: AnnData object containing the single-cell data and other related information.
        output_folder: Path to the folder where results will be saved.
        dimensionality_reduction: Method for dimensionality reduction.
        """
        self.adata = adata
        self.output_folder = output_folder
        self.typeclustering = tool
        self.Trajectory_folder = "/".join([self.output_folder, "Trajectory"])
        self.createdir(self.Trajectory_folder )
        self.runtime = runtime
        self.analysis_name = analysis_name
        self.figsize = (7, 7)
        self.palette28 = palette28
        self.palette102 = palette102
        self.epg_nodes_pos = None
        self.dimensionality_reduction= "X_umap"


# === Output Folder Management ===

    def createdir(self, directory):
        """
        Helper function to create a directory if it doesn't exist.
        :param directory: The path of the directory to create.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)


# === Gini Gene Selection ===

    def select_gini_genes(self, loess_frac=0.1, percentile=95, n_genes=None,
                          save_fig=False, fig_name='gini_vs_max.pdf', fig_path=None,
                          fig_size=(4, 4), pad=1.08, w_pad=None, h_pad=None):
        """Select high Gini genes for rare cell types."""

        if fig_path is None:
            fig_path = self.output_folder
        fig_size = mpl.rcParams['figure.figsize'] if fig_size is None else fig_size

        if 'gini' not in self.adata.var.columns:
            gini_values = np.array([self.select_gini_genes(self.adata[:, x].X) for x in self.adata.var_names])
            self.adata.var['gini'] = gini_values

        gini_values = self.adata.var['gini']
        max_genes = np.max(self.adata.X, axis = 0)
        loess_fitted = lowess(gini_values, max_genes, return_sorted = False, frac = loess_frac)
        residuals = gini_values - loess_fitted

        if n_genes is None:
            cutoff = np.percentile(residuals, percentile)
            id_gini_genes = np.where(residuals > cutoff)[0]
        else:
            id_gini_genes = np.argsort(residuals)[::-1][:n_genes]

        self.adata.uns['gini_genes'] = self.adata.var_names[id_gini_genes]
        print(f'{len(id_gini_genes)} Gini genes are selected.')
        fig = plt.figure(figsize = fig_size)
        plt.scatter(max_genes, gini_values, s = 5, alpha = 0.2, zorder = 1, color = '#6baed6')
        plt.scatter(max_genes[id_gini_genes], gini_values[id_gini_genes], s = 5, alpha = 0.9, zorder = 2,
                    color = '#EC4E4E')
        plt.plot(np.sort(max_genes), loess_fitted[np.argsort(max_genes)], linewidth = 3, zorder = 3, color = '#3182bd')
        plt.xlabel('Max Gene Expression')
        plt.ylabel('Gini Coefficient')
        plt.tight_layout(pad = pad, h_pad = h_pad, w_pad = w_pad)

        if save_fig:
            plt.savefig(os.path.join(fig_path, fig_name), pad_inches = 1, bbox_inches = 'tight')
            plt.close(fig)


# === Plotting Methods ===

    def plot_branches(self, n_components=None, comp1=0, comp2=1, comp3=2,
                      key_graph='epg', fig_size=None, pad=1.08, h_pad=1.0,  # Added h_pad here
                      show_text=False, save_fig=False, fig_path=None,
                      fig_name='branches.pdf'):
        """Plot branches from the elastic principal graph using matplotlib."""

        if fig_path is None:
            fig_path = self.output_folder
        fig_size = mpl.rcParams['figure.figsize'] if fig_size is None else fig_size

        if n_components is None:
            n_components = min(3, self.adata.obsm[self.dimensionality_reduction].shape[1])

        if n_components not in [2, 3]:
            raise ValueError("n_components should be 2 or 3")

        assert all(np.isin(['epg', 'flat_tree'], self.adata.uns_keys()))
        epg = self.adata.uns['epg'] if key_graph == 'epg' else self.adata.uns[key_graph.split('_')[0] + '_epg']
        flat_tree = self.adata.uns['flat_tree'] if key_graph == 'epg' else self.adata.uns[
            key_graph.split('_')[0] + '_flat_tree']

        ft_node_pos = nx.get_node_attributes(flat_tree, 'pos')
        epg_node_pos = nx.get_node_attributes(epg, 'pos')

        fig = plt.figure(figsize = fig_size)
        ax_i = fig.add_subplot(1, 1, 1, projection = '3d' if n_components == 3 else None)

        for edge_i in flat_tree.edges():
            branch_i_pos = np.array([epg_node_pos[i] for i in flat_tree.edges[edge_i]['nodes']])
            ax_i.plot(branch_i_pos[:, comp1], branch_i_pos[:, comp2], color = 'black')

        if show_text:
            for node_i in epg.nodes():
                ax_i.text(epg_node_pos[node_i][comp1], epg_node_pos[node_i][comp2], node_i,
                          color = 'black', fontsize = 0.8 * mpl.rcParams['font.size'], ha = 'left', va = 'bottom')

        ax_i.set_xlabel('Dim' + str(comp1 + 1), labelpad = -5, rotation = -15)
        ax_i.set_ylabel('Dim' + str(comp2 + 1), labelpad = 0, rotation = 45)

        if n_components == 3:
            ax_i.set_zlabel('Dim' + str(comp3 + 1), labelpad = 5, rotation = 90)
            ax_i.locator_params(axis = 'z', nbins = 4)

        ax_i.locator_params(axis = 'x', nbins = 4)
        ax_i.locator_params(axis = 'y', nbins = 4)
        ax_i.tick_params(axis = "x", pad = -4)
        ax_i.tick_params(axis = "y", pad = -1)

        plt.tight_layout(pad = pad, h_pad = h_pad)  # Correct usage of h_pad now

        if save_fig:
            plt.savefig(os.path.join(fig_path, fig_name), pad_inches = 1, bbox_inches = 'tight')
            plt.close(fig)

    def extract_branches(self, epg):
        # record the original degree(before removing nodes) for each node
        degrees_of_nodes = epg.degree()
        epg_copy = epg.copy()
        dict_branches = dict()
        clusters_to_merge = []
        while epg_copy.order() > 1:  # the number of vertices
            leaves = [n for n, d in epg_copy.degree() if d == 1]
            nodes_included = list(epg_copy.nodes())
            while leaves:
                leave = leaves.pop()
                nodes_included.remove(leave)
                nodes_to_merge = [leave]
                nodes_to_visit = list(epg_copy.nodes())
                self.dfs_from_leaf(epg_copy, leave, degrees_of_nodes, nodes_to_visit, nodes_to_merge)
                clusters_to_merge.append(nodes_to_merge)
                dict_branches[(nodes_to_merge[0], nodes_to_merge[-1])] = {}
                dict_branches[(nodes_to_merge[0], nodes_to_merge[-1])]['nodes'] = nodes_to_merge
                nodes_to_delete = nodes_to_merge[0:len(nodes_to_merge) - 1]
                if epg_copy.degree()[nodes_to_merge[-1]] == 1:  # avoid the single point
                    nodes_to_delete = nodes_to_merge
                    leaves = []
                epg_copy.remove_nodes_from(nodes_to_delete)
        dict_branches = self.add_branch_info(epg, dict_branches)
        # print('Number of branches: ' + str(len(clusters_to_merge)))
        return dict_branches

    def construct_flat_tree(self, dict_branches):
        flat_tree = nx.Graph()
        flat_tree.add_nodes_from(list(set(itertools.chain.from_iterable(dict_branches.keys()))))
        flat_tree.add_edges_from(dict_branches.keys())
        root = list(flat_tree.nodes())[0]
        edges = nx.bfs_edges(flat_tree, root)
        nodes = [root] + [v for u, v in edges]
        dict_nodes_label = dict()
        for i, node in enumerate(nodes):
            dict_nodes_label[node] = 'S' + str(i)
        nx.set_node_attributes(flat_tree, values = dict_nodes_label, name = 'label')
        dict_branches_color = dict()
        dict_branches_len = dict()
        dict_branches_nodes = dict()
        dict_branches_id = dict()  # the direction of nodes for each edge
        for x in dict_branches.keys():
            dict_branches_color[x] = dict_branches[x]['color']
            dict_branches_len[x] = dict_branches[x]['len']
            dict_branches_nodes[x] = dict_branches[x]['nodes']
            dict_branches_id[x] = dict_branches[x]['id']
        nx.set_edge_attributes(flat_tree, values = dict_branches_nodes, name = 'nodes')
        nx.set_edge_attributes(flat_tree, values = dict_branches_id, name = 'id')
        nx.set_edge_attributes(flat_tree, values = dict_branches_color, name = 'color')
        nx.set_edge_attributes(flat_tree, values = dict_branches_len, name = 'len')
        return flat_tree

# === Elastic Principal Graph Methods ===

    def seed_elastic_principal_graph(self, init_nodes_pos=None, init_edges=None, damping=0.75,
                                     pref_perc=50,n_neighbors=50, nb_pct=None,
                                     emb='X_umap'):
        """Seeding the initial elastic principal graph."""

        print('Seeding initial elastic principal graph...')
        input_data = self.adata.obsm[emb]
        if nb_pct is not None:
            n_neighbors = int(np.around(input_data.shape[0] * nb_pct))
        embedding = self.adata.obsm['X_umap']
        clusters = np.array(self.adata.obs['pheno_leiden'])
        tmp = pd.DataFrame(embedding)
        tmp['cluster'] = clusters
        init_nodes_pos = tmp.groupby('cluster').mean().values
        init_nodes_pos = init_nodes_pos[:, :2]
        self.epg_nodes_pos = init_nodes_pos
        print('The number of initial nodes is ' + str(self.epg_nodes_pos.shape[0]))
        if init_edges is None:
            # Minimum Spanning Tree
            print('Calculating minimum spanning tree...')
            D = pairwise_distances(self.epg_nodes_pos)
            G = nx.from_numpy_array(D)  # Use from_numpy_array instead
            mst = nx.minimum_spanning_tree(G)
            epg_edges = np.array(mst.edges())
        else:
            print('Setting initial edges...')
            epg_edges = init_edges
        # Store graph information and update self.adata
        epg = nx.Graph()
        epg.add_nodes_from(range(self.epg_nodes_pos.shape[0]))
        epg.add_edges_from(epg_edges)
        dict_nodes_pos = {i: x for i, x in enumerate(self.epg_nodes_pos)}
        nx.set_node_attributes(epg, values = dict_nodes_pos, name = 'pos')
        dict_branches = self.extract_branches(epg)
        flat_tree = self.construct_flat_tree(dict_branches)
        nx.set_node_attributes(flat_tree, values = {x: dict_nodes_pos[x] for x in flat_tree.nodes()}, name = 'pos')
        self.adata.uns['epg'] = deepcopy(epg)
        self.adata.uns['flat_tree'] = deepcopy(flat_tree)
        self.adata.uns['seed_epg'] = deepcopy(epg)
        self.adata.uns['seed_flat_tree'] = deepcopy(flat_tree)
        self.project_cells_to_epg()
        self.calculate_pseudotime()
        if 'params' not in self.adata.uns_keys():
            self.adata.uns['params'] = dict()
        print('Number of initial branches: ' + str(len(dict_branches)))

    def dfs_from_leaf(self, epg_copy, node, degrees_of_nodes, nodes_to_visit, nodes_to_merge):
        nodes_to_visit.remove(node)
        for n2 in epg_copy.neighbors(node):
            if n2 in nodes_to_visit:
                if degrees_of_nodes[n2] == 2:  # grow the branch
                    if n2 not in nodes_to_merge:
                        nodes_to_merge.append(n2)
                    self.dfs_from_leaf(epg_copy, n2, degrees_of_nodes, nodes_to_visit, nodes_to_merge)
                else:
                    nodes_to_merge.append(n2)
                    return

    def add_branch_info(self, epg, dict_branches):
        dict_nodes_pos = nx.get_node_attributes(epg, 'pos')
        sns_palette = sns.color_palette("hls", len(dict_branches)).as_hex()
        if dict_nodes_pos:
            for i, (br_key, br_value) in enumerate(dict_branches.items()):
                nodes = br_value['nodes']
                dict_branches[br_key]['id'] = (nodes[0], nodes[-1])  # the direction of nodes for each branch
                try:
                    br_nodes_pos = np.array([dict_nodes_pos[n] for n in nodes])
                    dict_branches[br_key]['len'] = sum(
                        np.sqrt(((br_nodes_pos[:-1, :] - br_nodes_pos[1:, :]) ** 2).sum(axis = 1)))
                    dict_branches[br_key]['color'] = sns_palette[i]
                except KeyError as e:
                    print(f"KeyError: Node {e} not found in dict_nodes_pos")
                    dict_branches[br_key]['color'] = sns_palette[i]  # Assign a default color or handle it as needed
                    dict_branches[br_key]['len'] = 0  # Assign a default length

        return dict_branches

    def project_cells_to_epg(self):
        input_data = self.adata.obsm['X_umap']
        epg = self.adata.uns['epg']
        dict_nodes_pos = nx.get_node_attributes(epg, 'pos')
        nodes_pos = np.empty((0, input_data.shape[1]))
        nodes = np.empty((0, 1), dtype = int)
        for key in dict_nodes_pos.keys():
            nodes_pos = np.vstack((nodes_pos, dict_nodes_pos[key]))
            nodes = np.append(nodes, key)
        indices = pairwise_distances_argmin_min(input_data, nodes_pos, axis = 1, metric = 'euclidean')[0]
        x_node = nodes[indices]
        self.adata.obs['node'] = x_node
        flat_tree = self.adata.uns['flat_tree']
        dict_branches_nodes = nx.get_edge_attributes(flat_tree, 'nodes')
        dict_branches_id = nx.get_edge_attributes(flat_tree, 'id')
        dict_node_state = nx.get_node_attributes(flat_tree, 'label')
        list_x_br_id = list()
        list_x_br_id_alias = list()
        list_x_lam = list()
        list_x_dist = list()
        for ix, xp in enumerate(input_data):
            list_br_id = [flat_tree.edges[br_key]['id'] for br_key, br_value in dict_branches_nodes.items() if
                          x_node[ix] in br_value]
            dict_br_matrix = dict()
            for br_id in list_br_id:
                dict_br_matrix[br_id] = np.array([dict_nodes_pos[i] for i in flat_tree.edges[br_id]['nodes']])
            dict_results = dict()
            list_dist_xp = list()
            for br_id in list_br_id:
                dict_results[br_id] = self.project_point_to_line_segment_matrix(dict_br_matrix[br_id], xp)
                list_dist_xp.append(dict_results[br_id][2])
            x_br_id = list_br_id[np.argmin(list_dist_xp)]
            x_br_id_alias = dict_node_state[x_br_id[0]], dict_node_state[x_br_id[1]]
            br_len = flat_tree.edges[x_br_id]['len']
            results = dict_results[x_br_id]
            x_dist = results[2]
            x_lam = results[3]
            if (x_lam > br_len):
                x_lam = br_len
            list_x_br_id.append(x_br_id)
            list_x_br_id_alias.append(x_br_id_alias)
            list_x_lam.append(x_lam)
            list_x_dist.append(x_dist)
        self.adata.obs['branch_id'] = list_x_br_id
        self.adata.obs['branch_id_alias'] = list_x_br_id_alias
        #     self.adata.uns['branch_id'] = list(set(self.adata.obs['branch_id'].tolist()))
        self.adata.obs['branch_lam'] = list_x_lam
        self.adata.obs['branch_dist'] = list_x_dist
        return None

    def project_point_to_line_segment_matrix(self, XP, p):
        XP = np.array(XP, dtype = float)
        p = np.array(p, dtype = float)
        AA = XP[:-1, :]
        BB = XP[1:, :]
        AB = (BB - AA)
        AB_squared = (AB * AB).sum(1)
        Ap = (p - AA)
        t = (Ap * AB).sum(1) / AB_squared
        t[AB_squared == 0] = 0
        Q = AA + AB * np.tile(t, (XP.shape[1], 1)).T
        Q[t <= 0, :] = AA[t <= 0, :]
        Q[t >= 1, :] = BB[t >= 1, :]
        kdtree = cKDTree(Q)
        d, idx_q = kdtree.query(p)
        dist_p_to_q = np.sqrt(np.inner(p - Q[idx_q, :], p - Q[idx_q, :]))
        XP_p = np.row_stack((XP[:idx_q + 1], Q[idx_q, :]))
        lam = np.sum(np.sqrt(np.square((XP_p[1:, :] - XP_p[:-1, :])).sum(1)))
        return list([Q[idx_q, :], idx_q, dist_p_to_q, lam])

    def calculate_pseudotime(self):
        flat_tree = self.adata.uns['flat_tree']
        dict_edge_len = nx.get_edge_attributes(flat_tree, 'len')
        # Clear any previous pseudotime columns in self.adata.obs
        self.adata.obs = self.adata.obs[self.adata.obs.columns.drop(list(self.adata.obs.filter(regex = '_pseudotime')))].copy()
        for root_node in flat_tree.nodes():
            df_pseudotime = pd.Series(index = self.adata.obs.index)
            list_bfs_edges = list(nx.bfs_edges(flat_tree, source = root_node))
            dict_bfs_predecessors = dict(nx.bfs_predecessors(flat_tree, source = root_node))
            for edge in list_bfs_edges:
                list_pre_edges = []
                pre_node = edge[0]
                while pre_node in dict_bfs_predecessors.keys():
                    pre_edge = (dict_bfs_predecessors[pre_node], pre_node)
                    list_pre_edges.append(pre_edge)
                    pre_node = dict_bfs_predecessors[pre_node]
                len_pre_edges = sum([flat_tree.edges[x]['len'] for x in list_pre_edges])
                indices = self.adata.obs[
                    (self.adata.obs['branch_id'] == edge) | (self.adata.obs['branch_id'] == (edge[1], edge[0]))].index
                if edge in flat_tree.edges:  # Ensure edge exists
                    df_pseudotime[indices] = len_pre_edges + (
                            flat_tree.edges[edge]['len'] - self.adata.obs.loc[indices, 'branch_lam'])
                else:
                    df_pseudotime[indices] = len_pre_edges + self.adata.obs.loc[indices, 'branch_lam']
            self.adata.obs[flat_tree.nodes[root_node]['label'] + '_pseudotime'] = df_pseudotime

        return None


# === Elastic Principal Graph Methods ===

    def elastic_principal_graph(self, epg_n_nodes=50, incr_n_nodes=30,
                                epg_alpha=0.01, epg_mu=0.05, epg_lambda=0.05,
                                epg_trimmingradius=float('inf'),  # Ensure infinity is correctly represented
                                epg_finalenergy='Penalized',
                                epg_beta=0.0, epg_n_processes=1,
                                save_fig=False, fig_name='ElPiGraph_analysis.pdf', fig_path=None, fig_size=(8, 8)):
        if fig_path is None:
            fig_path = self.output_folder

        if 'seed_epg' in self.adata.uns_keys():
            epg = self.adata.uns['seed_epg']
            dict_nodes_pos = nx.get_node_attributes(epg, 'pos')
            init_nodes_pos = np.array(list(dict_nodes_pos.values()))
            init_edges = np.array(list(epg.edges()))
            if ((init_nodes_pos.shape[0] + incr_n_nodes) >= epg_n_nodes):
                print('epg_n_nodes is too small. It is corrected to the initial number of nodes plus incr_n_nodes')
                epg_n_nodes = init_nodes_pos.shape[0] + incr_n_nodes
        else:
            print('No initial structure is seeded')
            if 'params' not in self.adata.uns_keys():
                self.adata.uns['params'] = dict()
            init_nodes_pos = None
            init_edges = None

        input_data = self.adata.obsm[self.dimensionality_reduction]
        print('Learning elastic principal graph...')
        pg_obj = elpigraph.computeElasticPrincipalTree(input_data, NumNodes = epg_n_nodes,
                                                       Lambda = epg_lambda, Mu = epg_mu,
                                                       TrimmingRadius = epg_trimmingradius,
                                                       alpha = epg_alpha, beta = epg_beta,
                                                       Do_PCA = False, CenterData = False,
                                                       n_cores = epg_n_processes)
        self.epg_nodes_pos = pg_obj[0]['NodePositions']  # Assuming pg_obj is a list
        epg_edges = pg_obj[0]['Edges']
        edge_indices = epg_edges[0]  # Extract the edge indices

        if isinstance(edge_indices, np.ndarray):
            epg_edges = [tuple(edge) for edge in edge_indices.tolist()]  # Convert to list of tuples
        else:
            raise ValueError("Unexpected format for edges. Must be a NumPy array.")

        # Store graph information and update self.adata
        epg = nx.Graph()
        epg.add_nodes_from(range(self.epg_nodes_pos.shape[0]))
        epg.add_edges_from(epg_edges)  # This should now work without errors
        dict_nodes_pos = {i: x for i, x in enumerate(self.epg_nodes_pos)}
        nx.set_node_attributes(epg, values = dict_nodes_pos, name = 'pos')

        dict_branches = self.extract_branches(epg)
        flat_tree = self.construct_flat_tree(dict_branches)
        nx.set_node_attributes(flat_tree, values = {x: dict_nodes_pos[x] for x in flat_tree.nodes()}, name = 'pos')

        self.adata.uns['epg'] = deepcopy(epg)
        self.adata.uns['ori_epg'] = deepcopy(epg)
        self.adata.uns['epg_obj'] = deepcopy(pg_obj)
        self.adata.uns['ori_epg_obj'] = deepcopy(pg_obj)
        self.adata.uns['flat_tree'] = deepcopy(flat_tree)

        self.project_cells_to_epg()
        self.calculate_pseudotime()
        print('Number of branches after learning elastic principal graph: ' + str(len(dict_branches)))

        # Ensure 'epg' key exists in 'params'
        if 'params' not in self.adata.uns_keys():
            self.adata.uns['params'] = dict()
        if 'epg' not in self.adata.uns['params']:
            self.adata.uns['params']['epg'] = dict()  # Initialize 'epg' key in 'params'

        # Update 'epg' parameters
        self.adata.uns['params']['epg'].update(
            {'epg_alpha': epg_alpha, 'epg_lambda': epg_lambda, 'epg_mu': epg_mu,
             'epg_trimmingradius': epg_trimmingradius, 'epg_beta': epg_beta})

        # Optionally save the figure
        if save_fig:
            f, ax = plt.subplots(figsize = fig_size)
            elpigraph.plot.PlotPG(input_data, pg_obj, Do_PCA = False, show_text = False, ax = ax)
            plt.savefig(os.path.join(fig_path, fig_name))
            plt.close(f)

    def add_flat_tree_node_pos(self):
        flat_tree = self.adata.uns['flat_tree']
        ft_node_pos = nx.spring_layout(flat_tree)  # Removed random_state argument
        bfs_root = list(flat_tree.nodes())[0]
        bfs_edges = list(nx.bfs_edges(flat_tree, bfs_root))
        bfs_nodes = [bfs_root] + [v for u, v in bfs_edges]
        ft_node_pos_updated = deepcopy(ft_node_pos)
        flat_tree_copy = deepcopy(flat_tree)
        flat_tree_copy.remove_node(bfs_root)
        for i, edge_i in enumerate(bfs_edges):
            dist_nodes = distance.euclidean(ft_node_pos_updated[edge_i[0]], ft_node_pos_updated[edge_i[1]])
            len_edge = flat_tree.edges[edge_i]['len']
            st_x = ft_node_pos_updated[edge_i[0]][0]
            ed_x = ft_node_pos_updated[edge_i[1]][0]
            st_y = ft_node_pos_updated[edge_i[0]][1]
            ed_y = ft_node_pos_updated[edge_i[1]][1]
            p_x = st_x + (ed_x - st_x) * (len_edge / dist_nodes)
            p_y = st_y + (ed_y - st_y) * (len_edge / dist_nodes)
            ft_node_pos_updated[edge_i[1]] = np.array([p_x, p_y])
            con_components = list(nx.connected_components(flat_tree_copy))
            for con_comp in con_components:
                if edge_i[1] in con_comp:
                    reachable_unvisited = con_comp - {edge_i[1]}
                    flat_tree_copy.remove_node(edge_i[1])
                    break
            for nd in reachable_unvisited:
                nd_x = ft_node_pos_updated[nd][0] + p_x - ed_x
                nd_y = ft_node_pos_updated[nd][1] + p_y - ed_y
                ft_node_pos_updated[nd] = np.array([nd_x, nd_y])
        nx.set_node_attributes(flat_tree, values = ft_node_pos_updated, name = 'pos_spring')

    def add_flat_tree_cell_pos(self,dist_scale):
        ## Update the positions of cells on flat tree
        cells_pos = np.empty([self.adata.shape[0], 2])
        flat_tree = self.adata.uns['flat_tree']
        ft_node_pos = nx.get_node_attributes(flat_tree, 'pos_spring')
        list_branch_id = nx.get_edge_attributes(flat_tree, 'id').values()
        for br_id in list_branch_id:
            s_pos = ft_node_pos[br_id[0]]  # start node position
            e_pos = ft_node_pos[br_id[1]]  # end node position
            dist_se = distance.euclidean(s_pos, e_pos)
            p_x = np.array(self.adata.obs[self.adata.obs['branch_id'] == br_id]['branch_lam'].tolist())
            dist_p = dist_scale * np.array(self.adata.obs[self.adata.obs['branch_id'] == br_id]['branch_dist'].tolist())
            np.random.seed(100)
            p_y = np.random.choice([1, -1], size = len(p_x)) * dist_p
            # rotation matrix
            ro_angle = np.arctan2((e_pos - s_pos)[1], (e_pos - s_pos)[0])  # counterclockwise angle
            p_x_prime = s_pos[0] + p_x * math.cos(ro_angle) - p_y * math.sin(ro_angle)
            p_y_prime = s_pos[1] + p_x * math.sin(ro_angle) + p_y * math.cos(ro_angle)
            p_pos = np.array((p_x_prime, p_y_prime)).T
            cells_pos[np.where(self.adata.obs['branch_id'] == br_id)[0], :] = [p_pos[i, :].tolist() for i in
                                                                          range(p_pos.shape[0])]
        self.adata.obsm['X_spring'] = cells_pos

    def slugify(value):
        return ''.join(e if e.isalnum() else '_' for e in value)

    def add_stream_sc_pos(self, root='S0', dist_scale=1, dist_pctl=95, preference=None):
        flat_tree = self.adata.uns['flat_tree']
        label_to_node = {value: key for key, value in nx.get_node_attributes(flat_tree, 'label').items()}

        # Check if the root is valid
        if root not in label_to_node:
            raise ValueError(f"Root '{root}' not found in flat tree labels.")

        root_node = label_to_node[root]
        dict_bfs_pre = dict(nx.bfs_predecessors(flat_tree, root_node))
        dict_bfs_suc = dict(nx.bfs_successors(flat_tree, root_node))
        dict_edge_shift_dist = self.calculate_shift_distance(root = root, dist_pctl = dist_pctl,
                                                             preference = preference)
        dict_path_len = nx.shortest_path_length(flat_tree, source = root_node, weight = 'len')

        # Initialize DataFrame for cell positions
        df_cells_pos = pd.DataFrame(index = self.adata.obs.index, columns = ['cells_pos'])
        dict_edge_pos = {}
        dict_node_pos = {}

        for edge in dict_edge_shift_dist.keys():

            node_pos_st = np.array([dict_path_len[edge[0]], dict_edge_shift_dist[edge]])
            node_pos_ed = np.array([dict_path_len[edge[1]], dict_edge_shift_dist[edge]])
            br_id = flat_tree.edges[edge]['id']
            id_cells = np.where(self.adata.obs['branch_id'] == br_id)[0]
            try:
                root_label = next((key for key, value in label_to_node.items() if value == root_node),
                                  None)
                pseudotime_col = f"{root_label}_pseudotime"
                if pseudotime_col not in self.adata.obs.columns:
                    raise KeyError(f"Column '{pseudotime_col}' not found in self.adata.obs")

                cells_pos_x = self.adata.obs[pseudotime_col].iloc[id_cells]  # Safely access the pseudotime column
            except KeyError as e:
                continue
            np.random.seed(100)
            cells_pos_y = node_pos_st[1] + dist_scale * self.adata.obs.iloc[id_cells]['branch_dist'] * np.random.choice(
                [1, -1],
                size =
                id_cells.shape[
                    0])
            cells_pos = np.array((cells_pos_x, cells_pos_y)).T
            df_cells_pos.iloc[id_cells, 0] = [cells_pos[i, :].tolist() for i in range(cells_pos.shape[0])]
            dict_edge_pos[edge] = np.array([node_pos_st, node_pos_ed])
            if edge[0] not in dict_bfs_pre.keys():
                dict_node_pos[edge[0]] = node_pos_st
            dict_node_pos[edge[1]] = node_pos_ed
        self.adata.obsm['X_stream_' + root] = np.array(df_cells_pos['cells_pos'].tolist())
        if flat_tree.degree(root_node) > 1:
            suc_nodes = dict_bfs_suc[root_node]
            edges = [(root_node, sn) for sn in suc_nodes]
            edge_y_pos = []
            for edge in edges:
                if edge in dict_edge_pos:
                    edge_y_pos.append(dict_edge_pos[edge][0, 1])  # Store the Y position

            if edge_y_pos:
                max_y_pos = max(edge_y_pos)
                min_y_pos = min(edge_y_pos)
                median_y_pos = np.median(edge_y_pos)
                x_pos = dict_edge_pos[edges[0]][0, 0]
                dict_node_pos[root_node] = np.array([x_pos, median_y_pos])
        self.adata.uns['stream_' + root] = {'nodes': dict_node_pos, 'edges': {}}

        for edge in dict_edge_pos.keys():
            edge_pos = dict_edge_pos[edge]
            if edge[0] in dict_bfs_pre.keys():
                pre_node = dict_bfs_pre[edge[0]]
                link_edge_pos = np.array([dict_edge_pos[(pre_node, edge[0])][1,], dict_edge_pos[edge][0,]])
                edge_pos = np.vstack((link_edge_pos, edge_pos))

            self.adata.uns['stream_' + root]['edges'][edge] = edge_pos
        if flat_tree.degree(root_node) > 1:
            suc_nodes = dict_bfs_suc[root_node]
            edges = [(root_node, sn) for sn in suc_nodes]
            edge_y_pos = []
            for edge in edges:
                if edge in dict_edge_pos:
                    edge_y_pos.append(dict_edge_pos[edge][0, 1])  # Store the Y position
            if edge_y_pos:
                max_y_pos = max(edge_y_pos)
                min_y_pos = min(edge_y_pos)
                x_pos = dict_node_pos[root_node][0]
                link_edge_pos = np.array([[x_pos, min_y_pos], [x_pos, max_y_pos]])
                self.adata.uns['stream_' + root]['edges'][(root_node, root_node)] = link_edge_pos

    def calculate_shift_distance(self,root='S0', dist_pctl=95, preference=None):
        flat_tree = self.adata.uns['flat_tree']
        dict_label_node = {value: key for key, value in nx.get_node_attributes(flat_tree, 'label').items()}
        root_node = dict_label_node[root]
        ##shift distance for each branch
        dict_edge_shift_dist = dict()
        max_dist = np.percentile(self.adata.obs['branch_dist'], dist_pctl)  ## maximum distance from cells to branch
        leaves = [k for k, v in flat_tree.degree() if v == 1]
        n_nonroot_leaves = len(list(set(leaves) - set([root_node])))
        dict_bfs_pre = dict(nx.bfs_predecessors(flat_tree, root_node))
        dict_bfs_suc = dict(nx.bfs_successors(flat_tree, root_node))
        # depth first search
        if (preference != None):
            preference_nodes = [dict_label_node[x] for x in preference]
        else:
            preference_nodes = None
        dfs_nodes = self.dfs_nodes_modified(flat_tree, root_node, preference = preference_nodes)
        dfs_nodes_copy = deepcopy(dfs_nodes)
        id_leaf = 0
        while (len(dfs_nodes_copy) > 1):
            node = dfs_nodes_copy.pop()
            pre_node = dict_bfs_pre[node]
            if (node in leaves):
                dict_edge_shift_dist[(pre_node, node)] = 2 * max_dist * (id_leaf - (n_nonroot_leaves / 2.0))
                id_leaf = id_leaf + 1
            else:
                suc_nodes = dict_bfs_suc[node]
                dict_edge_shift_dist[(pre_node, node)] = (sum([dict_edge_shift_dist[(node, sn)] for sn in
                                                               suc_nodes])) / float(len(suc_nodes))
        return dict_edge_shift_dist

    def dfs_nodes_modified(self, tree, source, preference=None):
        visited, stack = [], [source]
        bfs_tree = nx.bfs_tree(tree, source = source)
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.append(vertex)
                unvisited = set(tree[vertex]) - set(visited)
                if (preference != None):
                    weights = list()
                    for x in unvisited:
                        successors = dict(nx.bfs_successors(bfs_tree, source = x))
                        successors_nodes = list(itertools.chain.from_iterable(successors.values()))
                        weights.append(min([preference.index(s) if s in preference else len(preference) for s in
                                            successors_nodes + [x]]))
                    unvisited = [x for _, x in sorted(zip(weights, unvisited), reverse = True, key = lambda x: x[0])]
                stack.extend(unvisited)
        return visited

    def arrowed_spines(self, ax,
                       x_width_fraction=0.03,
                       x_height_fraction=0.02,
                       lw=None,
                       ohg=0.2,
                       locations=('bottom right', 'left up'),
                       **arrow_kwargs):
        arrow_kwargs.setdefault('overhang', ohg)
        arrow_kwargs.setdefault('clip_on', False)
        arrow_kwargs.update({'length_includes_head': True})
        if lw is None:
            lw = ax.spines['bottom'].get_linewidth() * 1e-4
        annots = {}
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        fig = ax.get_figure()
        dps = fig.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(dps)
        width, height = bbox.width, bbox.height
        hw = x_width_fraction * (ymax - ymin)
        hl = x_height_fraction * (xmax - xmin)
        yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
        yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height
        for loc_str in locations:
            side, direction = loc_str.split(' ')
            assert side in {'top', 'bottom', 'left', 'right'}, "Unsupported side"
            assert direction in {'up', 'down', 'left', 'right'}, "Unsupported direction"
            if side in {'bottom', 'top'}:
                if direction in {'up', 'down'}:
                    raise ValueError("Only left/right arrows supported on the bottom and top")
                dy = 0
                head_width = hw
                head_length = hl
                y = ymin if side == 'bottom' else ymax
                if direction == 'right':
                    x = xmin
                    dx = xmax - xmin
                else:
                    x = xmax
                    dx = xmin - xmax
            else:
                if direction in {'left', 'right'}:
                    raise ValueError("Only up/downarrows supported on the left and right")
                dx = 0
                head_width = yhw
                head_length = yhl
                x = xmin if side == 'left' else xmax
                if direction == 'up':
                    y = ymin
                    dy = ymax - ymin
                else:
                    y = ymax
                    dy = ymin - ymax
            annots[loc_str] = ax.arrow(x, y, dx, dy, fc = 'k', ec = 'k', width = lw,
                                       head_width = head_width, head_length = head_length, **arrow_kwargs)
        return annots

    def cal_stream_polygon_string(self, dict_ann, root='S0', preference=None, dist_scale=0.9,
                                  factor_num_win=10, factor_min_win=2.0, factor_width=2.5,
                                  log_scale=False, factor_zoomin=100.0):
        list_ann_string = [k for k, v in dict_ann.items() if is_string_dtype(v)]

        flat_tree = self.adata.uns['flat_tree']
        label_to_node = {value: key for key, value in nx.get_node_attributes(flat_tree, 'label').items()}
        if (preference != None):
            preference_nodes = [label_to_node[x] for x in preference]
        else:
            preference_nodes = None
        dict_branches = {x: flat_tree.edges[x] for x in flat_tree.edges()}
        dict_node_state = nx.get_node_attributes(flat_tree, 'label')

        root_node = label_to_node[root]
        bfs_edges = self.bfs_edges_modified(flat_tree, root_node, preference = preference_nodes)
        bfs_nodes = []
        for x in bfs_edges:
            if x[0] not in bfs_nodes:
                bfs_nodes.append(x[0])
            if x[1] not in bfs_nodes:
                bfs_nodes.append(x[1])

        dict_verts = dict()  ### coordinates of all vertices
        dict_extent = dict()  ### the extent of plot

        df_stream = self.adata.obs[['branch_id', 'branch_lam']].copy()
        df_stream = df_stream.astype('object')
        df_stream['edge'] = ''
        df_stream['lam_ordered'] = ''
        for x in bfs_edges:
            if x in nx.get_edge_attributes(flat_tree, 'id').values():
                id_cells = np.where(df_stream['branch_id'] == x)[0]
                df_stream.loc[df_stream.index[id_cells], 'edge'] = pd.Series(
                    index = df_stream.index[id_cells],
                    data = [x] * len(id_cells))
                df_stream.loc[df_stream.index[id_cells], 'lam_ordered'] = df_stream.loc[
                    df_stream.index[id_cells], 'branch_lam']
            else:
                id_cells = np.where(df_stream['branch_id'] == (x[1], x[0]))[0]
                df_stream.loc[df_stream.index[id_cells], 'edge'] = pd.Series(
                    index = df_stream.index[id_cells],
                    data = [x] * len(id_cells))
                df_stream.loc[df_stream.index[id_cells], 'lam_ordered'] = flat_tree.edges[x]['len'] - df_stream.loc[
                    df_stream.index[id_cells], 'branch_lam']
        for ann in list_ann_string:
            df_stream['CELL_LABEL'] = dict_ann[ann]
            len_ori = {}
            for x in bfs_edges:
                if (x in dict_branches.keys()):
                    len_ori[x] = dict_branches[x]['len']
                else:
                    len_ori[x] = dict_branches[(x[1], x[0])]['len']

            dict_tree = {}
            bfs_prev = dict(nx.bfs_predecessors(flat_tree, root_node))
            bfs_next = dict(nx.bfs_successors(flat_tree, root_node))
            for x in bfs_nodes:
                dict_tree[x] = {'prev': "", 'next': []}
                if (x in bfs_prev.keys()):
                    dict_tree[x]['prev'] = bfs_prev[x]
                if (x in bfs_next.keys()):
                    x_rank = [bfs_nodes.index(x_next) for x_next in bfs_next[x]]
                    dict_tree[x]['next'] = [y for _, y in sorted(zip(x_rank, bfs_next[x]), key = lambda y: y[0])]

            ##shift distance of each branch
            dict_shift_dist = dict()
            # modified depth first search
            dfs_nodes = self.dfs_nodes_modified(flat_tree, root_node, preference = preference_nodes)
            leaves = [n for n, d in flat_tree.degree() if d == 1]
            id_leaf = 0
            dfs_nodes_copy = deepcopy(dfs_nodes)
            num_nonroot_leaf = len(list(set(leaves) - set([root_node])))
            while len(dfs_nodes_copy) > 1:
                node = dfs_nodes_copy.pop()
                prev_node = dict_tree[node]['prev']
                if (node in leaves):
                    dict_shift_dist[(prev_node, node)] = -(float(1) / dist_scale) * (
                            num_nonroot_leaf - 1) / 2.0 + id_leaf * (float(1) / dist_scale)
                    id_leaf = id_leaf + 1
                else:
                    next_nodes = dict_tree[node]['next']
                    dict_shift_dist[(prev_node, node)] = (sum([dict_shift_dist[(node, next_node)] for next_node in
                                                               next_nodes])) / float(len(next_nodes))
            if (flat_tree.degree(root_node)) > 1:
                next_nodes = dict_tree[root_node]['next']
                dict_shift_dist[(root_node, root_node)] = (sum([dict_shift_dist[(root_node, next_node)] for next_node in
                                                                next_nodes])) / float(len(next_nodes))
            df_bins = pd.DataFrame(index = list(df_stream['CELL_LABEL'].unique()) + ['boundary', 'center', 'edge'])
            list_paths = self.find_root_to_leaf_paths(flat_tree, root_node)
            max_path_len = self.find_longest_path(list_paths, len_ori)
            size_w = max_path_len / float(factor_num_win)
            if (size_w > min(len_ori.values()) / float(factor_min_win)):
                size_w = min(len_ori.values()) / float(factor_min_win)
            step_w = size_w / 2  # step of sliding window (the divisor should be even)
            if (len(dict_shift_dist) > 1):
                max_width = (max_path_len / float(factor_width)) / (
                        max(dict_shift_dist.values()) - min(dict_shift_dist.values()))
            else:
                max_width = max_path_len / float(factor_width)
            # max_width = (max_path_len/float(factor_width))/(max(dict_shift_dist.values()) - min(dict_shift_dist.values()))
            dict_shift_dist = {x: dict_shift_dist[x] * max_width for x in dict_shift_dist.keys()}
            min_width = 0.0  # min width of branch
            min_cellnum = 0  # the minimal cell number in one branch
            min_bin_cellnum = 0  # the minimal cell number in each bin
            dict_edge_filter = dict()  # filter out cells whose total count on one edge is below the min_cellnum
            df_edge_cellnum = pd.DataFrame(index = df_stream['CELL_LABEL'].unique(), columns = bfs_edges, dtype = float)
            for i, edge_i in enumerate(bfs_edges):
                df_edge_i = df_stream[df_stream.edge == edge_i]
                cells_kept = df_edge_i.CELL_LABEL.value_counts()[
                    df_edge_i.CELL_LABEL.value_counts() > min_cellnum].index
                df_edge_i = df_edge_i[df_edge_i['CELL_LABEL'].isin(cells_kept)]
                dict_edge_filter[edge_i] = df_edge_i
                for cell_i in df_stream['CELL_LABEL'].unique():
                    df_edge_cellnum[edge_i][cell_i] = float(df_edge_i[df_edge_i['CELL_LABEL'] == cell_i].shape[0])
            for i, edge_i in enumerate(bfs_edges):
                # degree of the start node
                degree_st = flat_tree.degree(edge_i[0])
                # degree of the end node
                degree_end = flat_tree.degree(edge_i[1])
                # matrix of windows only appearing on one edge
                mat_w = np.vstack([np.arange(0, len_ori[edge_i] - size_w + (len_ori[edge_i] / 10 ** 6), step_w), \
                                   np.arange(size_w, len_ori[edge_i] + (len_ori[edge_i] / 10 ** 6), step_w)]).T
                mat_w[-1, -1] = len_ori[edge_i]
                if (degree_st == 1):
                    mat_w = np.insert(mat_w, 0, [0, size_w / 2.0], axis = 0)
                if (degree_end == 1):
                    mat_w = np.insert(mat_w, mat_w.shape[0], [len_ori[edge_i] - size_w / 2.0, len_ori[edge_i]],
                                      axis = 0)
                total_bins = df_bins.shape[1]  # current total number of bins
                df_bins = df_bins.copy()  # avoid warning "DataFrame is highly fragmented."
                if (degree_st > 1 and i == 0):
                    # matrix of windows appearing on multiple edges
                    mat_w_common = np.array([[0, size_w / 2.0], [0, size_w]])
                    # neighbor nodes
                    nb_nodes = list(flat_tree.neighbors(edge_i[0]))
                    index_nb_nodes = [bfs_nodes.index(x) for x in nb_nodes]
                    nb_nodes = np.array(nb_nodes)[np.argsort(index_nb_nodes)].tolist()
                    total_bins = df_bins.shape[1]  # current total number of bins
                    for i_win in range(mat_w_common.shape[0]):
                        df_bins["win" + str(total_bins + i_win)] = ""
                        df_bins.loc[df_bins.index[:-3], "win" + str(total_bins + i_win)] = 0
                        df_bins.at['edge', "win" + str(total_bins + i_win)] = [(root_node, root_node)]
                        for j in range(degree_st):
                            df_edge_j = dict_edge_filter[(edge_i[0], nb_nodes[j])]
                            cell_num_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered >= 0, \
                                                                        df_edge_j.lam_ordered <= mat_w_common[
                                                                            i_win, 1])]['CELL_LABEL'].value_counts()
                            df_bins.loc[cell_num_common2.index, "win" + str(total_bins + i_win)] = \
                                df_bins.loc[cell_num_common2.index, "win" + str(total_bins + i_win)] + cell_num_common2
                            df_bins.loc['edge', "win" + str(total_bins + i_win)].append((edge_i[0], nb_nodes[j]))
                        df_bins.at['boundary', "win" + str(total_bins + i_win)] = mat_w_common[i_win, :]
                        if (i_win == 0):
                            df_bins.loc['center', "win" + str(total_bins + i_win)] = 0
                        else:
                            df_bins.loc['center', "win" + str(total_bins + i_win)] = size_w / 2
                max_binnum = np.around((len_ori[edge_i] / 4.0 - size_w) / step_w)  # the maximal number of merging bins
                df_edge_i = dict_edge_filter[edge_i]
                total_bins = df_bins.shape[1]  # current total number of bins
                df_bins = df_bins.copy()  # avoid warning "DataFrame is highly fragmented."
                if (max_binnum <= 1):
                    for i_win in range(mat_w.shape[0]):
                        df_bins["win" + str(total_bins + i_win)] = ""
                        df_bins.loc[df_bins.index[:-3], "win" + str(total_bins + i_win)] = 0
                        cell_num = df_edge_i[np.logical_and(df_edge_i.lam_ordered >= mat_w[i_win, 0], \
                                                            df_edge_i.lam_ordered <= mat_w[i_win, 1])][
                            'CELL_LABEL'].value_counts()
                        df_bins.loc[cell_num.index, "win" + str(total_bins + i_win)] = cell_num
                        df_bins.at['boundary', "win" + str(total_bins + i_win)] = mat_w[i_win, :]
                        if (degree_st == 1 and i_win == 0):
                            df_bins.loc['center', "win" + str(total_bins + i_win)] = 0
                        elif (degree_end == 1 and i_win == (mat_w.shape[0] - 1)):
                            df_bins.loc['center', "win" + str(total_bins + i_win)] = len_ori[edge_i]
                        else:
                            df_bins.loc['center', "win" + str(total_bins + i_win)] = np.mean(mat_w[i_win, :])
                    col_wins = ["win" + str(total_bins + i_win) for i_win in range(mat_w.shape[0])]
                    df_bins.loc['edge', col_wins] = pd.Series(
                        index = col_wins,
                        data = [[edge_i]] * len(col_wins))
                df_bins = df_bins.copy()  # avoid warning "DataFrame is highly fragmented."
                if (max_binnum > 1):
                    id_stack = []
                    for i_win in range(mat_w.shape[0]):
                        id_stack.append(i_win)
                        bd_bins = [mat_w[id_stack[0], 0], mat_w[id_stack[-1], 1]]  # boundary of merged bins
                        cell_num = df_edge_i[np.logical_and(df_edge_i.lam_ordered >= bd_bins[0], \
                                                            df_edge_i.lam_ordered <= bd_bins[1])][
                            'CELL_LABEL'].value_counts()
                        if (len(id_stack) == max_binnum or any(cell_num > min_bin_cellnum) or i_win == mat_w.shape[
                            0] - 1):
                            df_bins["win" + str(total_bins)] = ""
                            df_bins.loc[df_bins.index[:-3], "win" + str(total_bins)] = 0
                            df_bins.loc[cell_num.index, "win" + str(total_bins)] = cell_num
                            df_bins.at['boundary', "win" + str(total_bins)] = bd_bins
                            df_bins.at['edge', "win" + str(total_bins)] = [edge_i]
                            if (degree_st == 1 and (0 in id_stack)):
                                df_bins.loc['center', "win" + str(total_bins)] = 0
                            elif (degree_end == 1 and i_win == (mat_w.shape[0] - 1)):
                                df_bins.loc['center', "win" + str(total_bins)] = len_ori[edge_i]
                            else:
                                df_bins.loc['center', "win" + str(total_bins)] = np.mean(bd_bins)
                            total_bins = total_bins + 1
                            id_stack = []

                df_bins = df_bins.copy()  # avoid warning "DataFrame is highly fragmented."
                if (degree_end > 1):
                    # matrix of windows appearing on multiple edges
                    mat_w_common = np.vstack([np.arange(len_ori[edge_i] - size_w + step_w,
                                                        len_ori[edge_i] + (len_ori[edge_i] / 10 ** 6), step_w), \
                                              np.arange(step_w, size_w + (len_ori[edge_i] / 10 ** 6), step_w)]).T
                    # neighbor nodes
                    nb_nodes = list(flat_tree.neighbors(edge_i[1]))
                    nb_nodes.remove(edge_i[0])
                    index_nb_nodes = [bfs_nodes.index(x) for x in nb_nodes]
                    nb_nodes = np.array(nb_nodes)[np.argsort(index_nb_nodes)].tolist()

                    # matrix of windows appearing on multiple edges
                    total_bins = df_bins.shape[1]  # current total number of bins
                    if (mat_w_common.shape[0] > 0):
                        for i_win in range(mat_w_common.shape[0]):
                            df_bins["win" + str(total_bins + i_win)] = ""
                            df_bins.loc[df_bins.index[:-3], "win" + str(total_bins + i_win)] = 0
                            cell_num_common1 = df_edge_i[np.logical_and(df_edge_i.lam_ordered > mat_w_common[i_win, 0], \
                                                                        df_edge_i.lam_ordered <= len_ori[edge_i])][
                                'CELL_LABEL'].value_counts()
                            df_bins.loc[cell_num_common1.index, "win" + str(total_bins + i_win)] = cell_num_common1
                            df_bins.at['edge', "win" + str(total_bins + i_win)] = [edge_i]
                            for j in range(degree_end - 1):
                                df_edge_j = dict_edge_filter[(edge_i[1], nb_nodes[j])]
                                cell_num_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered >= 0, \
                                                                            df_edge_j.lam_ordered <= mat_w_common[
                                                                                i_win, 1])]['CELL_LABEL'].value_counts()
                                df_bins.loc[cell_num_common2.index, "win" + str(total_bins + i_win)] = \
                                    df_bins.loc[
                                        cell_num_common2.index, "win" + str(total_bins + i_win)] + cell_num_common2
                                if abs(((sum(mat_w_common[i_win, :]) + len_ori[edge_i]) / 2) - (
                                        len_ori[edge_i] + size_w / 2.0)) < step_w / 100.0:
                                    df_bins.loc['edge', "win" + str(total_bins + i_win)].append(
                                        (edge_i[1], nb_nodes[j]))
                            df_bins.at['boundary', "win" + str(total_bins + i_win)] = mat_w_common[i_win, :]
                            df_bins.loc['center', "win" + str(total_bins + i_win)] = (sum(mat_w_common[i_win, :]) +
                                                                                      len_ori[edge_i]) / 2

            df_bins = df_bins.copy()  # avoid warning "DataFrame is highly fragmented."
            # order cell names by the index of first non-zero
            cell_list = df_bins.index[:-3]
            id_nonzero = []
            for i_cn, cellname in enumerate(cell_list):
                if (np.flatnonzero(df_bins.loc[cellname,]).size == 0):
                    print('Cell ' + cellname + ' does not exist')
                    break
                else:
                    id_nonzero.append(np.flatnonzero(df_bins.loc[cellname,])[0])
            cell_list_sorted = cell_list[np.argsort(id_nonzero)].tolist()
            # original count
            df_bins_ori = df_bins.reindex(cell_list_sorted + ['boundary', 'center', 'edge'])
            if (log_scale):
                df_n_cells = df_bins_ori.iloc[:-3, :].sum()
                df_n_cells = df_n_cells / df_n_cells.max() * factor_zoomin
                df_bins_ori.iloc[:-3, :] = df_bins_ori.iloc[:-3, :] * np.log2(df_n_cells + 1) / (df_n_cells + 1)

            df_bins_cumsum = df_bins_ori.copy()
            df_bins_cumsum.iloc[:-3, :] = df_bins_ori.iloc[:-3, :][::-1].cumsum()[::-1]

            # normalization
            df_bins_cumsum_norm = df_bins_cumsum.copy()
            df_bins_cumsum_norm.iloc[:-3, :] = min_width + max_width * (df_bins_cumsum.iloc[:-3, :]) / (
                df_bins_cumsum.iloc[:-3, :]).values.max()

            df_bins_top = df_bins_cumsum_norm.copy()
            df_bins_top.iloc[:-3, :] = df_bins_cumsum_norm.iloc[:-3, :].subtract(df_bins_cumsum_norm.iloc[0, :] / 2.0)
            df_bins_base = df_bins_top.copy()
            df_bins_base.iloc[:-4, :] = df_bins_top.iloc[1:-3, :].values
            df_bins_base.iloc[-4, :] = 0 - df_bins_cumsum_norm.iloc[0, :] / 2.0
            dict_forest = {cellname: {nodename: {'prev': "", 'next': "", 'div': ""} for nodename in bfs_nodes} \
                           for cellname in df_edge_cellnum.index}
            for cellname in cell_list_sorted:
                for node_i in bfs_nodes:
                    nb_nodes = list(flat_tree.neighbors(node_i))
                    index_in_bfs = [bfs_nodes.index(nb) for nb in nb_nodes]
                    nb_nodes_sorted = np.array(nb_nodes)[np.argsort(index_in_bfs)].tolist()
                    if node_i == root_node:
                        next_nodes = nb_nodes_sorted
                        prev_nodes = ''
                    else:
                        next_nodes = nb_nodes_sorted[1:]
                        prev_nodes = nb_nodes_sorted[0]
                    dict_forest[cellname][node_i]['next'] = next_nodes
                    dict_forest[cellname][node_i]['prev'] = prev_nodes
                    if (len(next_nodes) > 1):
                        pro_next_edges = []  # proportion of next edges
                        for nt in next_nodes:
                            id_wins = [ix for ix, x in enumerate(df_bins_cumsum_norm.loc['edge', :]) if
                                       x == [(node_i, nt)]]
                            pro_next_edges.append(df_bins_cumsum_norm.loc[cellname, 'win' + str(id_wins[0])])
                        if (sum(pro_next_edges) == 0):
                            dict_forest[cellname][node_i]['div'] = np.cumsum(
                                np.repeat(1.0 / len(next_nodes), len(next_nodes))).tolist()
                        else:
                            dict_forest[cellname][node_i]['div'] = (
                                    np.cumsum(pro_next_edges) / sum(pro_next_edges)).tolist()

            # Shift
            dict_ep_top = {cellname: dict() for cellname in cell_list_sorted}  # coordinates of end points
            dict_ep_base = {cellname: dict() for cellname in cell_list_sorted}
            dict_ep_center = dict()  # center coordinates of end points in each branch

            df_top_x = df_bins_top.copy()  # x coordinates in top line
            df_top_y = df_bins_top.copy()  # y coordinates in top line
            df_base_x = df_bins_base.copy()  # x coordinates in base line
            df_base_y = df_bins_base.copy()  # y coordinates in base line

            for edge_i in bfs_edges:
                id_wins = [i for i, x in enumerate(df_bins_cumsum_norm.loc['edge', :]) if x[0] == edge_i]
                prev_node = dict_tree[edge_i[0]]['prev']
                if (prev_node == ''):
                    x_st = 0
                    if (flat_tree.degree(root_node) > 1):
                        id_wins = id_wins[1:]
                else:
                    id_wins = id_wins[1:]  # remove the overlapped window
                    x_st = dict_ep_center[(prev_node, edge_i[0])][0] - step_w
                y_st = dict_shift_dist[edge_i]
                for cellname in cell_list_sorted:
                    ##top line
                    px_top = df_bins_top.loc['center', list(map(lambda x: 'win' + str(x), id_wins))]
                    py_top = df_bins_top.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))]
                    px_top_prime = x_st + px_top
                    py_top_prime = y_st + py_top
                    dict_ep_top[cellname][edge_i] = [px_top_prime[-1], py_top_prime[-1]]
                    df_top_x.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))] = px_top_prime
                    df_top_y.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))] = py_top_prime
                    ##base line
                    px_base = df_bins_base.loc['center', list(map(lambda x: 'win' + str(x), id_wins))]
                    py_base = df_bins_base.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))]
                    px_base_prime = x_st + px_base
                    py_base_prime = y_st + py_base
                    dict_ep_base[cellname][edge_i] = [px_base_prime[-1], py_base_prime[-1]]
                    df_base_x.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))] = px_base_prime
                    df_base_y.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))] = py_base_prime
                dict_ep_center[edge_i] = np.array([px_top_prime[-1], y_st])

            id_wins_start = [i for i, x in enumerate(df_bins_cumsum_norm.loc['edge', :]) if
                             x[0] == (root_node, root_node)]
            if (len(id_wins_start) > 0):
                mean_shift_dist = np.mean([dict_shift_dist[(root_node, x)] \
                                           for x in dict_forest[cell_list_sorted[0]][root_node]['next']])
                for cellname in cell_list_sorted:
                    ##top line
                    px_top = df_bins_top.loc['center', list(map(lambda x: 'win' + str(x), id_wins_start))]
                    py_top = df_bins_top.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_start))]
                    px_top_prime = 0 + px_top
                    py_top_prime = mean_shift_dist + py_top
                    df_top_x.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_start))] = px_top_prime
                    df_top_y.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_start))] = py_top_prime
                    ##base line
                    px_base = df_bins_base.loc['center', list(map(lambda x: 'win' + str(x), id_wins_start))]
                    py_base = df_bins_base.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_start))]
                    px_base_prime = 0 + px_base
                    py_base_prime = mean_shift_dist + py_base
                    df_base_x.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_start))] = px_base_prime
                    df_base_y.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_start))] = py_base_prime

            # determine joints points
            dict_joint_top = {cellname: dict() for cellname in cell_list_sorted}  # coordinates of joint points
            dict_joint_base = {cellname: dict() for cellname in cell_list_sorted}  # coordinates of joint points
            if (flat_tree.degree(root_node) == 1):
                id_joints = [i for i, x in enumerate(df_bins_cumsum_norm.loc['edge', :]) if len(x) > 1]
            else:
                id_joints = [i for i, x in enumerate(df_bins_cumsum_norm.loc['edge', :]) if
                             len(x) > 1 and x[0] != (root_node, root_node)]
                id_joints.insert(0, 1)
            for id_j in id_joints:
                joint_edges = df_bins_cumsum_norm.loc['edge', 'win' + str(id_j)]
                for id_div, edge_i in enumerate(joint_edges[1:]):
                    id_wins = [i for i, x in enumerate(df_bins_cumsum_norm.loc['edge', :]) if x == [edge_i]]
                    for cellname in cell_list_sorted:
                        if (len(dict_forest[cellname][edge_i[0]]['div']) > 0):
                            prev_node_top_x = df_top_x.loc[cellname, 'win' + str(id_j)]
                            prev_node_top_y = df_top_y.loc[cellname, 'win' + str(id_j)]
                            prev_node_base_x = df_base_x.loc[cellname, 'win' + str(id_j)]
                            prev_node_base_y = df_base_y.loc[cellname, 'win' + str(id_j)]
                            div = dict_forest[cellname][edge_i[0]]['div']
                            if (id_div == 0):
                                px_top_prime_st = prev_node_top_x
                                py_top_prime_st = prev_node_top_y
                            else:
                                px_top_prime_st = prev_node_top_x + (prev_node_base_x - prev_node_top_x) * div[
                                    id_div - 1]
                                py_top_prime_st = prev_node_top_y + (prev_node_base_y - prev_node_top_y) * div[
                                    id_div - 1]
                            px_base_prime_st = prev_node_top_x + (prev_node_base_x - prev_node_top_x) * div[id_div]
                            py_base_prime_st = prev_node_top_y + (prev_node_base_y - prev_node_top_y) * div[id_div]
                            df_top_x.loc[cellname, 'win' + str(id_wins[0])] = px_top_prime_st
                            df_top_y.loc[cellname, 'win' + str(id_wins[0])] = py_top_prime_st
                            df_base_x.loc[cellname, 'win' + str(id_wins[0])] = px_base_prime_st
                            df_base_y.loc[cellname, 'win' + str(id_wins[0])] = py_base_prime_st
                            dict_joint_top[cellname][edge_i] = np.array([px_top_prime_st, py_top_prime_st])
                            dict_joint_base[cellname][edge_i] = np.array([px_base_prime_st, py_base_prime_st])

            dict_tree_copy = deepcopy(dict_tree)
            dict_paths_top, dict_paths_base = self.find_paths(dict_tree_copy, bfs_nodes)

            # identify boundary of each edge
            dict_edge_bd = dict()
            for edge_i in bfs_edges:
                id_wins = [i for i, x in enumerate(df_top_x.loc['edge', :]) if edge_i in x]
                dict_edge_bd[edge_i] = [df_top_x.iloc[0, id_wins[0]], df_top_x.iloc[0, id_wins[-1]]]

            x_smooth = np.unique(
                np.arange(min(df_base_x.iloc[0, :]), max(df_base_x.iloc[0, :]), step = step_w / 20).tolist() \
                + [max(df_base_x.iloc[0, :])]).tolist()
            x_joints = df_top_x.iloc[0, id_joints].tolist()
            # replace nearest value in x_smooth by x axis of joint points
            for x in x_joints:
                x_smooth[np.argmin(np.abs(np.array(x_smooth) - x))] = x

            dict_smooth_linear = {cellname: {'top': dict(), 'base': dict()} for cellname in cell_list_sorted}
            # interpolation
            for edge_i_top in dict_paths_top.keys():
                path_i_top = dict_paths_top[edge_i_top]
                id_wins_top = [i_x for i_x, x in enumerate(df_top_x.loc['edge']) if
                               set(np.unique(x)).issubset(set(path_i_top))]
                if (flat_tree.degree(root_node) > 1 and \
                        edge_i_top == (root_node, dict_forest[cell_list_sorted[0]][root_node]['next'][0])):
                    id_wins_top.insert(0, 1)
                    id_wins_top.insert(0, 0)
                for cellname in cell_list_sorted:
                    x_top = df_top_x.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_top))].tolist()
                    y_top = df_top_y.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_top))].tolist()
                    f_top_linear = interpolate.interp1d(x_top, y_top, kind = 'linear')
                    x_top_new = [x for x in x_smooth if (x >= x_top[0]) and (x <= x_top[-1])] + [x_top[-1]]
                    x_top_new = np.unique(x_top_new).tolist()
                    y_top_new_linear = f_top_linear(x_top_new)
                    for id_node in range(len(path_i_top) - 1):
                        edge_i = (path_i_top[id_node], path_i_top[id_node + 1])
                        edge_i_bd = dict_edge_bd[edge_i]
                        id_selected = [i_x for i_x, x in enumerate(x_top_new) if
                                       x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                        dict_smooth_linear[cellname]['top'][edge_i] = pd.DataFrame([np.array(x_top_new)[id_selected], \
                                                                                    np.array(y_top_new_linear)[
                                                                                        id_selected]],
                                                                                   index = ['x', 'y'])
            for edge_i_base in dict_paths_base.keys():
                path_i_base = dict_paths_base[edge_i_base]
                id_wins_base = [i_x for i_x, x in enumerate(df_base_x.loc['edge']) if
                                set(np.unique(x)).issubset(set(path_i_base))]
                if (flat_tree.degree(root_node) > 1 and \
                        edge_i_base == (root_node, dict_forest[cell_list_sorted[0]][root_node]['next'][-1])):
                    id_wins_base.insert(0, 1)
                    id_wins_base.insert(0, 0)
                for cellname in cell_list_sorted:
                    x_base = df_base_x.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_base))].tolist()
                    y_base = df_base_y.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_base))].tolist()
                    f_base_linear = interpolate.interp1d(x_base, y_base, kind = 'linear')
                    x_base_new = [x for x in x_smooth if (x >= x_base[0]) and (x <= x_base[-1])] + [x_base[-1]]
                    x_base_new = np.unique(x_base_new).tolist()
                    y_base_new_linear = f_base_linear(x_base_new)
                    for id_node in range(len(path_i_base) - 1):
                        edge_i = (path_i_base[id_node], path_i_base[id_node + 1])
                        edge_i_bd = dict_edge_bd[edge_i]
                        id_selected = [i_x for i_x, x in enumerate(x_base_new) if
                                       x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                        dict_smooth_linear[cellname]['base'][edge_i] = pd.DataFrame([np.array(x_base_new)[id_selected], \
                                                                                     np.array(y_base_new_linear)[
                                                                                         id_selected]],
                                                                                    index = ['x', 'y'])

            # searching for edges which cell exists based on the linear interpolation
            dict_edges_CE = {cellname: [] for cellname in cell_list_sorted}
            for cellname in cell_list_sorted:
                for edge_i in bfs_edges:
                    if (sum(abs(dict_smooth_linear[cellname]['top'][edge_i].loc['y'] - \
                                dict_smooth_linear[cellname]['base'][edge_i].loc['y']) > 1e-12)):
                        dict_edges_CE[cellname].append(edge_i)

            # determine paths which cell exists
            dict_paths_CE_top = {cellname: {} for cellname in cell_list_sorted}
            dict_paths_CE_base = {cellname: {} for cellname in cell_list_sorted}
            dict_forest_CE = dict()
            for cellname in cell_list_sorted:
                edges_cn = dict_edges_CE[cellname]
                nodes = [nodename for nodename in bfs_nodes if nodename in set(itertools.chain(*edges_cn))]
                dict_forest_CE[cellname] = {nodename: {'prev': "", 'next': []} for nodename in nodes}
                for node_i in nodes:
                    prev_node = dict_tree[node_i]['prev']
                    if ((prev_node, node_i) in edges_cn):
                        dict_forest_CE[cellname][node_i]['prev'] = prev_node
                    next_nodes = dict_tree[node_i]['next']
                    for x in next_nodes:
                        if (node_i, x) in edges_cn:
                            (dict_forest_CE[cellname][node_i]['next']).append(x)
                dict_paths_CE_top[cellname], dict_paths_CE_base[cellname] = self.find_paths(dict_forest_CE[cellname],
                                                                                            nodes)

            dict_smooth_new = deepcopy(dict_smooth_linear)
            for cellname in cell_list_sorted:
                paths_CE_top = dict_paths_CE_top[cellname]
                for edge_i_top in paths_CE_top.keys():
                    path_i_top = paths_CE_top[edge_i_top]
                    edges_top = [x for x in bfs_edges if set(np.unique(x)).issubset(set(path_i_top))]
                    id_wins_top = [i_x for i_x, x in enumerate(df_top_x.loc['edge']) if
                                   set(np.unique(x)).issubset(set(path_i_top))]

                    x_top = []
                    y_top = []
                    for e_t in edges_top:
                        if (e_t == edges_top[-1]):
                            py_top_linear = dict_smooth_linear[cellname]['top'][e_t].loc['y']
                            px = dict_smooth_linear[cellname]['top'][e_t].loc['x']
                        else:
                            py_top_linear = dict_smooth_linear[cellname]['top'][e_t].iloc[1, :-1]
                            px = dict_smooth_linear[cellname]['top'][e_t].iloc[0, :-1]
                        x_top = x_top + px.tolist()
                        y_top = y_top + py_top_linear.tolist()
                    x_top_new = x_top
                    y_top_new = savgol_filter(y_top, 11, polyorder = 1)
                    for id_node in range(len(path_i_top) - 1):
                        edge_i = (path_i_top[id_node], path_i_top[id_node + 1])
                        edge_i_bd = dict_edge_bd[edge_i]
                        id_selected = [i_x for i_x, x in enumerate(x_top_new) if
                                       x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                        dict_smooth_new[cellname]['top'][edge_i] = pd.DataFrame([np.array(x_top_new)[id_selected], \
                                                                                 np.array(y_top_new)[id_selected]],
                                                                                index = ['x', 'y'])

                paths_CE_base = dict_paths_CE_base[cellname]
                for edge_i_base in paths_CE_base.keys():
                    path_i_base = paths_CE_base[edge_i_base]
                    edges_base = [x for x in bfs_edges if set(np.unique(x)).issubset(set(path_i_base))]
                    id_wins_base = [i_x for i_x, x in enumerate(df_base_x.loc['edge']) if
                                    set(np.unique(x)).issubset(set(path_i_base))]

                    x_base = []
                    y_base = []
                    for e_b in edges_base:
                        if (e_b == edges_base[-1]):
                            py_base_linear = dict_smooth_linear[cellname]['base'][e_b].loc['y']
                            px = dict_smooth_linear[cellname]['base'][e_b].loc['x']
                        else:
                            py_base_linear = dict_smooth_linear[cellname]['base'][e_b].iloc[1, :-1]
                            px = dict_smooth_linear[cellname]['base'][e_b].iloc[0, :-1]
                        x_base = x_base + px.tolist()
                        y_base = y_base + py_base_linear.tolist()
                    x_base_new = x_base
                    y_base_new = savgol_filter(y_base, 11, polyorder = 1)
                    for id_node in range(len(path_i_base) - 1):
                        edge_i = (path_i_base[id_node], path_i_base[id_node + 1])
                        edge_i_bd = dict_edge_bd[edge_i]
                        id_selected = [i_x for i_x, x in enumerate(x_base_new) if
                                       x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                        dict_smooth_new[cellname]['base'][edge_i] = pd.DataFrame([np.array(x_base_new)[id_selected], \
                                                                                  np.array(y_base_new)[id_selected]],
                                                                                 index = ['x', 'y'])

            # find all edges of polygon
            poly_edges = []
            dict_tree_copy = deepcopy(dict_tree)
            cur_node = root_node
            next_node = dict_tree_copy[cur_node]['next'][0]
            dict_tree_copy[cur_node]['next'].pop(0)
            poly_edges.append((cur_node, next_node))
            cur_node = next_node
            while (not (next_node == root_node and cur_node == dict_tree[root_node]['next'][-1])):
                while (len(dict_tree_copy[cur_node]['next']) != 0):
                    next_node = dict_tree_copy[cur_node]['next'][0]
                    dict_tree_copy[cur_node]['next'].pop(0)
                    poly_edges.append((cur_node, next_node))
                    if (cur_node == dict_tree[root_node]['next'][-1] and next_node == root_node):
                        break
                    cur_node = next_node
                while (len(dict_tree_copy[cur_node]['next']) == 0):
                    next_node = dict_tree_copy[cur_node]['prev']
                    poly_edges.append((cur_node, next_node))
                    if (cur_node == dict_tree[root_node]['next'][-1] and next_node == root_node):
                        break
                    cur_node = next_node

            verts = {cellname: np.empty((0, 2)) for cellname in cell_list_sorted}
            for cellname in cell_list_sorted:
                for edge_i in poly_edges:
                    if edge_i in bfs_edges:
                        x_top = dict_smooth_new[cellname]['top'][edge_i].loc['x']
                        y_top = dict_smooth_new[cellname]['top'][edge_i].loc['y']
                        pxy = np.array([x_top, y_top]).T
                    else:
                        edge_i = (edge_i[1], edge_i[0])
                        x_base = dict_smooth_new[cellname]['base'][edge_i].loc['x']
                        y_base = dict_smooth_new[cellname]['base'][edge_i].loc['y']
                        x_base = x_base[::-1]
                        y_base = y_base[::-1]
                        pxy = np.array([x_base, y_base]).T
                    verts[cellname] = np.vstack((verts[cellname], pxy))
            dict_verts[ann] = verts

            extent = {'xmin': "", 'xmax': "", 'ymin': "", 'ymax': ""}
            for cellname in cell_list_sorted:
                for edge_i in bfs_edges:
                    xmin = dict_smooth_new[cellname]['top'][edge_i].loc['x'].min()
                    xmax = dict_smooth_new[cellname]['top'][edge_i].loc['x'].max()
                    ymin = dict_smooth_new[cellname]['base'][edge_i].loc['y'].min()
                    ymax = dict_smooth_new[cellname]['top'][edge_i].loc['y'].max()
                    if (extent['xmin'] == ""):
                        extent['xmin'] = xmin
                    else:
                        if (xmin < extent['xmin']):
                            extent['xmin'] = xmin

                    if (extent['xmax'] == ""):
                        extent['xmax'] = xmax
                    else:
                        if (xmax > extent['xmax']):
                            extent['xmax'] = xmax

                    if (extent['ymin'] == ""):
                        extent['ymin'] = ymin
                    else:
                        if (ymin < extent['ymin']):
                            extent['ymin'] = ymin

                    if (extent['ymax'] == ""):
                        extent['ymax'] = ymax
                    else:
                        if (ymax > extent['ymax']):
                            extent['ymax'] = ymax
            dict_extent[ann] = extent
        return dict_verts, dict_extent

    def cal_stream_polygon_numeric(self,dict_ann, root='S0', preference=None, dist_scale=0.9,
                                   factor_num_win=10, factor_min_win=2.0, factor_width=2.5,
                                   factor_nrow=200, factor_ncol=400,
                                   log_scale=False, factor_zoomin=100.0):
        list_ann_numeric = [k for k, v in dict_ann.items() if is_numeric_dtype(v)]

        flat_tree = self.adata.uns['flat_tree']
        label_to_node = {value: key for key, value in nx.get_node_attributes(flat_tree, 'label').items()}
        if (preference != None):
            preference_nodes = [label_to_node[x] for x in preference]
        else:
            preference_nodes = None
        dict_branches = {x: flat_tree.edges[x] for x in flat_tree.edges()}
        dict_node_state = nx.get_node_attributes(flat_tree, 'label')

        root_node = label_to_node[root]
        bfs_edges = self.bfs_edges_modified(flat_tree, root_node, preference = preference_nodes)
        bfs_nodes = []
        for x in bfs_edges:
            if x[0] not in bfs_nodes:
                bfs_nodes.append(x[0])
            if x[1] not in bfs_nodes:
                bfs_nodes.append(x[1])

        df_stream = self.adata.obs[['branch_id', 'branch_lam']].copy()
        df_stream = df_stream.astype('object')
        df_stream['edge'] = ''
        df_stream['lam_ordered'] = ''
        for x in bfs_edges:
            if x in nx.get_edge_attributes(flat_tree, 'id').values():
                id_cells = np.where(df_stream['branch_id'] == x)[0]
                df_stream.loc[df_stream.index[id_cells], 'edge'] = pd.Series(
                    index = df_stream.index[id_cells],
                    data = [x] * len(id_cells))
                df_stream.loc[df_stream.index[id_cells], 'lam_ordered'] = df_stream.loc[
                    df_stream.index[id_cells], 'branch_lam']
            else:
                id_cells = np.where(df_stream['branch_id'] == (x[1], x[0]))[0]
                df_stream.loc[df_stream.index[id_cells], 'edge'] = pd.Series(
                    index = df_stream.index[id_cells],
                    data = [x] * len(id_cells))
                df_stream.loc[df_stream.index[id_cells], 'lam_ordered'] = flat_tree.edges[x]['len'] - df_stream.loc[
                    df_stream.index[id_cells], 'branch_lam']

        df_stream['CELL_LABEL'] = 'unknown'
        for ann in list_ann_numeric:
            df_stream[ann] = dict_ann[ann]

        len_ori = {}
        for x in bfs_edges:
            if (x in dict_branches.keys()):
                len_ori[x] = dict_branches[x]['len']
            else:
                len_ori[x] = dict_branches[(x[1], x[0])]['len']

        dict_tree = {}
        bfs_prev = dict(nx.bfs_predecessors(flat_tree, root_node))
        bfs_next = dict(nx.bfs_successors(flat_tree, root_node))
        for x in bfs_nodes:
            dict_tree[x] = {'prev': "", 'next': []}
            if (x in bfs_prev.keys()):
                dict_tree[x]['prev'] = bfs_prev[x]
            if (x in bfs_next.keys()):
                x_rank = [bfs_nodes.index(x_next) for x_next in bfs_next[x]]
                dict_tree[x]['next'] = [y for _, y in sorted(zip(x_rank, bfs_next[x]), key = lambda y: y[0])]

        ##shift distance of each branch
        dict_shift_dist = dict()
        # modified depth first search
        dfs_nodes = self.dfs_nodes_modified(flat_tree, root_node, preference = preference_nodes)
        leaves = [n for n, d in flat_tree.degree() if d == 1]
        id_leaf = 0
        dfs_nodes_copy = deepcopy(dfs_nodes)
        num_nonroot_leaf = len(list(set(leaves) - set([root_node])))
        while len(dfs_nodes_copy) > 1:
            node = dfs_nodes_copy.pop()
            prev_node = dict_tree[node]['prev']
            if (node in leaves):
                dict_shift_dist[(prev_node, node)] = -(float(1) / dist_scale) * (
                        num_nonroot_leaf - 1) / 2.0 + id_leaf * (float(1) / dist_scale)
                id_leaf = id_leaf + 1
            else:
                next_nodes = dict_tree[node]['next']
                dict_shift_dist[(prev_node, node)] = (sum([dict_shift_dist[(node, next_node)] for next_node in
                                                           next_nodes])) / float(len(next_nodes))
        if (flat_tree.degree(root_node)) > 1:
            next_nodes = dict_tree[root_node]['next']
            dict_shift_dist[(root_node, root_node)] = (sum([dict_shift_dist[(root_node, next_node)] for next_node in
                                                            next_nodes])) / float(len(next_nodes))

        # dataframe of bins
        df_bins = pd.DataFrame(index = list(df_stream['CELL_LABEL'].unique()) + ['boundary', 'center', 'edge'])
        dict_ann_df = {ann: pd.DataFrame(index = list(df_stream['CELL_LABEL'].unique())) for ann in list_ann_numeric}
        dict_merge_num = {ann: [] for ann in list_ann_numeric}  # number of merged sliding windows
        list_paths = self.find_root_to_leaf_paths(flat_tree, root_node)
        max_path_len = self.find_longest_path(list_paths, len_ori)
        size_w = max_path_len / float(factor_num_win)
        if (size_w > min(len_ori.values()) / float(factor_min_win)):
            size_w = min(len_ori.values()) / float(factor_min_win)

        step_w = size_w / 2  # step of sliding window (the divisor should be even)
        if (len(dict_shift_dist) > 1):
            max_width = (max_path_len / float(factor_width)) / (
                    max(dict_shift_dist.values()) - min(dict_shift_dist.values()))
        else:
            max_width = max_path_len / float(factor_width)
        # max_width = (max_path_len/float(factor_width))/(max(dict_shift_dist.values()) - min(dict_shift_dist.values()))
        dict_shift_dist = {x: dict_shift_dist[x] * max_width for x in dict_shift_dist.keys()}
        min_width = 0.0  # min width of branch
        min_cellnum = 0  # the minimal cell number in one branch
        min_bin_cellnum = 0  # the minimal cell number in each bin
        dict_edge_filter = dict()  # filter out cells whose total count on one edge is below the min_cellnum
        df_edge_cellnum = pd.DataFrame(index = df_stream['CELL_LABEL'].unique(), columns = bfs_edges, dtype = float)

        for i, edge_i in enumerate(bfs_edges):
            df_edge_i = df_stream[df_stream.edge == edge_i]
            cells_kept = df_edge_i.CELL_LABEL.value_counts()[df_edge_i.CELL_LABEL.value_counts() > min_cellnum].index
            df_edge_i = df_edge_i[df_edge_i['CELL_LABEL'].isin(cells_kept)]
            dict_edge_filter[edge_i] = df_edge_i
            for cell_i in df_stream['CELL_LABEL'].unique():
                df_edge_cellnum[edge_i][cell_i] = float(df_edge_i[df_edge_i['CELL_LABEL'] == cell_i].shape[0])

        for i, edge_i in enumerate(bfs_edges):
            # degree of the start node
            degree_st = flat_tree.degree(edge_i[0])
            # degree of the end node
            degree_end = flat_tree.degree(edge_i[1])
            # matrix of windows only appearing on one edge
            mat_w = np.vstack([np.arange(0, len_ori[edge_i] - size_w + (len_ori[edge_i] / 10 ** 6), step_w), \
                               np.arange(size_w, len_ori[edge_i] + (len_ori[edge_i] / 10 ** 6), step_w)]).T
            mat_w[-1, -1] = len_ori[edge_i]
            if (degree_st == 1):
                mat_w = np.insert(mat_w, 0, [0, size_w / 2.0], axis = 0)
            if (degree_end == 1):
                mat_w = np.insert(mat_w, mat_w.shape[0], [len_ori[edge_i] - size_w / 2.0, len_ori[edge_i]], axis = 0)
            total_bins = df_bins.shape[1]  # current total number of bins

            df_bins = df_bins.copy()  # avoid warning "DataFrame is highly fragmented."
            if (degree_st > 1 and i == 0):
                # matrix of windows appearing on multiple edges
                mat_w_common = np.array([[0, size_w / 2.0], [0, size_w]])
                # neighbor nodes
                nb_nodes = list(flat_tree.neighbors(edge_i[0]))
                index_nb_nodes = [bfs_nodes.index(x) for x in nb_nodes]
                nb_nodes = np.array(nb_nodes)[np.argsort(index_nb_nodes)].tolist()
                # matrix of windows appearing on multiple edges
                total_bins = df_bins.shape[1]  # current total number of bins
                for i_win in range(mat_w_common.shape[0]):
                    df_bins["win" + str(total_bins + i_win)] = ""
                    df_bins.loc[df_bins.index[:-3], "win" + str(total_bins + i_win)] = 0
                    df_bins.at['edge', "win" + str(total_bins + i_win)] = [(root_node, root_node)]
                    dict_df_ann_common = dict()
                    for ann in list_ann_numeric:
                        dict_df_ann_common[ann] = list()
                    for j in range(degree_st):
                        df_edge_j = dict_edge_filter[(edge_i[0], nb_nodes[j])]
                        cell_num_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered >= 0, \
                                                                    df_edge_j.lam_ordered <= mat_w_common[i_win, 1])][
                            'CELL_LABEL'].value_counts()
                        df_bins.loc[cell_num_common2.index, "win" + str(total_bins + i_win)] = \
                            df_bins.loc[cell_num_common2.index, "win" + str(total_bins + i_win)] + cell_num_common2
                        for ann in list_ann_numeric:
                            dict_df_ann_common[ann].append(df_edge_j[np.logical_and(df_edge_j.lam_ordered >= 0, \
                                                                                    df_edge_j.lam_ordered <=
                                                                                    mat_w_common[i_win, 1])])
                        df_bins.loc['edge', "win" + str(total_bins + i_win)].append((edge_i[0], nb_nodes[j]))
                    for ann in list_ann_numeric:
                        ann_values_common = pd.concat(dict_df_ann_common[ann]).groupby(['CELL_LABEL'])[ann].mean()
                        dict_ann_df[ann].loc[
                            ann_values_common.index, "win" + str(total_bins + i_win)] = ann_values_common
                        dict_ann_df[ann] = dict_ann_df[ann].copy()  # avoid warning "DataFrame is highly fragmented."
                    df_bins.at['boundary', "win" + str(total_bins + i_win)] = mat_w_common[i_win, :]
                    if (i_win == 0):
                        df_bins.loc['center', "win" + str(total_bins + i_win)] = 0
                    else:
                        df_bins.loc['center', "win" + str(total_bins + i_win)] = size_w / 2
            max_binnum = np.around((len_ori[edge_i] / 4.0 - size_w) / step_w)  # the maximal number of merging bins
            df_edge_i = dict_edge_filter[edge_i]
            total_bins = df_bins.shape[1]  # current total number of bins
            df_bins = df_bins.copy()  # avoid warning "DataFrame is highly fragmented."
            if (max_binnum <= 1):
                for i_win in range(mat_w.shape[0]):
                    df_bins["win" + str(total_bins + i_win)] = ""
                    df_bins.loc[df_bins.index[:-3], "win" + str(total_bins + i_win)] = 0
                    cell_num = df_edge_i[np.logical_and(df_edge_i.lam_ordered >= mat_w[i_win, 0], \
                                                        df_edge_i.lam_ordered <= mat_w[i_win, 1])][
                        'CELL_LABEL'].value_counts()
                    df_bins.loc[cell_num.index, "win" + str(total_bins + i_win)] = cell_num
                    df_bins.at['boundary', "win" + str(total_bins + i_win)] = mat_w[i_win, :]
                    for ann in list_ann_numeric:
                        dict_ann_df[ann]["win" + str(total_bins + i_win)] = 0
                        ann_values = df_edge_i[np.logical_and(df_edge_i.lam_ordered >= mat_w[i_win, 0], \
                                                              df_edge_i.lam_ordered <= mat_w[i_win, 1])].groupby(
                            ['CELL_LABEL'])[ann].mean()
                        dict_ann_df[ann].loc[ann_values.index, "win" + str(total_bins + i_win)] = ann_values
                        dict_ann_df[ann] = dict_ann_df[ann].copy()  # avoid warning "DataFrame is highly fragmented."
                        dict_merge_num[ann].append(1)
                    if (degree_st == 1 and i_win == 0):
                        df_bins.loc['center', "win" + str(total_bins + i_win)] = 0
                    elif (degree_end == 1 and i_win == (mat_w.shape[0] - 1)):
                        df_bins.loc['center', "win" + str(total_bins + i_win)] = len_ori[edge_i]
                    else:
                        df_bins.loc['center', "win" + str(total_bins + i_win)] = np.mean(mat_w[i_win, :])
                col_wins = ["win" + str(total_bins + i_win) for i_win in range(mat_w.shape[0])]
                df_bins.loc['edge', col_wins] = pd.Series(
                    index = col_wins,
                    data = [[edge_i]] * len(col_wins))

            df_bins = df_bins.copy()  # avoid warning "DataFrame is highly fragmented."
            if (max_binnum > 1):
                id_stack = []
                for i_win in range(mat_w.shape[0]):
                    id_stack.append(i_win)
                    bd_bins = [mat_w[id_stack[0], 0], mat_w[id_stack[-1], 1]]  # boundary of merged bins
                    cell_num = df_edge_i[np.logical_and(df_edge_i.lam_ordered >= bd_bins[0], \
                                                        df_edge_i.lam_ordered <= bd_bins[1])][
                        'CELL_LABEL'].value_counts()
                    if (len(id_stack) == max_binnum or any(cell_num > min_bin_cellnum) or i_win == mat_w.shape[0] - 1):
                        df_bins["win" + str(total_bins)] = ""
                        df_bins.loc[df_bins.index[:-3], "win" + str(total_bins)] = 0
                        df_bins.loc[cell_num.index, "win" + str(total_bins)] = cell_num
                        df_bins.at['boundary', "win" + str(total_bins)] = bd_bins
                        df_bins.at['edge', "win" + str(total_bins)] = [edge_i]
                        for ann in list_ann_numeric:
                            dict_ann_df[ann]["win" + str(total_bins)] = 0
                            ann_values = df_edge_i[np.logical_and(df_edge_i.lam_ordered >= bd_bins[0], \
                                                                  df_edge_i.lam_ordered <= bd_bins[1])].groupby(
                                ['CELL_LABEL'])[ann].mean()
                            dict_ann_df[ann].loc[ann_values.index, "win" + str(total_bins)] = ann_values
                            dict_ann_df[ann] = dict_ann_df[
                                ann].copy()  # avoid warning "DataFrame is highly fragmented."
                            dict_merge_num[ann].append(len(id_stack))
                        if (degree_st == 1 and (0 in id_stack)):
                            df_bins.loc['center', "win" + str(total_bins)] = 0
                        elif (degree_end == 1 and i_win == (mat_w.shape[0] - 1)):
                            df_bins.loc['center', "win" + str(total_bins)] = len_ori[edge_i]
                        else:
                            df_bins.loc['center', "win" + str(total_bins)] = np.mean(bd_bins)
                        total_bins = total_bins + 1
                        id_stack = []

            df_bins = df_bins.copy()  # avoid warning "DataFrame is highly fragmented."
            if (degree_end > 1):
                mat_w_common = np.vstack([np.arange(len_ori[edge_i] - size_w + step_w,
                                                    len_ori[edge_i] + (len_ori[edge_i] / 10 ** 6), step_w), \
                                          np.arange(step_w, size_w + (len_ori[edge_i] / 10 ** 6), step_w)]).T
                nb_nodes = list(flat_tree.neighbors(edge_i[1]))
                nb_nodes.remove(edge_i[0])
                index_nb_nodes = [bfs_nodes.index(x) for x in nb_nodes]
                nb_nodes = np.array(nb_nodes)[np.argsort(index_nb_nodes)].tolist()
                total_bins = df_bins.shape[1]  # current total number of bins
                if (mat_w_common.shape[0] > 0):
                    for i_win in range(mat_w_common.shape[0]):
                        df_bins["win" + str(total_bins + i_win)] = ""
                        df_bins.loc[df_bins.index[:-3], "win" + str(total_bins + i_win)] = 0
                        cell_num_common1 = df_edge_i[np.logical_and(df_edge_i.lam_ordered > mat_w_common[i_win, 0], \
                                                                    df_edge_i.lam_ordered <= len_ori[edge_i])][
                            'CELL_LABEL'].value_counts()
                        df_bins.loc[cell_num_common1.index, "win" + str(total_bins + i_win)] = cell_num_common1
                        dict_df_ann_common = dict()
                        for ann in list_ann_numeric:
                            dict_ann_df[ann]["win" + str(total_bins + i_win)] = 0
                            dict_df_ann_common[ann] = list()
                            dict_df_ann_common[ann].append(
                                df_edge_i[np.logical_and(df_edge_i.lam_ordered > mat_w_common[i_win, 0], \
                                                         df_edge_i.lam_ordered <= len_ori[edge_i])])
                            dict_merge_num[ann].append(1)
                        df_bins.at['edge', "win" + str(total_bins + i_win)] = [edge_i]
                        for j in range(degree_end - 1):
                            df_edge_j = dict_edge_filter[(edge_i[1], nb_nodes[j])]
                            cell_num_common2 = df_edge_j[np.logical_and(df_edge_j.lam_ordered >= 0, \
                                                                        df_edge_j.lam_ordered <= mat_w_common[
                                                                            i_win, 1])]['CELL_LABEL'].value_counts()
                            df_bins.loc[cell_num_common2.index, "win" + str(total_bins + i_win)] = \
                                df_bins.loc[cell_num_common2.index, "win" + str(total_bins + i_win)] + cell_num_common2
                            for ann in list_ann_numeric:
                                dict_df_ann_common[ann].append(df_edge_j[np.logical_and(df_edge_j.lam_ordered >= 0, \
                                                                                        df_edge_j.lam_ordered <=
                                                                                        mat_w_common[i_win, 1])])
                            if abs(((sum(mat_w_common[i_win, :]) + len_ori[edge_i]) / 2) - (
                                    len_ori[edge_i] + size_w / 2.0)) < step_w / 100.0:
                                df_bins.loc['edge', "win" + str(total_bins + i_win)].append((edge_i[1], nb_nodes[j]))
                        for ann in list_ann_numeric:
                            ann_values_common = pd.concat(dict_df_ann_common[ann]).groupby(['CELL_LABEL'])[ann].mean()
                            dict_ann_df[ann].loc[
                                ann_values_common.index, "win" + str(total_bins + i_win)] = ann_values_common
                            dict_ann_df[ann] = dict_ann_df[
                                ann].copy()  # avoid warning "DataFrame is highly fragmented."s
                        df_bins.at['boundary', "win" + str(total_bins + i_win)] = mat_w_common[i_win, :]
                        df_bins.loc['center', "win" + str(total_bins + i_win)] = (sum(mat_w_common[i_win, :]) + len_ori[
                            edge_i]) / 2

        df_bins = df_bins.copy()  # avoid warning "DataFrame is highly fragmented."
        # order cell names by the index of first non-zero
        cell_list = df_bins.index[:-3]
        id_nonzero = []
        for i_cn, cellname in enumerate(cell_list):
            if (np.flatnonzero(df_bins.loc[cellname,]).size == 0):
                print('Cell ' + cellname + ' does not exist')
                break
            else:
                id_nonzero.append(np.flatnonzero(df_bins.loc[cellname,])[0])
        cell_list_sorted = cell_list[np.argsort(id_nonzero)].tolist()

        for ann in list_ann_numeric:
            dict_ann_df[ann] = dict_ann_df[ann].reindex(cell_list_sorted)

        # original count
        df_bins_ori = df_bins.reindex(cell_list_sorted + ['boundary', 'center', 'edge'])
        if (log_scale):
            df_n_cells = df_bins_ori.iloc[:-3, :].sum()
            df_n_cells = df_n_cells / df_n_cells.max() * factor_zoomin
            df_bins_ori.iloc[:-3, :] = df_bins_ori.iloc[:-3, :] * np.log2(df_n_cells + 1) / (df_n_cells + 1)

        df_bins_cumsum = df_bins_ori.copy()
        df_bins_cumsum.iloc[:-3, :] = df_bins_ori.iloc[:-3, :][::-1].cumsum()[::-1]

        # normalization
        df_bins_cumsum_norm = df_bins_cumsum.copy()
        df_bins_cumsum_norm.iloc[:-3, :] = min_width + max_width * (df_bins_cumsum.iloc[:-3, :]) / (
            df_bins_cumsum.iloc[:-3, :]).values.max()

        df_bins_top = df_bins_cumsum_norm.copy()
        df_bins_top.iloc[:-3, :] = df_bins_cumsum_norm.iloc[:-3, :].subtract(df_bins_cumsum_norm.iloc[0, :] / 2.0)
        df_bins_base = df_bins_top.copy()
        df_bins_base.iloc[:-4, :] = df_bins_top.iloc[1:-3, :].values
        df_bins_base.iloc[-4, :] = 0 - df_bins_cumsum_norm.iloc[0, :] / 2.0

        df_bins_top = df_bins_cumsum_norm.copy()
        df_bins_top.iloc[:-3, :] = df_bins_cumsum_norm.iloc[:-3, :].subtract(df_bins_cumsum_norm.iloc[0, :] / 2.0)
        df_bins_base = df_bins_top.copy()
        df_bins_base.iloc[:-4, :] = df_bins_top.iloc[1:-3, :].values
        df_bins_base.iloc[-4, :] = 0 - df_bins_cumsum_norm.iloc[0, :] / 2.0

        dict_forest = {cellname: {nodename: {'prev': "", 'next': "", 'div': ""} for nodename in bfs_nodes} \
                       for cellname in df_edge_cellnum.index}
        for cellname in cell_list_sorted:
            for node_i in bfs_nodes:
                nb_nodes = list(flat_tree.neighbors(node_i))
                index_in_bfs = [bfs_nodes.index(nb) for nb in nb_nodes]
                nb_nodes_sorted = np.array(nb_nodes)[np.argsort(index_in_bfs)].tolist()
                if node_i == root_node:
                    next_nodes = nb_nodes_sorted
                    prev_nodes = ''
                else:
                    next_nodes = nb_nodes_sorted[1:]
                    prev_nodes = nb_nodes_sorted[0]
                dict_forest[cellname][node_i]['next'] = next_nodes
                dict_forest[cellname][node_i]['prev'] = prev_nodes
                if (len(next_nodes) > 1):
                    pro_next_edges = []  # proportion of next edges
                    for nt in next_nodes:
                        id_wins = [ix for ix, x in enumerate(df_bins_cumsum_norm.loc['edge', :]) if x == [(node_i, nt)]]
                        pro_next_edges.append(df_bins_cumsum_norm.loc[cellname, 'win' + str(id_wins[0])])
                    if (sum(pro_next_edges) == 0):
                        dict_forest[cellname][node_i]['div'] = np.cumsum(
                            np.repeat(1.0 / len(next_nodes), len(next_nodes))).tolist()
                    else:
                        dict_forest[cellname][node_i]['div'] = (
                                np.cumsum(pro_next_edges) / sum(pro_next_edges)).tolist()

        # Shift
        dict_ep_top = {cellname: dict() for cellname in cell_list_sorted}  # coordinates of end points
        dict_ep_base = {cellname: dict() for cellname in cell_list_sorted}
        dict_ep_center = dict()  # center coordinates of end points in each branch

        df_top_x = df_bins_top.copy()  # x coordinates in top line
        df_top_y = df_bins_top.copy()  # y coordinates in top line
        df_base_x = df_bins_base.copy()  # x coordinates in base line
        df_base_y = df_bins_base.copy()  # y coordinates in base line

        for edge_i in bfs_edges:
            id_wins = [i for i, x in enumerate(df_bins_cumsum_norm.loc['edge', :]) if x[0] == edge_i]
            prev_node = dict_tree[edge_i[0]]['prev']
            if (prev_node == ''):
                x_st = 0
                if (flat_tree.degree(root_node) > 1):
                    id_wins = id_wins[1:]
            else:
                id_wins = id_wins[1:]  # remove the overlapped window
                x_st = dict_ep_center[(prev_node, edge_i[0])][0] - step_w
            y_st = dict_shift_dist[edge_i]
            for cellname in cell_list_sorted:
                ##top line
                px_top = df_bins_top.loc['center', list(map(lambda x: 'win' + str(x), id_wins))]
                py_top = df_bins_top.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))]
                px_top_prime = x_st + px_top
                py_top_prime = y_st + py_top
                dict_ep_top[cellname][edge_i] = [px_top_prime[-1], py_top_prime[-1]]
                df_top_x.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))] = px_top_prime
                df_top_y.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))] = py_top_prime
                ##base line
                px_base = df_bins_base.loc['center', list(map(lambda x: 'win' + str(x), id_wins))]
                py_base = df_bins_base.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))]
                px_base_prime = x_st + px_base
                py_base_prime = y_st + py_base
                dict_ep_base[cellname][edge_i] = [px_base_prime[-1], py_base_prime[-1]]
                df_base_x.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))] = px_base_prime
                df_base_y.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))] = py_base_prime
            dict_ep_center[edge_i] = np.array([px_top_prime[-1], y_st])
        id_wins_start = [i for i, x in enumerate(df_bins_cumsum_norm.loc['edge', :]) if x[0] == (root_node, root_node)]
        if (len(id_wins_start) > 0):
            mean_shift_dist = np.mean([dict_shift_dist[(root_node, x)] \
                                       for x in dict_forest[cell_list_sorted[0]][root_node]['next']])
            for cellname in cell_list_sorted:
                ##top line
                px_top = df_bins_top.loc['center', list(map(lambda x: 'win' + str(x), id_wins_start))]
                py_top = df_bins_top.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_start))]
                px_top_prime = 0 + px_top
                py_top_prime = mean_shift_dist + py_top
                df_top_x.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_start))] = px_top_prime
                df_top_y.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_start))] = py_top_prime
                ##base line
                px_base = df_bins_base.loc['center', list(map(lambda x: 'win' + str(x), id_wins_start))]
                py_base = df_bins_base.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_start))]
                px_base_prime = 0 + px_base
                py_base_prime = mean_shift_dist + py_base
                df_base_x.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_start))] = px_base_prime
                df_base_y.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_start))] = py_base_prime

        # determine joints points
        dict_joint_top = {cellname: dict() for cellname in cell_list_sorted}  # coordinates of joint points
        dict_joint_base = {cellname: dict() for cellname in cell_list_sorted}  # coordinates of joint points
        if (flat_tree.degree(root_node) == 1):
            id_joints = [i for i, x in enumerate(df_bins_cumsum_norm.loc['edge', :]) if len(x) > 1]
        else:
            id_joints = [i for i, x in enumerate(df_bins_cumsum_norm.loc['edge', :]) if
                         len(x) > 1 and x[0] != (root_node, root_node)]
            id_joints.insert(0, 1)
        for id_j in id_joints:
            joint_edges = df_bins_cumsum_norm.loc['edge', 'win' + str(id_j)]
            for id_div, edge_i in enumerate(joint_edges[1:]):
                id_wins = [i for i, x in enumerate(df_bins_cumsum_norm.loc['edge', :]) if x == [edge_i]]
                for cellname in cell_list_sorted:
                    if (len(dict_forest[cellname][edge_i[0]]['div']) > 0):
                        prev_node_top_x = df_top_x.loc[cellname, 'win' + str(id_j)]
                        prev_node_top_y = df_top_y.loc[cellname, 'win' + str(id_j)]
                        prev_node_base_x = df_base_x.loc[cellname, 'win' + str(id_j)]
                        prev_node_base_y = df_base_y.loc[cellname, 'win' + str(id_j)]
                        div = dict_forest[cellname][edge_i[0]]['div']
                        if (id_div == 0):
                            px_top_prime_st = prev_node_top_x
                            py_top_prime_st = prev_node_top_y
                        else:
                            px_top_prime_st = prev_node_top_x + (prev_node_base_x - prev_node_top_x) * div[id_div - 1]
                            py_top_prime_st = prev_node_top_y + (prev_node_base_y - prev_node_top_y) * div[id_div - 1]
                        px_base_prime_st = prev_node_top_x + (prev_node_base_x - prev_node_top_x) * div[id_div]
                        py_base_prime_st = prev_node_top_y + (prev_node_base_y - prev_node_top_y) * div[id_div]
                        df_top_x.loc[cellname, 'win' + str(id_wins[0])] = px_top_prime_st
                        df_top_y.loc[cellname, 'win' + str(id_wins[0])] = py_top_prime_st
                        df_base_x.loc[cellname, 'win' + str(id_wins[0])] = px_base_prime_st
                        df_base_y.loc[cellname, 'win' + str(id_wins[0])] = py_base_prime_st
                        dict_joint_top[cellname][edge_i] = np.array([px_top_prime_st, py_top_prime_st])
                        dict_joint_base[cellname][edge_i] = np.array([px_base_prime_st, py_base_prime_st])

        dict_tree_copy = deepcopy(dict_tree)
        dict_paths_top, dict_paths_base = self.find_paths(dict_tree_copy, bfs_nodes)

        # identify boundary of each edge
        dict_edge_bd = dict()
        for edge_i in bfs_edges:
            id_wins = [i for i, x in enumerate(df_top_x.loc['edge', :]) if edge_i in x]
            dict_edge_bd[edge_i] = [df_top_x.iloc[0, id_wins[0]], df_top_x.iloc[0, id_wins[-1]]]

        x_smooth = np.unique(
            np.arange(min(df_base_x.iloc[0, :]), max(df_base_x.iloc[0, :]), step = step_w / 20).tolist() \
            + [max(df_base_x.iloc[0, :])]).tolist()
        x_joints = df_top_x.iloc[0, id_joints].tolist()
        # replace nearest value in x_smooth by x axis of joint points
        for x in x_joints:
            x_smooth[np.argmin(np.abs(np.array(x_smooth) - x))] = x

        dict_smooth_linear = {cellname: {'top': dict(), 'base': dict()} for cellname in cell_list_sorted}
        # interpolation
        for edge_i_top in dict_paths_top.keys():
            path_i_top = dict_paths_top[edge_i_top]
            id_wins_top = [i_x for i_x, x in enumerate(df_top_x.loc['edge']) if
                           set(np.unique(x)).issubset(set(path_i_top))]
            if (flat_tree.degree(root_node) > 1 and \
                    edge_i_top == (root_node, dict_forest[cell_list_sorted[0]][root_node]['next'][0])):
                id_wins_top.insert(0, 1)
                id_wins_top.insert(0, 0)
            for cellname in cell_list_sorted:
                x_top = df_top_x.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_top))].tolist()
                y_top = df_top_y.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_top))].tolist()
                f_top_linear = interpolate.interp1d(x_top, y_top, kind = 'linear')
                x_top_new = [x for x in x_smooth if (x >= x_top[0]) and (x <= x_top[-1])] + [x_top[-1]]
                x_top_new = np.unique(x_top_new).tolist()
                y_top_new_linear = f_top_linear(x_top_new)
                for id_node in range(len(path_i_top) - 1):
                    edge_i = (path_i_top[id_node], path_i_top[id_node + 1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x, x in enumerate(x_top_new) if x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                    dict_smooth_linear[cellname]['top'][edge_i] = pd.DataFrame([np.array(x_top_new)[id_selected], \
                                                                                np.array(y_top_new_linear)[
                                                                                    id_selected]], index = ['x', 'y'])
        for edge_i_base in dict_paths_base.keys():
            path_i_base = dict_paths_base[edge_i_base]
            id_wins_base = [i_x for i_x, x in enumerate(df_base_x.loc['edge']) if
                            set(np.unique(x)).issubset(set(path_i_base))]
            if (flat_tree.degree(root_node) > 1 and \
                    edge_i_base == (root_node, dict_forest[cell_list_sorted[0]][root_node]['next'][-1])):
                id_wins_base.insert(0, 1)
                id_wins_base.insert(0, 0)
            for cellname in cell_list_sorted:
                x_base = df_base_x.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_base))].tolist()
                y_base = df_base_y.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins_base))].tolist()
                f_base_linear = interpolate.interp1d(x_base, y_base, kind = 'linear')
                x_base_new = [x for x in x_smooth if (x >= x_base[0]) and (x <= x_base[-1])] + [x_base[-1]]
                x_base_new = np.unique(x_base_new).tolist()
                y_base_new_linear = f_base_linear(x_base_new)
                for id_node in range(len(path_i_base) - 1):
                    edge_i = (path_i_base[id_node], path_i_base[id_node + 1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x, x in enumerate(x_base_new) if x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                    dict_smooth_linear[cellname]['base'][edge_i] = pd.DataFrame([np.array(x_base_new)[id_selected], \
                                                                                 np.array(y_base_new_linear)[
                                                                                     id_selected]], index = ['x', 'y'])
        dict_edges_CE = {cellname: [] for cellname in cell_list_sorted}
        for cellname in cell_list_sorted:
            for edge_i in bfs_edges:
                if (sum(abs(dict_smooth_linear[cellname]['top'][edge_i].loc['y'] - \
                            dict_smooth_linear[cellname]['base'][edge_i].loc['y']) > 1e-12)):
                    dict_edges_CE[cellname].append(edge_i)

        # determine paths which cell exists
        dict_paths_CE_top = {cellname: {} for cellname in cell_list_sorted}
        dict_paths_CE_base = {cellname: {} for cellname in cell_list_sorted}
        dict_forest_CE = dict()
        for cellname in cell_list_sorted:
            edges_cn = dict_edges_CE[cellname]
            nodes = [nodename for nodename in bfs_nodes if nodename in set(itertools.chain(*edges_cn))]
            dict_forest_CE[cellname] = {nodename: {'prev': "", 'next': []} for nodename in nodes}
            for node_i in nodes:
                prev_node = dict_tree[node_i]['prev']
                if ((prev_node, node_i) in edges_cn):
                    dict_forest_CE[cellname][node_i]['prev'] = prev_node
                next_nodes = dict_tree[node_i]['next']
                for x in next_nodes:
                    if (node_i, x) in edges_cn:
                        (dict_forest_CE[cellname][node_i]['next']).append(x)
            dict_paths_CE_top[cellname], dict_paths_CE_base[cellname] = self.find_paths(dict_forest_CE[cellname], nodes)

        dict_smooth_new = deepcopy(dict_smooth_linear)
        for cellname in cell_list_sorted:
            paths_CE_top = dict_paths_CE_top[cellname]
            for edge_i_top in paths_CE_top.keys():
                path_i_top = paths_CE_top[edge_i_top]
                edges_top = [x for x in bfs_edges if set(np.unique(x)).issubset(set(path_i_top))]
                id_wins_top = [i_x for i_x, x in enumerate(df_top_x.loc['edge']) if
                               set(np.unique(x)).issubset(set(path_i_top))]

                x_top = []
                y_top = []
                for e_t in edges_top:
                    if (e_t == edges_top[-1]):
                        py_top_linear = dict_smooth_linear[cellname]['top'][e_t].loc['y']
                        px = dict_smooth_linear[cellname]['top'][e_t].loc['x']
                    else:
                        py_top_linear = dict_smooth_linear[cellname]['top'][e_t].iloc[1, :-1]
                        px = dict_smooth_linear[cellname]['top'][e_t].iloc[0, :-1]
                    x_top = x_top + px.tolist()
                    y_top = y_top + py_top_linear.tolist()
                x_top_new = x_top
                y_top_new = savgol_filter(y_top, 11, polyorder = 1)
                for id_node in range(len(path_i_top) - 1):
                    edge_i = (path_i_top[id_node], path_i_top[id_node + 1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x, x in enumerate(x_top_new) if x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                    dict_smooth_new[cellname]['top'][edge_i] = pd.DataFrame([np.array(x_top_new)[id_selected], \
                                                                             np.array(y_top_new)[id_selected]],
                                                                            index = ['x', 'y'])

            paths_CE_base = dict_paths_CE_base[cellname]
            for edge_i_base in paths_CE_base.keys():
                path_i_base = paths_CE_base[edge_i_base]
                edges_base = [x for x in bfs_edges if set(np.unique(x)).issubset(set(path_i_base))]
                id_wins_base = [i_x for i_x, x in enumerate(df_base_x.loc['edge']) if
                                set(np.unique(x)).issubset(set(path_i_base))]

                x_base = []
                y_base = []
                for e_b in edges_base:
                    if (e_b == edges_base[-1]):
                        py_base_linear = dict_smooth_linear[cellname]['base'][e_b].loc['y']
                        px = dict_smooth_linear[cellname]['base'][e_b].loc['x']
                    else:
                        py_base_linear = dict_smooth_linear[cellname]['base'][e_b].iloc[1, :-1]
                        px = dict_smooth_linear[cellname]['base'][e_b].iloc[0, :-1]
                    x_base = x_base + px.tolist()
                    y_base = y_base + py_base_linear.tolist()
                x_base_new = x_base
                y_base_new = savgol_filter(y_base, 11, polyorder = 1)
                for id_node in range(len(path_i_base) - 1):
                    edge_i = (path_i_base[id_node], path_i_base[id_node + 1])
                    edge_i_bd = dict_edge_bd[edge_i]
                    id_selected = [i_x for i_x, x in enumerate(x_base_new) if x >= edge_i_bd[0] and x <= edge_i_bd[1]]
                    dict_smooth_new[cellname]['base'][edge_i] = pd.DataFrame([np.array(x_base_new)[id_selected], \
                                                                              np.array(y_base_new)[id_selected]],
                                                                             index = ['x', 'y'])
        poly_edges = []
        dict_tree_copy = deepcopy(dict_tree)
        cur_node = root_node
        next_node = dict_tree_copy[cur_node]['next'][0]
        dict_tree_copy[cur_node]['next'].pop(0)
        poly_edges.append((cur_node, next_node))
        cur_node = next_node
        while (not (next_node == root_node and cur_node == dict_tree[root_node]['next'][-1])):
            while (len(dict_tree_copy[cur_node]['next']) != 0):
                next_node = dict_tree_copy[cur_node]['next'][0]
                dict_tree_copy[cur_node]['next'].pop(0)
                poly_edges.append((cur_node, next_node))
                if (cur_node == dict_tree[root_node]['next'][-1] and next_node == root_node):
                    break
                cur_node = next_node
            while (len(dict_tree_copy[cur_node]['next']) == 0):
                next_node = dict_tree_copy[cur_node]['prev']
                poly_edges.append((cur_node, next_node))
                if (cur_node == dict_tree[root_node]['next'][-1] and next_node == root_node):
                    break
                cur_node = next_node

        verts = {cellname: np.empty((0, 2)) for cellname in cell_list_sorted}
        for cellname in cell_list_sorted:
            for edge_i in poly_edges:
                if edge_i in bfs_edges:
                    x_top = dict_smooth_new[cellname]['top'][edge_i].loc['x']
                    y_top = dict_smooth_new[cellname]['top'][edge_i].loc['y']
                    pxy = np.array([x_top, y_top]).T
                else:
                    edge_i = (edge_i[1], edge_i[0])
                    x_base = dict_smooth_new[cellname]['base'][edge_i].loc['x']
                    y_base = dict_smooth_new[cellname]['base'][edge_i].loc['y']
                    x_base = x_base[::-1]
                    y_base = y_base[::-1]
                    pxy = np.array([x_base, y_base]).T
                verts[cellname] = np.vstack((verts[cellname], pxy))

        extent = {'xmin': "", 'xmax': "", 'ymin': "", 'ymax': ""}
        for cellname in cell_list_sorted:
            for edge_i in bfs_edges:
                xmin = dict_smooth_new[cellname]['top'][edge_i].loc['x'].min()
                xmax = dict_smooth_new[cellname]['top'][edge_i].loc['x'].max()
                ymin = dict_smooth_new[cellname]['base'][edge_i].loc['y'].min()
                ymax = dict_smooth_new[cellname]['top'][edge_i].loc['y'].max()
                if (extent['xmin'] == ""):
                    extent['xmin'] = xmin
                else:
                    if (xmin < extent['xmin']):
                        extent['xmin'] = xmin

                if (extent['xmax'] == ""):
                    extent['xmax'] = xmax
                else:
                    if (xmax > extent['xmax']):
                        extent['xmax'] = xmax

                if (extent['ymin'] == ""):
                    extent['ymin'] = ymin
                else:
                    if (ymin < extent['ymin']):
                        extent['ymin'] = ymin

                if (extent['ymax'] == ""):
                    extent['ymax'] = ymax
                else:
                    if (ymax > extent['ymax']):
                        extent['ymax'] = ymax
        dict_im_array = dict()
        for ann in list_ann_numeric:
            im_nrow = factor_nrow
            im_ncol = factor_ncol
            xmin = extent['xmin']
            xmax = extent['xmax']
            ymin = extent['ymin'] - (extent['ymax'] - extent['ymin']) * 0.1
            ymax = extent['ymax'] + (extent['ymax'] - extent['ymin']) * 0.1
            im_array = {cellname: np.zeros((im_nrow, im_ncol)) for cellname in cell_list_sorted}
            df_bins_ann = dict_ann_df[ann]
            for cellname in cell_list_sorted:
                for edge_i in bfs_edges:
                    id_wins_all = [i for i, x in enumerate(df_bins_cumsum_norm.loc['edge', :]) if x[0] == edge_i]
                    prev_edge = ''
                    id_wins_prev = []
                    if (flat_tree.degree(root_node) > 1):
                        if (edge_i == bfs_edges[0]):
                            id_wins = [0, 1]
                            im_array = self.fill_im_array(im_array, df_bins_ann, flat_tree, df_base_x, df_base_y,
                                                          df_top_x,
                                                          df_top_y, xmin, xmax, ymin, ymax, im_nrow, im_ncol, step_w,
                                                          dict_shift_dist, id_wins, edge_i, cellname, id_wins_prev,
                                                          prev_edge)
                        id_wins = id_wins_all
                        if (edge_i[0] == root_node):
                            prev_edge = (root_node, root_node)
                            id_wins_prev = [0, 1]
                        else:
                            prev_edge = (dict_tree[edge_i[0]]['prev'], edge_i[0])
                            id_wins_prev = [i for i, x in enumerate(df_bins_cumsum_norm.loc['edge', :]) if
                                            x[0] == prev_edge]
                        im_array = self.fill_im_array(im_array, df_bins_ann, flat_tree, df_base_x, df_base_y, df_top_x,
                                                      df_top_y, xmin, xmax, ymin, ymax, im_nrow, im_ncol, step_w,
                                                      dict_shift_dist, id_wins, edge_i, cellname, id_wins_prev,
                                                      prev_edge)
                    else:
                        id_wins = id_wins_all
                        if (edge_i[0] != root_node):
                            prev_edge = (dict_tree[edge_i[0]]['prev'], edge_i[0])
                            id_wins_prev = [i for i, x in enumerate(df_bins_cumsum_norm.loc['edge', :]) if
                                            x[0] == prev_edge]
                        im_array = self.fill_im_array(im_array, df_bins_ann, flat_tree, df_base_x, df_base_y, df_top_x,
                                                      df_top_y, xmin, xmax, ymin, ymax, im_nrow, im_ncol, step_w,
                                                      dict_shift_dist, id_wins, edge_i, cellname, id_wins_prev,
                                                      prev_edge)
            dict_im_array[ann] = im_array
        return verts, extent, cell_list_sorted, dict_ann_df, dict_im_array

    def bfs_edges_modified(self, tree, source, preference=None):
        visited, queue = [], [source]
        bfs_tree = nx.bfs_tree(tree, source = source)
        predecessors = dict(nx.bfs_predecessors(bfs_tree, source))
        edges = []
        while queue:
            vertex = queue.pop()
            if vertex not in visited:
                visited.append(vertex)
                if (vertex in predecessors.keys()):
                    edges.append((predecessors[vertex], vertex))
                unvisited = set(tree[vertex]) - set(visited)
                if (preference != None):
                    weights = list()
                    for x in unvisited:
                        successors = dict(nx.bfs_successors(bfs_tree, source = x))
                        successors_nodes = list(itertools.chain.from_iterable(successors.values()))
                        weights.append(min([preference.index(s) if s in preference else len(preference) for s in
                                            successors_nodes + [x]]))
                    unvisited = [x for _, x in sorted(zip(weights, unvisited), reverse = True, key = lambda x: x[0])]
                queue.extend(unvisited)
        return edges

    def find_root_to_leaf_paths(self, flat_tree, root):
        list_paths = list()
        for x in flat_tree.nodes():
            if ((x != root) & (flat_tree.degree(x) == 1)):
                path = list(nx.all_simple_paths(flat_tree, root, x))[0]
                list_edges = list()
                for ft, sd in zip(path, path[1:]):
                    list_edges.append((ft, sd))
                list_paths.append(list_edges)
        return list_paths

    def find_longest_path(self, list_paths, len_ori):
        list_lengths = list()
        for x in list_paths:
            list_lengths.append(sum([len_ori[x_i] for x_i in x]))
        return max(list_lengths)

    def find_paths(self, dict_tree, bfs_nodes):
        dict_paths_top = dict()
        dict_paths_base = dict()
        for node_i in bfs_nodes:
            prev_node = dict_tree[node_i]['prev']
            next_nodes = dict_tree[node_i]['next']
            if (prev_node == '') or (len(next_nodes) > 1):
                if (prev_node == ''):
                    cur_node_top = node_i
                    cur_node_base = node_i
                    stack_top = [cur_node_top]
                    stack_base = [cur_node_base]
                    while (len(dict_tree[cur_node_top]['next']) > 0):
                        cur_node_top = dict_tree[cur_node_top]['next'][0]
                        stack_top.append(cur_node_top)
                    dict_paths_top[(node_i, next_nodes[0])] = stack_top
                    while (len(dict_tree[cur_node_base]['next']) > 0):
                        cur_node_base = dict_tree[cur_node_base]['next'][-1]
                        stack_base.append(cur_node_base)
                    dict_paths_base[(node_i, next_nodes[-1])] = stack_base
                for i_mid in range(len(next_nodes) - 1):
                    cur_node_base = next_nodes[i_mid]
                    cur_node_top = next_nodes[i_mid + 1]
                    stack_base = [node_i, cur_node_base]
                    stack_top = [node_i, cur_node_top]
                    while (len(dict_tree[cur_node_base]['next']) > 0):
                        cur_node_base = dict_tree[cur_node_base]['next'][-1]
                        stack_base.append(cur_node_base)
                    dict_paths_base[(node_i, next_nodes[i_mid])] = stack_base
                    while (len(dict_tree[cur_node_top]['next']) > 0):
                        cur_node_top = dict_tree[cur_node_top]['next'][0]
                        stack_top.append(cur_node_top)
                    dict_paths_top[(node_i, next_nodes[i_mid + 1])] = stack_top
        return dict_paths_top, dict_paths_base

    def fill_im_array(self, dict_im_array, df_bins_gene, flat_tree, df_base_x, df_base_y, df_top_x, df_top_y, xmin,
                      xmax,
                      ymin, ymax, im_nrow, im_ncol, step_w, dict_shift_dist, id_wins, edge_i, cellname, id_wins_prev,
                      prev_edge):
        pad_ratio = 0.008
        xmin_edge = df_base_x.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))].min()
        xmax_edge = df_base_x.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))].max()
        id_st_x = int(np.floor(((xmin_edge - xmin) / (xmax - xmin)) * (im_ncol - 1)))
        id_ed_x = int(np.floor(((xmax_edge - xmin) / (xmax - xmin)) * (im_ncol - 1)))
        if (flat_tree.degree(edge_i[1]) == 1):
            id_ed_x = id_ed_x + 1
        if (id_st_x < 0):
            id_st_x = 0
        if (id_st_x > 0):
            id_st_x = id_st_x + 1
        if (id_ed_x > (im_ncol - 1)):
            id_ed_x = im_ncol - 1
        if (prev_edge != ''):
            shift_dist = dict_shift_dist[edge_i] - dict_shift_dist[prev_edge]
            gene_color = df_bins_gene.loc[
                cellname, list(map(lambda x: 'win' + str(x), [id_wins_prev[-1]] + id_wins[1:]))].tolist()
        else:
            gene_color = df_bins_gene.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))].tolist()
        x_axis = df_base_x.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))].tolist()
        x_base = np.linspace(x_axis[0], x_axis[-1], id_ed_x - id_st_x + 1)
        gene_color_new = np.interp(x_base, x_axis, gene_color)
        y_axis_base = df_base_y.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))].tolist()
        y_axis_top = df_top_y.loc[cellname, list(map(lambda x: 'win' + str(x), id_wins))].tolist()
        f_base_linear = interpolate.interp1d(x_axis, y_axis_base, kind = 'linear')
        f_top_linear = interpolate.interp1d(x_axis, y_axis_top, kind = 'linear')
        y_base = f_base_linear(x_base)
        y_top = f_top_linear(x_base)
        id_y_base = np.ceil((1 - (y_base - ymin) / (ymax - ymin)) * (im_nrow - 1)).astype(int) + int(
            im_ncol * pad_ratio)
        id_y_base[id_y_base < 0] = 0
        id_y_base[id_y_base > (im_nrow - 1)] = im_nrow - 1
        id_y_top = np.floor((1 - (y_top - ymin) / (ymax - ymin)) * (im_nrow - 1)).astype(int) - int(im_ncol * pad_ratio)
        id_y_top[id_y_top < 0] = 0
        id_y_top[id_y_top > (im_nrow - 1)] = im_nrow - 1
        id_x_base = range(id_st_x, (id_ed_x + 1))
        for x in range(len(id_y_base)):
            if (x in range(int(step_w / xmax * im_ncol)) and prev_edge != ''):
                if (shift_dist > 0):
                    id_y_base[x] = id_y_base[x] - int(im_ncol * pad_ratio)
                    id_y_top[x] = id_y_top[x] + int(im_ncol * pad_ratio) - \
                                  int(abs(shift_dist) / abs(ymin - ymax) * im_nrow * 0.3)
                    if (id_y_top[x] < 0):
                        id_y_top[x] = 0
                if (shift_dist < 0):
                    id_y_base[x] = id_y_base[x] - int(im_ncol * pad_ratio) + \
                                   int(abs(shift_dist) / abs(ymin - ymax) * im_nrow * 0.3)
                    id_y_top[x] = id_y_top[x] + int(im_ncol * pad_ratio)
                    if (id_y_base[x] > im_nrow - 1):
                        id_y_base[x] = im_nrow - 1
            dict_im_array[cellname][id_y_top[x]:(id_y_base[x] + 1), id_x_base[x]] = np.tile(gene_color_new[x], \
                                                                                            (id_y_base[x] - id_y_top[
                                                                                                x] + 1))
        return dict_im_array


# === Plotting Methods ===

    def plot_stream_all(self):
            plot_stream_folder = os.path.join(self.Trajectory_folder, "plot_stream")
            if not os.path.exists(plot_stream_folder):
                os.makedirs(plot_stream_folder)

            all_roots = list(set(nx.get_node_attributes(self.adata.uns['flat_tree'],
                                                        'label').values()))
            for _ in range(len(all_roots)):
                root = all_roots[_]
                plot_root_folder = os.path.join(plot_stream_folder, root)
                if not os.path.exists(plot_root_folder):
                    os.makedirs(plot_root_folder)
                self.adata.obs['pheno_leiden'] = self.adata.obs['pheno_leiden'].astype('str')
                self.plot_stream(root=root, color=['pheno_leiden'],
                                 fig_path=plot_root_folder,
                                 fig_name=f"{self.analysis_name}_{'S' + root[-1] + '_'+ self.typeclustering}.pdf")
                # Plot pseudotime for the root
                self.plot_stream(root = root, color = ['S' + root[-1] + '_pseudotime'],
                                 fig_path = plot_root_folder,
                                 fig_name = f"{self.analysis_name}_{'S' + root[-1] + '_pseudotime'}.pdf")
                for feat in ['Cell_type', 'EXP', 'ID', 'Time_point', 'Condition', 'Count']:
                    if len(self.adata.obs[feat].unique()):
                        self.plot_stream(root = root, color = [feat],
                                         fig_path = plot_root_folder,
                                         fig_name = f"{self.analysis_name}_{root}_{feat}.pdf")
                for _ in self.adata.var.index:
                    self.plot_stream(root = root, color = [_],
                                     fig_path = plot_root_folder,
                                     fig_name = f"{self.analysis_name}_{'S' + root[-1] + '_'+_}.pdf")

    def plot_flat_tree_all(self):
        plot_tree_folder = os.path.join(self.Trajectory_folder, "plot_flat_tree")
        if not os.path.exists(plot_tree_folder):
            os.makedirs(plot_tree_folder)

        all_roots = list(set(nx.get_node_attributes(self.adata.uns['flat_tree'], 'label').values()))

        for _ in range(len(all_roots)):
            root = all_roots[_]
            plot_root_folder = os.path.join(plot_tree_folder, root)
            if not os.path.exists(plot_root_folder):
                os.makedirs(plot_root_folder)
            self.adata.obs['pheno_leiden'] = self.adata.obs['pheno_leiden'].astype('str')
            self.plot_flat_tree(root=root, color=['pheno_leiden'],
                                fig_path=plot_root_folder,
                                fig_name=f"{self.analysis_name}_{'S' + root[-1] + '_'+ self.typeclustering}.pdf")
            # Plot pseudotime for the root
            self.plot_flat_tree(root = root, color = ['S' + root[-1] + '_pseudotime'],
                                fig_path = plot_root_folder,
                                fig_name = f"{self.analysis_name}_{'S' + root[-1] + '_pseudotime'}.pdf")
            # Plot for 'Cell_type', 'EXP', 'ID', 'Time_point', 'Condition', 'Count' in obs
            for feat in ['Cell_type', 'EXP', 'ID', 'Time_point', 'Condition', 'Count']:
                if len(self.adata.obs[feat].unique()) > 1:
                    self.plot_flat_tree(root = root, color = [feat],
                                        fig_path = plot_root_folder,
                                        fig_name = f"{self.analysis_name}_{root}_{feat}.pdf")
            # Plot for each gene in var
        plot_tree_gene_folder = os.path.join(plot_tree_folder, "plot_flat_tree_markers")
        if not os.path.exists(plot_tree_gene_folder):
            os.makedirs(plot_tree_gene_folder)
        for gene in self.adata.var.index:
            self.plot_flat_tree(root = root, color = [gene],
                                    fig_path = plot_tree_gene_folder,
                                    fig_name = f"{self.analysis_name}_{gene}.pdf")


    def plot_stream(self, root='S0', color=None, preference=None, dist_scale=0.5,
                    factor_num_win=10, factor_min_win=2.0, factor_width=2.5, factor_nrow=200, factor_ncol=400,
                    log_scale=False, factor_zoomin=100.0,
                    fig_legend_order=None, fig_legend_ncol=1,
                    fig_colorbar_aspect=30,
                    vmin=None, vmax=None, fig_path=None, fig_name=None,
                    pad=1.08, w_pad=None, h_pad=None):
        # Set default fig_path to output_folder if not provided
        if fig_path is None:
            fig_path = self.output_folder

        fig_size = self.figsize

        if color is None:
            color = ['label']
        ### Remove duplicate keys
        color = list(dict.fromkeys(color))

        dict_ann = dict()
        use_turbo = False  # To track if we should use the 'turbo' colormap
        for ann in color:
            if ann in self.adata.obs.columns:
                dict_ann[ann] = self.adata.obs[ann]
            elif ann in self.adata.var_names:
                dict_ann[ann] = self.adata.obs_vector(ann)
                use_turbo = True  # Set turbo color map for genes from self.adata.var
            else:
                raise ValueError(f"Could not find '{ann}' in `self.adata.obs.columns` or `self.adata.var_names`")

        flat_tree = self.adata.uns['flat_tree']
        ft_node_label = nx.get_node_attributes(flat_tree, 'label')
        label_to_node = {value: key for key, value in nx.get_node_attributes(flat_tree, 'label').items()}
        if root not in label_to_node:
            raise ValueError(f"There is no root '{root}'")

        preference_nodes = [label_to_node[x] for x in preference] if preference else None

        # Handle legend order
        legend_order = {ann: np.unique(dict_ann[ann]) for ann in color if is_string_dtype(dict_ann[ann])}
        if fig_legend_order is not None:
            if not isinstance(fig_legend_order, dict):
                raise TypeError("`fig_legend_order` must be a dictionary")
            for ann in fig_legend_order:
                if ann in legend_order:
                    legend_order[ann] = fig_legend_order[ann]
                else:
                    print(f"'{ann}' is ignored for ordering legend labels due to incorrect name or data type")

        dict_plot = dict()

        # For categorical data
        list_string_type = [k for k, v in dict_ann.items() if is_string_dtype(v)]
        if len(list_string_type) > 0:
            dict_verts, dict_extent = self.cal_stream_polygon_string(dict_ann, root = root, preference = preference_nodes,
                                                                     dist_scale = dist_scale,
                                                                     factor_num_win = factor_num_win,
                                                                     factor_min_win = factor_min_win,
                                                                     factor_width = factor_width, log_scale = log_scale,
                                                                     factor_zoomin = factor_zoomin)
            dict_plot['string'] = [dict_verts, dict_extent]

        # For numeric data
        list_numeric_type = [k for k, v in dict_ann.items() if is_numeric_dtype(v)]
        if len(list_numeric_type) > 0:
            verts, extent, ann_order, dict_ann_df, dict_im_array = self.cal_stream_polygon_numeric(dict_ann, root = root,
                                                                                                   preference = preference_nodes,
                                                                                                   dist_scale = dist_scale,
                                                                                                   factor_num_win = factor_num_win,
                                                                                                   factor_min_win = factor_min_win,
                                                                                                   factor_width = factor_width,
                                                                                                   factor_nrow = factor_nrow,
                                                                                                   factor_ncol = factor_ncol,
                                                                                                   log_scale = log_scale,
                                                                                                   factor_zoomin = factor_zoomin)
            dict_plot['numeric'] = [verts, extent, ann_order, dict_ann_df, dict_im_array]

        # Start plotting
        for ann in color:
            if is_string_dtype(dict_ann[ann]):
                # Handle color mapping for categorical variables
                unique_vals = np.unique(dict_ann[ann])
                num_categories = len(unique_vals)

                # Choose the right palette
                if num_categories <= 28:
                    dict_palette = self.palette28[:num_categories]  # Use palette28 if 28 or fewer categories
                else:
                    dict_palette = self.palette102[:num_categories]  # Use palette102 if more than 28 categories
                # Create color mapping
                color_mapping = {str(val): dict_palette[i] for i, val in enumerate(unique_vals)}
                verts = dict_plot['string'][0][ann]
                extent = dict_plot['string'][1][ann]
                xmin = extent['xmin']
                xmax = extent['xmax']
                ymin = extent['ymin'] - (extent['ymax'] - extent['ymin']) * 0.1
                ymax = extent['ymax'] + (extent['ymax'] - extent['ymin']) * 0.1

                fig, ax = plt.subplots(figsize = fig_size)
                legend_labels = []

                # Plot each category with the corresponding color
                for ann_i in unique_vals:
                    ann_i_str = str(ann_i)
                    if ann_i_str in color_mapping:
                        legend_labels.append(ann_i_str)
                        verts_cell = verts[ann_i]
                        polygon = Polygon(verts_cell, closed = True, color = color_mapping[ann_i_str], alpha = 0.8, lw = 0)
                        ax.add_patch(polygon)
                    else:
                        print(f"Warning: Color for '{ann_i}' not found in palette. Using default gray color.")
                        polygon = Polygon(verts_cell, closed = True, color = 'gray', alpha = 0.8, lw = 0)
                        ax.add_patch(polygon)

                # Add legend
                ax.legend(legend_labels, bbox_to_anchor = (1.03, 0.5), loc = 'center left', ncol = fig_legend_ncol,
                          frameon = False, columnspacing = 0.4, borderaxespad = 0.2, handletextpad = 0.3)

            else:
                # Handle numeric data (continuous)
                verts = dict_plot['numeric'][0]
                extent = dict_plot['numeric'][1]
                ann_order = dict_plot['numeric'][2]
                dict_ann_df = dict_plot['numeric'][3]
                dict_im_array = dict_plot['numeric'][4]
                xmin = extent['xmin']
                xmax = extent['xmax']
                ymin = extent['ymin'] - (extent['ymax'] - extent['ymin']) * 0.1
                ymax = extent['ymax'] + (extent['ymax'] - extent['ymin']) * 0.1

                fig, ax = plt.subplots(figsize = fig_size)
                cmap = 'turbo' if use_turbo else None  # Use 'turbo' if plotting a gene from self.adata.var

                for ann_i in ann_order:
                    vmin_i = dict_ann_df[ann].loc[ann_i, :].min() if vmin is None else vmin
                    vmax_i = dict_ann_df[ann].loc[ann_i, :].max() if vmax is None else vmax
                    im = ax.imshow(dict_im_array[ann][ann_i], interpolation = 'bicubic',
                                   extent = [xmin, xmax, ymin, ymax], vmin = vmin_i, vmax = vmax_i, aspect = 'auto',
                                   cmap = cmap)
                    verts_cell = verts[ann_i]
                    clip_path = Polygon(verts_cell, facecolor = 'none', edgecolor = 'none', closed = True)
                    ax.add_patch(clip_path)
                    im.set_clip_path(clip_path)
                    cbar = plt.colorbar(im, ax = ax, pad = 0.04, fraction = 0.02, aspect = fig_colorbar_aspect)
                    cbar.ax.locator_params(nbins = 5)

                if use_turbo:
                    ax.set_title(f'{ann} (intensity)', fontsize = 12)

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_xlabel("pseudotime", labelpad = 2)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.locator_params(axis = 'x', nbins = 8)
            ax.tick_params(axis = "x", pad = -1)

            # Set title with root and color
            ax.set_title(f'Root: {root}, Color: {ann}', fontsize = 14)

            plt.tight_layout(pad = pad, h_pad = h_pad, w_pad = w_pad)

            # Generate fig_name if not provided
            if fig_name is None:
                fig_name = f"{root}_{ann}.pdf"

            # Save figure with path
            file_path_S = os.path.join(fig_path, fig_name)
            if not os.path.exists(os.path.dirname(file_path_S)):
                os.makedirs(os.path.dirname(file_path_S))
            plt.savefig(file_path_S, pad_inches = 1, bbox_inches = 'tight', dpi = 100)
            plt.close(fig)

    def plot_flat_tree(self, root='S0', color=None, dist_scale=0.5,
                       fig_legend_order=None, fig_legend_ncol=1, fig_ncol=3,
                       vmin=None, vmax=None, fig_path=None, fig_name=None,
                       pad=1.08, w_pad=None, h_pad=None,
                       alpha=0.8, dot_size=20):  # Added dot_size parameter
        """Plot flat tree layout with cells colored by a specified annotation."""

        if fig_path is None:
            fig_path = self.output_folder

        fig_size = self.figsize

        if color is None:
            color = ['label']
        ### Remove duplicate keys
        color = list(dict.fromkeys(color))

        dict_ann = dict()
        use_turbo = False  # To track if we should use the 'turbo' colormap for genes
        for ann in color:
            if ann in self.adata.obs.columns:
                dict_ann[ann] = self.adata.obs[ann]
            elif ann in self.adata.var_names:
                dict_ann[ann] = self.adata.obs_vector(ann)
                use_turbo = True  # Set turbo color map for genes from self.adata.var
            else:
                raise ValueError(f"Could not find '{ann}' in `self.adata.obs.columns` or `self.adata.var_names`")

        # Add positions of nodes and cells on the flat tree
        self.add_flat_tree_node_pos()
        flat_tree = self.adata.uns['flat_tree']
        self.add_flat_tree_cell_pos(dist_scale)

        # Retrieve node positions and cell positions
        ft_node_pos = nx.get_node_attributes(flat_tree, 'pos_spring')
        ft_node_label = nx.get_node_attributes(flat_tree, 'label')
        df_plot = pd.DataFrame(index = self.adata.obs.index, data = self.adata.obsm['X_spring'],
                               columns = ['FlatTree1', 'FlatTree2'])

        # Add annotation data to the plot DataFrame
        for ann in color:
            df_plot[ann] = dict_ann[ann]

        df_plot_shuf = df_plot.sample(frac = 1, random_state = 100)  # Shuffle for better plotting

        # Handle legend ordering for categorical variables
        legend_order = {ann: np.unique(df_plot_shuf[ann]) for ann in color if is_string_dtype(df_plot_shuf[ann])}
        if fig_legend_order is not None:
            if not isinstance(fig_legend_order, dict):
                raise TypeError("`fig_legend_order` must be a dictionary")
            for ann in fig_legend_order:
                if ann in legend_order:
                    legend_order[ann] = fig_legend_order[ann]
                else:
                    print(f"'{ann}' is ignored for ordering legend labels due to incorrect name or data type")

        dict_plot = dict()

        # For categorical data
        list_string_type = [k for k, v in dict_ann.items() if is_string_dtype(v)]
        if len(list_string_type) > 0:
            dict_plot['string'] = list_string_type

        # For numeric data
        list_numeric_type = [k for k, v in dict_ann.items() if is_numeric_dtype(v)]
        if len(list_numeric_type) > 0:
            dict_plot['numeric'] = list_numeric_type

        # Set up the figure layout
        fig_ncol = min(fig_ncol, len(color))  # Ensure fig_ncol is initialized and set correctly
        fig_nrow = int(np.ceil(len(color) / fig_ncol))
        fig = plt.figure(figsize = (fig_size[0] * fig_ncol * 1.05, fig_size[1] * fig_nrow))

        # Plot for each annotation
        for i, ann in enumerate(color):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i + 1)

            if is_string_dtype(df_plot[ann]):
                # Handle color mapping for categorical variables
                unique_vals = np.unique(df_plot[ann])
                num_categories = len(unique_vals)

                # Choose the right palette
                if num_categories <= 28:
                    dict_palette = self.palette28[:num_categories]  # Use palette28 if 28 or fewer categories
                else:
                    dict_palette = self.palette102[:num_categories]  # Use palette102 if more than 28 categories
                # Create color mapping
                color_mapping = {str(val): dict_palette[i] for i, val in enumerate(unique_vals)}

                # Categorical variable: apply a color palette
                sc_i = sns.scatterplot(ax = ax_i,
                                       x = 'FlatTree1', y = 'FlatTree2',
                                       hue = ann, hue_order = legend_order[ann],
                                       data = df_plot_shuf, alpha = alpha, linewidth = 0,
                                       palette = color_mapping, s = dot_size)  # Adjust dot size here

                legend_handles, legend_labels = ax_i.get_legend_handles_labels()
                ax_i.legend(handles = legend_handles, labels = legend_labels,
                            bbox_to_anchor = (1, 0.5), loc = 'center left', ncol = fig_legend_ncol,
                            frameon = False)

            else:
                # Continuous variable: use a colormap
                vmin_i = df_plot[ann].min() if vmin is None else vmin
                vmax_i = df_plot[ann].max() if vmax is None else vmax
                cmap = 'turbo' if use_turbo else None  # Use 'turbo' if plotting a gene from self.adata.var

                sc_i = ax_i.scatter(df_plot_shuf['FlatTree1'], df_plot_shuf['FlatTree2'],
                                    c = df_plot_shuf[ann], vmin = vmin_i, vmax = vmax_i, alpha = alpha, cmap = cmap,
                                    s = dot_size)
                cbar = plt.colorbar(sc_i, ax = ax_i, pad = 0.01, fraction = 0.05, aspect = 40)
                cbar.solids.set_edgecolor("face")
                cbar.ax.locator_params(nbins = 5)

            # Optionally plot the graph structure (edges) - now always included
            for edge_i in flat_tree.edges():
                branch_i_pos = np.array([ft_node_pos[i] for i in edge_i])
                ax_i.plot(branch_i_pos[:, 0], branch_i_pos[:, 1], c = 'black', alpha = 0.8)

            # Optionally show node labels
            for node_i in flat_tree.nodes():
                ax_i.text(ft_node_pos[node_i][0], ft_node_pos[node_i][1], ft_node_label[node_i],
                          color = 'black', fontsize = 1 * mpl.rcParams['font.size'],
                          ha = 'left', va = 'bottom')

            # Set axis labels and layout adjustments
            ax_i.set_xlabel("FlatTree1", labelpad = 2)
            ax_i.set_ylabel("FlatTree2", labelpad = 0)
            ax_i.locator_params(axis = 'x', nbins = 5)
            ax_i.locator_params(axis = 'y', nbins = 5)
            ax_i.tick_params(axis = "x", pad = -1)
            ax_i.tick_params(axis = "y", pad = -3)

            # Set title
            ax_i.set_title(f'Root: {root}, Color: {ann}', fontsize = 14)

        # Tight layout for the whole figure
        plt.tight_layout(pad = pad, h_pad = h_pad, w_pad = w_pad)

        # Save the figure
        if fig_name is None:
            fig_name = f"{self.analysis_name}_{root}_{color[0]}.pdf"
        plt.savefig(os.path.join(fig_path, fig_name), pad_inches = 1, bbox_inches = 'tight', dpi = 100)
        plt.close(fig)

    def plot_stream_sc(self, root='S0', color=None, dist_scale=0.5, dist_pctl=95,
                       fig_legend_ncol=1, fig_legend_order=None,
                       vmin=None, vmax=None, alpha=0.8, dot_size=20,fig_name=None,
                       pad=1.08, w_pad=None, h_pad=None, fig_path=None):
        """Plot stream using matplotlib and seaborn with palette selection."""

        if fig_path is None:
            fig_path = self.output_folder

        fig_size = self.figsize
        color = list(dict.fromkeys(color))  # Remove duplicates in color
        dict_ann = dict()

        # Prepare annotations (color)
        for ann in color:
            if ann in self.adata.obs.columns:
                dict_ann[ann] = self.adata.obs[ann]
            elif ann in self.adata.var_names:
                dict_ann[ann] = self.adata.obs_vector(ann)
            else:
                raise ValueError(f"Could not find '{ann}' in `self.adata.obs.columns` or `self.adata.var_names`.")

        # Prepare tree structure
        flat_tree = self.adata.uns['flat_tree']
        ft_node_label = nx.get_node_attributes(flat_tree, 'label')
        label_to_node = {value: key for key, value in nx.get_node_attributes(flat_tree, 'label').items()}
        if root not in label_to_node.keys():
            raise ValueError(f"There is no root '{root}'")

        self.add_stream_sc_pos(root=root, dist_scale=dist_scale, dist_pctl=dist_pctl)
        stream_nodes = self.adata.uns['stream_' + root]['nodes']
        stream_edges = self.adata.uns['stream_' + root]['edges']
        df_plot = pd.DataFrame(index=self.adata.obs.index, data=self.adata.obsm['X_stream_' + root],
                               columns=['pseudotime', 'dist'])

        # Add annotations to the dataframe
        for ann in color:
            df_plot[ann] = dict_ann[ann]

        df_plot_shuf = df_plot.sample(frac=1, random_state=100)

        # Legend order handling
        legend_order = {ann: np.unique(df_plot_shuf[ann]) for ann in color if is_string_dtype(df_plot_shuf[ann])}
        if fig_legend_order is not None:
            if not isinstance(fig_legend_order, dict):
                raise TypeError("`fig_legend_order` must be a dictionary")
            for ann in fig_legend_order.keys():
                if ann in legend_order.keys():
                    legend_order[ann] = fig_legend_order[ann]
                else:
                    print(f"'{ann}' is ignored for ordering legend labels due to incorrect name or data type.")

        # Plot using matplotlib and seaborn
        for i, ann in enumerate(color):
            fig = plt.figure(figsize=fig_size)
            ax_i = fig.add_subplot(1, 1, 1)

            if is_string_dtype(df_plot[ann]):
                # Handle palette based on the number of categories
                unique_vals = np.unique(df_plot[ann])
                num_categories = len(unique_vals)

                # Choose palette based on the number of categories
                if num_categories <= 28:
                    palette = self.palette28[:num_categories]  # Use palette28 for 28 or fewer categories
                else:
                    palette = self.palette102[:num_categories]  # Use palette102 for more than 28 categories

                # Seaborn scatterplot for categorical data
                sc_i = sns.scatterplot(ax=ax_i,
                                       x='pseudotime', y='dist',
                                       hue=ann, hue_order=legend_order[ann],
                                       data=df_plot_shuf,
                                       alpha=alpha, linewidth=0, s=dot_size,  # Use dot_size for scatterplot size
                                       palette=palette)

                legend_handles, legend_labels = ax_i.get_legend_handles_labels()
                ax_i.legend(handles=legend_handles, labels=legend_labels,
                            bbox_to_anchor=(1, 0.5), loc='center left', ncol=fig_legend_ncol, frameon=False)

                if ann + '_color' not in self.adata.uns_keys():
                    colors_sns = sc_i.get_children()[0].get_facecolors()
                    colors_sns_scaled = (255 * colors_sns).astype(int)
                    self.adata.uns[ann + '_color'] = {df_plot_shuf[ann][i]: '#%02x%02x%02x' % (
                        colors_sns_scaled[i][0], colors_sns_scaled[i][1], colors_sns_scaled[i][2])
                        for i in np.unique(df_plot_shuf[ann], return_index=True)[1]}
            else:
                # Scatterplot for continuous data
                vmin_i = df_plot[ann].min() if vmin is None else vmin
                vmax_i = df_plot[ann].max() if vmax is None else vmax
                sc_i = ax_i.scatter(df_plot_shuf['pseudotime'], df_plot_shuf['dist'],
                                    c=df_plot_shuf[ann], vmin=vmin_i, vmax=vmax_i, alpha=alpha, s=dot_size)
                cbar = plt.colorbar(sc_i, ax=ax_i, pad=0.01, fraction=0.05, aspect=40)
                cbar.solids.set_edgecolor("face")
                cbar.ax.locator_params(nbins=5)

            # Plot the graph edges (always enabled)
            for edge_i in stream_edges.keys():
                branch_i_pos = stream_edges[edge_i]
                branch_i = pd.DataFrame(branch_i_pos, columns=range(branch_i_pos.shape[1]))
                for ii in np.arange(start=0, stop=branch_i.shape[0], step=2):
                    if branch_i.iloc[ii, 0] == branch_i.iloc[ii + 1, 0]:
                        ax_i.plot(branch_i.iloc[[ii, ii + 1], 0], branch_i.iloc[[ii, ii + 1], 1],
                                  c='#767070', alpha=0.8)
                    else:
                        ax_i.plot(branch_i.iloc[[ii, ii + 1], 0], branch_i.iloc[[ii, ii + 1], 1],
                                  c='black', alpha=1)

            # Display text labels on graph nodes (always enabled)
            for node_i in flat_tree.nodes():
                ax_i.text(stream_nodes[node_i][0], stream_nodes[node_i][1], ft_node_label[node_i],
                          color='black', fontsize=0.9 * mpl.rcParams['font.size'],
                          ha='left', va='bottom')

            ax_i.set_xlabel("pseudotime", labelpad=2)
            ax_i.spines['left'].set_visible(False)
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
            ax_i.get_yaxis().set_visible(False)
            ax_i.locator_params(axis='x', nbins=8)
            ax_i.tick_params(axis="x", pad=-1)

            ax_i.set_title(ann)
            plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)

            # Generate fig_name if not provided
            if fig_name is None:
                fig_name = f"{root}_{ann}.pdf"

            # Save figure with path
            file_path_S = os.path.join(fig_path, fig_name)
            if not os.path.exists(os.path.dirname(file_path_S)):
                os.makedirs(os.path.dirname(file_path_S))
            plt.savefig(file_path_S, pad_inches=1, bbox_inches='tight', dpi=100)
            plt.close(fig)

    def clean_uns(self):

            self.adata.obs['branch_id'] = self.adata.obs['branch_id'].astype('str')
            self.adata.obs['branch_id_alias'] = self.adata.obs['branch_id_alias'].astype('str')
            # Delete all keys containing 'stream_S'
            keys_to_delete = [key for key in self.adata.uns.keys() if "stream_S" in key]
            for key in keys_to_delete:
                del self.adata.uns[key]

            # Delete specific keys
            for key in ['flat_tree', 'seed_epg', 'seed_flat_tree', 'ori_epg', 'epg_obj', 'ori_epg_obj','epg']:
                if key in self.adata.uns:
                    del self.adata.uns[key]

    def plot_stream_sc_all(self):
        """ Plots the stream trajectories for different roots with multiple visualizations for each root.
        It generates and saves plots for:
        - Pheno Leiden clusters.
        - Pseudotime for each root.
        - Cell type, Experiment (EXP), ID, Time point, Condition, and Count (if applicable).
        - Each gene in the AnnData object's variable index.
        All plots are saved to the `plot_stream_sc` folder, with subfolders for each root.
        Returns: None """
        plot_stream_folder = os.path.join(self.Trajectory_folder, "plot_stream_sc")
        os.makedirs(plot_stream_folder, exist_ok=True)  # Create the folder if it doesn't exist
        all_roots = list(set(nx.get_node_attributes(self.adata.uns['flat_tree'], 'label').values()))
        self.adata.obs['pheno_leiden'] = self.adata.obs['pheno_leiden'].astype(str)  # Ensure type conversion before loop
        for root in all_roots:
            plot_root_folder = os.path.join(plot_stream_folder, root)
            os.makedirs(plot_root_folder, exist_ok=True)  # Create folder for each root
            # Plot Pheno Leiden clusters
            self.plot_stream_sc(root=root, color=['pheno_leiden'],
                                fig_path=plot_root_folder,
                                fig_name=f"{self.analysis_name}_S{root[-1]}_{self.typeclustering}.pdf")
            # Plot pseudotime for the current root
            self.plot_stream_sc(root=root, color=[f'S{root[-1]}_pseudotime'],
                                fig_path=plot_root_folder,
                                fig_name=f"{self.analysis_name}_S{root[-1]}_pseudotime.pdf")
            # Plot for 'Cell_type', 'EXP', 'ID', 'Time_point', 'Condition', 'Count' in obs if feature has more than 1 unique value
            for feat in ['Cell_type', 'EXP', 'ID', 'Time_point', 'Condition', 'Count']:
                if len(self.adata.obs[feat].unique()) > 1:
                    self.plot_stream_sc(root=root, color=[feat],
                                        fig_path=plot_root_folder,
                                        fig_name=f"{self.analysis_name}_{root}_{feat}.pdf")
            # Plot for each gene in var
            for gene in self.adata.var.index:
                self.plot_stream_sc(root=root, color=[gene],
                                    fig_path=plot_root_folder,
                                    fig_name=f"{self.analysis_name}_S{root[-1]}_{gene}.pdf")