##exocrine GCNG with normalized graph matrix 
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from scipy import sparse

import torch
from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa
from torch_geometric.data import Data, DataLoader

from CCST import get_graph, train_DGI, PCA_process, Kmeans_cluster


def get_data(args):
    data_folder = args.data_path + args.data_name + '/' + args.cell_type + '/'
    with open(data_folder + 'Adjacent', 'rb') as fp:
        adj_0 = pickle.load(fp)
    X_data = np.load(data_folder + 'features.npy')
    num_points = X_data.shape[0]
    adj_I = np.eye(num_points)
    adj_I = sparse.csr_matrix(adj_I)

    adj = (1-args.lambda_I)*adj_0 + args.lambda_I*adj_I

    return adj_0, adj, X_data


def draw_map(args, draw_constructed = 0, draw_nerighbor=0):
    data_folder = args.data_path + args.data_name + '/' + args.cell_type + '/'
    df_cell_cluster_type = pd.read_csv(args.result_path+ 'types.txt', header=None, index_col=None, delimiter='\t')
    cell_sub_points = np.load(data_folder + 'cell_sub_points.npy')
    df_cell_cluster_type = df_cell_cluster_type.to_numpy()
    cell_cluster_type_list = df_cell_cluster_type[:,1] #
    n_clusters = max(cell_cluster_type_list) + 1 # start from 0
    # Draw cluster
    interneuron_x_points = cell_sub_points[:, 0]  # location of interneuron x_points
    interneuron_y_points = cell_sub_points[:, 1]  # location of interneuron y_points

    sc_cluster = plt.scatter(x=interneuron_x_points, y=interneuron_y_points, s=20, c= cell_cluster_type_list)
    plt.legend(handles=sc_cluster.legend_elements()[0], bbox_to_anchor=(1, 0.8), loc='center left')
    # else:
    #     sc_cluster = plt.scatter(x=sheet_X, y=sheet_Y, s=10, c=colors[cell_cluster_batch_list])
    #     cb_cluster = plt.colorbar(all_cluster, boundaries=np.arange(n_clusters + 1) - 0.5).set_ticks(np.arange(n_clusters))

    plt.xlabel('Dimension1')
    plt.ylabel('Dimension2')
    plt.savefig(args.result_path + '/' + args.cell_type + '_cluster'  + '.png', dpi=400, bbox_inches='tight')
    plt.clf()

    # Draw constructed_cluster
    if draw_constructed:
        from scipy.spatial.distance import pdist
        from matplotlib.collections import LineCollection
    # Draw barplot
    if draw_nerighbor:
        from scipy.spatial.distance import pdist
    return

def CCST_on_seqFISH(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lambda_I = args.lambda_I # intracellular (gene) and extracellular (spatial) information balance
    n_clusters = args.n_clusters  # num_cell_types
    # Parameters
    # batch_size = 10  # Batch size ???
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adj_0, adj, X_data = get_data(args)
    num_cell = X_data.shape[0]
    num_feature = X_data.shape[1]
    print('Adj:', adj.shape, 'Edges:', len(adj.data))
    print('Features:', X_data.shape) # (num_cell; num_feature)

    if args.DGI:
        print("-----------Deep Graph Infomax-------------")
        data_list = get_graph(adj, X_data)
        data_loader = DataLoader(data_list)
        DGI_model = train_DGI(args, data_loader=data_loader, in_channels=num_feature)
        for data in data_loader:
            data.to(device)
            X_embedding, _, _ = DGI_model(data)
            X_embedding = X_embedding.cpu().detach().numpy()
            X_embedding_filename =  args.embedding_data_path+'lambdaI' + str(lambda_I) + '_epoch' + str(args.num_epoch) + '_Embed_X.npy'
            np.save(X_embedding_filename, X_embedding)

    if args.cluster:
        print("-----------Clustering-------------")
        X_embedding_filename =  args.embedding_data_path+'lambdaI' + str(lambda_I) + '_epoch' + str(args.num_epoch) + '_Embed_X.npy'
        X_embedding = np.load(X_embedding_filename)

        if args.PCA:
            X_embedding = PCA_process(X_embedding, nps=30)
        print('Shape of data to cluster:', X_embedding.shape)
        cluster_labels, score = Kmeans_cluster(X_embedding, n_clusters)
        # n_clusters = cluster_labels.max()+1 # cluster_labels = {0;n_clusters-1} -> + 1
        # Umap(args, X_embedding, cluster_labels, n_clusters, score)

        all_data = [] # txt: cell_id, cell batch, cluster type 
        for index in range(num_cell):
            all_data.append([index, cluster_labels[index]])

        np.savetxt(args.result_path+'/types.txt', np.array(all_data), fmt='%3d', delimiter='\t')

    if args.draw_map:
        print("-----------Drawing map-------------")
        draw_map(args)

    if args.diff_gene:
        print("-----------Differential gene expression analysis-------------")
        sys.setrecursionlimit(1000000)
        gene = get_gene(args)
        all_cate_info = np.loadtxt(args.result_path+'/types.txt', dtype=int)
        cell_index_all_cluster=[[] for i in range(n_clusters)]
        for cate_info in all_cate_info:
            cell_index_all_cluster[cate_info[2]].append(cate_info[0])

        for i in range(n_clusters): #num_cell_types
            cell_index_cur_cluster = cell_index_all_cluster[i]
            cur_cate_num = len(cell_index_cur_cluster)
            cells_cur_type = []
            cells_features_cur_type = []
            cells_other_type = []
            cells_features_other_type = []
            for cell_index in range(num_cell):
                if cell_index in cell_index_cur_cluster:
                    cells_cur_type.append(cell_index)
                    cells_features_cur_type.append(X_data[cell_index].tolist())
                else:
                    cells_other_type.append(cell_index)
                    cells_features_other_type.append(X_data[cell_index].tolist())

            cells_features_cur_type = np.array(cells_features_cur_type)
            cells_features_other_type = np.array(cells_features_other_type)
            pvalues_features = np.zeros(num_feature)
            for k in range(num_feature):
                cells_curfeature_cur_type = cells_features_cur_type[:,k]
                cells_curfeature_other_type = cells_features_other_type[:,k]
                if np.all(cells_curfeature_cur_type == 0) and np.all(cells_curfeature_other_type == 0):
                    pvalues_features[k] = 1
                else:
                    pvalues_features[k] = test(cells_curfeature_cur_type, cells_curfeature_other_type)

            num_diff_gene = 200
            num_gene = int(len(gene))
            print('Cluster', i, '. Cell num:', cur_cate_num)
            min_p_index = pvalues_features.argsort()
            min_p = pvalues_features[min_p_index]
            min_p_gene = gene[min_p_index] # all candidate gene
            gene_cur_type = []
            gene_other_type = []
            diff_gene_sub = []
            for ind, diff_gene_index in enumerate(min_p_index):
                flag_0 = (len(gene_cur_type) >= num_diff_gene)
                flag_1 = (len(gene_other_type) >= num_diff_gene)
                if (flag_0 and flag_1) or (min_p[ind] > 0.05):
                    break
                if cells_features_cur_type[:,diff_gene_index].mean() > cells_features_other_type[:, diff_gene_index].mean():
                    if not flag_0:
                        gene_cur_type.append(gene[diff_gene_index])  # diff gene that higher in subpopulation 0
                        diff_gene_sub.append('cur '+str(len(gene_cur_type)-1))
                    else:
                        diff_gene_sub.append('None')
                if cells_features_cur_type[:,diff_gene_index].mean() < cells_features_other_type[:, diff_gene_index].mean():
                    if not flag_1:
                        gene_other_type.append(gene[diff_gene_index])  # diff gene that higher in subpopulation 1
                        diff_gene_sub.append('other '+str(len(gene_other_type)-1))
                    else:
                        diff_gene_sub.append('None')
            print('diff gene for cur type:', len(gene_cur_type))
            print('diff gene for other type:', len(gene_other_type))

            file_handle=open(args.result_path+'/cluster' + str(i) + '_diff_gene.txt', mode='w')
            for m in range(len(diff_gene_sub)):
                file_handle.write(min_p_gene[m] + '\t' + str(min_p[m]) + '\t' + diff_gene_sub[m] + '\n')
            file_handle.close()
            file_handle_0 = open(args.result_path+'/cluster' + str(i) + '_gene_cur.txt', mode='w')
            for m in range(len(gene_cur_type)):
                file_handle_0.write(gene_cur_type[m]+'\n')
            file_handle_0.close()
            file_handle_1 = open(args.result_path+'/cluster' + str(i) + '_gene_other.txt', mode='w')
            for m in range(len(gene_other_type)):
                file_handle_1.write(gene_other_type[m]+'\n')
            file_handle_1.close()
    return 

# For diff_gene test
def test(data1, data2):
    from scipy import stats
    # mannwhitneyu
    stat, p = stats.mannwhitneyu(data1, data2)
    return p
