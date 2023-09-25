import numpy as np
import matplotlib
matplotlib.use('Agg')
import os
import pandas as pd
from scipy import sparse
import pickle

####################  get the whole training dataset
def get_seqFISH_adj(cell_type, cell_type_dict, cell_type_annotations_path, spatial_location_path, generated_data_path):
    # this part is specifically designed for "MERFISH dataset".
    # For other datasets, please change the sheet name accordingly.
    print("------ Get Adjacency matrix ------")
    cell_type = cell_type_dict[cell_type]

    df_annotations = pd.read_csv(cell_type_annotations_path)  # (2050, 10000) feature #(2050) spatital # louvain -> 25
    df_annotations = df_annotations.to_numpy()
    all_cell_subpopulations = df_annotations[:, 1]
    cell_sub_idx = []
    for idx, cell_sub in enumerate(all_cell_subpopulations):
        if cell_sub in cell_type: # all_cell_sub in cell_type
            cell_sub_idx.append(idx)

    num_cells = len(cell_sub_idx) # 146 cell is interneuron
    # Spatial_location
    df_spatial = pd.read_csv(spatial_location_path)
    df_spatial = df_spatial.to_numpy()
    all_points = df_spatial[:, 2:] # list location of points
    cell_sub_points = all_points[cell_sub_idx]
    cell_sub_x_points = cell_sub_points[:,0]
    cell_sub_y_points = cell_sub_points[:,1]
    print("*** create cell_sub_idx.npy ***")
    np.save(generated_data_path + 'cell_sub_idx.npy', cell_sub_idx)
    print("*** create cell_sub_points.npy ***")
    np.save(generated_data_path + 'cell_sub_points.npy', cell_sub_points)
    distance_list = []
    distance_matrix = np.zeros((num_cells, num_cells))
    for i in range(num_cells):
        for j in range(num_cells):
            if i != j:
                dist = np.linalg.norm(np.array([cell_sub_x_points[i], cell_sub_y_points[i]]) - np.array([cell_sub_x_points[j], cell_sub_y_points[j]]))
                distance_list.append(dist)
                distance_matrix[i, j] = dist
    threshold = 180
    distance_matrix_threshold_I = np.where(distance_matrix <= threshold, distance_matrix, 0)
    distance_matrix_threshold_I = np.where(distance_matrix_threshold_I != 0, 1, 0)
    distance_matrix_threshold_I_N = np.float32(distance_matrix_threshold_I)  # don't normalize adjcent matrix
    ## get normalized sparse adjacent matrix
    distance_matrix_threshold_I_N_crs = sparse.csr_matrix(distance_matrix_threshold_I_N)  # normalize adjcent matrix
    print("*** create Adjacent ***")
    with open(generated_data_path + 'Adjacent', 'wb') as fp:
        pickle.dump(distance_matrix_threshold_I_N_crs, fp)
    return cell_sub_idx

def get_seqFISH_attribute(cell_sub_idx, data_path, generated_data_path):

    all_data = np.loadtxt(data_path, str, delimiter=",")
    print("------ Get all_gene ------")
    print("*** create all_genes.txt ***")
    all_genes_file = generated_data_path+'all_genes.txt'
    all_gene_names = all_data[0,:]

    file_handle = open(all_genes_file, mode='w')
    for gene_name in all_gene_names:
        file_handle.write(gene_name+'\n')
    file_handle.close()
    print("------ Get features ------")
    # this part is specifically designed for the demo merfish dataset.
    # For other datasets, please modify the 'B1', 'B2', 'B3'.
    all_features = all_data[1:, :] # (cell; feature_of_cell) -> (2050; 10000)
    cell_sub_features = all_features[cell_sub_idx]
    cell_sub_features[cell_sub_features == ''] = 0.0
    cell_sub_features = cell_sub_features.astype(np.float32)
    print('Feature: ', cell_sub_features.shape)
    print("*** create features.npy ***")
    np.save( generated_data_path + 'features.npy', cell_sub_features)
    return all_gene_names, cell_sub_features


def remove_low_gene(all_features, all_gene_names, generated_data_path, thres=0.2):
    print("------ Remove low_gene (Mean) ------")
    each_feature_mean = all_features.mean(0)
    tmp_mean = np.arange(len(each_feature_mean))
    indeces_after_removal = tmp_mean[each_feature_mean>=thres]
    features_after_removal = all_features[:,indeces_after_removal]
    gene_names_after_removal = all_gene_names[indeces_after_removal]
    print("new Feature: ", features_after_removal.shape)

    print("*** create gene_names.txt ***")
    genes_file = generated_data_path+'gene_names.txt'
    file_handle = open(genes_file, mode='w')
    for gene_name in gene_names_after_removal:
        file_handle.write(gene_name+'\n')
    file_handle.close()
    print("*** create new features.npy ***")
    np.save( generated_data_path + 'features.npy', features_after_removal)
    return features_after_removal, gene_names_after_removal

def remove_low_var_gene(all_features, all_gene_names, generated_data_path, thres=0.05):
    print("------ Remove low_gene (Var) ------")
    each_feature_var = all_features.var(0)
    tmp_var = np.arange(len(each_feature_var))
    indeces_after_removal = tmp_var[each_feature_var>=thres]
    features_after_removal = all_features[:,indeces_after_removal]
    gene_names_after_removal = all_gene_names[indeces_after_removal]
    print("new Features: ", features_after_removal.shape)

    print("*** create new gene_names.txt ***")
    genes_file = generated_data_path+'gene_names.txt'
    file_handle=open(genes_file, mode='w')
    for gene_name in gene_names_after_removal:
        file_handle.write(gene_name+'\n')
    file_handle.close()

    print("*** create new features.npy ***")
    np.save( generated_data_path + 'features.npy', features_after_removal)
    return features_after_removal, gene_names_after_removal

def normalization(features, generated_data_path):
    print("------ Normalization ------")
    node_num = features.shape[0]
    feature_num = features.shape[1]
    #max_in_each_node = features.max(1)
    sum_in_each_node = features.sum(1)
    sum_in_each_node = sum_in_each_node.reshape(-1, 1)
    sum_matrix = (np.repeat(sum_in_each_node, axis=1, repeats=feature_num))
    feature_normed = features/sum_matrix * 10000
    print("*** create new features.npy ***")
    np.save( generated_data_path + 'features.npy', feature_normed)
    return feature_normed


def main(args):
    cell_type_dict = {
        'Excitatory_neuron': [3, 4, 5, 6, 9, 11, 13, 18],
        'Interneuron': [7, 23],
        'Neural_Stem': [8, 14],
        'Neural_Progenitors': [15, 24],
        'Neuroblasts': [12, 17, 21],
        'Endothelial': [2],
        'Astrocytes': [10, 19],
        'Oligodendrocytes': [16],
        'Microglia': [20],
        'Ependymal': [22],
        'Oligo_Precursor': [25],
    }
    cell_sub_idx = get_seqFISH_adj(args.cell_type, cell_type_dict, args.cell_type_annotations_path, args.spatial_location_path, args.generated_data_path)
    all_gene_names, all_features = get_seqFISH_attribute(cell_sub_idx, args.gene_expression_path, args.generated_data_path)
    # Remove low_mean
    if args.remove_low_gene == 'mean':
        features_after_removal_mean, gene_names_after_removal_mean = remove_low_gene(all_features, all_gene_names, args.generated_data_path, thres=args.min_gene)
        normalized_features = normalization(features_after_removal_mean, args.generated_data_path)
    # Remove low_var
    elif args.remove_low_gene == 'var':
        features_after_removal_val, gene_names_after_removal_val = remove_low_var_gene(all_features, all_gene_names, args.generated_data_path, thres=args.min_gene_var)
        normalized_features = normalization(features_after_removal_mean, args.generated_data_path) # normalized_features = normalization(features_after_removal_val, args.generated_data_path)
    else:
        print("Don't remove gene")
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell_type', type=str, default= 'Interneuron', help='11 type_cell of OB mouse')
    parser.add_argument('--min_gene', type=float, default=0.2, help='Lowly expressed genes whose mean value is lower than this threshold will be filtered out') # 0.2 | 10000 -> 1593
    parser.add_argument('--min_gene_var', type=float, default=0.05, help='Low variance genes whose variance value is lower than this threshold will be filtered out') #0.05 | 10000 -> 7438
    parser.add_argument('--remove_low_gene', type=str, default='mean',help='remove with mean per cell or remove with val per cell')
    parser.add_argument('--gene_expression_path', type=str, default='dataset/seqFISH/ob_counts.csv',help='file path for gene expression')
    parser.add_argument('--spatial_location_path', type=str, default='dataset/seqFISH/ob_cellcentroids.csv',help='file path for cell location')
    parser.add_argument('--cell_type_annotations_path', type=str, default='dataset/seqFISH/OB_cell_type_annotations.csv',help='file path for cell location')
    parser.add_argument('--generated_data_path', type=str, default='generated_data/seqFISH/', help='The folder that generated data will be put in')
    args = parser.parse_args()

    args.generated_data_path = args.generated_data_path + '/' + args.cell_type + '/'
    if not os.path.exists(args.generated_data_path):
        os.makedirs(args.generated_data_path)
    print("Cell type: ", args.cell_type)
    main(args)
