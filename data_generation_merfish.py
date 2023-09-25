import numpy as np
import openpyxl
import matplotlib
matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import os
import scipy
from scipy import sparse
import pickle
import scipy.linalg
####################  get the whole training dataset


def get_MERFISH_adj(data_path, generated_data_path):
    # this part is specifically designed for "MERFISH dataset".
    # For other datasets, please change the sheet name accordingly.
    print("------ Get Adjacency matrix ------")
    if 1:
        workbook = openpyxl.load_workbook(data_path)
        distance_matrix_list = []
        all_distance_list = []
        num_cells = 0
        for sheet_name in ['Batch 1', 'Batch 2', 'Batch 3']:
            worksheet = workbook[sheet_name]
            sheet_X = [item.value for item in list(worksheet.columns)[1]]  # Get x_col spatial
            sheet_X = sheet_X[1:]  # Drop head
            sheet_Y = [item.value for item in list(worksheet.columns)[2]]  # Get y_col spatial
            sheet_Y = sheet_Y[1:]  # Drop head
            sheet_X = np.array(sheet_X).astype(float)
            sheet_Y = np.array(sheet_Y).astype(float)
            sheet_num_cells = len(sheet_X)  # B1 645 cell | B2 400 cell | B3 323 cell
            num_cells += sheet_num_cells  # -> 645+400+323 =
            distance_list = []
            sheet_distance_matrix = np.zeros((sheet_num_cells, sheet_num_cells))
            for i in range(sheet_num_cells):
                for j in range(sheet_num_cells):
                    if i != j:
                        dist = np.linalg.norm(np.array([sheet_X[i], sheet_Y[i]]) - np.array([sheet_X[j], sheet_Y[j]]))
                        distance_list.append(dist)
                        sheet_distance_matrix[i, j] = dist
            all_distance_list += distance_list #B1:(655*654) | B2:(400*399) | B3:(323*322)
            distance_matrix_list.append(sheet_distance_matrix)
        all_distance_array = np.array(all_distance_list)  # Batch: shape: num_cell * (num_cell-1)
        ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.block_diag.html
        """ scipy.linalg.block_diag
        Batch 1 : distance_matrix_list[0] (645, 645) 
        Batch 2 : distance_matrix_list[0] (400, 400)
        Batch 3 : distance_matrix_list[0] (323, 323)
        -> all_distance_matrix = (1368, 1368) 
        """
        all_distance_matrix = scipy.linalg.block_diag(distance_matrix_list[0],
                                                      distance_matrix_list[1],
                                                      distance_matrix_list[2])
        # print("*** create all_distance_array.npy ***")
        np.save( generated_data_path + 'all_distance_array.npy', all_distance_array)
        # print("*** create all_distance_matrix.npy ***")
        np.save( generated_data_path + 'all_distance_matrix.npy', all_distance_matrix)
    else:
        all_distance_array = np.load( generated_data_path + 'all_distance_array.npy')
        all_distance_matrix = np.load( generated_data_path + 'all_distance_matrix.npy')
        num_cells = len(all_distance_matrix)

    ###try different distance threshold, so that on average, each cell has x neighbor cells, see Tab. S1 for results
    # print('num_cells:', num_cells)
    # distance < threhold -> constructed
    threshold = 200
    num_big = np.where(all_distance_array<threshold)[0].shape[0] # return length
    # print("num_distance: ", len(all_distance_array))
    # print("threshold: ", threshold)
    # print("num_big_distance: ", num_big)
    # print("num_neighbor of cell:", str(num_big/num_cells)) # 5.8
    distance_matrix_threshold_I = np.where(all_distance_matrix <= threshold, all_distance_matrix, 0)
    distance_matrix_threshold_I = np.where(distance_matrix_threshold_I != 0, 1, 0)
    distance_matrix_threshold_I_N = np.float32(distance_matrix_threshold_I) # don't normalize adjcent matrix
    ## get normalized sparse adjacent matrix
    distance_matrix_threshold_I_N_crs = sparse.csr_matrix(distance_matrix_threshold_I_N) # normalize adjcent matrix
    print("*** create Adjacent ***")
    with open( generated_data_path + 'Adjacent', 'wb') as fp:
        pickle.dump(distance_matrix_threshold_I_N_crs, fp)

def get_all_gene(p, generated_data_path):
    print("------ Get all_gene ------")
    with open(p, encoding = 'utf-8') as f:
        # all_gene_names = np.loadtxt(f, str, delimiter = ",")[1:, 0]
        all_gene_names = np.loadtxt(f, str, delimiter=",")[1:, 0]
    print("*** create all_genes.txt ***")
    all_genes_file = generated_data_path+'all_genes.txt'
    file_handle=open(all_genes_file, mode='w')
    for gene_name in all_gene_names:
        file_handle.write(gene_name+'\n')
    file_handle.close()
    return all_gene_names

def get_attribute(data_path, generated_data_path):
    print("------ Get features ------")
    # this part is specifically designed for the demo merfish dataset. 
    # For other datasets, please modify the 'B1', 'B2', 'B3'.
    all_data = np.loadtxt(data_path, str, delimiter = ",")
    all_features = all_data[1:, 1:] # (feature_of_cell; cell)
    all_features[all_features == ''] = 0.0
    all_features = all_features.astype(np.float32)
    # np. swapaxes to swap any two axes
    all_features = np.swapaxes(all_features, 0, 1) # (cell; feature_of_cell)
    print('Feature: ', all_features.shape)
    print("*** create features.npy ***")
    np.save( generated_data_path + 'features.npy', all_features)
    all_features_batch_1, all_features_batch_2, all_features_batch_3 = [], [], []
    all_cell_names = all_data[0, 1:]
    all_batch_info = []
    for i,cell_name in enumerate(all_cell_names):
        if 'B1' in cell_name:
            all_batch_info.append(1)
            all_features_batch_1.append(all_features[i,:])
        elif 'B2' in cell_name:
            all_batch_info.append(2)
            all_features_batch_2.append(all_features[i,:])
        elif 'B3' in cell_name:
            all_batch_info.append(3)
            all_features_batch_3.append(all_features[i,:])
    all_features_all_batch = [np.array(all_features_batch_1), np.array(all_features_batch_2), np.array(all_features_batch_3)]
    print("*** create cell_batch_info.npy ***")
    np.save( generated_data_path + 'cell_batch_info.npy', all_batch_info)
    return all_features, all_features_all_batch


def remove_low_gene(all_features, all_features_all_batch, all_gene_names, generated_data_path, thres=1):
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
    features_all_batch_after_removal = []
    for all_features_each_batch in all_features_all_batch:
        features_all_batch_after_removal.append(all_features_each_batch[:,indeces_after_removal])
    return features_all_batch_after_removal, gene_names_after_removal

def batch_correction(datasets, genes_list, generated_data_path):
    # List of datasets (matrices of cells-by-genes):
    #datasets = [ list of scipy.sparse.csr_matrix or numpy.ndarray ]
    # List of gene lists:
    genes_lists = [genes_list, genes_list, genes_list]
    print("------ Batch_correction ------")
    import scanorama
    # Batch correction.
    corrected, _ = scanorama.correct(datasets, genes_lists)
    features_corrected = []
    for i, corrected_each_batch in enumerate(corrected):
        #features_corrected.append(np.array(corrected_each_batch.A))
        if i == 0:
            features_corrected = corrected_each_batch.A
        else:
            features_corrected = np.vstack((features_corrected, corrected_each_batch.A))
    features_corrected = np.array(features_corrected)
    print('Corrected size: ', features_corrected.shape)

    print("*** create new features.npy ***")
    np.save( generated_data_path + 'features.npy', features_corrected)
    return features_corrected

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

def remove_low_var_gene(features_input, gene_names, generated_data_path, thres=0.4):
    print("------ Remove low_gene (Var) ------")
    each_feature_var = features_input.var(0)
    tmp_var = np.arange(len(each_feature_var))
    indeces_after_removal = tmp_var[each_feature_var>=thres] 
    features_after_removal = features_input[:,indeces_after_removal]
    gene_names_after_removal = gene_names[indeces_after_removal]
    print("new Features: ", features_after_removal.shape)

    print("*** create new gene_names.txt ***")
    genes_file = generated_data_path+'gene_names.txt'
    file_handle=open(genes_file, mode='w')
    for gene_name in gene_names_after_removal:
        file_handle.write(gene_name+'\n')
    file_handle.close()

    print("*** create new features.npy ***")
    np.save( generated_data_path + 'features.npy', features_after_removal)
    print()
    return gene_names_after_removal


def main(args):
    ## for default merfish dataset:
    # B1：645, B2:400, B3:323
    # args.gene_expression_path = 'dataset/merfish/pnas.1912459116.sd12.csv'
    # args.spatial_location_path = 'dataset/merfish/pnas.1912459116.sd15.xlsx'
    get_MERFISH_adj(args.spatial_location_path, args.generated_data_path)

    all_gene_names = get_all_gene(args.gene_expression_path, args.generated_data_path)
    all_features, all_features_all_batch = get_attribute(args.gene_expression_path, args.generated_data_path)
    features_all_batch_after_removal, gene_names_after_removal = remove_low_gene(all_features, all_features_all_batch, all_gene_names, args.generated_data_path, thres=args.min_gene)
    if args.batch_correct:
        features_corrected = batch_correction(features_all_batch_after_removal, gene_names_after_removal, args.generated_data_path)
        normalized_features = normalization(features_corrected, args.generated_data_path)
    else:
        normalized_features = normalization(features_all_batch_after_removal, args.generated_data_path)
    remove_low_var_gene(normalized_features, gene_names_after_removal, args.generated_data_path, thres=args.min_gene_var)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--batch_correct', type=int, default=1, help='run batch correction or not')
    parser.add_argument( '--min_gene', type=float, default=1, help='Lowly expressed genes whose mean value is lower than this threshold will be filtered out')
    parser.add_argument( '--min_gene_var', type=float, default=0.4, help='Low variance genes whose variance value is lower than this threshold will be filtered out')
    parser.add_argument( '--gene_expression_path', type=str, default='dataset/MERFISH/pnas.1912459116.sd12.csv', help='file path for gene expression')
    parser.add_argument( '--spatial_location_path', type=str, default='dataset/MERFISH/pnas.1912459116.sd15.xlsx', help='file path for cell location')
    parser.add_argument( '--generated_data_path', type=str, default='generated_data/MERFISH/', help='The folder that generated data will be put in')
    args = parser.parse_args() 

    if not os.path.exists(args.generated_data_path):
        os.makedirs(args.generated_data_path) 
    main(args)
