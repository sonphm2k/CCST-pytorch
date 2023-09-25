import os
import pandas as pd
import scanpy as sc # pip install
import numpy as np
import stlearn # pip install
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import sparse
import pickle
####################  get the whole training dataset

def adata_preprocess(args, i_adata, min_cells=3, pca_n_comps=300):
    print('===== Preprocessing Data: ', args.data_name)
    sc.pp.filter_genes(i_adata, min_cells=min_cells)
    adata_X = sc.pp.normalize_total(i_adata, target_sum=1, exclude_highly_expressed=True, inplace=False)['X']
    adata_X = sc.pp.scale(adata_X)
    adata_X = sc.pp.pca(adata_X, n_comps=pca_n_comps)
    return adata_X

def get_ST_adj(generated_data_fold):
    coordinates = np.load(generated_data_fold + 'coordinates.npy')
    if not os.path.exists(generated_data_fold):
        os.makedirs(generated_data_fold) 
    ############# get batch adjacent matrix
    cell_num = len(coordinates)

    ############ the distribution of distance 
    if 1:
        distance_list = []
        print ('calculating distance matrix, it takes a while')
        for j in range(cell_num):
            for i in range (cell_num):
                if i!=j:
                    distance_list.append(np.linalg.norm(coordinates[j]-coordinates[i]))

        distance_array = np.array(distance_list)
        np.save(generated_data_fold + 'distance_array.npy', distance_array)
    else:
        distance_array = np.load(generated_data_fold + 'distance_array.npy')

    ###try different distance threshold, so that on average, each cell has x neighbor cells, see Tab. S1 for results

    for threshold in [300]:#range (210,211):#(100,400,40):
        # num_big = np.where(distance_array<threshold)[0].shape[0]
        # print (threshold,num_big,str(num_big/(cell_num*2))) #300 | 22064 | 2.9046866771985256
        from sklearn.metrics.pairwise import euclidean_distances
        distance_matrix = euclidean_distances(coordinates, coordinates)
        distance_matrix_threshold_I = np.zeros(distance_matrix.shape)
        distance_matrix_threshold_W = np.zeros(distance_matrix.shape)
        for i in range(distance_matrix_threshold_I.shape[0]):
            for j in range(distance_matrix_threshold_I.shape[1]):
                if distance_matrix[i,j] <= threshold and distance_matrix[i,j] > 0:
                    distance_matrix_threshold_I[i,j] = 1
                    distance_matrix_threshold_W[i,j] = distance_matrix[i,j]
        ############### get normalized sparse adjacent matrix
        distance_matrix_threshold_I_N = np.float32(distance_matrix_threshold_I) ## do not normalize adjcent matrix
        distance_matrix_threshold_I_N_crs = sparse.csr_matrix(distance_matrix_threshold_I_N)
        with open(generated_data_fold + 'Adjacent', 'wb') as fp:
            pickle.dump(distance_matrix_threshold_I_N_crs, fp)

def get_type(args, cell_types, generated_data_fold):
    types_dic = []
    types_idx = []
    for t in cell_types:
        if not t in types_dic:
            types_dic.append(t) 
        id = types_dic.index(t)
        types_idx.append(id)
    n_types = max(types_idx) + 1 # start from 0
    # For human breast cancer dataset, sort the cells for better visualization
    if args.data_name == 'V1_Breast_Cancer_Block_A_Section_1':
        types_dic_sorted = ['Healthy_1', 'Healthy_2',
                            'Tumor_edge_1', 'Tumor_edge_2', 'Tumor_edge_3', 'Tumor_edge_4', 'Tumor_edge_5', 'Tumor_edge_6',
                            'DCIS/LCIS_1', 'DCIS/LCIS_2', 'DCIS/LCIS_3', 'DCIS/LCIS_4', 'DCIS/LCIS_5',
                            'IDC_1', 'IDC_2', 'IDC_3', 'IDC_4', 'IDC_5', 'IDC_6', 'IDC_7']
    elif args.data_name == 'DLPFC':
        types_dic_sorted = ['Layer1', 'Layer2', 'Layer3', 'Layer4', 'Layer5', 'Layer6', 'WM', np.nan]
    else:
        np.save(generated_data_fold+'cell_types.npy', np.array(cell_types))
        np.savetxt(generated_data_fold+'types_dic.txt', np.array(types_dic), fmt='%s', delimiter='\t')
        return
    relabel_map = {}
    cell_types_relabel=[]
    for i in range(n_types):
        relabel_map[i]= types_dic_sorted.index(types_dic[i])
    for old_index in types_idx:
        cell_types_relabel.append(relabel_map[old_index])
    np.save(generated_data_fold+'cell_types.npy', np.array(cell_types_relabel))
    np.savetxt(generated_data_fold+'types_dic.txt', np.array(types_dic_sorted), fmt='%s', delimiter='\t')
    # create types_gt.txt -> valid pred and gt
    gt_data = []
    for index, cell_types in enumerate(cell_types_relabel):
        gt_data.append([index, cell_types])
    np.savetxt(generated_data_fold + '/types_gt.txt', np.array(gt_data), fmt='%3d', delimiter='\t')
    return

def draw_map(generated_data_fold, cmap):
    coordinates = np.load(generated_data_fold + 'coordinates.npy')
    cell_types = np.load(generated_data_fold+'cell_types.npy')
    n_types = max(cell_types) + 1 # start from 0

    types_dic = np.loadtxt(generated_data_fold+'types_dic.txt', dtype='|S15', delimiter='\t').tolist()
    for i,tmp in enumerate(types_dic):
        types_dic[i] = tmp.decode()
    print("Types_dict: ", types_dic)
    sc_cluster = plt.scatter(x=coordinates[:,0], y=-coordinates[:,1], s=5, c=cell_types, cmap= cmap)
    plt.legend(handles = sc_cluster.legend_elements(num=n_types)[0],labels=types_dic, bbox_to_anchor=(1,0.5), loc='center left', prop={'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.axis('scaled')
    plt.title('Annotation')
    plt.savefig(generated_data_fold+'/spacial.png', dpi=400, bbox_inches='tight') 
    plt.clf()

def main(args):
    data_fold = args.data_path+args.data_name+'/'
    generated_data_fold = args.generated_data_path + args.data_name+'/'
    if not os.path.exists(generated_data_fold):
        os.makedirs(generated_data_fold)
    df_meta = pd.read_csv(data_fold + 'metadata.tsv', sep='\t')
    if args.data_name == 'V1_Breast_Cancer_Block_A_Section_1':
        ## The cell_type are put in df_meta['fine_annot_type'] in V1_Breast_Cancer_Block_A_Section_1 dataset. This is labeled by SEDR
        cell_types = df_meta['fine_annot_type']  ## BRCA1 - 10x Visium spatial transcriptomics data of human breast cance # 3798 cell_types
        adata_h5 = stlearn.Read10X(path=data_fold, count_file=args.data_name+ '_filtered_feature_bc_matrix.h5')
        cmap = 'rainbow' # Draw annotation
    elif args.data_name == 'DLPFC':
        cell_types = df_meta['layer_guess'] ## DLPFC-151676 # 3460 cell_types ; 33538 features
        adata_h5 = stlearn.Read10X(path=data_fold, count_file= '151676_filtered_feature_bc_matrix.h5')
        cmap = 'viridis' # Draw annotation
    print(adata_h5)

    features = adata_preprocess(args, adata_h5, min_cells=args.min_cells, pca_n_comps=args.Dim_PCA)
    print("Features: ", features.shape)
    coordinates = adata_h5.obsm['spatial']

    np.save(generated_data_fold + 'features.npy', features)
    np.save(generated_data_fold + 'coordinates.npy', np.array(coordinates))

    get_ST_adj(generated_data_fold)
    get_type(args, cell_types, generated_data_fold)
    draw_map(generated_data_fold, cmap)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument( '--min_cells', type=float, default=5, help='Lowly expressed genes which appear in fewer than this number of cells will be filtered out')
    parser.add_argument( '--Dim_PCA', type=int, default=200, help='The output dimention of PCA')
    parser.add_argument( '--data_path', type=str, default='dataset/', help='The path to dataset')
    parser.add_argument( '--data_name', type=str, default='V1_Breast_Cancer_Block_A_Section_1', help=" 'V1_Breast_Cancer_Block_A_Section_1' or 'DLPFC' ")
    parser.add_argument( '--generated_data_path', type=str, default='generated_data/', help='The folder to store the generated data')
    args = parser.parse_args() 

    main(args)

# python data_generation_ST.py --data_name V1_Breast_Cancer_Block_A_Section_1 # 3798 × 36601 -> 3798 × 200 (Dim PCA)

# python data_generation_ST.py --data_name DLPFC # 3460 × 33538 -> 3460 x Dim_PCA