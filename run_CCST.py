##exocrine GCNG with normalized graph matrix 
import os
from torch_geometric.nn import GCNConv, ChebConv, GATConv, DeepGraphInfomax, global_mean_pool, global_max_pool  # noqa

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # ================Specify data type firstly===============
    parser.add_argument( '--data_type', type=str, default='sc', help='"sc" or "nsc", refers to single cell resolution datasets(e.g. MERFISH, seqFISH) and non single cell resolution data(e.g. ST) respectively')
    # =========================== args ===============================
    parser.add_argument( '--data_name', type=str, default='V1_Breast_Cancer_Block_A_Section_1', help=" 'MERFISH', 'seqFISH', 'V1_Breast_Cancer_Block_A_Section_1' or 'DLPFC' ")
    parser.add_argument('--cell_type', type=str, default='Interneuron', help='seqFISH datasets: 11 type_cell of OB mouse')
    parser.add_argument( '--lambda_I', type=float, default=0.3) #0.8 on MERFISH, 0.3 on ST
    parser.add_argument( '--DGI', type=int, default=1, help='run Deep Graph Infomax(DGI) model, otherwise direct load embeddings')
    parser.add_argument( '--load', type=int, default=0, help='Load pretrained DGI model')
    parser.add_argument( '--num_epoch', type=int, default=5000, help='numebr of epoch in training DGI')
    parser.add_argument( '--hidden', type=int, default=256, help='hidden channels in DGI')
    parser.add_argument( '--cluster', type=int, default=1, help='run cluster or not')
    parser.add_argument('--PCA', type=int, default=1, help='run PCA or not')
    parser.add_argument( '--n_clusters', type=int, default=5, help='number of clusters in Kmeans, when ground truth label is not avalible.') #5 on MERFISH, 20 on Breast
    parser.add_argument( '--draw_map', type=int, default=1, help='run drawing map')
    parser.add_argument( '--diff_gene', type=int, default=0, help='Run differential gene expression analysis')
    # =========================== path ===============================
    parser.add_argument( '--data_path', type=str, default='generated_data/', help='data path')
    parser.add_argument( '--model_path', type=str, default='model')
    parser.add_argument( '--embedding_data_path', type=str, default='Embedding_data')
    parser.add_argument( '--result_path', type=str, default='results')
    args = parser.parse_args()

    args.model_path = args.model_path + '/' + args.data_name + '/'
    args.embedding_data_path = args.embedding_data_path +'/'+ args.data_name +'/'
    args.result_path = args.result_path +'/'+ args.data_name +'/' + 'lambdaI' + str(args.lambda_I) + '/'


    if not os.path.exists(args.embedding_data_path):
        os.makedirs(args.embedding_data_path) 
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    print ('------------------------Model and Training Details--------------------------')
    print(args)

    if args.data_type == 'sc': # should input a single cell resolution dataset, e.g. MERFISH
        if args.data_name == 'MERFISH':
            from CCST_merfish_utils import CCST_on_MERFISH
            CCST_on_MERFISH(args) #-> CCST_merfish_util.py
        elif args.data_name == 'seqFISH':
            from CCST_seqFISH_utils import CCST_on_seqFISH
            args.model_path = args.model_path + '/' + args.cell_type + '/'
            args.embedding_data_path = args.embedding_data_path + '/' + args.cell_type + '/'
            args.result_path = args.result_path + '/' + args.cell_type + '/'
            if not os.path.exists(args.embedding_data_path):
                os.makedirs(args.embedding_data_path)
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            if not os.path.exists(args.result_path):
                os.makedirs(args.result_path)
            CCST_on_seqFISH(args)
    elif args.data_type == 'nsc': # should input a non-single cell resolution dataset, e.g. V1_Breast_Cancer_Block_A_Section_1
        from CCST_ST_utils import CCST_on_ST
        CCST_on_ST(args)
    else:
        print('Data type not specified')

# # # MERFISH
# # train 5000 epoch
# python run_CCST.py --data_type sc --data_name MERFISH --lambda_I 0.8 --DGI 1 --load 0 --num_epoch 5000 --cluster 0 --draw_map 0
# # load + cluster -> PCA -> K_means with n_clusters == 5
# python run_CCST.py --data_type sc --data_name MERFISH --lambda_I 0.8 --DGI 1 --load 1 --num_epoch 5000 --cluster 1  --PCA 30 --n_clusters 5 --draw_map 1

# # # seqFISH
# # train 2000 epoch
# python run_CCST.py --data_type sc --data_name seqFISH --cell_type Interneuron --lambda_I 0.8 --DGI 1 --load 0 --num_epoch 1000 --cluster 0 --draw_map 0
# python run_CCST.py --data_type sc --data_name seqFISH --cell_type Neural_Stem --lambda_I 0.8 --DGI 1 --load 0 --num_epoch 1000 --cluster 0 --draw_map 0
# # load + cluster -> PCA -> K_means with n_clusters == 2
# python run_CCST.py --data_type sc --data_name seqFISH --cell_type Interneuron --lambda_I 0.8 --DGI 1 --load 1 --num_epoch 1000 --cluster 1  --PCA 30 --n_clusters 2 --draw_map 1
# python run_CCST.py --data_type sc --data_name seqFISH --cell_type Neural_Stem --lambda_I 0.8 --DGI 1 --load 1 --num_epoch 1000 --cluster 1  --PCA 30 --n_clusters 2 --draw_map 1

# # # DLPFC
# # train 5000 epoch
# python run_CCST.py --data_type nsc --data_name DLPFC --lambda_I 0.3 --DGI 1 --load 0 --num_epoch 5000 --cluster 0 --draw_map 0
# # load + cluster -> PCA -> K_means with n_clusters == 8
# python run_CCST.py --data_type nsc --data_name DLPFC --lambda_I 0.3 --DGI 1 --load 1 --num_epoch 5000 --cluster 1  --PCA 30 --n_clusters 8 --draw_map 1

# # # V1_Breast_Cancer_Block_A_Section_1
# # train 5000 epoch
# python run_CCST.py --data_type nsc --data_name V1_Breast_Cancer_Block_A_Section_1 --lambda_I 0.3 --DGI 1 --load 0 --num_epoch 5000 --cluster 0 --draw_map 0
# # load + cluster -> PCA -> K_means with n_clusters == 20
# python run_CCST.py --data_type nsc --data_name V1_Breast_Cancer_Block_A_Section_1 --lambda_I 0.3 --DGI 1 --load 1 --num_epoch 5000 --cluster 1  --PCA 30 --n_clusters 20 --draw_map 1
# python run_CCST.py --data_type nsc --data_name V1_Breast_Cancer_Block_A_Section_1 --lambda_I 0.3 --DGI 0 --load 1 --num_epoch 5000 --cluster 1  --PCA 30 --n_clusters 20 --draw_map 1
