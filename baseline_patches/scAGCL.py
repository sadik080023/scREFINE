#scAGCL
import argparse
import random
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import Data
from model import Encoder, AGCLModel
from utils import load_h5_data1, load_h5_data2, GraphConstruction
from train import Train
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize



if __name__ == '__main__':

    def clustering_accuracy(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.int64)
        y_pred = np.asarray(y_pred, dtype=np.int64)

        assert y_true.size == y_pred.size

        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)

        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1

        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        return w[row_ind, col_ind].sum() / y_pred.size

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='Pollen.h5')
    parser.add_argument('--num_cluster', type=int, default=8)
    parser.add_argument('--lam', type=float, default=1.0)
    parser.add_argument('--subgraph_size', type=int, default=400)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--random_seed', type=int, default=12345)   
    parser.add_argument('--splits_npz', type=str, default=None, help='PATCH: scREFINE split file')
    parser.add_argument('--num_epochs', type=int, default=500)   
    parser.add_argument('--num_itersAdv', type=int, default=10)
    args = parser.parse_args()

    config = {
        'num_hidden': 256,
        'num_proj_hidden': 256,
        'num_layers': 2,
        'weight_decay': 0.00001,
        'alpha': 100,
        'beta' : 0.01,
        'edge_r1': 0.4,
        'edge_r2': 0.3,
        'feature_r1': 0.3,
        'feature_r2': 0.4,
        'tau': 0.5
    }
        
    num_hidden = config['num_hidden']
    num_proj_hidden =config['num_proj_hidden']
    num_layers = config['num_layers']
    weight_decay = config['weight_decay']
    alpha = config["alpha"] 
    beta = config["beta"] 
    edge_r1 = config['edge_r1']
    edge_r2 = config['edge_r2']
    feature_r1 = config['feature_r1']
    feature_r2 = config['feature_r2']
    tau=config['tau']

    data_file=args.data_file
    num_cluster=args.num_cluster
    lam = args.lam #adversarial weight
    subgraph_size=args.subgraph_size
    learning_rate=args.learning_rate

    num_epochs = args.num_epochs
    num_itersAdv=args.num_itersAdv
    random_seed=args.random_seed

    torch.manual_seed(random_seed)
    random.seed(random_seed) 
    np.random.seed(random_seed)

    # Read input data
    data_path='./data/' + data_file
    X, Y = load_h5_data1(data_path)   # h5 file has X and Y field
    #X, Y = load_h5_data2(data_path)   #h5 file in which X in "exprs" and Y in "obs" field

    # PATCH: align with scREFINE v2 splits (same cells; unsupervised training; test-only eval)
    idx_test = None
    if args.splits_npz:
        import numpy as _np
        sp = _np.load(args.splits_npz, allow_pickle=True)
        assert X.shape[0] == len(sp['y_full']), f"cell count mismatch: {X.shape[0]} vs {len(sp['y_full'])}"
        idx_test = sp['idx_test']
        print(f"PATCH: scREFINE splits loaded, test={len(idx_test)}")

    data = Data(x=torch.from_numpy(X))
    cellGraph=GraphConstruction(data, X, num_cluster)
    data.y=torch.tensor(Y, dtype=torch.int64)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    

    #Build a model
    encoder = Encoder(data.num_features, num_hidden, num_layers=num_layers).to(device)
    model = AGCLModel(encoder, num_hidden, num_proj_hidden, tau).to(device)
    #optimizer   
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    #Training
    embeddings=Train(model, data, cellGraph, device, num_epochs,lam, alpha, beta, num_itersAdv, optimizer, edge_r1, edge_r2, feature_r1, feature_r2, subgraph_size)
    #Cell Assigment and Evaluation
    Z_eval = embeddings.clone()
    Y=data.y
    Z = Z_eval.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    Z = normalize(Z, norm='l2')
    kmeans = KMeans(n_clusters=num_cluster, init="k-means++", random_state=random_seed)  # PATCH
    pred = kmeans.fit_predict(Z)

    # PATCH: test-only evaluation (same test cells as scREFINE)
    if idx_test is not None:
        Y_eval, pred_eval = Y[idx_test], pred[idx_test]
    else:
        Y_eval, pred_eval = Y, pred
    ari_score = adjusted_rand_score(Y_eval, pred_eval)
    nmi_score = normalized_mutual_info_score(Y_eval, pred_eval)
    ami_score = adjusted_mutual_info_score(Y_eval, pred_eval)
    acc_score = clustering_accuracy(Y_eval, pred_eval)

    print(
        'Final result: '
        + 'ARI= ' + str(ari_score)
        + ', NMI=' + str(nmi_score)
        + ', AMI=' + str(ami_score)
        + ', ACC=' + str(acc_score)
    )

    output_file = data_file + '_output.csv'
    with open("./result/" + output_file, "a") as f:
        print(
            "Final result is: "
            + 'ARI= ' + str(ari_score)
            + ', NMI=' + str(nmi_score)
            + ', AMI=' + str(ami_score)
            + ', ACC=' + str(acc_score),
            file=f
        )    
  
    y_pred_out = pd.DataFrame(pred)
    y_pred_out.to_csv("./result/" + data_file + "_labels.csv", sep='\t')