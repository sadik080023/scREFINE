#aae_pytorch_basic.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# scgpt+scsemiaae
import argparse
from datetime import datetime
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.nn.parameter import Parameter
from layers import ZINBLoss, MeanAct, DispAct
from filter import load_data
from utils import cluster_acc
from sklearn.mixture import GaussianMixture   

# Top-level argument parsing disabled; arguments are parsed in the main block.
cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}   # GPU的一些设置


class SAAE_net(nn.Module):
    def __init__(self,X_dim, N,M,z_dim,n_clusters,sigma=1.,gamma=1.):
        super().__init__()
        self.sigma = sigma
        self.gamma = gamma
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, z_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)        
       
        self.Q_net = nn.Sequential( 
            nn.Linear(X_dim, N),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(N, M),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(M, z_dim)  
        )
        
        self.P_net = nn.Sequential(                           
            nn.Linear(z_dim, M),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(M, N)
        )
       
        self.out_net = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(N, N),
            nn.Sigmoid(),             
        )
        self._lin_mean = nn.Sequential(nn.Linear(N,X_dim), MeanAct())
        self._lin_disp = nn.Sequential(nn.Linear(N,X_dim), DispAct())
        self._lin_pi = nn.Sequential(nn.Linear(N,X_dim), nn.Sigmoid())
        # degree
        self.v = 1
        self.zinb_loss = ZINBLoss()
        self.out_y = nn.Linear(z_dim,n_clusters)
        self.D_net_gaus = nn.Sequential(
            nn.Linear(z_dim, N), 
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(N, M),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(M, 1),
            nn.Sigmoid()
        )
        self.n_clusters = n_clusters
        self.loss_ce = nn.CrossEntropyLoss()
    def gen(self,z):
        z = z.to(torch.float32)
        return self.D_net_gaus(z)
    def forward(self,x):
        x  = x.float()
        z_noise  = self.Q_net((x+torch.randn_like(x) * self.sigma).float())
        z_noise_2 = self.P_net(z_noise.float())
        h = self.out_net(z_noise_2)
        mean = self._lin_mean(h)
        disp = self._lin_disp(h)
        pi = self._lin_pi(h)      
        y_hat=F.softmax(self.out_y(z_noise),dim=1)#训练使用带噪声的z,预测使用不带噪声的z
        latent = torch.cat((z_noise,y_hat),dim=1)#cat函数 将z和y按行样本合并，共同输入解码器 
         
        z = self.Q_net(x.float())
        return z,mean,disp,pi,z_noise,y_hat,latent

    #对于模型进行预训练
    def pretrian(self,x, X_raw, size_factor,epoch =100,batch_size=128,lr=0.001):    
        self.train()
        time = datetime.now().strftime('%Y%m%d')
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(X_raw), torch.Tensor(size_factor))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
        for epoch in range(epoch):
            for batch_idx, (x_batch, x_raw_batch, sf_batch) in enumerate(dataloader):
                x_tensor = Variable(x_batch)
                x_raw_tensor = Variable(x_raw_batch)
                sf_tensor = Variable(sf_batch)
                _, mean_tensor, disp_tensor, pi_tensor,_,_ ,_= self.forward(x_tensor)
                loss = self.zinb_loss(x=x_raw_tensor, mean=mean_tensor, disp=disp_tensor, pi=pi_tensor, scale_factor=sf_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print('Pretrain epoch [{}/{}], ZINB loss:{:.4f}'.format(batch_idx+1, epoch+1, loss.item()))
        torch.save({
            'ae_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        },'./预训练模型/'+time+'.pth.tar')
    def encodeBatch(self, X, batch_size=128):
        # Keep model and tensors on the same device. CPU mode is safer for this legacy code.
        use_cuda = False
        encoded = []
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        for batch_idx in range(num_batch):
            xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]
            inputs = Variable(xbatch)
            z= self.Q_net(inputs.float())
            encoded.append(z.data)

        encoded = torch.cat(encoded, dim=0)
        return encoded
    def save_checkpoint(self, state, index, filename):
        newfilename = os.path.join(filename, 'FTcheckpoint_%d.pth.tar' % index)
        torch.save(state, newfilename)
   
     #对模型进行训练
    def fit(self,x,y,X_raw, sf,lr=0.01,lr_d=0.0001 ,batch_size=128, num_epochs=100,save_dir="", use_scgpt=False, labeled_mask=None, seed=0, idx_test=None):#PATCH
        #y-=1
        Y = torch.tensor(y).long()
        if labeled_mask is None:
            labeled_mask_t = torch.ones(len(y), dtype=torch.bool)
        else:
            labeled_mask_t = torch.tensor(labeled_mask).bool()
        TINY = 1e-15
        X = torch.tensor(x)
        X_raw = torch.tensor(X_raw)
        sf = torch.tensor(sf)
        optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)
        #sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=0.001, last_epoch=-1, verbose=False)#学习率衰减
        optimizer_D = optim.Adam(self.D_net_gaus.parameters(),lr=lr_d)
        kmeans = KMeans(self.n_clusters, n_init=20, random_state=seed)  # PATCH
        data = self.encodeBatch(X)
        self.y_pred = kmeans.fit_predict(data.data.cpu().numpy())
       
        yt = y[idx_test] if idx_test is not None else y  # PATCH: test-only eval
        pt = self.y_pred[idx_test] if idx_test is not None else self.y_pred
        acc = np.round(cluster_acc(yt, pt), 5)
        nmi = np.round(metrics.normalized_mutual_info_score(yt, pt), 5)
        ari = np.round(metrics.adjusted_rand_score(yt, pt), 5)
        print('Initializing k-means: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
        
        num = X.shape[0]
        num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
        np.random.seed(seed)  # PATCH
        
        for epoch in range(num_epochs):
            self.eval()
            latent = self.encodeBatch(X)
            y_ = F.softmax(latent,dim=1).data.cpu().numpy()
            newlatent=np.concatenate((latent,y_), axis=1)
            newlatent = np.nan_to_num(newlatent, nan=0.0, posinf=1e4, neginf=-1e4)
            self.final_latent =newlatent
            self.y_pred =GaussianMixture(n_components=n_clusters, random_state=seed).fit_predict(newlatent)#PATCH
            yt = y[idx_test] if idx_test is not None else y  # PATCH: test-only eval
            pt = self.y_pred[idx_test] if idx_test is not None else self.y_pred
            acc = np.round(cluster_acc(yt, pt), 5)
            nmi = np.round(metrics.normalized_mutual_info_score (yt, pt), 5)
            ari = np.round(metrics.adjusted_rand_score(yt, pt), 5)
            print('Clustering   %d: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (epoch+1, acc, nmi, ari))
          

            train_loss = 0.0
            recon_loss_val = 0.0
            self.train()
            for batch_idx in range(num_batch):
                start = batch_idx * batch_size
                end = min((batch_idx + 1) * batch_size, num)

                xbatch = X[start:end]
                xrawbatch = X_raw[start:end]
                sfbatch = sf[start:end]
                y_batch = Y[start:end]
                mask_batch = labeled_mask_t[start:end]

                # Use true exported 5% labeled mask for CE supervision.
                xbatch_use = xbatch[mask_batch]
                y_use = y_batch[mask_batch]

                # Use all cells for embedding-space reconstruction/adversarial training in scGPT mode.
                xbatch_unlabel = xbatch
                xrawbatch = xrawbatch
                sfbatch = sfbatch

                optimizer.zero_grad()
                inputs = Variable(xbatch_unlabel)
                inputs_label = Variable(xbatch_use)
                rawinputs = Variable(xrawbatch)
                sfinputs = Variable(sfbatch)
        
                
                z, meanbatch, dispbatch, pibatch,z_noise,_,_= self.forward(inputs)
                if use_scgpt:
                    # For dense scGPT embeddings, count-based ZINB is not appropriate.
                    # Use embedding-space reconstruction loss instead.
                    meanbatch = torch.nan_to_num(meanbatch, nan=0.0, posinf=1e4, neginf=-1e4)
                    inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e4, neginf=-1e4)
                    zinb_loss = F.mse_loss(meanbatch.float(), inputs.float())
                else:
                    zinb_loss = self.zinb_loss(rawinputs.float(), meanbatch, dispbatch, pibatch, sfinputs)
                if inputs_label.shape[0] > 0:
                    _,_,_,_,_,y_hat_use,_=self.forward(inputs_label)
                    ce_loss = self.loss_ce(y_hat_use, y_use)
                else:
                    ce_loss = zinb_loss * 0.0
                recon_loss = zinb_loss + self.gamma * ce_loss
                recon_loss.backward()                                                                            
                optimizer.step()
                #sch.step()
                
                if batch_idx % 5 == 0:
                    z = self.Q_net(inputs.float()) 
                    z2 = z.detach().numpy()
                    z_real_gauss = Variable(torch.from_numpy(np.random.normal(float(z2.mean()),float(z2.std()),size = (z.shape[0], z_dim))))
                    D_real_gauss = self.gen(z_real_gauss)
                
                    z, meanbatch, dispbatch, pibatch,z_noise,_,_ = self.forward(inputs)
                    D_fake_gauss = self.gen(z_noise)
                    # D_loss = -(torch.mean(torch.log((D_real_gauss + TINY-1)**2) + torch.log((D_fake_gauss + TINY)**2)))
                    # G_loss = -torch.mean(torch.log((D_fake_gauss + TINY-1)**2))
                    D_loss = -torch.mean(torch.log(D_real_gauss + TINY)) -torch.mean(torch.log((1-D_fake_gauss + TINY)))
                    optimizer_D.zero_grad()
                    D_loss.backward()
                    optimizer_D.step()
                
                z,  meanbatch, dispbatch, pibatch,z_noise,_,_ = self.forward(inputs)
                D_fake_gauss = self.gen(z_noise)
                G_loss = -torch.mean(torch.log(D_fake_gauss + TINY))
                G_loss.backward()
                optimizer.step()
           

                recon_loss_val += recon_loss.data * len(inputs)

            print("#Epoch %3d: Recon_loss_val: %.4f,Recon Loss: %.4f D_loss: %.4f,G_loss:%.4f" % (
                epoch + 1, recon_loss_val / num, zinb_loss,D_loss,G_loss))
        self.save_checkpoint({'epoch': epoch+1,
                'state_dict': self.state_dict(),
                'y_pred': self.y_pred,
                'y': y
                }, epoch+1, filename=save_dir)
        return self.y_pred, acc, nmi, ari
if __name__ == '__main__':
 # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', default='data/Brain.h5', type=str)
    parser.add_argument('--use_scgpt', action='store_true', help='Use precomputed scGPT embedding h5 and skip raw-count preprocessing')
    parser.add_argument('--splits_npz', type=str, default=None, help='PATCH: scREFINE split file')
    parser.add_argument('--seed', type=int, default=42, help='PATCH: random seed')
    parser.add_argument('--label_cells', default=0.2, type=float)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--fit_epochs', default=100, type=int)
    parser.add_argument('--gamma', default=1., type=float,
                        help='coefficient of cross entropy loss')
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--z_dim', default=128, type=int,
                        help='The number of neurons in the inner layer of the encoder')
    parser.add_argument('--M', default=256, type=int,
                        help='The number of neurons in the outer layer of the encoder')
    parser.add_argument('--N', default=512, type=int,
                        help='The number of neurons in the outest layer of the encoder')
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/scSemiAAE_p0_1/')
    parser.add_argument('--ae_weight_file', default='AE_weights_p0_1.pth.tar')

    args = parser.parse_args()

    import random as _random
    torch.manual_seed(args.seed); np.random.seed(args.seed); _random.seed(args.seed)  # PATCH

    x, y, raw_x, sf, labeled_mask = load_data(args.data_path, use_scgpt=args.use_scgpt)

    # PATCH: exact scREFINE v2 splits — same labeled cells, test-only evaluation
    idx_test = None; y_eval = y
    if args.splits_npz:
        import numpy as _np
        sp = _np.load(args.splits_npz, allow_pickle=True)
        assert x.shape[0] == len(sp['y_full']), f"cell count mismatch: {x.shape[0]} vs {len(sp['y_full'])}"
        y = sp['y_full'].astype(int); y_eval = y
        labeled_mask = _np.zeros(len(y), dtype=bool)
        labeled_mask[sp['idx_train_labeled']] = True
        idx_test = sp['idx_test']
        print(f"PATCH: labeled={labeled_mask.sum()}, test={len(idx_test)} (scREFINE splits)")

    n_clusters = len(set(y_eval))
    z_dim = args.z_dim

    os.makedirs('./预训练模型', exist_ok=True)
    os.makedirs('./数据模型', exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    saae = SAAE_net(
        X_dim=x.shape[1],
        N=args.N,
        M=args.M,
        z_dim=args.z_dim,
        n_clusters=n_clusters,
        gamma=args.gamma
    )

    if args.use_scgpt or args.pretrain_epochs <= 0:
        print('Skipping ZINB pretraining for scGPT embedding input')
    else:
        print('Pretraining model')
        saae.pretrian(
            x,
            raw_x,
            sf,
            epoch=args.pretrain_epochs,
            batch_size=args.batch_size
        )

    print('Training model')
    y_pred, acc, nmi, ari = saae.fit(
        x,
        y,
        raw_x,
        sf,
        batch_size=args.batch_size,
        num_epochs=args.fit_epochs,
        save_dir='./数据模型',
        use_scgpt=args.use_scgpt,     # PATCH
        labeled_mask=labeled_mask,    # PATCH: only scREFINE's 5% labeled cells supervise
        seed=args.seed,               # PATCH
        idx_test=idx_test             # PATCH: test-only evaluation
    )

    from sklearn.metrics import adjusted_mutual_info_score
    ami = adjusted_mutual_info_score(y_eval[idx_test], y_pred[idx_test])  # PATCH
    acc = cluster_acc(y_eval[idx_test], y_pred[idx_test])
    nmi = metrics.normalized_mutual_info_score(y_eval[idx_test], y_pred[idx_test])
    ari = metrics.adjusted_rand_score(y_eval[idx_test], y_pred[idx_test])
    print('Final result: ACC= %.4f, NMI= %.4f, AMI= %.4f, ARI= %.4f' % (acc, nmi, ami, ari))


   