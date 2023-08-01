import torch
import sys
import torch.nn.functional as fn
import dgl
import wandb
from itertools import product

from dgl import DGLGraph
import dgl.function as dgl_fn
from dgl.nn import EGNNConv
import numpy as np
import scipy
import scipy.spatial
import scipy.stats
import argparse
from dgl.data import DGLDataset

from dgl import DGLGraph
import dgl.function as dgl_fn
from dgl.nn import EGNNConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter
from e3nn import o3
from e3nn.nn import BatchNorm
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import HeteroConv
import torch_geometric.transforms as T
from torch.distributions.normal import Normal
from scipy.spatial.transform import Rotation as R

sys.path.append('Transition1x')
from transition1x import Dataloader

def makebasis(dist):
    return scipy.stats.norm.pdf(dist,range(10),0.5)



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ns',type=int,default=32)
    parser.add_argument('--nv',type=int,default=8)
    parser.add_argument('--num_conv_layers',type=int,default=6)
    parser.add_argument('--clip',type=float,default=10)
    parser.add_argument('--lr',type=float,default=0.00001)
    parser.add_argument('--dropout',type=float,default=0)
    parser.add_argument('--weight_decay',type=float,default=0)
    parser.add_argument('--batch_norm',type=bool,default=False)
    parser.add_argument('--residual',type=bool,default=True)
    parser.add_argument('--model_weights',type=str,default='')
    parser.add_argument('--n_times',type=int,default=1)
    args = parser.parse_args()
    return args

class TransStateDataset(DGLDataset):
    """ Template for customizing graph datasets in DGL.

    Parameters
    ----------
    url : str
        URL to download the raw dataset
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    save_dir : str
        Directory to save the processed dataset.
        Default: the value of `raw_dir`
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose : bool
        Whether to print out progress information
    """
    def __init__(self,
                 h5_file_path,
                 datasplit="data",
                 verbose=False):
        super(TransStateDataset, self).__init__(name='transition_states',                                        
                                        verbose=verbose)
        loader = Dataloader(hdf5_file=h5_file_path, datasplit=datasplit, only_final=True)
        data = [x for x in loader]

        self.reactant_graphs, self.ts_graphs, self.product_graphs = [],[],[]
        for system_dict in data:
            reactant_graph = self.graph_from_datadict(system_dict['reactant'])
            ts_graph = self.graph_from_datadict(system_dict['transition_state'])
            product_graph = self.graph_from_datadict(system_dict['product'])

            self.reactant_graphs.append(reactant_graph)
            self.ts_graphs.append(ts_graph)
            self.product_graphs.append(product_graph)        

    def get_fc_edges(self, n_atoms):
        atom_idxs = list(range(n_atoms))
        edges = [  (src_idx, dst_idx) for src_idx, dst_idx in product(atom_idxs, atom_idxs) if src_idx != dst_idx ]
        src_idxs, dst_idxs = list(zip(*edges))
        return src_idxs, dst_idxs
    
    def graph_from_datadict(self, data_dict):

        atom_idx_map = {1:0,6:1,7:2,8:3}

        atomic_numbers = data_dict['atomic_numbers']
        mapped_atomic_numbers = list(map(lambda x: atom_idx_map[x], atomic_numbers))
        mapped_atomic_numbers = torch.tensor(mapped_atomic_numbers)

        # one-hot encode the mapped atomic numbers
        atom_types_onehot = fn.one_hot(mapped_atomic_numbers, num_classes=4)
        
        edges = self.get_fc_edges(len(atomic_numbers))
        g = dgl.graph(edges)

        g.ndata['x0'] = torch.tensor(data_dict['positions']).float()
        g.ndata['h0'] = atom_types_onehot
        
        with torch.no_grad():
            src,dst = g.all_edges()
            g.edata['dist'] = edgedist = torch.sqrt(torch.sum(torch.square(g.ndata['x0'][src]-g.ndata['x0'][dst]),axis=1)).float()
            g.edata['dfeat'] = torch.tensor(np.array([makebasis(d) for d in edgedist])).float()
        
        
        return g

    def process(self):
        # process raw data to graphs, labels, splitting masks
        pass

    def __getitem__(self, idx):
        # get one example by index
        return self.reactant_graphs[idx], self.ts_graphs[idx], self.product_graphs[idx]

    def __len__(self):
        # number of data examples
        return len(self.reactant_graphs)
    
class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features,dropout=0, residual=True, batch_norm=True):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        self.dropout = torch.nn.Dropout(dropout)
        self.doubleleakyrelu=DoubleleakyRelu.apply
        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.ReLU(),
            nn.Linear(n_edge_features, tp.weight_numel),
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='sum',scatter_message=True):

        edge_src, edge_dst = edge_index
        if type(node_attr) is tuple:
            tp = self.tp(node_attr[0][edge_src], edge_sh, self.dropout(self.doubleleakyrelu(self.fc(edge_attr))))
        else:
            tp = self.tp(node_attr[edge_src], edge_sh, self.dropout(self.doubleleakyrelu(self.fc(edge_attr))))
        if scatter_message:
            out = scatter(tp, edge_dst, dim=0, dim_size=out_nodes, reduce=reduce)
        else:
            out=tp
        if self.residual:
            if type(node_attr) is tuple:
                padded = F.pad(node_attr[1], (0, out.shape[-1] - node_attr[1].shape[-1]))
            else:
                padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)

        return out
    
class DoubleleakyRelu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return torch.where(torch.abs(input)<=10.0,input,0.01*input)
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.abs(input) > 10.0] *= 0.01
        return grad_input

class Se3NN(torch.nn.Module):
    def __init__(self, num_node_features, ns=32,nv=8, n_layers=6,sh_lmax=2,dropout= 0,batch_norm=False,residual=True,n_times=1):
        super().__init__()
        self.ns=ns
        self.nv=nv
        self.n_times=n_times
        self.dropout=torch.nn.Dropout(dropout)
        self.embedding = torch.nn.Linear(num_node_features, ns)
        self.eembedding = torch.nn.Linear(10, ns)
        self.reembedding = torch.nn.Linear(10, ns)
        self.peembedding = torch.nn.Linear(10, ns)
        self.thresh=nn.Threshold(10,10)
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.doubleleakyrelu=DoubleleakyRelu.apply
        self.dist_basis=[Normal(i,0.5) for i in range(10)]
        if sh_lmax==2:
             self.irrep_seq = [
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o + {nv}x2e',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o',
                f'{ns}x0e + {nv}x1o + {nv}x2e + {nv}x1e + {nv}x2o + {ns}x0o',
                f'{ns}x0e'
            ]
        else:
            self.irrep_seq=[
                f'{ns}x0e',
                f'{ns}x0e + {nv}x1o',
                f'{ns}x0e + {nv}x1o + {nv}x1e',
                f'{ns}x0e + {nv}x1o + {nv}x1e + {ns}x0o',
                f'{ns}x0e'
            ]
        conv_layers = []
        for i in range(n_layers):
            in_irreps = self.irrep_seq[min(i, len(self.irrep_seq) - 1)]
            out_irreps = self.irrep_seq[min(i + 1, len(self.irrep_seq) - 1)]
            layer=TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=out_irreps,
                n_edge_features=5 * ns,
                dropout=dropout,
                residual=residual,
                batch_norm=batch_norm)
            conv_layers.append(layer)
        self.conv_layers = torch.nn.ModuleList(conv_layers)

        coord_conv_layers=[]
        for i in range(n_layers):
            in_irreps = self.irrep_seq[min(i+1, len(self.irrep_seq) - 1)]
            layer=TensorProductConvLayer(
                in_irreps=in_irreps,
                sh_irreps=self.sh_irreps,
                out_irreps=f'{1}x1o',
                n_edge_features=5 * ns,
                dropout=dropout,
                residual=False,
                batch_norm=False)
            coord_conv_layers.append(layer)
        self.coord_conv_layers = torch.nn.ModuleList(coord_conv_layers)
        self.elinear = torch.nn.Linear(ns*2,ns)
        self.elinear2 = torch.nn.Linear(ns,ns)
        self.finale = torch.nn.Linear(ns,1)

    def forward(self, g,product):
        node_features = fn.elu(self.embedding(g.ndata['h0']))
        edge_features = fn.elu(self.eembedding(g.edata['dfeat']))
        reactant_edge_features=fn.elu(self.reembedding(g.edata['dfeat']))
        coord_features = g.ndata['x0']
        src,dst = g.edges()
        product_edge_features=fn.elu(self.peembedding(product.edata['dfeat']))
        edge_index=g.edges()
        edge_vectors=coord_features[src]-coord_features[dst]
        edge_sh=o3.spherical_harmonics(self.sh_irreps, edge_vectors, normalize=True, normalization='component')
        for t in range(self.n_times):
            if t>0:
                node_features=node_features[:,:self.ns]
                node_features=fn.elu(node_features)
            for conv_layer,coord_conv_layer in zip(self.conv_layers,self.coord_conv_layers):
                node_edge_features=torch.cat((node_features[src,:self.ns],node_features[dst,:self.ns],edge_features,reactant_edge_features,product_edge_features),dim=1)
                node_features= conv_layer(node_features,edge_index,node_edge_features,edge_sh)
                node_edge_features=torch.cat((node_features[src,:self.ns],node_features[dst,:self.ns],edge_features,reactant_edge_features,product_edge_features),dim=1)
                coord_updates=coord_conv_layer(node_features,edge_index,node_edge_features,edge_sh,scatter_message=True)
                #coord_updates=self.thresh(coord_updates)
                coord_updates= self.doubleleakyrelu(coord_updates)
                #coord_updates=-coord_updates
                #coord_updates=self.thresh(coord_updates)
                #coord_updates=-coord_updates
                #print(coord_features.shape,coord_updates.shape)
                coord_features=coord_features-torch.mean(coord_features,axis=0)+coord_updates
                edge_vectors=coord_features[src]-coord_features[dst]
                #vector_signs=torch.sign(edge_vectors)
                #edge_vectors=torch.abs(edge_vectors)
                #edge_vectors=vector_signs*edge_vectors.clamp_(min=1e-6)
                edge_sh=o3.spherical_harmonics(self.sh_irreps, edge_vectors, normalize=True, normalization='component')
                edge_dist=torch.linalg.norm(edge_vectors,axis=1,ord=2).float()
                #edge_dist=torch.sqrt(torch.sum(torch.square(coord_features[src]-coord_features[dst]),axis=1)).float()
                edge_basis=torch.stack([self.dist_basis[i].log_prob(edge_dist).exp() for i in range(10)]).float().T
                edge_features=fn.elu(self.eembedding(edge_basis))
        edge_features=torch.cat((node_features[src,:self.ns],node_features[dst,:self.ns]),dim=1)
        edge_features=self.dropout(fn.elu(self.elinear(edge_features)))
        edge_features=self.dropout(fn.elu(self.elinear2(edge_features)))
        edge_features=fn.elu(self.finale(edge_features))
        node_features=fn.elu(node_features)
        return node_features,coord_features,edge_features
    


def align_scipy(coord1, coord2):
    coord1=coord1.detach().cpu().numpy()
    coord2=coord2.detach().cpu().numpy()
    center1 = np.mean(coord1, axis=0)
    center2 = np.mean(coord2, axis=0)
    coord1_centered = coord1 - center1
    coord2_centered = coord2 - center2
    rotation,rssd=R.align_vectors(coord1_centered,coord2_centered)
    rmse=np.sqrt((rssd*rssd)/coord1.shape[0])
    return rmse

def align_mse(coord1, coord2):
    center1 = torch.mean(coord1, dim=0)
    center2 = torch.mean(coord2, dim=0)
    coord1_centered = coord1 - center1
    coord2_centered = coord2 - center2

    # Calculating the covariance matrix
    covariance = torch.matmul(coord1_centered.T, coord2_centered)

    # Performing Singular Value Decomposition (SVD)
    u, _, vt = torch.svd(covariance)

    # Calculating the rotation matrix
    rotation = torch.matmul(vt.T, u.T)

    if torch.linalg.det(rotation) < 0:
        vt[:, -1] *= -1
        rotation = torch.matmul(vt.T, u.T)

    # Aligning coord2 to coord1
    coord2_aligned = torch.matmul(coord2_centered, rotation)

    # Calculating MSE
    mse = torch.mean(torch.square(coord1_centered - coord2_aligned).sum(axis=1))

    return mse

def evaluate_model(model,eval_dataloader):
    model.eval()
    with torch.no_grad():
        total_loss=0
        rmse=0
        aligned_rmse=0
        for reactants, ts, products in eval_dataloader:
            reactants.ndata['h0'] = reactants.ndata['h0'].float()
            reactants = reactants.to('cuda')
            ts = ts.to('cuda')
            products = products.to('cuda')
            # products = products.to('cuda')

            #labels=ts.ndata['x0']
            pred_nodes, pred_coords,_ = model(reactants,products)
            #loss = fn.l1_loss(pred_coords, labels)
            src,dst = reactants.edges()
            true_edist = ts.edata['dist']
            pred_edist = torch.sum(torch.square(pred_coords[src]-pred_coords[dst]),axis=1).float()
            true_edist_2 = true_edist*true_edist
            loss = fn.l1_loss(pred_edist, true_edist_2)
            total_loss += loss.item()
            rmse += fn.l1_loss(torch.sqrt(pred_edist),true_edist)
            prev_batch_idx = 0
            next_batch_idx = 0
            for batch_idx in reactants.batch_num_nodes():
                next_batch_idx += batch_idx.item()
                pred_coords_align = pred_coords[prev_batch_idx:next_batch_idx,:]
                ts_coords=ts.ndata['x0'][prev_batch_idx:next_batch_idx,:]
                prev_batch_idx = next_batch_idx
                aligned_rmse += align_scipy(ts_coords,pred_coords_align)
    aligned_rmse = aligned_rmse/len(eval_dataloader.dataset)
    rmse = rmse/len(eval_dataloader)
    wandb.log({'eval_loss': total_loss/len(eval_dataloader), 'eval_rmse': rmse, 'eval_coord_rmse': aligned_rmse})
    return total_loss/len(eval_dataloader),rmse,aligned_rmse

if __name__=='__main__':

    args = parse_arguments()
    wandb.init(project='transition_state',name='')
    wandb.config.update(args)
    gnn=Se3NN(num_node_features=4,ns=args.ns,nv=args.nv,n_layers=args.num_conv_layers,dropout=args.dropout,batch_norm=args.batch_norm,residual=args.residual,sh_lmax=2,n_times=args.n_times).to('cuda')
    if args.model_weights!='':
        gnn.load_state_dict(torch.load(args.model_weights))
    traindataset = TransStateDataset('Transition1x/data/transition1x.h5','train')
    testdataset = TransStateDataset('Transition1x/data/transition1x.h5','test')
    loader = Dataloader(hdf5_file='Transition1x/data/transition1x.h5', datasplit='data', only_final=True)
    lr = args.lr
    opt = torch.optim.Adam(gnn.parameters(), lr=lr, weight_decay=args.weight_decay)
    train_dataloader = dgl.dataloading.GraphDataLoader(traindataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = dgl.dataloading.GraphDataLoader(testdataset, batch_size=args.batch_size, shuffle=True)
    edge_loss = 1
    batches_per_epoch = len(train_dataloader)
    train_metrics_marker = 0.0
    train_metrics_interval = 5
    train_losses = []
    test_losses = []
    test_idx = []
    best_test_rmse=100
    losses = []
    n_epochs = 3000
    idx = 0
    test_loss = 0

    first = []

    for epoch in range(n_epochs):

        for batch_idx, (reactants, ts, products) in enumerate(train_dataloader):


            reactants.ndata['h0'] = reactants.ndata['h0'].float()
            reactants = reactants.to('cuda')
            ts = ts.to('cuda')
            products = products.to('cuda')
            # products = products.to('cuda')

            epoch_exact = epoch + batch_idx/batches_per_epoch

            gnn.train()
            opt.zero_grad()
            pred_nodes, pred_coords, edge_feat = gnn(reactants,products)

            #first.append(pred_coords[0][0].item())
            if edge_loss == 2:
                loss = fn.l1_loss(edge_feat.squeeze().flatten(), ts.edata['dist'].flatten())
                print(edge_feat[0][0].item(),ts.edata['dist'][0].item())
            elif edge_loss == 1: #use coordinates to get edge distance
                src,dst = reactants.edges()
                true_edist = ts.edata['dist']
                pred_edist = torch.sum(torch.square(pred_coords[src]-pred_coords[dst]),axis=1).float()
                true_edist_2 = true_edist**2
                loss = fn.l1_loss(pred_edist, true_edist_2)
                wandb.log({'train_edge_loss': loss.item()})
                pred_edist = torch.sqrt(pred_edist)
                rmse = fn.l1_loss(pred_edist,true_edist)
                wandb.log({'train_rmse': rmse.item()})
            else:
                loss = fn.l1_loss(pred_coords, ts.ndata['x0'])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gnn.parameters(), args.clip)
            opt.step()

            # record current train loss
            train_losses.append(loss.item())
            losses.append(loss.item())
            idx += 1


            # wandb.log({
            #     'Train Loss':loss.item(),
            #     })

            if epoch_exact - train_metrics_marker >= train_metrics_interval:
                train_metrics_marker = epoch_exact
                print(f'Epoch = {epoch_exact:.4f}')
                print(f'train loss = {np.mean(train_losses):.4f}')
                wandb.log({'train_epoch_loss': loss.item(), 'epoch': epoch_exact})
                train_losses = []

        test_loss,rmse,aligned_rmse = evaluate_model(gnn,test_dataloader)
        if aligned_rmse<best_test_rmse:
            best_test_rmse=aligned_rmse
            torch.save(gnn.state_dict(), wandb.run.dir+'/best_model.pt')
            wandb.log({'best_test_coord_rmse': best_test_rmse})
            wandb.run.summary["best_test_coord_rmse"] = best_test_rmse
        test_losses.append(test_loss)
        print(f'test loss = {test_loss:.4f}')




