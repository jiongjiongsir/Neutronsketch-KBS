import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN_NeighSampler(torch.nn.Module):
    def __init__(self
                 ,device, subgraph_loader
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True):
        super(GCN_NeighSampler, self).__init__()

        self.layer_loader = subgraph_loader
        self.device = device
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False,normalize=False))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        self.num_layers = num_layers
        # self.normalize = False
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False,normalize=False))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=False,normalize=False))

        self.dropout = dropout
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()        
        
        
    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            # print(x.shape,edge_index.shape)
            # print('max: ',edge_index.max().item())
            # print(edge_index)
            # x_target = x[:size[1]]
            # x = self.convs[i]((x, x_target), edge_index)
            x = self.convs[i](x, edge_index)
            # print(x.shape)
            if i != self.num_layers-1:
                if self.batchnorm:
                    x = self.bns[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        x = x[:size[1]]
        return x.log_softmax(dim=-1)
    
    '''
    subgraph_loader: size = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=**, shuffle=False,
                                  num_workers=12)
    You can also sample the complete k-hop neighborhood, but this is rather expensive (especially for Reddit). 
    We apply here trick here to compute the node embeddings efficiently: 
       Instead of sampling multiple layers for a mini-batch, we instead compute the node embeddings layer-wise. 
       Doing this exactly k times mimics a k-layer GNN.  
    '''
    
    def inference_all(self, data):
        x, adj_t = data.x, data.adj_t
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.batchnorm: 
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x.log_softmax(dim=-1)
    
    def inference(self, x_all):
        # pbar = tqdm(total=x_all.size(0) * self.num_layers, ncols=80)
        # pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch.
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in self.layer_loader:
                edge_index, _, size = adj.to(self.device)
                x = x_all[n_id].to(self.device)
                x = self.convs[i](x, edge_index)
                if i != self.num_layers - 1:
                    x = F.relu(x)
                    if self.batchnorm: 
                        x = self.bns[i](x)
                # xs.append(x)
                x_target = x[:size[1]]
                xs.append(x_target.cpu())

                # pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        # pbar.close()

        return x_all.log_softmax(dim=-1)
