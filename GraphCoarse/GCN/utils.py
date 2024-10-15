from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import DglNodePropPredDataset,PygNodePropPredDataset
import torch
import dgl
from torch_geometric.utils import to_dense_adj
from graph_coarsening.coarsening_utils import *
from torch_geometric.datasets import Coauthor
from torch_geometric.datasets import CitationFull
from torch_geometric.datasets import Reddit
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import networkx as nx
import time
import psutil
def one_hot(x, class_count):
    return torch.eye(class_count)[x, :]

def extract_components(H):
        if H.A.shape[0] != H.A.shape[1]:
            H.logger.error('Inconsistent shape to extract components. '
                           'Square matrix required.')
            return None

        if H.is_directed():
            raise NotImplementedError('Directed graphs not supported yet.')

        graphs = []

        visited = np.zeros(H.A.shape[0], dtype=bool)

        while not visited.all():
            stack = set([np.nonzero(~visited)[0][0]])
            comp = []

            while len(stack):
                v = stack.pop()
                if not visited[v]:
                    comp.append(v)
                    visited[v] = True

                    stack.update(set([idx for idx in H.A[v, :].nonzero()[1]
                                      if not visited[idx]]))

            comp = sorted(comp)
            G = H.subgraph(comp)
            G.info = {'orig_idx': comp}
            graphs.append(G)

        return graphs

def coarsening(dataset, coarsening_ratio, coarsening_method):
    if dataset == 'dblp':
        dataset = CitationFull(root='./dataset', name=dataset)
        data = dataset[0]
    elif dataset == 'Physics':
        dataset = Coauthor(root='./dataset/Physics', name=dataset)
        data = dataset[0]
    elif dataset == 'ogbn-arxiv':
        print(dataset)
        datasets = PygNodePropPredDataset(root='./dataset', name=dataset)
        data = datasets[0]
        print(data.is_undirected())
        # data['x'] = data['x'].squeeze()
        data['y'].squeeze_()
        idx_split = datasets.get_idx_split()
        a = torch.zeros(data['y'].shape[0])
        b = torch.zeros(data['y'].shape[0])
        c = torch.zeros(data['y'].shape[0])
        a[idx_split['train']] = 1
        a = a.bool()
        b[idx_split['valid']] = 1
        b = b.bool()
        c[idx_split['test']] = 1
        c = c.bool()
        data['train_mask'] = a
        data['val_mask'] = b
        data['test_mask'] = c
        # print(idx_split)
        mem = psutil.virtual_memory()
        print('first load data mem usage: ',mem)
    elif dataset == 'reddit':
        dataset = Reddit(root='./dataset/reddit')
        data = dataset[0]
    else:
        dataset = Planetoid(root='./dataset', name=dataset)
        data = dataset[0]

    print(data.edge_index)
    print(to_dense_adj(data.edge_index)[0])
    # dataset_dgl = DglNodePropPredDataset(root='./dataset', name=dataset)
    # data_dgl ,label= dataset_dgl[0]
    # data_dgl = dgl.to_bidirected(data_dgl)
    # G = gsp.graphs.Graph(W=data_dgl.adjacency_matrix().to_dense())
    # if dataset == 'ogbn-arxiv':
    #     mem = psutil.virtual_memory()
    #     print('undirected_graph mem usage: ',mem)
    #     undirected_graph = to_networkx(data=data,to_undirected=True)
    #     mem = psutil.virtual_memory()
    #     print('to_networkx mem usage: ',mem)
    #     a = nx.adjacency_matrix(undirected_graph)
    #     mem = psutil.virtual_memory()
    #     print('a mem usage: ',mem)
    #     b = a.todense()
    #     mem = psutil.virtual_memory()
    #     print('b mem usage: ',mem)
    #     print(b)
    #     dense_adj = torch.tensor(b)
    #     mem = psutil.virtual_memory()
    #     print('c mem usage: ',mem)
    #     print(dense_adj.shape)
    #     # dense_adj = torch.tensor(nx.adjacency_matrix(undirected_graph).todense())
    #     # mem = psutil.virtual_memory()
    #     # print('dense_adj mem usage: ',mem)
    #     G = gsp.graphs.Graph(W=dense_adj)
    #     mem = psutil.virtual_memory()
    #     print('gsp mem usage: ',mem)
    # else:
    #     G = gsp.graphs.Graph(W=to_dense_adj(data.edge_index)[0])
    G = gsp.graphs.Graph(W=to_dense_adj(data.edge_index)[0])
    print(G)
    mem = psutil.virtual_memory()
    print('G graph mem usage: ',mem)
    components = extract_components(G)
    print('the number of subgraphs is', len(components))
    candidate = sorted(components, key=lambda x: len(x.info['orig_idx']), reverse=True)
    number = 0
    C_list=[]
    Gc_list=[]
    time1 = time.time()
    while number < len(candidate):
        H = candidate[number]
        if len(H.info['orig_idx']) > 10:
            C, Gc, Call, Gall = coarsen(H, r=coarsening_ratio, method=coarsening_method)
            C_list.append(C)
            Gc_list.append(Gc)
        number += 1
    print('Coarsening time: ',time.time()-time1)
    return data.x.shape[1], len(set(np.array(data.y))), candidate, C_list, Gc_list

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def splits(data, num_classes, exp):
    if exp!='fixed':
        indices = []
        for i in range(num_classes):
            index = (data.y == i).nonzero().view(-1)
            index = index[torch.randperm(index.size(0))]
            indices.append(index)

        if exp == 'random':
            train_index = torch.cat([i[:20] for i in indices], dim=0)
            val_index = torch.cat([i[20:50] for i in indices], dim=0)
            test_index = torch.cat([i[50:] for i in indices], dim=0)
        else:
            train_index = torch.cat([i[:5] for i in indices], dim=0)
            val_index = torch.cat([i[5:10] for i in indices], dim=0)
            test_index = torch.cat([i[10:] for i in indices], dim=0)

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(test_index, size=data.num_nodes)

    return data

def load_data(datasets, candidate, C_list, Gc_list, exp):
    if datasets == 'dblp':
        dataset = CitationFull(root='./dataset', name=datasets)
        # data = splits(dataset[0], n_classes, exp)
    elif datasets == 'Physics':
        dataset = Coauthor(root='./dataset/Physics', name=datasets)
        # data = splits(dataset[0], n_classes, exp)
    elif datasets == 'ogbn-arxiv':
        dataset = PygNodePropPredDataset(root='./dataset', name=datasets)
        data = dataset[0]
        # dataset[0]['x'] = dataset[0]['x'].squeeze()
        dataset[0]['y'].squeeze_()
        print(data)
    elif datasets == 'reddit':
        dataset = Reddit(root='./dataset/reddit')
        data = dataset[0]
    else:
        dataset = Planetoid(root='./dataset', name=datasets)
        # data = splits(dataset[0], n_classes, exp)
    print(dataset[0].y)
    n_classes = len(set(np.array(dataset[0].y)))

    data = splits(dataset[0], n_classes, exp)  
    print(data)
    print(n_classes)
    if datasets == 'ogbn-arxiv':
        idx_split = dataset.get_idx_split()
        a = torch.zeros(dataset[0]['y'].shape[0])
        b = torch.zeros(dataset[0]['y'].shape[0])
        c = torch.zeros(dataset[0]['y'].shape[0])
        a[idx_split['train']] = 1
        a = a.bool()
        b[idx_split['valid']] = 1
        b = b.bool()
        c[idx_split['test']] = 1
        c = c.bool()
        data['train_mask'] = a
        data['val_mask'] = b
        data['test_mask'] = c
    print(data)
    train_mask = data.train_mask
    val_mask = data.val_mask
    labels = data.y
    features = data.x

    coarsen_node = 0
    number = 0
    coarsen_row = None
    coarsen_col = None
    coarsen_features = torch.Tensor([])
    coarsen_train_labels = torch.Tensor([])
    coarsen_train_mask = torch.Tensor([]).bool()
    coarsen_val_labels = torch.Tensor([])
    coarsen_val_mask = torch.Tensor([]).bool()

    while number < len(candidate):
        H = candidate[number]
        keep = H.info['orig_idx']
        H_features = features[keep]
        H_labels = labels[keep]
        H_train_mask = train_mask[keep]
        H_val_mask = val_mask[keep]
        if len(H.info['orig_idx']) > 10 and torch.sum(H_train_mask)+torch.sum(H_val_mask) > 0:
            train_labels = one_hot(H_labels, n_classes)
            train_labels[~H_train_mask] = torch.Tensor([0 for _ in range(n_classes)])
            val_labels = one_hot(H_labels, n_classes)
            val_labels[~H_val_mask] = torch.Tensor([0 for _ in range(n_classes)])
            C = C_list[number]
            Gc = Gc_list[number]

            new_train_mask = torch.BoolTensor(np.sum(C.dot(train_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(train_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_train_mask[mix_mask > 1] = False

            new_val_mask = torch.BoolTensor(np.sum(C.dot(val_labels), axis=1))
            mix_label = torch.FloatTensor(C.dot(val_labels))
            mix_label[mix_label > 0] = 1
            mix_mask = torch.sum(mix_label, dim=1)
            new_val_mask[mix_mask > 1] = False

            coarsen_features = torch.cat([coarsen_features, torch.FloatTensor(C.dot(H_features))], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, torch.argmax(torch.FloatTensor(C.dot(train_labels)), dim=1).float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, new_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, torch.argmax(torch.FloatTensor(C.dot(val_labels)), dim=1).float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, new_val_mask], dim=0)

            if coarsen_row is None:
                coarsen_row = Gc.W.tocoo().row
                coarsen_col = Gc.W.tocoo().col
            else:
                current_row = Gc.W.tocoo().row + coarsen_node
                current_col = Gc.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += Gc.W.shape[0]

        elif torch.sum(H_train_mask)+torch.sum(H_val_mask)>0:

            coarsen_features = torch.cat([coarsen_features, H_features], dim=0)
            coarsen_train_labels = torch.cat([coarsen_train_labels, H_labels.float()], dim=0)
            coarsen_train_mask = torch.cat([coarsen_train_mask, H_train_mask], dim=0)
            coarsen_val_labels = torch.cat([coarsen_val_labels, H_labels.float()], dim=0)
            coarsen_val_mask = torch.cat([coarsen_val_mask, H_val_mask], dim=0)

            if coarsen_row is None:
                raise Exception('The graph does not need coarsening.')
            else:
                current_row = H.W.tocoo().row + coarsen_node
                current_col = H.W.tocoo().col + coarsen_node
                coarsen_row = np.concatenate([coarsen_row, current_row], axis=0)
                coarsen_col = np.concatenate([coarsen_col, current_col], axis=0)
            coarsen_node += H.W.shape[0]
        number += 1

    print('the size of coarsen graph features:', coarsen_features.shape)

    coarsen_edge = torch.LongTensor([coarsen_row, coarsen_col])
    coarsen_train_labels = coarsen_train_labels.long()
    coarsen_val_labels = coarsen_val_labels.long()

    return data, coarsen_features, coarsen_train_labels, coarsen_train_mask, coarsen_val_labels, coarsen_val_mask, coarsen_edge


# # import os
# # import torch
# # import dgl
# # import torch.nn.functional as F
# # from tqdm import tqdm
# # from torch_geometric.loader import NeighborLoader
# # from torch.optim.lr_scheduler import ReduceLROnPlateau
# # from torch_geometric.nn import MessagePassing, SAGEConv
# # from ogb.nodeproppred import Evaluator, DglNodePropPredDataset
# # from torch_geometric.utils import to_dense_adj
# # from graph_coarsening.coarsening_utils import *
# # import torch_geometric.transforms as T
# # target_dataset = 'ogbn-arxiv'#我们将把ogbn-arxiv下载到当前示例工程的'networks'文件夹下
# # dataset = DglNodePropPredDataset(name=target_dataset, root='./dataset')
# # data,node_labels  = dataset[0]
# # print(data)
# # # g = dgl.graph((data.edge_index[0], data.edge_index[1]))
# # dense = data.adjacency_matrix().to_dense()
# # G = gsp.graphs.Graph(W=dense)
# # print(G)
# # print(torch.nonzero(dense).size(0))
# # # print(data['edge_index'])
# # # data['x'] = data['x'].squeeze()
# # data['y'] = data['y'].squeeze()
# # idx_split = dataset.get_idx_split()
# # a = torch.zeros(data['y'].shape[0])
# # b = torch.zeros(data['y'].shape[0])
# # c = torch.zeros(data['y'].shape[0])
# # a[idx_split['train']] = 1
# # a = a.bool()
# # b[idx_split['valid']] = 1
# # b = b.bool()
# # c[idx_split['test']] = 1
# # c = c.bool()
# # data['train_mask'] = a
# # data['val_mask'] = b
# # data['test_mask'] = c
# # print(to_dense_adj(data['edge_index'].sort(dim=1)[0])[0][0][0:20])
# # # G = gsp.graphs.Graph(W=to_dense_adj(data.edge_index)[0])
# # # print(G)
# # # components = extract_components(G)
# # # print('the number of subgraphs is', len(components))

# # print(idx_split['valid'].shape)


# import os
# import torch
# import dgl
# import torch.nn.functional as F
# from tqdm import tqdm
# from torch_geometric.loader import NeighborLoader
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# from torch_geometric.nn import MessagePassing, SAGEConv
# from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
# from torch_geometric.utils import to_dense_adj
# from graph_coarsening.coarsening_utils import *
# import torch_geometric.transforms as T
# from torch_geometric.utils import to_networkx
# import networkx as nx
# target_dataset = 'ogbn-arxiv'#我们将把ogbn-arxiv下载到当前示例工程的'networks'文件夹下
# dataset = PygNodePropPredDataset(name=target_dataset, root='./dataset')
# dataset[0]['y'].squeeze_()
# # dataset[0]['y'] = tmp
# data = dataset[0]
# dataset[0]['y'].squeeze_()
# idx_split = dataset.get_idx_split()
# a = torch.zeros(dataset[0]['y'].shape[0])
# b = torch.zeros(dataset[0]['y'].shape[0])
# c = torch.zeros(dataset[0]['y'].shape[0])
# a[idx_split['train']] = 1
# a = a.bool()
# b[idx_split['valid']] = 1
# b = b.bool()
# c[idx_split['test']] = 1
# c = c.bool()
# data['train_mask'] = a
# data['val_mask'] = b
# data['test_mask'] = c
# print(data)
# # print(tmp)
# print(dataset[0]['y'])
# print(data['y'].shape)
# # undirected_graph = to_networkx(data=data,to_undirected=True)
# # dense_adj = torch.tensor(nx.adjacency_matrix(undirected_graph).todense())
# # print(undirected_graph.is_directed())
# # print(dense_adj)
# # print(torch.nonzero(dense_adj).size(0))