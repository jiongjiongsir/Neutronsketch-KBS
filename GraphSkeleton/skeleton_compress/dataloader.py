import numpy as np
# from boxprint import bprint
import os
import scipy.sparse as sp
import json
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import CoauthorPhysicsDataset, AmazonCoBuyComputerDataset
from dgl.data import RedditDataset, CoraFullDataset, YelpDataset, FlickrDataset
from dgl.data import CoauthorCSDataset, AmazonCoBuyPhotoDataset

from ogb.nodeproppred import DglNodePropPredDataset

import torch
import dgl
from dgl import DGLGraph
import time
import networkx as nx
def dataloader(dataset):

    if dataset == 'dgraph':
        # bprint("DGraph-Fin", width=20)
        root = '../datasets/DGraphFin'
        file_path = root + '/dgraphfin-tmp.npz'
        data = np.load(file_path)
        save_path = root + '/skeleton'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif dataset =='ogbn-arxiv':
        root = '../datasets/ogbn-arxiv'
        file_path = root + '/ogbn-arxiv.npz'
        data = np.load(file_path)
        save_path = root + '/skeleton'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif dataset =='ogbn-products':
        root = '../datasets/ogbn-products'
        file_path = root + '/ogbn-products.npz'
        data = np.load(file_path)
        save_path = root + '/skeleton'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif dataset =='pubmed':
        root = '../datasets/pubmed'
        file_path = root + '/pubmed.npz'
        data = np.load(file_path)
        save_path = root + '/skeleton'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif dataset=='cora':
        root = '../datasets/cora'
        file_path = root + '/cora.npz'
        data = np.load(file_path)
        save_path = root + '/skeleton'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    elif dataset=='reddit':
        root = '../datasets/reddit'
        file_path = root + '/reddit.npz'
        data = np.load(file_path)
        save_path = root + '/skeleton'
        if not os.path.exists(save_path):
            os.makedirs(save_path)  
    return data, save_path
def load_graph(dataset):
    if dataset == 'cora':
        dataset = CoraGraphDataset()
        graph = dataset[0]
    elif dataset == 'citeseer':
        dataset = CiteseerGraphDataset()
        graph = dataset[0]
    elif dataset == 'pubmed':
        dataset = PubmedGraphDataset()
        graph = dataset[0]
    elif dataset == 'reddit':
        dataset = RedditDataset()
        graph = dataset[0]
        # print('after load graph', graph)
    elif dataset == 'corafull':
        dataset = CoraFullDataset()
        graph = dataset[0]
        graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = split_dataset(
            graph.num_nodes(), 6, 3, 1
        )
    elif dataset == 'physics':
        dataset = CoauthorPhysicsDataset()
        graph = dataset[0]
        graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = split_dataset(
            graph.num_nodes(), 6, 3, 1
        )
    elif dataset == 'computer':
        dataset = AmazonCoBuyComputerDataset()
        graph = dataset[0]
        graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = split_dataset(
            graph.num_nodes(), 6, 3, 1
        )
    elif dataset == 'Coauthor_cs':
        # dataset = CoauthorCSDataset('cs')
        dataset = CoauthorCSDataset()
        graph = dataset[0]
        graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = split_dataset(
            graph.num_nodes(), 6, 3, 1
        )
    elif dataset == 'photo':
        dataset = AmazonCoBuyPhotoDataset()
        graph = dataset[0]
        graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = split_dataset(
            graph.num_nodes(), 6, 3, 1
        )

    elif 'ogbn' in dataset:
        print("dataset", dataset)
        # elif dataset in ['ogbn-arxiv', 'ogbn-proteins', 'ogbn-products']:
        data = DglNodePropPredDataset(name=dataset)
        graph, node_labels = data[0]
        if dataset == 'ogbn-arxiv':
            feat = graph.ndata['feat']
            # graph = dgl.to_bidirected(graph)
            graph.ndata['feat'] = feat
        graph.ndata['label'] = node_labels[:, 0]
        idx_split = data.get_idx_split()
        num_nodes = graph.num_nodes()
        graph.ndata['train_mask'] = create_mask(idx_split['train'], num_nodes)
        graph.ndata['val_mask'] = create_mask(idx_split['valid'], num_nodes)
        graph.ndata['test_mask'] = create_mask(idx_split['test'], num_nodes)
    elif dataset == 'yelp':
        dataset = YelpDataset()
        graph = dataset[0]
    elif dataset == 'flickr':
        dataset = FlickrDataset()
        graph = dataset[0]
    elif dataset in ['ppi', 'ppi-large', 'amazon', 'reddit-s']:
        print(dataset)
        prefix = '/home/yuanh/data/' + dataset
        adj_full = sp.load_npz('{}/adj_full.npz'.format(prefix)).astype(bool)
        role = json.load(open('{}/role.json'.format(prefix)))
        feats = np.load('{}/feats.npy'.format(prefix))
        class_map = json.load(open('{}/class_map.json'.format(prefix)))
        class_map = {int(k): v for k, v in class_map.items()}
        assert len(class_map) == feats.shape[0]
        edges = adj_full.nonzero()
        # print(len(edges[0]))
        # assert False
        graph = dgl.graph((edges[0], edges[1]))
        # if args.self_loop:
        # graph = dgl.remove_self_loop(graph)
        # graph = dgl.add_self_loop(graph)
        num_nodes = adj_full.shape[0]
        train_mask = create_mask(role['tr'], num_nodes)
        val_mask = create_mask(role['va'], num_nodes)
        test_mask = create_mask(role['te'], num_nodes)
        # find onehot label if multiclass or not
        if isinstance(list(class_map.values())[0], list):
            is_multiclass = True
            num_classes = len(list(class_map.values())[0])
            class_arr = np.zeros((num_nodes, num_classes))
            for k, v in class_map.items():
                class_arr[k] = v
            labels = class_arr
            # non_zero_labels = []
            # for row in labels:
            #     non_zero_labels.append(np.nonzero(row)[0].tolist())
            # labels = non_zero_labels
        else:
            num_classes = max(class_map.values()) - min(class_map.values()) + 1
            class_arr = np.zeros((num_nodes, num_classes))
            offset = min(class_map.values())
            is_multiclass = False
            for k, v in class_map.items():
                class_arr[k][v - offset] = 1
            labels = np.where(class_arr)[1]

        # print(type(train_mask), train_mask)
        graph.ndata['train_mask'] = torch.tensor(train_mask)
        graph.ndata['val_mask'] = torch.tensor(val_mask)
        graph.ndata['test_mask'] = torch.tensor(test_mask)
        graph.ndata['feat'] = torch.tensor(feats, dtype=torch.float32)
        graph.ndata['label'] = torch.tensor(labels, dtype=torch.int64)
    else:
        raise NotImplemented
    # graph = dgl.remove_self_loop(graph)
    # graph = dgl.add_self_loop(graph)
    print('nodes:', graph.num_nodes(), 'edges:', graph.number_of_edges())
    return graph


def get_nids(graph):
    train_nids = torch.nonzero(graph.ndata['train_mask']).view(-1)
    valid_nids = torch.nonzero(graph.ndata['val_mask']).view(-1)
    test_nids = torch.nonzero(graph.ndata['test_mask']).view(-1)
    return train_nids, valid_nids, test_nids


def get_dataset(args, ratio=0):
    graph = load_graph(args.dataset)
    # Add reverse edges since ogbn-arxiv is unidirectional.
    # graph = dgl.add_reverse_edges(graph)
    if ratio > 0:
        has_remove = hasattr(args, 'remove')
        has_pagerank = hasattr(args, 'pagerank')
        if has_pagerank and args.pagerank:
            pr_time = time.time()
            pr = PageRank(graph)
            pr.run_pagerank(200)
            pr_time = time.time() - pr_time
            print(f'pr cost {pr_time:.3f}')

        rm_list = torch.nonzero(graph.ndata['train_mask']).view(-1).tolist()

        if has_pagerank and args.pagerank:
            print('use pagerank', graph.ndata['pv'][:10])
            rm_list = sorted(
                rm_list,
                key=lambda x: graph.ndata['pv'][x],
                reverse=True if has_remove and args.remove == 'large' else False,
            )
        else:
            rm_list = sorted(
                rm_list,
                key=lambda x: graph.in_degrees(x),
                reverse=True if has_remove and args.remove == 'large' else False,
            )

        rm_list = rm_list[: int(len(rm_list) * ratio)]

        split_graph(graph, rm_list)
        graph = dgl.remove_self_loop(graph)
        graph = dgl.add_self_loop(graph)

        print(graph.ndata['feat'].shape, graph.ndata['label'].shape, graph.ndata['train_mask'].shape)

    train_nids = torch.nonzero(graph.ndata['train_mask']).view(-1)
    valid_nids = torch.nonzero(graph.ndata['val_mask']).view(-1)
    test_nids = torch.nonzero(graph.ndata['test_mask']).view(-1)
    return graph, train_nids, valid_nids, test_nids


def create_mask(idx, l):
    """Create mask."""
    mask = torch.zeros(l, dtype=bool)
    # mask = np.zeros(l, dtype=bool)
    mask[idx] = True
    return mask


def split_dataset(num_nodes, x, y, z):
    '''
    x: train nodes, y: val nodes, z: test nodes
    '''
    train_mask = torch.tensor([False for i in range(num_nodes)], dtype=torch.bool)
    val_mask = torch.tensor([False for i in range(num_nodes)], dtype=torch.bool)
    test_mask = torch.tensor([False for i in range(num_nodes)], dtype=torch.bool)
    step = int(num_nodes / (x + y + z))
    train_mask[: int(x * step)] = True
    val_mask[int(x * step) : int((x + y) * step)] = True
    test_mask[int((x + y) * step) :] = True
    assert train_mask.sum() + val_mask.sum() + test_mask.sum() == num_nodes
    return train_mask, val_mask, test_mask


# def split_graph(graph, n_nodes, n_edges, features, labels, train_mask, val_mask, test_mask, fraction):
def split_graph(graph, rm_list):
    if len(rm_list) == 0:
        return
    n_nodes = graph.num_nodes()
    indices = torch.ones(n_nodes, dtype=bool)
    indices[rm_list] = False
    indices = torch.nonzero(indices).view(-1)
    # print(n_nodes, len(indices) + len(rm_list), len(indices))

    if isinstance(graph, nx.classes.digraph.DiGraph):
        print('graph is DiGraph')
        graph.remove_nodes_from(rm_list)
    elif isinstance(graph, DGLGraph):
        print('g is DGLGraph')
        graph.remove_nodes(rm_list)
    # torch.index_select(x, 0, indices)
    # print(graph.ndata['feat'].shape, graph.ndata['label'].shape, graph.ndata['train_mask'].shape)
    # graph.ndata['feat'] = torch.index_select(graph.ndata['feat'], 0, indices)
    # print(graph.ndata['feat'].shape)
    # graph.ndata['label'] = torch.index_select(graph.ndata['label'], 0, indices)
    # graph.ndata['train_mask'] = torch.index_select(graph.ndata['train_mask'], 0, indices)
    # graph.ndata['val_mask'] = torch.index_select(graph.ndata['val_mask'], 0, indices)
    # graph.ndata['test_mask'] = torch.index_select(graph.ndata['test_mask'], 0, indices)

    # features = features[:new_n_nodes]
    # labels = labels[:new_n_nodes]
    # train_mask = train_mask[:new_n_nodes]
    # val_mask = val_mask[:new_n_nodes]
    # test_mask = test_mask[:new_n_nodes]

    # return graph, features, labels, train_mask, val_mask, test_mask


class AddNids:
    def __init__(self, args):
        self.dataset = graph, train_nids, val_nids, test_nids = get_dataset(args)
        self.train_idx = train_nids
        self.val_idx = val_nids
        self.test_idx = test_nids
        labels = graph.ndata['label']
        if labels.dim() > 1:
            self.num_classes = labels.shape[1]
        else:
            self.num_classes = torch.max(labels) + 1
        print(self.num_classes)


if __name__ == '__main__':
    # graph = load_graph('cora')
    # graph = load_graph('reddit')
    # graph = load_graph('reddit')
    # print(graph)
   
   
    graph = load_graph('reddit')
    print('hh')
    print(graph)
    print(graph.edges()[0].shape)
    
    # print(edge_index.shape)
    x = graph.ndata['feat'].numpy()
    y = graph.ndata['label'].numpy()
    # print(graph.ndata)
    print(torch.nonzero(graph.ndata['train_mask']).shape)
    print(torch.nonzero(graph.ndata['val_mask']).shape)
    print(torch.nonzero(graph.ndata['test_mask']).shape)
    edge_index = torch.cat((graph.edges()[0].unsqueeze(0),graph.edges()[1].unsqueeze(0)),dim=0).numpy().T

    
    # num_elements_80_percent = int(0.8 * torch.nonzero(graph.ndata['val_mask']).size(0))
    # # 生成随机索引
    # indices = torch.randperm(torch.nonzero(graph.ndata['val_mask']).size(0))

    # # 选择80%的元素
    # selected_indices = indices[:num_elements_80_percent]
    # train_mask = torch.squeeze(torch.nonzero(graph.ndata['val_mask'])[selected_indices]).numpy()

    # # 剩余20%的元素
    # remaining_indices = indices[num_elements_80_percent:]
    # valid_mask = torch.squeeze(torch.nonzero(graph.ndata['val_mask'])[remaining_indices]).numpy()

    train_mask = torch.squeeze(torch.nonzero(graph.ndata['train_mask'])).numpy()
    valid_mask = torch.squeeze(torch.nonzero(graph.ndata['val_mask'])).numpy()
    test_mask = torch.squeeze(torch.nonzero(graph.ndata['test_mask'])).numpy()
    print(train_mask.shape)
    print(valid_mask.shape)
    npz_data = { 'x': x, 'y': y, 
                'edge_index': edge_index, 
                'train_mask': train_mask, 'valid_mask': valid_mask, 'test_mask': test_mask}
    print(edge_index.shape)
    np.savez('../datasets/reddit/reddit.npz', x=x,y=y,train_mask=train_mask,valid_mask=valid_mask,test_mask=test_mask,edge_index=edge_index)
  
  
    # graph = load_graph('ogbn-products')
    # print(graph)
    # graph = load_graph('reddit-s')
    # print('nodes:', graph.num_nodes())
    # print('nodes', graph.nodes(), len(graph.nodes()))
    # graph = load_graph('reddit')
    # print(graph)
    # pass
    # train_nids = torch.nonzero(graph.ndata['train_mask']).view(-1)
    # dgs = graph.in_degrees(train_nids)
    # print(max(dgs))
    # print(min(dgs))

    # root = '../datasets/DGraphFin'
    # file_path = root + '/dgraphfin.npz'
    # data = np.load(file_path)
    # x = data['x']
    # y = data['y']
    # edge_index = data['edge_index']
    # train_mask = data['train_mask']
    # valid_mask = data['valid_mask']
    # test_mask = data['test_mask']
    # np.savez('../datasets/DGraphFin/dgraphfin-tmp.npz', x=x,y=y,train_mask=train_mask,valid_mask=valid_mask,test_mask=test_mask,edge_index=edge_index)
    # data = np.load('../datasets/DGraphFin/dgraphfin-tmp.npz')
    # print(data.files)