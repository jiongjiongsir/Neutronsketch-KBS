# from config import *
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from scipy.io import loadmat
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

# import sys, os
# import torch.nn.functional as F

# import faulthandler
# faulthandler.enable()

# sys.path.append(os.path.join(os.getcwd(), '..'))
# from pagerank import PageRank


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
            graph = dgl.to_bidirected(graph)
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
        prefix = '/home/sanzo/neutron-sanzo/data/' + dataset
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
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
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



def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename,'rb'):
        # print(line)
        index.append(int(line.strip()))
    return index

def get_test_index(filename):
    index = []
    with open(filename,'rb') as file_name:
        index =pkl.load(file_name)
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    # if dataset_str=='syn':
    #     return load_syn_data()

    # names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    # objects = []
    # for i in range(len(names)):
    #     with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
    #         if sys.version_info > (3, 0):
    #             objects.append(pkl.load(f, encoding='latin1'))
    #         else:
    #             objects.append(pkl.load(f))

    # x, y, tx, ty, allx, ally, graph = tuple(objects)
    # if dataset_str == 'ogbn-arxiv':
    #     test_idx_reorder = get_test_index("./data/ind.{}.test.index".format(dataset_str))
    # else:
    #     test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str))

    # # print(len(test_idx_reorder))
    # test_idx_range = np.sort(test_idx_reorder)
    # print(test_idx_range)
    # if dataset_str == 'citeseer':
    #     # Fix citeseer dataset (there are some isolated nodes in the graph)
    #     # Find isolated nodes, add them as zero-vecs into the right position
    #     test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
    #     tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    #     tx_extended[test_idx_range-min(test_idx_range), :] = tx
    #     tx = tx_extended
    #     ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
    #     ty_extended[test_idx_range-min(test_idx_range), :] = ty
    #     ty = ty_extended

    # features = sp.vstack((allx, tx)).tolil()
    # # print(features)
    # features[test_idx_reorder, :] = features[test_idx_range, :]
    # # print(features) 
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    # # print(adj)
    # labels = np.vstack((ally, ty))
    # labels[test_idx_reorder, :] = labels[test_idx_range, :]

    # idx_test = test_idx_range.tolist()
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)
    # print(len(idx_test),len(idx_train),len(idx_val))
    # print(features.shape)
    # train_mask = sample_mask(idx_train, labels.shape[0])
    # val_mask = sample_mask(idx_val, labels.shape[0])
    # test_mask = sample_mask(idx_test, labels.shape[0])
    # print(train_mask)
    # y_train = np.zeros(labels.shape)
    # y_val = np.zeros(labels.shape)
    # y_test = np.zeros(labels.shape)
    # y_train[train_mask, :] = labels[train_mask, :]
    # y_val[val_mask, :] = labels[val_mask, :]
    # y_test[test_mask, :] = labels[test_mask, :]
    # # print(adj,y_train.shape)
    # print(y_val.shape,y_test.shape,y_train.shape)
    graph = load_graph(dataset_str)
    x = graph.ndata['feat'][torch.nonzero(graph.ndata['train_mask']).squeeze()].numpy()  # Replace 'feat' with the actual node feature name
    tx =  graph.ndata['feat'][torch.nonzero(graph.ndata['test_mask']).squeeze()].numpy()
    indices_allx =torch.nonzero(graph.ndata['test_mask'] == 0).squeeze()
    # print(indices_allx.size())
    allx = graph.ndata['feat'][indices_allx].numpy()
    indices_test = torch.nonzero(graph.ndata['test_mask']).squeeze()
    print(graph.ndata['test_mask'].size())
    print(graph.ndata['feat'][indices_test].size())
    print(graph.ndata['feat'][indices_allx].size())
    # Extract labels for labeled instances
    one_hot_label = torch.nn.functional.one_hot(graph.ndata['label'], num_classes=47)
    y = one_hot_label[torch.nonzero(graph.ndata['train_mask']).squeeze()].numpy()  # Replace 'label' with the actual label name
    ty = one_hot_label[torch.nonzero(graph.ndata['test_mask']).squeeze()].numpy()
    ally = one_hot_label[torch.nonzero(graph.ndata['test_mask'] == 0).squeeze()].numpy()
    print(one_hot_label[torch.nonzero(graph.ndata['train_mask']).squeeze()].size())
    print(one_hot_label[torch.nonzero(graph.ndata['test_mask']).squeeze()].size())
    print(one_hot_label[torch.nonzero(graph.ndata['test_mask'] == 0).squeeze()].size())
    test_indices = graph.ndata['test_mask'].nonzero().squeeze().tolist()

    dicts = {}
    with open("./data/ind.{}.{}".format(dataset_str, 'graph'), 'rb') as f:
        dicts = pkl.load(f)
    # print(graph.sample_neighbors(0, -1).edges())
    # for i in range(graph.nodes().size()[0]):
    #     dicts[i] = graph.sample_neighbors(i, -1).edges()[0].tolist()
    # print(len(dicts.items()))
    features =  graph.ndata['feat'].numpy()
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(dicts))
    # idx_test = indices_test.tolist()
    # idx_train = torch.nonzero(graph.ndata['train_mask']).squeeze().tolist()
    # idx_val = torch.nonzero(graph.ndata['val_mask']).squeeze().tolist()
    labels = one_hot_label.numpy()
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    train_mask = graph.ndata['train_mask'].numpy()
    val_mask = graph.ndata['val_mask'].numpy()
    test_mask = graph.ndata['test_mask'].numpy()
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    print(features.shape,y_train.shape,y_val.shape,y_test.shape,train_mask.shape,val_mask.shape,test_mask.shape)
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask

# def load_data(dataset_str):
#     """
#     Loads input data from gcn/data directory

#     ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
#     ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
#     ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
#         (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
#     ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
#     ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
#     ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
#     ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
#         object;
#     ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

#     All objects above must be saved using python pickle module.

#     :param dataset_str: Dataset name
#     :return: All data input files loaded (as well the training/test data).
#     """
#     if dataset_str=='syn':
#         return load_syn_data()

#     names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
#     objects = []
#     for i in range(len(names)):
#         with open("./data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
#             if sys.version_info > (3, 0):
#                 objects.append(pkl.load(f, encoding='latin1'))
#             else:
#                 objects.append(pkl.load(f))

#     x, y, tx, ty, allx, ally, graph = tuple(objects)
#     test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str))
#     test_idx_range = np.sort(test_idx_reorder)

#     if dataset_str == 'citeseer':
#         # Fix citeseer dataset (there are some isolated nodes in the graph)
#         # Find isolated nodes, add them as zero-vecs into the right position
#         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
#         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
#         tx_extended[test_idx_range-min(test_idx_range), :] = tx
#         tx = tx_extended
#         ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
#         ty_extended[test_idx_range-min(test_idx_range), :] = ty
#         ty = ty_extended

#     features = sp.vstack((allx, tx)).tolil()
#     features[test_idx_reorder, :] = features[test_idx_range, :]
#     adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

#     labels = np.vstack((ally, ty))
#     labels[test_idx_reorder, :] = labels[test_idx_range, :]

#     idx_test = test_idx_range.tolist()
#     idx_train = range(len(y))
#     idx_val = range(len(y), len(y)+500)

#     train_mask = sample_mask(idx_train, labels.shape[0])
#     val_mask = sample_mask(idx_val, labels.shape[0])
#     test_mask = sample_mask(idx_test, labels.shape[0])

#     y_train = np.zeros(labels.shape)
#     y_val = np.zeros(labels.shape)
#     y_test = np.zeros(labels.shape)
#     y_train[train_mask, :] = labels[train_mask, :]
#     y_val[val_mask, :] = labels[val_mask, :]
#     y_test[test_mask, :] = labels[test_mask, :]

#     return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_brain_data():
    fun = loadmat('../conn_parcel23_460.mat')['r_part1'].transpose([2, 0, 1])

    myelin = loadmat('../thk_myl_parcel23_460.mat')['myelin_460_ica23_norm'].transpose()
    print(myelin.shape)
    thk = loadmat('../thk_myl_parcel23_460.mat')['thk_460_ica23_norm'].transpose()

    const_features = np.ones([myelin.shape[0], myelin.shape[1], 10])
    # node features
    feas = np.concatenate([np.expand_dims(myelin, -1), np.expand_dims(thk, -1), const_features], -1)
    # feas = np.concatenate([thk_200,myelin_200],-1)

    adjs = []
    for adj in fun:
        adj[adj < 0.0] = 0
        sqrt_deg = np.diag(1.0 / np.sqrt(np.sum(adj, axis=0, dtype=float).squeeze()))
        adj = np.matmul(np.matmul(sqrt_deg, adj), sqrt_deg)
        adjs.append(np.expand_dims(adj, 0))

    with open('../bev460.txt') as f:
        lines = (line for line in f if not line.startswith('#'))
        bev = np.loadtxt(lines, delimiter='\t')
        bev = np.nan_to_num(bev, 0)

    with open('../bev460.txt') as f:
        heads = f.readline()[1:].split('\t')
        head_to_id = {}
        for head in heads:
            head_to_id[head] = len(head_to_id)
    print('bev shape', bev.shape)

    labels = loadmat('../cca_460.mat')['cca_460'].flatten()
    labels[labels > 0] = 1
    labels[labels < 0] = 0

    adjs = np.concatenate(adjs, 0)
    labels = np.array(labels).astype(int)

    b = np.zeros((labels.size, labels.max() + 1))
    b[np.arange(labels.size), labels] = 1
    labels = b

    order = np.random.permutation(adjs.shape[0])
    shuffle_adjs = adjs[order]
    shuffle_feas = feas[order]
    shuffle_labels = labels[order]

    train_split = int(adjs.shape[0] * 0.8)
    val_split = int(adjs.shape[0] * 0.9)

    train_adjs = shuffle_adjs[:train_split]
    train_feas = shuffle_feas[:train_split]
    train_labels = shuffle_labels[:train_split]

    val_adjs = shuffle_adjs[train_split:val_split]
    val_feas = shuffle_feas[train_split:val_split]
    val_labels = shuffle_labels[train_split:val_split]

    test_adjs = shuffle_adjs[val_split:]
    test_feas = shuffle_feas[val_split:]
    test_labels = shuffle_labels[val_split:]

    return train_adjs, train_feas, train_labels, \
           val_adjs, val_feas, val_labels, \
           test_adjs, test_feas, test_labels

def load_syn_data():

    with open('../../data/syn.pkl','rb') as fin:
        syn = pkl.load(fin)
    features = syn['features']
    labels = syn['labels']
    adj = syn['adj']

    sizes = features.shape[0]
    nodes = np.array(list(range(sizes)))
    np.random.shuffle(nodes)
    train = int(sizes*0.6)
    val = int(sizes*0.2)
    idx_train = nodes[:train]
    idx_val = nodes[train:train+val]
    idx_test = nodes[train+val:]
    labels = np.array(labels)
    b = np.zeros((labels.size, labels.max() + 1))
    b[np.arange(labels.size), labels] = 1
    labels = b
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        values = values.astype(np.float32)
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features).astype(np.float32)
    try:
        return features.todense() # [coordinates, data, shape], []
    except:
        return features

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


if __name__ == '__main__':
    # graph = load_graph('cora')
    # graph = load_graph('reddit')
    # graph = load_graph('reddit')
    # print(graph)
    load_data('ogbn-arxiv')