# import scipy.sparse as sp
# import json
# from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
# from dgl.data import CoauthorPhysicsDataset, AmazonCoBuyComputerDataset
# from dgl.data import RedditDataset, CoraFullDataset, YelpDataset, FlickrDataset
# from dgl.data import CoauthorCSDataset, AmazonCoBuyPhotoDataset

# from ogb.nodeproppred import DglNodePropPredDataset

# import numpy as np
# import torch
# import dgl
# from dgl import DGLGraph
# import time
# import networkx as nx

# import pickle
# import numpy as np
# import scipy.sparse as sp
# import collections
# # import sys, os
# # import torch.nn.functional as F

# # import faulthandler
# # faulthandler.enable()

# # sys.path.append(os.path.join(os.getcwd(), '..'))
# # from pagerank import PageRank


# def load_graph(dataset):
#     if dataset == 'cora':
#         dataset = CoraGraphDataset()
#         graph = dataset[0]
#     elif dataset == 'citeseer':
#         dataset = CiteseerGraphDataset()
#         graph = dataset[0]
#     elif dataset == 'pubmed':
#         dataset = PubmedGraphDataset()
#         graph = dataset[0]
#     elif dataset == 'reddit':
#         dataset = RedditDataset()
#         graph = dataset[0]
#         # print('after load graph', graph)
#     elif dataset == 'corafull':
#         dataset = CoraFullDataset()
#         graph = dataset[0]
#         graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = split_dataset(
#             graph.num_nodes(), 6, 3, 1
#         )
#     elif dataset == 'physics':
#         dataset = CoauthorPhysicsDataset()
#         graph = dataset[0]
#         graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = split_dataset(
#             graph.num_nodes(), 6, 3, 1
#         )
#     elif dataset == 'computer':
#         dataset = AmazonCoBuyComputerDataset()
#         graph = dataset[0]
#         graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = split_dataset(
#             graph.num_nodes(), 6, 3, 1
#         )
#     elif dataset == 'Coauthor_cs':
#         # dataset = CoauthorCSDataset('cs')
#         dataset = CoauthorCSDataset()
#         graph = dataset[0]
#         graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = split_dataset(
#             graph.num_nodes(), 6, 3, 1
#         )
#     elif dataset == 'photo':
#         dataset = AmazonCoBuyPhotoDataset()
#         graph = dataset[0]
#         graph.ndata['train_mask'], graph.ndata['val_mask'], graph.ndata['test_mask'] = split_dataset(
#             graph.num_nodes(), 6, 3, 1
#         )

#     elif 'ogbn' in dataset:
#         print("dataset", dataset)
#         # elif dataset in ['ogbn-arxiv', 'ogbn-proteins', 'ogbn-products']:
#         data = DglNodePropPredDataset(name=dataset)
#         graph, node_labels = data[0]
#         if dataset == 'ogbn-arxiv':
#             feat = graph.ndata['feat']
#             graph = dgl.to_bidirected(graph)
#             graph.ndata['feat'] = feat
#         graph.ndata['label'] = node_labels[:, 0]
#         idx_split = data.get_idx_split()
#         num_nodes = graph.num_nodes()
#         graph.ndata['train_mask'] = create_mask(idx_split['train'], num_nodes)
#         graph.ndata['val_mask'] = create_mask(idx_split['valid'], num_nodes)
#         graph.ndata['test_mask'] = create_mask(idx_split['test'], num_nodes)
#     elif dataset == 'yelp':
#         dataset = YelpDataset()
#         graph = dataset[0]
#     elif dataset == 'flickr':
#         dataset = FlickrDataset()
#         graph = dataset[0]
#     elif dataset in ['ppi', 'ppi-large', 'amazon', 'reddit-s']:
#         print(dataset)
#         prefix = '/home/sanzo/neutron-sanzo/data/' + dataset
#         adj_full = sp.load_npz('{}/adj_full.npz'.format(prefix)).astype(bool)
#         role = json.load(open('{}/role.json'.format(prefix)))
#         feats = np.load('{}/feats.npy'.format(prefix))
#         class_map = json.load(open('{}/class_map.json'.format(prefix)))
#         class_map = {int(k): v for k, v in class_map.items()}
#         assert len(class_map) == feats.shape[0]
#         edges = adj_full.nonzero()
#         # print(len(edges[0]))
#         # assert False
#         graph = dgl.graph((edges[0], edges[1]))
#         # if args.self_loop:
#         # graph = dgl.remove_self_loop(graph)
#         # graph = dgl.add_self_loop(graph)
#         num_nodes = adj_full.shape[0]
#         train_mask = create_mask(role['tr'], num_nodes)
#         val_mask = create_mask(role['va'], num_nodes)
#         test_mask = create_mask(role['te'], num_nodes)
#         # find onehot label if multiclass or not
#         if isinstance(list(class_map.values())[0], list):
#             is_multiclass = True
#             num_classes = len(list(class_map.values())[0])
#             class_arr = np.zeros((num_nodes, num_classes))
#             for k, v in class_map.items():
#                 class_arr[k] = v
#             labels = class_arr
#             # non_zero_labels = []
#             # for row in labels:
#             #     non_zero_labels.append(np.nonzero(row)[0].tolist())
#             # labels = non_zero_labels
#         else:
#             num_classes = max(class_map.values()) - min(class_map.values()) + 1
#             class_arr = np.zeros((num_nodes, num_classes))
#             offset = min(class_map.values())
#             is_multiclass = False
#             for k, v in class_map.items():
#                 class_arr[k][v - offset] = 1
#             labels = np.where(class_arr)[1]

#         # print(type(train_mask), train_mask)
#         graph.ndata['train_mask'] = torch.tensor(train_mask)
#         graph.ndata['val_mask'] = torch.tensor(val_mask)
#         graph.ndata['test_mask'] = torch.tensor(test_mask)
#         graph.ndata['feat'] = torch.tensor(feats, dtype=torch.float32)
#         graph.ndata['label'] = torch.tensor(labels, dtype=torch.int64)
#     else:
#         raise NotImplemented
#     graph = dgl.remove_self_loop(graph)
#     graph = dgl.add_self_loop(graph)
#     print('nodes:', graph.num_nodes(), 'edges:', graph.number_of_edges())
#     return graph


# def get_nids(graph):
#     train_nids = torch.nonzero(graph.ndata['train_mask']).view(-1)
#     valid_nids = torch.nonzero(graph.ndata['val_mask']).view(-1)
#     test_nids = torch.nonzero(graph.ndata['test_mask']).view(-1)
#     return train_nids, valid_nids, test_nids


# def get_dataset(args, ratio=0):
#     graph = load_graph(args.dataset)
#     # Add reverse edges since ogbn-arxiv is unidirectional.
#     # graph = dgl.add_reverse_edges(graph)
#     if ratio > 0:
#         has_remove = hasattr(args, 'remove')
#         has_pagerank = hasattr(args, 'pagerank')
#         if has_pagerank and args.pagerank:
#             pr_time = time.time()
#             pr = PageRank(graph)
#             pr.run_pagerank(200)
#             pr_time = time.time() - pr_time
#             print(f'pr cost {pr_time:.3f}')

#         rm_list = torch.nonzero(graph.ndata['train_mask']).view(-1).tolist()

#         if has_pagerank and args.pagerank:
#             print('use pagerank', graph.ndata['pv'][:10])
#             rm_list = sorted(
#                 rm_list,
#                 key=lambda x: graph.ndata['pv'][x],
#                 reverse=True if has_remove and args.remove == 'large' else False,
#             )
#         else:
#             rm_list = sorted(
#                 rm_list,
#                 key=lambda x: graph.in_degrees(x),
#                 reverse=True if has_remove and args.remove == 'large' else False,
#             )

#         rm_list = rm_list[: int(len(rm_list) * ratio)]

#         split_graph(graph, rm_list)
#         graph = dgl.remove_self_loop(graph)
#         graph = dgl.add_self_loop(graph)

#         print(graph.ndata['feat'].shape, graph.ndata['label'].shape, graph.ndata['train_mask'].shape)

#     train_nids = torch.nonzero(graph.ndata['train_mask']).view(-1)
#     valid_nids = torch.nonzero(graph.ndata['val_mask']).view(-1)
#     test_nids = torch.nonzero(graph.ndata['test_mask']).view(-1)
#     return graph, train_nids, valid_nids, test_nids


# def create_mask(idx, l):
#     """Create mask."""
#     mask = torch.zeros(l, dtype=bool)
#     # mask = np.zeros(l, dtype=bool)
#     mask[idx] = True
#     return mask


# def split_dataset(num_nodes, x, y, z):
#     '''
#     x: train nodes, y: val nodes, z: test nodes
#     '''
#     train_mask = torch.tensor([False for i in range(num_nodes)], dtype=torch.bool)
#     val_mask = torch.tensor([False for i in range(num_nodes)], dtype=torch.bool)
#     test_mask = torch.tensor([False for i in range(num_nodes)], dtype=torch.bool)
#     step = int(num_nodes / (x + y + z))
#     train_mask[: int(x * step)] = True
#     val_mask[int(x * step) : int((x + y) * step)] = True
#     test_mask[int((x + y) * step) :] = True
#     assert train_mask.sum() + val_mask.sum() + test_mask.sum() == num_nodes
#     return train_mask, val_mask, test_mask


# # def split_graph(graph, n_nodes, n_edges, features, labels, train_mask, val_mask, test_mask, fraction):
# def split_graph(graph, rm_list):
#     if len(rm_list) == 0:
#         return
#     n_nodes = graph.num_nodes()
#     indices = torch.ones(n_nodes, dtype=bool)
#     indices[rm_list] = False
#     indices = torch.nonzero(indices).view(-1)
#     # print(n_nodes, len(indices) + len(rm_list), len(indices))

#     if isinstance(graph, nx.classes.digraph.DiGraph):
#         print('graph is DiGraph')
#         graph.remove_nodes_from(rm_list)
#     elif isinstance(graph, DGLGraph):
#         print('g is DGLGraph')
#         graph.remove_nodes(rm_list)
#     # torch.index_select(x, 0, indices)
#     # print(graph.ndata['feat'].shape, graph.ndata['label'].shape, graph.ndata['train_mask'].shape)
#     # graph.ndata['feat'] = torch.index_select(graph.ndata['feat'], 0, indices)
#     # print(graph.ndata['feat'].shape)
#     # graph.ndata['label'] = torch.index_select(graph.ndata['label'], 0, indices)
#     # graph.ndata['train_mask'] = torch.index_select(graph.ndata['train_mask'], 0, indices)
#     # graph.ndata['val_mask'] = torch.index_select(graph.ndata['val_mask'], 0, indices)
#     # graph.ndata['test_mask'] = torch.index_select(graph.ndata['test_mask'], 0, indices)

#     # features = features[:new_n_nodes]
#     # labels = labels[:new_n_nodes]
#     # train_mask = train_mask[:new_n_nodes]
#     # val_mask = val_mask[:new_n_nodes]
#     # test_mask = test_mask[:new_n_nodes]

#     # return graph, features, labels, train_mask, val_mask, test_mask


# class AddNids:
#     def __init__(self, args):
#         self.dataset = graph, train_nids, val_nids, test_nids = get_dataset(args)
#         self.train_idx = train_nids
#         self.val_idx = val_nids
#         self.test_idx = test_nids
#         labels = graph.ndata['label']
#         if labels.dim() > 1:
#             self.num_classes = labels.shape[1]
#         else:
#             self.num_classes = torch.max(labels) + 1
#         print(self.num_classes)


# def save_data_to_pickle(filename, data):
#     with open(filename, 'wb') as f:
#         pickle.dump(data, f)

# def dgl_graph_to_custom_format_and_save(graph, filename_prefix):
#     # Assuming graph is a DGLGraph
    
#     # Extract node features
#     x = graph.ndata['feat'][torch.nonzero(graph.ndata['train_mask']).squeeze()].numpy()  # Replace 'feat' with the actual node feature name
#     tx =  graph.ndata['feat'][torch.nonzero(graph.ndata['test_mask']).squeeze()].numpy()
#     indices_allx =torch.nonzero(graph.ndata['test_mask'] == 0).squeeze()
#     # print(indices_allx.size())
#     allx = graph.ndata['feat'][indices_allx].numpy()
#     indices_test = torch.nonzero(graph.ndata['test_mask']).squeeze()
#     print(graph.ndata['test_mask'].size())
#     print(graph.ndata['feat'][indices_test].size())
#     print(graph.ndata['feat'][indices_allx].size())
#     # Extract labels for labeled instances
#     one_hot_label = torch.nn.functional.one_hot(graph.ndata['label'], num_classes=41)
#     y = one_hot_label[torch.nonzero(graph.ndata['train_mask']).squeeze()].numpy()  # Replace 'label' with the actual label name
#     ty = one_hot_label[torch.nonzero(graph.ndata['test_mask']).squeeze()].numpy()
#     ally = one_hot_label[torch.nonzero(graph.ndata['test_mask'] == 0).squeeze()].numpy()
#     print(one_hot_label[torch.nonzero(graph.ndata['train_mask']).squeeze()].size())
#     print(one_hot_label[torch.nonzero(graph.ndata['test_mask']).squeeze()].size())
#     print(one_hot_label[torch.nonzero(graph.ndata['test_mask'] == 0).squeeze()].size())
#     test_indices = graph.ndata['test_mask'].nonzero().squeeze().tolist()
#     dicts = {}
#     for i in range(graph.nodes().size()[0]):
#         dicts[i] = graph.sample_neighbors(i, -1).edges()[0].tolist()
#     print(len(dicts.items()))

#     # save_data_to_pickle(f'{filename_prefix}ind.ogbn-arxiv.x', sp.csr_matrix(x))
#     # save_data_to_pickle(f'{filename_prefix}ind.ogbn-arxiv.tx', sp.csr_matrix(tx))
#     # save_data_to_pickle(f'{filename_prefix}ind.ogbn-arxiv.allx', sp.csr_matrix(allx))
#     # save_data_to_pickle(f'{filename_prefix}ind.ogbn-arxiv.y', y)
#     # save_data_to_pickle(f'{filename_prefix}ind.ogbn-arxiv.ty', ty)
#     # save_data_to_pickle(f'{filename_prefix}ind.ogbn-arxiv.ally', ally)
#     # save_data_to_pickle(f'{filename_prefix}ind.ogbn-arxiv.test.index', test_indices)
#     save_data_to_pickle(f'{filename_prefix}ind.reddit.graph', dicts)

#     # index = []
#     # with open(f'{filename_prefix}ind.ogbn-arxiv.test.index','rb') as file_name:
#     #     unpickled_dict =pickle.load(file_name)
#     #     print(unpickled_dict)
#     # for line in open(f'{filename_prefix}ind.ogbn-arxiv.test.index','rb'):
#     #     print(line)
#     #     index.append(int(line.strip()))

#     # # Extract one-hot labels for test instances
#     # ty = graph.ndata['test_label'].numpy()  # Replace 'test_label' with the actual test label name
    
#     # # Extract labels for all instances (labeled and unlabeled)
#     # ally = graph.ndata['all_label'].numpy()  # Replace 'all_label' with the actual all label name
    
#     # Extract the adjacency matrix
#     # adj = sp.coo_matrix(graph.adjacency_matrix().to_dense().numpy())  # Convert to a sparse matrix
    
#     # # Extract the graph structure in the specified format
#     # graph_structure = collections.defaultdict(list)
#     # for src, dst in zip(adj.row, adj.col):
#     #     graph_structure[src].append(dst)
    
#     # # Extract the indices of test instances
#     # test_indices = graph.ndata['test_mask'].nonzero().squeeze().numpy()  # Replace 'test_mask' with the actual test mask name
    
#     # # Create a dictionary for the main graph in the specified format
#     # custom_format_data_main = {
#     #     'x': sp.csr_matrix(x),
#     #     'y': y,
#     #     # 'ty': ty,
#     #     # 'ally': ally,
#     #     'graph': graph_structure,
#     #     'test_indices': test_indices.tolist()
#     # }
    
#     # # Save the data for the main graph to a pickle file
#     # print(sp.csr_matrix(x))
#     # save_data_to_pickle(f'{filename_prefix}_main.x', sp.csr_matrix(x))
    
#     # Extract node features for the test instances
#     # tx = graph.ndata['test_feat'].numpy()  # Replace 'test_feat' with the actual test node feature name
    
#     # # Extract node features for all instances (labeled and unlabeled)
#     # allx = graph.ndata['all_feat'].numpy()  # Replace 'all_feat' with the actual all node feature name
    
#     # # Create a dictionary for the additional features
#     # custom_format_data_additional = {
#     #     'tx': sp.csr_matrix(tx),
#     #     'allx': sp.csr_matrix(allx)
#     # }
    
#     # Save the additional data to a pickle file
#     # save_data_to_pickle(f'{filename_prefix}_additional.pkl', custom_format_data_additional)


# if __name__ == '__main__':
#     # graph = load_graph('cora')
#     # graph = load_graph('reddit')
#     # graph = load_graph('reddit')
#     # print(graph)
#     graph = load_graph('reddit')
#     print(graph.ndata['train_mask'].shape)
#     # graph = load_graph('ogbn-products')
#     # print(graph)
#     # Example usage
# # Assuming your DGL graph is named 'g' and you want to save the data to 'output_file'
#     dgl_graph_to_custom_format_and_save(graph, './data-gcn/')
#     # graph = load_graph('reddit-s')
#     # print('nodes:', graph.num_nodes())
#     # print('nodes', graph.nodes(), len(graph.nodes()))
#     # graph = load_graph('reddit')
#     # print(graph)
#     # pass
#     # train_nids = torch.nonzero(graph.ndata['train_mask']).view(-1)
#     # dgs = graph.in_degrees(train_nids)
#     # print(max(dgs))
#     # print(min(dgs))


import os
import re
import struct
import time
from functools import wraps
import numpy as np
import pandas as pd
import scipy.sparse as ss
from collections import Counter
import matplotlib.pyplot as plt


def parse_num(filename, mode):
  if not os.path.exists(filename):
    print(f'{filename} not exist!')
    assert False
  if not os.path.isfile(filename):
    print(f'{filename} not a file!')
    assert False
  ret = []
  with open(filename) as f:
    for line in f.readlines():
      if line.find(mode) >= 0:
        nums = re.findall(r"\d+\.?\d*", line[line.find(mode):])
        ret.append(float(nums[0]))
  return ret


def create_dir(path):
  if path and not os.path.exists(path):
    os.makedirs(path)


# def create_fi(path):
#   if path and not os.path.exists(path):
#     os.makedirs(path)


def read_edge_list_fron_binfile(filepath):
  binfile = open(filepath, 'rb')
  edge_num = os.path.getsize(filepath) // 8
  edge_list = []
  for _ in range(edge_num):
    u = struct.unpack('i', binfile.read(4))[0]
    v = struct.unpack('i', binfile.read(4))[0]
    edge_list.append((u, v))
  binfile.close()
  return edge_list    


def show_time(func):
  @wraps(func) # need add this for multiprocessing, keep the __name__ attribute of func
  def with_time(*args, **kwargs):
    time_cost = time.time()
    func(*args, **kwargs)
    time_cost = time.time() - time_cost
    print('function {} cost {:.2f}s'.format(func.__name__, time_cost))
  return with_time


def read_edgelist(filename, sep='\t'):
  data = pd.read_csv(filename, sep=sep, encoding = 'utf-8', header=None)
  return np.array(data)


def edgelist_to_coo_matrix(edgelist):
  "Read data file and return sparse matrix in coordinate format."
  # if the nodes are integers, use 'dtype = np.uint32'
  rows = edgelist[:, 0]
  cols = edgelist[:, 1]
  n_nodes = np.max(edgelist) + 1
  ones = np.ones(len(rows), np.uint32)
  matrix = ss.coo_matrix((ones, (rows, cols)), shape=(n_nodes, n_nodes))
  # print(matrix.shape)
  return matrix


def read_edlist_to_coo_matrix(filename='edges.txt'):
  "Read data file and return sparse matrix in coordinate format."
  # if the nodes are integers, use 'dtype = np.uint32'
  data = pd.read_csv(filename, sep = '\t', encoding = 'utf-8', header=None)
  rows = data.iloc[:, 0]  # Not a copy, just a reference.
  cols = data.iloc[:, 1]
  n_nodes = max(data.max()) + 1
  ones = np.ones(len(rows), np.uint32)
  matrix = ss.coo_matrix((ones, (rows, cols)), shape=(n_nodes, n_nodes))
  # print(matrix.shape)
  return matrix  


def draw_power_law(dataset, edge_list, lock=None):
  # convert to networkx is too slowly
  # G = nx.from_edgelist(edge_list, create_using=nx.DiGraph())
  # assert len(edge_list) == G.number_of_edges()
  # print('{} has {} edges'.format(filepath, G.number_of_edges()))
  # degree_freq = nx.degree_histogram(G)
  # degrees = range(len(degree_freq))

  # compute
  degree_dict = {}
  for u, v in edge_list:
    if u not in degree_dict:
      degree_dict[u] = 1
    else:
      degree_dict[u] += 1
  degree_list = degree_dict.values()
  max_degree = max(degree_list)
  tmp = Counter(degree_list)
  degrees = np.array(list(tmp.keys()))
  degree_freq = np.array(list(tmp.values()))
  # print(type(degrees), degrees)

  # save
  with open(f'./degree-distribute/npz/{dataset}.npz', 'wb') as f:
    np.save(f, degrees)
    np.save(f, degree_freq)
  
  # # load
  # with open(f'./degree-distribute/npz/{dataset}.npz', 'rb') as f:
  #   degrees = np.load(f)
  #   degree_freq = np.load(f)

  try:
    if lock is not None:
      lock.acquire()
    plt.figure(figsize=(8, 6)) 
    plt.loglog(degrees[:], degree_freq[:], '.') 
    plt.xlabel('Degree', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    # plt.xlim(1, max(degrees))
    # plt.ylim(1, max(degree_freq))
    print(f'save to {os.getcwd()}/degree-distribute/{dataset}.pdf')
    plt.savefig(f'./degree-distribute/{dataset}.pdf', format='pdf')
  except Exception as err:
      raise err
  finally:
    if lock is not None:
      lock.release()  


def plot_bar(x_name, y_name, datas, labels, filename='bar.pdf', color=None):
  assert (len(datas[0]) == len(x_name))
  #  == len(labels)
  # x_name = ['cora', 'citeseer', 'pubmed', 'arxiv', 'reddit', 'orkut', 'wiki']
  # aligraph = [0.289, 0.463, 2.459, 1.733,0, 0, 0]
  # nts = [0.027, 0.053, 0.059, 0.270, 2.1180, 9.133, 20.435]  

  # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
  # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
  # 线型：-  --   -.  :    ,
  # marker：.  ,   o   v    <    *    +    1
  plt.figure(figsize=(6, 3))
  plt.grid(linestyle="-.")  # 设置背景网格线为虚线
  # linestyle = "-"
  x = np.arange(len(x_name))
  fontsize = 12
  # n 为有几个柱子
  # total_width, n = 0.8, 2
  total_width, n = 0.8, len(datas)
  width = total_width / n
  offset = (total_width - width) / 2 
  x = x - offset
  # x = x - total_width /2

  # low = 0.05
  # up = 0.44
  low = 0
  up = np.max(datas)
  up = np.max(datas) + 1
  plt.ylim(low, up)
  # plt.xlabel("Amount of Data", fontsize=15)
  # plt.ylabel(f"Time (s)", fontsize=20)
  plt.ylabel(y_name, fontsize=fontsize)
  # labels = ['GraphScope', 'NTS']

  # 'tomato', 'blue', 'orange', 'green', 'purple', 'deepskyblue'
  if color is None:
    color = ['blue', 'green', 'orange', 'tomato', 'purple', 'deepskyblue']

  for i, data in enumerate(datas):
    plt.bar(x + width * i, data, width=width, color=color[i], edgecolor='w')  # , edgecolor='k',)
    

  plt.xticks(x + offset, labels=x_name, fontsize=fontsize, rotation=0)
  plt.yticks(fontsize=fontsize)

  # num1, num2 = 1, 1.1
  # plt.legend(labels=labels, ncol=4, prop={'size': 11}, bbox_to_anchor=(num1, num2))
  plt.legend(labels=labels, ncol=4, prop={'size': fontsize}, loc='best')
  # num1, num2 = 0.9, 1.2
  # plt.legend(labels=labels, ncol=4, prop={'size': 11}, bbox_to_anchor=(num1, num2))

  plt.tight_layout()
  print(f"save to {filename}")
  plt.savefig(filename, format='pdf')
  plt.show()
  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中      


def plot_stack_bar(x_name, y_name, datas, labels, filename='bar.pdf', color=None):
  assert (len(datas[0]) == len(x_name))
  #  == len(labels)
  # x_name = ['cora', 'citeseer', 'pubmed', 'arxiv', 'reddit', 'orkut', 'wiki']
  # aligraph = [0.289, 0.463, 2.459, 1.733,0, 0, 0]
  # nts = [0.027, 0.053, 0.059, 0.270, 2.1180, 9.133, 20.435]  

  # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
  # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
  # 线型：-  --   -.  :    ,
  # marker：.  ,   o   v    <    *    +    1
  plt.figure(figsize=(6, 3))
  plt.grid(linestyle="-.")  # 设置背景网格线为虚线
  # linestyle = "-"
  x = np.arange(len(x_name))
  # n 为有几个柱子
  # total_width, n = 0.8, 2
  total_width, n = 0.8, len(datas)
  width = total_width / n
  offset = (total_width - width) / 2 
  x = x - offset
  # x = x - total_width /2

  # low = 0.05
  # up = 0.44
  low = 0
  up = 1
  plt.ylim(low, up)
  # plt.xlabel("Amount of Data", fontsize=15)
  # plt.ylabel(f"Time (s)", fontsize=20)
  plt.ylabel(y_name, fontsize=12)
  # labels = ['GraphScope', 'NTS']

  # 'tomato', 'blue', 'orange', 'green', 'purple', 'deepskyblue'
  if color is None:
    color = ['blue', 'green', 'orange', 'tomato', 'purple', 'deepskyblue']

  pre_bottom = np.zeros_like(datas[0])
  for i, data in enumerate(datas):
    plt.bar(x + width, data, label=labels[i], width=width, color=color[i], edgecolor=None, bottom=pre_bottom)# , edgecolor='k',)  
    pre_bottom += data
    # plt.bar(x + width * i, data, width=width, color=color[i], edgecolor='w')  # , edgecolor='k',)
    

  plt.xticks(x + offset, labels=x_name, fontsize=12, rotation=0)
  y_ticks = [f'{x:.0%}' for x in np.linspace(start=0, stop=1, num=6)]
  plt.yticks(np.linspace(start=0, stop=1, num=6), labels=y_ticks, fontsize=12)

  num1, num2 = 1, 1.2
  plt.legend(labels=labels, ncol=4, prop={'size': 11}, bbox_to_anchor=(num1, num2))
  # plt.legend(labels=labels, ncol=2, prop={'size': 11}, loc='best')

  plt.tight_layout()
  print(f"save to {filename}")
  plt.savefig(filename, format='pdf')
  plt.show()
  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中        


def plot_line(X, Y, labels, savefile=None, color=None, x_label=None, y_label=None, show=False, x_ticks=None, 
              x_name=None, loc=None, y_ticks=None, y_name=None, high_mark='.', ylim=None, draw_small=False,
              xscale=None):
  assert(len(X) == len(Y) == len(labels))
  # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
  # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
  # 线型：-  --   -.  :    ,
  # marker：.  ,   o   v    <    *    +    1
  # plt.figure(figsize=(8, 6))
  # linestyle = "-"
  # fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=400) 
  fig, ax = plt.subplots(1, 1, figsize=(5, 4)) 
  ax.grid(linestyle="-.")  # 设置背景网格线为虚线
  # ax = ax.gca()
  # ax.spines['top'].set_visible(False)  # 去掉上边框
  # ax.spines['right'].set_visible(False)  # 去掉右边框
  fontsize = 13
  linewidth = 1.2
  markersize = 7

  if color is None:
    # color = ['blue', 'green', 'tomato', 'orange', 'purple', 'deepskyblue', 'red', 'cyan', 'magenta', 'yellow', 'black']
    # color = ['blue', 'green', 'orange', 'purple', 'red', 'black', 'yellow', 'cyan', 'magenta', 'pink',  'deepskyblue', 'tomato']
    color = ['orange', 'blue', 'green', 'tomato', 'purple', 'deepskyblue', 'red']
  
  for i in range(len(X)):
    if len(X[i]) == 0:
      continue
    ax.plot(X[i], Y[i], marker='', markersize=markersize, color=color[i], alpha=1, label=labels[i], linewidth=linewidth)
    # ax.plot(X[i], Y[i], marker='', markersize=markersize, alpha=1, label=labels[i], linewidth=linewidth)

    # plot max point
    # pos = np.where(np.amax(Y[i]) == Y[i])[0].tolist()
    # pos = pos[0]
    # ax.plot(X[i][pos], Y[i][pos], marker='x', markersize=markersize, color='red', alpha=1, linewidth=linewidth)
    # ax.plot(X[i][pos], Y[i][pos], marker=high_mark, markersize=markersize-2, alpha=1, linewidth=linewidth)

  if x_ticks is not None and x_name is not None:
    # print(x_ticks)
    ax.set_xticks(x_ticks, x_name, fontsize=fontsize - 2)  # 默认字体大小为10
    ax.set_xlim(np.min(x_ticks), np.max(x_ticks))
  else:
    max_xticks = max(max(x) if len(x) > 0 else 0 for x in X)
    x_ticks = np.linspace(0, max_xticks, 6).tolist()
    ax.set_xlim(np.min(x_ticks), np.max(x_ticks))
    x_name = [f'{x:.2f}' for x in x_ticks]
    ax.set_xticks(x_ticks, x_name, fontsize=fontsize - 2)  # 默认字体大小为10


  if y_ticks is not None and y_name is not None:
    # print(y_ticks)
    ax.set_yticks(y_ticks, y_name, fontsize=fontsize - 2)  # 默认字体大小为10
    ax.set_ylim(np.min(y_ticks), np.max(y_ticks))
  else:
    max_xticks = max(max(x) if len(x) > 0 else 0 for x in Y)
    y_ticks = np.linspace(0, max_xticks, 6).tolist()
    ax.set_ylim(np.min(y_ticks), np.max(y_ticks))
    y_name = [f'{x:.2f}' for x in y_ticks]
    ax.set_yticks(y_ticks, y_name, fontsize=fontsize - 2)  # 默认字体大小为10

  ax.set_ylabel(y_label, fontsize=fontsize)
  ax.set_xlabel(x_label, fontsize=fontsize)
  # ax.xlim(0, np.max(X) + 1)  # 设置x轴的范围

  if xscale is not None:
    ax.set_xscale('log', base=xscale)

  # ax.legend()
  # 显示各曲线的图例 loc=3 lower left
  if not loc:
    loc = 'best'
  ax.legend(loc=loc, numpoints=1, ncol=1, prop={'size': fontsize})
  # plt.legend(labels=labels, ncol=4, prop={'size': 11}, bbox_to_anchor=(num1, num2))
  # leg = ax.gca().get_legend()
  # ltext = leg.get_texts()
  # ax.setp(ltext, fontsize=15)
  # ax.setp(ltext, fontsize=25, fontweight='bold')  # 设置图例字体的大小和粗细
  # ax.tight_layout()

  if not savefile:
    savefile = 'plot_line.pdf'
  print(f'save to {savefile}')
  fig.savefig(f'{savefile}', format='pdf')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
  if show:
    plt.show()
  plt.close()


if __name__ == '__main__':
  print("this utils fuction tools.")



