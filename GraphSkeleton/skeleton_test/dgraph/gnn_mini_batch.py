from dataloader import dataloader 
import os
from torch_geometric.utils import to_undirected
import torch
import torch.nn.functional as F
import time
from torch_geometric.data import Data
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch_geometric.loader import NeighborSampler
import numpy as np
from sklearn import metrics
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from models import GATv2_NeighSampler, SAGE_NeighSampler, GIN_NeighSampler, GCN_NeighSampler
from config import config, update_config


def eval_acc(y_true, y_pred):
        y_pred = y_pred.argmax(1)

        correct = y_true == y_pred
        print(correct.shape)
        print(correct.sum().item())
        acc = float(correct.sum().item())/len(correct)

        return acc

def train():
    model.train()
    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        n_id = n_id.to(device)
        # print(adjs)
        optimizer.zero_grad()
        out = model(x[n_id], adjs)
        # print(batch_size)
        loss = F.nll_loss(out, y[n_id[:batch_size]])
        loss.backward()
        optimizer.step()
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(y[n_id[:batch_size]]).sum())

    loss = total_loss / len(train_loader)
    approx_acc = total_correct / data.train_mask.size(0)
    return loss, approx_acc


@torch.no_grad()
def test(masks):
    model.eval()
    out = model.inference(x)
    aucs = []
    # print(out.shape)
    # print(out[masks[0]].shape)
    # print(masks[0].shape,masks[1].shape,data.y[masks[0]],data.y[masks[1]])
    
    output = out.argmax(1).cpu().numpy()
    accuracy_val = metrics.accuracy_score(data.y[masks[0]], output[masks[0]])
    accuracy_test = metrics.accuracy_score(data.y[masks[1]], output[masks[1]])
    # accuracy_val = eval_acc(data.y[masks[0]],out[masks[0]])
    # accuracy_test = eval_acc(data.y[masks[1]],out[masks[1]])
    aucs.append(accuracy_val)
    aucs.append(accuracy_test)

    # for mask in masks:
    #     test_auc = metrics.roc_auc_score(data.y[mask].cpu().numpy(), out[mask,1].detach().cpu().numpy(),multi_class='ovo')
    #     aucs.append(test_auc)
    return aucs


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument('--device', default='gpu',
                    help='device for computing')
parser.add_argument('--cut', default='skeleton_gamma', choices = ['no', 'skeleton_alpha','skeleton_beta','skeleton_gamma'],
                    help='use skeleton')
parser.add_argument('--model', default='sage', choices = ['sage', 'gat', 'gin'],
                    help='model')
parser.add_argument('--mlp', action='store_true',
                    help='if use mlp for classification')
parser.add_argument('--iter', default=10, type=int, 
                    help='iteration for running')
args = parser.parse_args()
update_config(config, args.model)
print(args)

if args.device == 'cpu':
    print('| Using cpu |')
    device = torch.device('cpu')
else:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Used >tmp')
    memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    print('| Using GPU:' + str(np.argmin(memory_gpu)), ' |')
    device = torch.device('cuda:'+str(np.argmin(memory_gpu)))

data = dataloader(args.cut)
# print(data.train_mask.max())
train_loader = NeighborSampler(data.edge_index, node_idx=data.train_mask,
                               sizes=config['sample_size'], batch_size=config['batch_size'],
                               shuffle=True, num_workers=12)
subgraph_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1],
                                  batch_size=config['batch_size'], shuffle=False,
                                  num_workers=12)

if args.model == 'sage':
    model = SAGE_NeighSampler(device, subgraph_loader, data.x.shape[1], 
                                config['hidden_channels'], 
                                out_channels=40, 
                                num_layers=config['num_layers'], 
                                dropout=config['dropout'], 
                                batchnorm=config['batchnorm'])
elif args.model == 'gat':
    model = GATv2_NeighSampler(device, subgraph_loader, data.x.shape[1], 
                                config['hidden_channels'], 
                                out_channels=2, 
                                num_layers=config['num_layers'], 
                                dropout=config['dropout'], 
                                layer_heads=[4,2,1], 
                                batchnorm=config['batchnorm'])
elif args.model == 'gin':
    model = GIN_NeighSampler(device, subgraph_loader, data.x.shape[1], 
                                config['hidden_channels'], 
                                out_channels=2, 
                                num_layers=config['num_layers'], 
                                dropout=config['dropout'], 
                                batchnorm=config['batchnorm'])
elif args.model == 'gcn':
    model = GCN_NeighSampler(device, subgraph_loader, data.x.shape[1], 
                                config['hidden_channels'], 
                                out_channels=40, 
                                num_layers=config['num_layers'], 
                                dropout=config['dropout'], 
                                batchnorm=config['batchnorm'])
print('-'*8, args.model, '-'*8)
print(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
x, edge_index, y = data.x.to(torch.float32).to(device), data.edge_index.to(torch.float32).to(device), data.y.to(device)
# print('hh',data.validate())
best_val_acc = best_test_acc = 0
test_accs = []
Time = []
Log_train = []
Log_valid = []
Log_test = []
for run in range(args.iter):
    print('')
    print(f'Run {run:02d}:')
    print('')

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    start = time.perf_counter()
    best_val_acc = final_test_acc = 0
    log_train = []
    log_valid = []
    log_test = []
    count = 0
    for epoch in range(config['epochs']):
        loss, acc = train()
        if epoch % 1 == 0:
            print(f'Train Epoch {epoch:02d}, Loss: {loss:.4f}')
            val_acc, test_acc = test([data.valid_mask, data.test_mask])
            print(f'Evaluation Epoch:{epoch:02d}: Val: {val_acc:.4f},'
                  f'Test: {test_acc:.4f}')
            print('Time: ',time.perf_counter()-start)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
                count=0
            else:
                count+=1
                if(count == 10):
                    break

            
            log_train.append(acc)
            log_valid.append(val_acc)
            log_test.append(test_acc)

    test_accs.append(final_test_acc)
    end = time.perf_counter()
    Time.append(end-start)
    Log_train.append(log_train)
    Log_valid.append(log_valid)
    Log_test.append(log_test)
    print('-----------------------')
    print('Consuming time{}:'.format(end-start))
    print('best acc{}:'.format(final_test_acc))

test_acc = torch.tensor(test_accs)
aver_time = torch.tensor(Time)
print('============================')
print(f'Final Test: {test_acc.mean():.4f} ± {test_acc.std():.4f}')
print(f'aver time: {aver_time.mean():.4f} ± {aver_time.std():.4f}')
print(test_accs)