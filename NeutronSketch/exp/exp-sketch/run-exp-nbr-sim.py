import sys
import os
import time
import utils
import numpy as np
import matplotlib.pyplot as plt

init_command = [
    "WEIGHT_DECAY:0.0001",
    "DROP_RATE:0.5",
    "DECAY_RATE:0.97",
    "DECAY_EPOCH:100",
    "PROC_OVERLAP:0",
    "PROC_LOCAL:0",
    "PROC_CUDA:0",
    "PROC_REP:0",
    "LOCK_FREE:1",
    "TIME_SKIP:0",
    "MINI_PULL:1",
    "BATCH_NORM:0",
    "PROC_REP:0",
    "LOCK_FREE:1",
    "CACHE_TYPE:rate",
    "CACHE_POLICY:degree",
    "CACHE_RATE:1",
    "SAMPLE_MODE:4",
    "DEGREE_SWITCH:-1",
    "TARGET_DEGREE:100",
    "PIPELINES:1",
    "TIME_SKIP:0",
    "BEST_PARAMETER:0",
    "SKETCH_MODE:6",
    "NODES_NUM:3",
    "RUN_MODE:turn",
    "MODE:zerocopy",
]

graph_config = {
    'reddit': "VERTICES:232965\nEDGE_FILE:../../data/reddit/reddit.edge\nFEATURE_FILE:../../data/reddit/reddit.feat\nLABEL_FILE:../../data/reddit/reddit.label\nMASK_FILE:../../data/reddit/reddit.mask\nLAYERS:602-128-41\n",
    'ogbn-arxiv': "VERTICES:169343\nEDGE_FILE:../../data/ogbn-arxiv/ogbn-arxiv.edge\nFEATURE_FILE:../../data/ogbn-arxiv/ogbn-arxiv.feat\nLABEL_FILE:../../data/ogbn-arxiv/ogbn-arxiv.label\nMASK_FILE:../../data/ogbn-arxiv/ogbn-arxiv.mask\nLAYERS:128-128-40\n",
    'ogbn-products': "VERTICES:2449029\nEDGE_FILE:../../data/ogbn-products/ogbn-products.edge\nFEATURE_FILE:../../data/ogbn-products/ogbn-products.feat\nLABEL_FILE:../../data/ogbn-products/ogbn-products.label\nMASK_FILE:../../data/ogbn-products/ogbn-products.mask\nLAYERS:100-128-47\n",
    # 'AmazonCoBuy_computers': "VERTICES:13752\nEDGE_FILE:../../data/AmazonCoBuy_computers/AmazonCoBuy_computers.edge\nFEATURE_FILE:../../data/AmazonCoBuy_computers/AmazonCoBuy_computers.feat\nLABEL_FILE:../../data/AmazonCoBuy_computers/AmazonCoBuy_computers.label\nMASK_FILE:../../data/AmazonCoBuy_computers/AmazonCoBuy_computers.mask\nLAYERS:767-128-10\n",
    # 'AmazonCoBuy_photo': "VERTICES:7650\nEDGE_FILE:../../data/AmazonCoBuy_photo/AmazonCoBuy_photo.edge\nFEATURE_FILE:../../data/AmazonCoBuy_photo/AmazonCoBuy_photo.feat\nLABEL_FILE:../../data/AmazonCoBuy_photo/AmazonCoBuy_photo.label\nMASK_FILE:../../data/AmazonCoBuy_photo/AmazonCoBuy_photo.mask\nLAYERS:745-128-8\n",
    # 'enwiki-links': "VERTICES:13593032\nEDGE_FILE:../../data/enwiki-links/enwiki-links.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    # 'livejournal': "VERTICES:4846609\nEDGE_FILE:../../data/livejournal/livejournal.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    # 'lj-large': "VERTICES:7489073\nEDGE_FILE:../../data/lj-large/lj-large.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    # 'lj-links': "VERTICES:5204175\nEDGE_FILE:../../data/lj-links/lj-links.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    # 'europe_osm': "VERTICES:50912018\nEDGE_FILE:../../data/europe_osm/europe_osm.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    # 'dblp-2011': "VERTICES:933258\nEDGE_FILE:../../data/dblp-2011/dblp-2011.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    # 'frwiki-2013': "VERTICES:1350986\nEDGE_FILE:../../data/frwiki-2013/frwiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    # 'dewiki-2013': "VERTICES:1510148\nEDGE_FILE:../../data/dewiki-2013/dewiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    # 'itwiki-2013': "VERTICES:1016179\nEDGE_FILE:../../data/itwiki-2013/itwiki-2013.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    # 'hollywood-2011': "VERTICES:1985306\nEDGE_FILE:../../data/hollywood-2011/hollywood-2011.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
    # 'enwiki-2016': "VERTICES:5088560\nEDGE_FILE:../../data/enwiki-2016/enwiki-2016.edge\nFEATURE_FILE:random\nLABEL_FILE:random\nMASK_FILE:random\nLAYERS:600-128-60\n",
}
# SKETCH_MODE=6,
#                         NODES_NUM=3,
#                         RUN_MODE='turn',
#                         LOW1=low_1,
#                         HIGH1=high_1,
#                         LOW2=low_2,
#                         HIGH2=high_2,

def new_command(
    dataset,
    fanout='2,2',
    valfanout='-1,-1',
    batch_size='6000',
    val_batch_size='6000',
    algo='GCNNEIGHBORGPU',
    epochs='10',
    batch_type='random',
    lr='0.01',
    run='1',
    classes='1',
    lower_fanout=4,
    # sketch_mode=0,
    # nodes_num=0,
    # run_mode='turn',
    low_1=0.0,
    high_1=0.0,
    low_2=0.0,
    high_2=0.0,
    **kw,
):
    other_config = init_command[:]
    print('length:',len(other_config))
    print('init length:',len(init_command))
    other_config.append(f'ALGORITHM:{algo}')
    other_config.append(f'FANOUT:{fanout}')
    other_config.append(f'VALFANOUT:{valfanout}')
    other_config.append(f'BATCH_SIZE:{batch_size}')
    other_config.append(f'VALBATCH_SIZE:{val_batch_size}')
    other_config.append(f'EPOCHS:{epochs}')
    other_config.append(f'BATCH_TYPE:{batch_type}')
    other_config.append(f'LEARN_RATE:{lr}')
    other_config.append(f'RUNS:{lr}')
    other_config.append(f'CLASSES:{classes}')
    other_config.append(f'RUNS:{run}')
    other_config.append(f'LOW1:{low_1}')
    other_config.append(f'HIGH1:{high_1}')
    other_config.append(f'LOW2:{low_2}')
    other_config.append(f'HIGH2:{high_2}')
    other_config.append(f'LOWER_FANOUT:{lower_fanout}')
    print('length:',len(other_config))
    # for k, v in kw.items():
    #     other_config.append(f'{k}:{v}')
    #     print(k, v)
    # assert False
    ret = graph_config[dataset] + '\n'.join(other_config)
    return ret


def run(dataset, cmd, log_path, suffix=''):
    if not os.path.exists(log_path):
        utils.create_dir(log_path)

    run_time = time.time()
    with open('exp.cfg', 'w') as f:
        f.write(cmd)

    run_command = f'mpiexec -np 1 ../build/nts exp.cfg > {log_path}/{dataset}{suffix}.log'
    print('running: ', run_command)
    os.system(run_command)

    run_time = time.time() - run_time
    print(f'done! cost {run_time:.2f}s')




if __name__ == '__main__':
    utils.create_dir('./build')
    os.system('cd ../build && cmake ../.. && make -j $(nproc) && cd ..')

    # # datasets = ['AmazonCoBuy_computers', 'AmazonCoBuy_photo', 'reddit', 'ogbn-arxiv','ogbn-products']
    datasets = ['reddit']
    # datasets = ['ogbn-products']
    # # datasets = ['reddit', 'ogbn-arxiv', 'ogbn-products']
    # # datasets = ['reddit', 'ogbn-arxiv']
    # datasets = ['AmazonCoBuy_computers', 'ogbn-products', 'AmazonCoBuy_photo']
    # datasets = ['ogbn-arxiv']
    # datasets = [ 'ogbn-products','reddit']
    # datasets = ['ogbn-arxiv', 'reddit', 'ogbn-products']
    

    # sample_rate = ['1.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    # batch_sizes = {
    #     'AmazonCoBuy_computers': 512,
    #     'AmazonCoBuy_photo': 512,
    #     'reddit': 65536,
    #     'ogbn-arxiv': 2048,
    #     'ogbn-products': 2048,
    # }
    # learn_rates = {
    #     'AmazonCoBuy_computers': 0.001,
    #     'AmazonCoBuy_photo': 0.001,
    #     'reddit': 0.01,
    #     'ogbn-arxiv': 0.001,
    #     'ogbn-arxiv': 0.01,
    #     'ogbn-products': 0.01,
    #     'ogbn-products': 0.001,
    # }
    
    # run_times = {
    #     'reddit': 600,
    #     'ogbn-arxiv': 50,
    #     'ogbn-products': 600,
    #     # 'AmazonCoBuy_computers': 200,
    #     # 'AmazonCoBuy_photo': 200,
    # }

    print(123)
    lists = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
    # low_1='0.0'
    # high_1='0.1'
    # low_2='0.5'
    # high_2='0.7'
    # for i in range(10):
    #     low_1 = lists[i]
    #     high_1 = lists[i+1]
    #     for j in range(i,10):
    #         low_2 = lists[j]
    #         high_2 = lists[j+1]
    #         print(low_1,' ',high_1, ' ',low_2,' ',high_2)
    for ds in datasets:
        file_path = f'./log/{ds}-nbr-sim'
        utils.create_dir(file_path)
            # cmd = new_command(ds, batch_type='shuffle', fanout='10,25', valfanout='10,25', epochs=3, batch_size=bs, RUN_TIME=run_times[ds])
            # cmd = new_command(ds, batch_type='shuffle', fanout='10,25', lr=0.01, epochs=3, batch_size=bs, RUN_TIME=run_times[ds], valfanout='10,25')
        for i in range(10):
            low_1 = lists[i]
            high_1 = lists[i+1]
            for j in range(i,10):
                low_2 = lists[j]
                high_2 = lists[j+1]
                    # print(sr)
                cmd = new_command(
                                ds,
                                fanout='10,25',
                                batch_size=1024,
                                # valfanout='10,25',
                                valfanout='10,25',
                                val_batch_size=1000000000,
                                # valfanout='10,25',
                                # val_batch_size=batch_sizes[ds],
                                algo='GCNNEIGHBORGPUDEGREE',
                                epochs=200,
                                batch_type='shuffle',
                                lr=0.001,
                                RUN_TIME=300,
                                # CACHE_POLICY='sample',
                                # CACHE_TYPE='gpu_memory',
                                # MODE='pipeline',
                                # lower_fanout=15,
                                # SKETCH_MODE=6,
                                # NODES_NUM=3,
                                low_1=low_1,
                                high_1=high_1,
                                low_2=low_2,
                                high_2=high_2,
                                lower_fanout=1,
                            )
                # print(cmd)
                run(ds, cmd, file_path, suffix=f'-{low_1}-{high_1}-and-{low_2}-{high_2}')