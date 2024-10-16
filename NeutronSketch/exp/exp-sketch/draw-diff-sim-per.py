import os
import re
import matplotlib.pyplot as plt
import numpy as np
def parse_nodes_num(file_path):
    assert file_path
    val_acc = []
    if not os.path.exists(file_path):
        print(f'{file_path} not exist')    
    with open(file_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if(re.findall(r"Sketch train nodes: \((\d+)\)", line)!=[]):
                # val_acc.append(re.findall(r'Sketch train nodes: \((\d+)\)', line)[0])
                return int(re.findall(r'Sketch train nodes: \((\d+)\)', line)[0])
    # return float(max(val_acc))
    return

def cal_data():
    dataset = ['ogbn-arxiv','reddit','ogbn-products']
    train_nodes = {'reddit':153431,
                    'ogbn-products':196615,
                    'ogbn-arxiv':90941}
    res = []
    lists = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
    for ds in dataset:
        tmp = []
        for i in range(len(lists)-1):
            file_path = f'./log/{ds}-nbr-sim'
            suffix = f'-{lists[i]}-{lists[i+1]}-and-{lists[i]}-{lists[i+1]}'
            log_file = f'{file_path}/{ds}{suffix}.log'
            tmp.append(parse_nodes_num(log_file))
        res.append([x/train_nodes[ds] for x in tmp])
        # res.append(tmp)
    # print(res)
    return res

def draw_bar():
    data = cal_data()
    # x = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9']
    x = np.linspace(0, 1, 11)
    print(x)
    bar_width = 0.35
    # plt.figure(figsize=(5, 4))  # 设置图形大小

    myparams = {
                'axes.labelsize': '12',
                'xtick.labelsize': '12',
                'ytick.labelsize': '12',
                'font.family': 'Times New Roman',
                'figure.figsize': '5, 4',  #图片尺寸
                'lines.linewidth': 2,
                'legend.fontsize': '20',
                # 'legend.loc': 'best', #[]"upper right", "upper left"]
                'legend.loc': 'upper left', #[]"upper right", "upper left"]
                'legend.numpoints': 1,
                'legend.frameon': False,
                # 'lines.ncol': 2,
                }
    plt.rcParams.update(myparams)

    fig, (ax1, ax3, ax5) = plt.subplots(nrows=1, ncols=3, figsize=(10, 2.5))
    plt.subplots_adjust(wspace=0.4)
    widths = x[1] - x[0]

    ax1.bar( x[:-1],data[0],width=widths, align='edge', color = '#0080FF',edgecolor='black')  # 绘制柱形图
    ax1.set_xlim(0.0,1.0)
    ax1.set_ylim(0.0,1.0)
    ax1.set_xticks(np.arange(0, 1.1, 0.1))
    # 添加标题和标签
    # ax1.set_title('Reddit',fontsize=20)
    ax1.set_xlabel('Neighbor Similarity',fontsize=20)
    ax1.set_ylabel('Percentage(%)',fontsize=20)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.tick_params(axis='y', labelsize=11)

    ax3.bar( x[:-1],data[1],width=widths, align='edge', color = '#0080FF',edgecolor='black')  # 绘制柱形图
    ax3.set_xlim(0.0,1.0)
    ax3.set_ylim(0.0,1.0)
    ax3.set_xticks(np.arange(0, 1.1, 0.1))
    # 添加标题和标签
    # ax1.set_title('Reddit',fontsize=20)
    ax3.set_xlabel('Neighbor Similarity',fontsize=20)
    ax3.set_ylabel('Percentage(%)',fontsize=20)
    ax3.tick_params(axis='x', labelsize=11)
    ax3.tick_params(axis='y', labelsize=11)

    ax5.bar( x[:-1],data[2],width=widths, align='edge', color = '#0080FF',edgecolor='black')  # 绘制柱形图
    ax5.set_xlim(0.0,1.0)
    ax5.set_ylim(0.0,1.0)
    ax5.set_xticks(np.arange(0, 1.1, 0.1))
    # ax3.set_xticklabels(x,['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'])
    # 添加标题和标签
    # ax3.set_title('Ogbn-Products',fontsize=20)
    ax5.set_xlabel('Neighbor Similarity',fontsize=20)
    ax5.set_ylabel('Percentage(%)',fontsize=20)
    ax5.tick_params(axis='x', labelsize=11)
    ax5.tick_params(axis='y', labelsize=11)
    ax1.annotate('(a) Ogbn-Arxiv', xy=(0.5, -0.37), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=20)
    ax3.annotate('(b) Reddit', xy=(0.5, -0.37), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=20)
    ax5.annotate('(c) Ogbn-Products', xy=(0.5, -0.37), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=20)
    plt.savefig(f'./picture/diff-sim-per.pdf',bbox_inches='tight')
    # 显示图形
    plt.show() 

def draw_plot():
    data = cal_data()
    sums = sum(data[0])
    tmp_sum = 0
    res = []
    for i in data[0]:
        tmp_sum+=i
        res.append(tmp_sum/sums)
    print(res)

if __name__ == '__main__':
    # draw_bar()
    draw_bar()



# import matplotlib.pyplot as plt
# import numpy as np

# # 设置区间和密度
# intervals = np.linspace(0, 1, 11)  # 生成11个点，从0到1，包括0和1，即10个区间
# densities = np.linspace(0.1, 1.0, 10)  # 生成每个区间的密度，从0.1到1.0

# # 生成数据
# data = []
# for i in range(len(densities)):
#     data.extend([intervals[i]] * int(1000 * densities[i]))

# # 绘制直方图
# plt.figure(figsize=(10, 6))
# plt.hist(data, bins=intervals, color='blue', edgecolor='black')
# plt.title('Density Distribution Plot')
# plt.xlabel('Interval')
# plt.ylabel('Density')
# plt.xticks(intervals)  # 设置x轴的刻度显示为区间
# plt.grid(True)  # 显示网格
# plt.savefig(f'./picture/diff-sim-per.pdf',bbox_inches='tight')
# plt.show()