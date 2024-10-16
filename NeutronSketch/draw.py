# # -*- coding: utf-8 -*-

# from ast import parse
# from calendar import c
# import numpy as np
# import matplotlib
# import os, re
# import itertools

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

# import matplotlib.pyplot as plt
# plt.rcParams["font.family"] = "Times New Roman"


# def plot_bar(x_name, y_name, datas, labels, filename='bar.pdf', color=None):
#   # print(x_name, y_name)
#   # print(datas)
#   # print(labels)
#   assert (len(datas[0]) == len(x_name))
#   #  == len(labels)
#   # x_name = ['cora', 'citeseer', 'pubmed', 'arxiv', 'reddit', 'orkut', 'wiki']
#   # aligraph = [0.289, 0.463, 2.459, 1.733,0, 0, 0]
#   # nts = [0.027, 0.053, 0.059, 0.270, 2.1180, 9.133, 20.435]  

#   # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
#   # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
#   # 线型：-  --   -.  :    ,
#   # marker：.  ,   o   v    <    *    +    1
#   plt.figure(figsize=(7, 4))
#   # linestyle = "-"
#   x = np.arange(len(x_name))
#   # n 为有几个柱子
#   # total_width, n = 0.8, 2
#   total_width, n = 0.8, len(datas)
#   width = total_width / n
#   offset = (total_width - width) / 2 
#   x = x - offset
#   # x = x - total_width /2

#   # low = 0.05
#   # up = 0.44
#   low = 0
#   up = np.max(datas)
#   plt.ylim(low, up + 1)
#   # plt.xlabel("Amount of Data", fontsize=15)
#   # plt.ylabel(f"Time (s)", fontsize=20)
#   plt.ylabel(y_name, fontsize=20)
#   # labels = ['GraphScope', 'NTS']

#   # 'tomato', 'blue', 'orange', 'green', 'purple', 'deepskyblue'
#   if color is None:
#     color = ['blue', 'green', 'tomato', 'orange', 'purple', 'deepskyblue']
  

#   for i, data in enumerate(datas):
#     plt.bar(x + width * i, data, width=width, color=color[i], edgecolor='w')  # , edgecolor='k',)
    

#   plt.xticks(x + offset, labels=x_name, fontsize=15)

#   plt.legend(labels=labels, ncol=2, prop={'size': 14})

#   plt.tight_layout()
#   plt.savefig(filename, format='pdf')
#   plt.show()
#   # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中

# def get_time_acc(accs, times, best, early_stop=True):
#   # print(times)
#   if not isinstance(times, list):
#     times = [times for _ in range(len(accs))]
#   # print(times)
#   # assert(False)
#   # print(times)
#   # print(accs,best, 'find')
#   idx = len(accs)
  
#   if early_stop:
#     idx = 0
#     while accs[idx] < best:
#       idx += 1
#   # idx = bisect.bisect(accs, best)
#   idx = min(idx+10, len(accs))
#   accs_ret = accs[:idx+1]
#   times_ret = list(itertools.accumulate(times[:idx+1]))
#   # print(len(accs_ret))
#   # print(accs_ret[-1], best)
#   # assert accs_ret[-1] >= best
#   assert len(accs_ret) == len(times_ret)
#   # print(len(accs_ret))
#   return [times_ret, accs_ret]


# def plot_line(X, Y, labels, savefile=None, color=None, y_label=None):
#   assert(len(X) == len(Y) == len(labels))
#   # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
#   # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
#   # 线型：-  --   -.  :    ,
#   # marker：.  ,   o   v    <    *    +    1
#   plt.figure(figsize=(8, 6))
#   # linestyle = "-"
#   plt.grid(linestyle="-.")  # 设置背景网格线为虚线
#   # ax = plt.gca()
#   # ax.spines['top'].set_visible(False)  # 去掉上边框
#   # ax.spines['right'].set_visible(False)  # 去掉右边框

#   linewidth = 2.0
#   markersize = 7

#   if color is None:
#     color = ['blue', 'green', 'tomato', 'orange', 'purple', 'deepskyblue']
  
#   for i in range(len(X)):
#     plt.plot(X[i], Y[i], marker='', markersize=markersize, color=color[i], alpha=1, label=labels[i], linewidth=linewidth)
#     pos = np.where(np.amax(Y[i]) == Y[i])[0].tolist()
#     pos = pos[0]
#     # print(pos)
#     # print(Y[i][pos[0]], Y[i][pos[1]])

#     plt.plot(X[i][pos], Y[i][pos], marker='x', markersize=markersize, color='red', alpha=1, linewidth=linewidth)
#     plt.plot(X[i][pos], Y[i][pos], marker='.', markersize=markersize-2, color=color[i], alpha=1, linewidth=linewidth)


  
#   x_ticks = np.linspace(0, np.max(X), 5).tolist()
#   y_labels = [f'{x:.2f}' for x in x_ticks]
#   plt.xticks(x_ticks, y_labels, fontsize=15)  # 默认字体大小为10

#   y_ticks = [0.1, 0.3, 0.5, 0.7, 0.9]
#   y_lables = ['10%', '30%', '50%', '70%', '90%']
#   plt.yticks(np.array(y_ticks), y_lables, fontsize=15)
#   # plt.title("example", fontsize=12, fontweight='bold')  # 默认字体大小为12
#   # plt.text(1, label_position, dataset,fontsize=25, fontweight='bold')
#   # plt.xlabel("Edge Miss Rate", fontsize=15)
#   if not y_label:
#     y_label = "Val"
#   plt.ylabel(f"{y_label} Acc", fontsize=15)
#   plt.xlabel(f"Time (s)", fontsize=15)
#   plt.xlim(0, np.max(X) + 1)  # 设置x轴的范围
#   plt.ylim(0, 1)

#   # plt.legend()
#   # 显示各曲线的图例 loc=3 lower left
#   plt.legend(loc=0, numpoints=1, ncol=2)
#   leg = plt.gca().get_legend()
#   ltext = leg.get_texts()
#   plt.setp(ltext, fontsize=15)
#   # plt.setp(ltext, fontsize=25, fontweight='bold')  # 设置图例字体的大小和粗细
#   plt.tight_layout()
#   if not savefile:
#     savefile = 'plot_line.png'
#   plt.savefig(f'./{savefile}', format='pdf')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
#   plt.show()



# def parse_log(filename=None):
#   assert filename
#   if not os.path.exists(filename):
#     print(f'{filename} not exist')
#   train_acc = []
#   val_acc = []
#   test_acc = []
#   avg_time_list = []
#   time_cost = dict()
#   # avg_train_time = None
#   # avg_val_time = None
#   # avg_test_time = None
#   dataset = None
#   with open(filename) as f:
#     while True:
#       line = f.readline()
#       if not line:
#         break
#       # print(line)
#       if line.find('Epoch ') >= 0:
#         nums = re.findall(r"\d+\.?\d*", line)
#         # print(nums)
#         train_acc.append(float(nums[1]))
#         val_acc.append(float(nums[2]))
#         test_acc.append(float(nums[3]))
#       elif line.find('edge_file') >= 0:
#         l, r = line.rfind('/'), line.rfind('.')
#         dataset = line[l+1:r]
#       elif line.find('Avg') >= 0:
#         nums = re.findall(r"\d+\.?\d*", line)
#         avg_time_list.append(float(nums[0]))
#         avg_time_list.append(float(nums[1]))
#         avg_time_list.append(float(nums[2]))
#       elif line.find('TIME') >= 0:
#         nums = re.findall(r"\d+\.?\d*", line)
#         time_cost[int(nums[0])] = [float(x) for x in nums[1:]]
#         # TIME(0) sample 0.000 compute_time 2.977 comm_time 0.003 mpi_comm 0.302 rpc_comm 0.000 rpc_wait_time 2.675
#   return dataset, [train_acc, val_acc, test_acc], avg_time_list, time_cost

# X, Y = [], []
# labels = []

# # files = ['cora_seq.log', '_rand.log']
# datasets = ['cora', 'pubmed', 'citeseer', 'arxiv', 'reddit']
# modes = ['seq', 'shuffle', 'rand', 'low', 'upper']
# pre_path = './log/'
# host_num = 8

# for ds in datasets:
#   for type in ['Val', 'Test']:
#     X, Y, labels = [], [], []
#     T, T1= [], []
#     idx = 1 if type == 'Val' else 2
#     for ms in modes:
#       name = pre_path + ds + '_' + ms + '.log'
#       if not os.path.exists(name):
#         print(name, 'not exist.')
#         continue
#       dataset, acc_list, time_list, time_cost = parse_log(name)
#       # print(time_cost)
#       # TIME(2) sample 0.236 compute_time 0.930 comm_time 1.966 mpi_comm 0.117 rpc_comm 0.812 rpc_wait_time 0.000
#       # print(time_cost)
#       # print(ds, ms)
#       if ds in ['arxiv', 'reddit']:
#         # print(time_cost)
#         compute_time = [time_cost[i][1] for i in range(host_num)]
#         comm_time = [time_cost[i][2] for i in range(host_num)]
#         all_time = [time_cost[i][1] + time_cost[i][2] for i in range(host_num)]
#         T.append(compute_time)
#         T1.append(comm_time)
#       # print(compute_time)
#       ret = get_time_acc(acc_list[idx], time_list[0], max(acc_list[idx]), False)
#       X.append(ret[0])
#       Y.append(ret[1])
#       labels.append(ds + '-' +ms)
#       print(ds+'_'+ms+'_'+type, max(ret[1]))
#     plot_line(X, Y, labels, pre_path+ds+'-'+type+'.pdf', y_label= type,)

#     if ds in ['arxiv', 'reddit']:
#       plot_bar([f'host {i}' for i in range(host_num)], 'Compute Time (s)', T, labels, pre_path+ds+'-compute'+'.pdf')
#       plot_bar([f'host {i}' for i in range(host_num)], 'Comm Time (s)', T1, labels, pre_path+ds+'-comm'+'.pdf')


# # x_name = ['cora', 'citeseer', 'pubmed', 'arxiv', 'reddit', 'orkut', 'wiki']
# # labels = ['GraphScope', 'NTS']
# # aligraph = [0.289, 0.463, 2.459, 1.733,0, 0, 0]
# # nts = [0.027, 0.053, 0.059, 0.270, 2.1180, 9.133, 20.435] 
# # plot_bar(x_name, 'Time (s)', [aligraph, aligraph], labels, 'xx.pdf')

# assert(False)


import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

import numpy as np
import matplotlib.ticker as mtick
import os
import re


# https://blog.csdn.net/ddpiccolo/article/details/89892449
def plot_line(plot_params, X, Y, labels, xlabel, ylabel, yticks, figpath=None):

  pylab.rcParams.update(plot_params)  #更新自己的设置
  # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
  # plt.style.use("seaborn-deep")
  # ["seaborn-deep", "grayscale", "bmh", "ggplot"]
  
  # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #线型设置
  # https://matplotlib.org/stable/api/markers_api.html  'o', 's', 'v', 'p', '*', 'd', 'X', 'D',
  makrer_list = [ 'o', 's', 'v', 'p', '*', 'd', 'X', 'D']
#   marker_every = [[10,8],[5,12],[5,14],50,70,180,60]
  marker_every = [10,10,10,10,10,10,10]
  # fig1 = plt.figure(1)
  axes1 = plt.subplot(111)#figure1的子图1为axes1
  for i, (x, y) in enumerate(zip(X, Y)):
    # plt.plot(x, y, label = labels[i], marker=makrer_list[i], markersize=3,markevery=marker_every[i])
    plt.plot(x, y, label = labels[i], markersize=5)
  axes1.set_yticks(yticks)
  axes1.set_xticks([0,150,300])  
  ############################
  # axes1.set_ylim(0.60, 0.70)
  # axes1.set_ylim(0.86, 0.91)
  axes1.set_ylim(0.88, 0.94)
  # axes1.set_ylim(0.92, 0.94)
  # axes1.set_ylim(0.85, 0.92)

  # axes1.set_ylim(50, 450)
  axes1.set_xlim(0, 100)

  # axes1.set_ylim(50, 600)
  # axes1.set_xlim(0, 300)

  plt.legend(ncol=2)
  ############################

  # axes1 = plt.gca()
  # axes1.grid(True)  # add grid

  plt.ylabel(ylabel) 
  plt.xlabel(xlabel) 
  plt.title('ogbn-arxiv')
  figpath = './log/batch-size/reddit-exp5/plot.pdf' if not figpath else figpath
  plt.savefig(figpath, dpi=1000, bbox_inches='tight', format='pdf')#bbox_inches='tight'会裁掉多余的白边
  print(figpath, 'is plot.')
  plt.close()


# 每隔time_skip对acc取一个平均值
def split_list(X, Y, time_skip):
    retX, retY = [], []
    for arrx,arry in zip(X, Y):
        tmpx, tmpy = [], []
        pre, idx = 0, 0
        print(len(arrx),len(arry))
        for i in range(len(arrx)):
            
            x, y = arrx[i], arry[i]
            if x >= idx * time_skip:
                tmpx.append(x)
                tmpy.append(np.average(arry[pre : i + 1]))
                pre = i + 1
                idx += 1
        if pre < len(arrx):
            tmpx.append(arrx[-1])
            tmpy.append(np.average(arry[pre : ]))

        retX.append(tmpx)
        retY.append(tmpy)
    return retX, retY


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
                # print(line)
                nums = re.findall(r"\d+\.?\d*", line[line.find(mode)+len(mode) :])
                ret.append(float(nums[0]))
    return ret



if __name__ == '__main__':
  # def print_val_acc(mode, datasets, batch_sizes, suffix=None):
  batch_sizes = {
        'reddit': (128,512, 1024,2048,4096,  8192, 16384, 32768,65536,  153431,'mix7'),
        'ogbn-arxiv': (128,  512,1024,2048,  4096,  8192,16384, 32768, 65536,90941,'mix3'),
        'ogbn-products': (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,'mix4'),
    }
  suffix = ['shuffle','low','high','metis']
  log_path = './log/batch-type'
  val_acc = []
  run_time = []
  for ds in ['reddit']:
    for sf in suffix:
        log_file = f'{log_path}/{ds}/{ds}-{sf}.log'
        val_acc.append(parse_num(log_file, 'val_acc'))
        run_time.append(parse_num(log_file, 'gcn_run_time'))

  # diff0 = []
  # run_time = []
  # diff1 = []
 
  # diff0.append(parse_num(f'./log/tmp-four-layers.log','diff layer 0'))
  # diff0.append(parse_num(f'./log/tmp-four-layers.log','diff layer 1'))
  # run_time.append(parse_num(f'./log/tmp-four-layers.log', 'gcn_run_time'))
  # run_time.append(parse_num(f'./log/tmp-four-layers.log', 'gcn_run_time'))
  # print(len(diff0[0]))
  # print(len(run_time[0]))
  # index = 0
  # sumX = 0
  # sumY = 0
  # skip = [5,5,20,20,20,20,30,30,20,5,10,10]
  # skip = [10,10,10,10,10,10,35,30,40,10,10,10]
  # run_time_com = []
  # val_acc_com = []
  # for i in range(len(val_acc)):
  #   listX = []
  #   listY = []
  #   sumX = 0
  #   sumY = 0
  #   index = 0
  #   if(i ==0):
  #     print(val_acc[i])
  #     print(run_time[i])
  #   for j in range(len(val_acc[i])):
  #       index += 1
  #       sumX += run_time[i][j]
  #       sumY += val_acc[i][j]
  #       if index == skip[i]:
  #           # print(index)
  #           listX.append(sumX/index)
  #           listY.append(sumY/index)
  #           sumX = 0
  #           sumY = 0
  #           index = 0
  #   run_time_com.append(listX)
  #   val_acc_com.append(listY)
  # print(val_acc_com[9])
  # print(run_time_com[9])
  

  # X = [np.arange(2, 40 + 2, 2), np.arange(2, 40 + 2, 2), np.arange(2, 40 + 2, 2)]

  # yticks = [0.92,0.925,0.93,0.935,0.94]
  # xticks = [0,150,300,450,600]
  # labels = batch_sizes['reddit']

  # yticks = [0.60,0.63,0.66,0.70]
  # xticks = [0,50,100]

  yticks = [0.86,0.87,0.88,0.89,0.90,0.91]
  xticks = [0,50,100]


  
  yticks = [0.88,0.91,0.92,0.93,0.94]
  xticks = [0,50,100]

  labels = suffix

  # yticks = [0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92]
  # xticks = [0,150,300,450,500]
  # labels = batch_sizes['ogbn-products']

  xlabel = 'run time'
  ylabel = 'Accuracy'

  myparams = {
    'axes.labelsize': '10',
    'xtick.labelsize': '10',
    'ytick.labelsize': '10',
    'font.family': 'Times New Roman',
    'figure.figsize': '4, 3',  #图片尺寸
    'lines.linewidth': 1,
    'legend.fontsize': '8',
    'legend.loc': 'right', #[]"upper right", "upper left"]
    'legend.numpoints': 1,
    # 'lines.ncol': 2,
  }
  # print(plt.rcParams.keys())
#   print(run_time_com[4])
#   print(val_acc_com[4])
  run_time_skip,val_acc_skip = split_list(run_time,val_acc,5)
  plot_line(myparams, run_time, val_acc, labels, xlabel, ylabel, yticks, 'line-reddit-diff-type-no-avg.pdf')



  # yticks = [50,150,250,350,450]
  # xticks = [0,150,300]
  # labels = ('input layer','output layer')
  # plot_line(myparams, run_time, diff0, labels, xlabel, ylabel, yticks, 'line-w-diff-4.pdf')