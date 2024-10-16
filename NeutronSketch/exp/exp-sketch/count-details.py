import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np

import numpy as np
import matplotlib.ticker as mtick
import os
import re


# https://blog.csdn.net/ddpiccolo/article/details/89892449
def plot_line(plot_params, title,X, Y, Z,labels, xlabel, ylabel, xticks,yticks,ylim, figpath=None):

  pylab.rcParams.update(plot_params)  #更新自己的设置
  # https://tonysyu.github.io/raw_content/matplotlib-style-gallery/gallery.html
  # plt.style.use("seaborn-deep")
  # ["seaborn-deep", "grayscale", "bmh", "ggplot"]
  
  # line_styles=['ro-','b^-','gs-','ro--','b^--','gs--']  #线型设置
  # https://matplotlib.org/stable/api/markers_api.html  'o', 's', 'v', 'p', '*', 'd', 'X', 'D',
  # color_list = ['#bdddf2','#8e8e8e','#f3ec8a','#f0633a','#fdd37a','#e6f397','#bfd2bb','#d394de','#b0dbce','#99093d','#73c49a','#415aa4']
  color_list = ['#7A57D1','#FF731D','#004d61','#bc8420','#CF0A0A','#83FFE6','#0000A1','#fff568','#0080ff','#81C6E8',         '#385098','#ffb5ba','#EA047E','#B1AFFF','#425F57','#CFFF8D','#100720','#18978F','#F9CEEE','#7882A4','#E900FF','#84DFFF','#B2EA70','#FED2AA','#49FF00','#14279B','#911F27','#00dffc']
  makrer_list = [ 'o', 's', 'v', 'p', '*', 'd', 'X', 'D']
#   marker_every = [[10,8],[5,12],[5,14],50,70,180,60]
  marker_every = [10,10,10,10,10,10,10]
  # fig1 = plt.figure(1)
  axes1 = plt.subplot(111)#figure1的子图1为axes1
  for i, (x, y) in enumerate(zip(X, Y)):
    # plt.plot(x, y, label = labels[i], marker=makrer_list[i], markersize=3,markevery=marker_every[i])
    plt.plot(x, y, label = labels[i], color=color_list[i],markersize=5)
  axes1.set_yticks(yticks)
  axes1.set_xticks(xticks)  
  ############################
#   axes1.set_ylim(0.65, 0.72)
  # axes1.set_ylim(0.92, 0.94)
  # axes1.set_ylim(0.85, 0.92)
#   print(ylim)
  axes1.set_ylim(ylim[0],ylim[1])
  # axes1.set_ylim(50, 450)
  axes1.set_xlim(0, 45)
  # axes1.set_ylim(50, 600)
  # axes1.set_xlim(0, 300)

  # axes1.spines['right'].set_visible(False)
  # z_ax = axes1.twinx()
  # z_ax.plot(X[0],Z[0],color=color_list[-1],label = 'nodes num')
  # z_ax.set_ylabel('z')
  # z_ax.set_ylim(-10000,10000)
  # z_ax.set_yticks([0,1000,3000,5000,10000])
  # plt.legend(ncol=2)
  ############################

  # axes1 = plt.gca()
  # axes1.grid(True)  # add grid

  plt.ylabel(ylabel) 
  plt.xlabel(xlabel) 
  plt.title(title)
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

def parse_num(filename, mode, switch):
    if not os.path.exists(filename):
        print(f'{filename} not exist!')
        assert False
    if not os.path.isfile(filename):
        print(f'{filename} not a file!')
        assert False
    ret = []
    i = 0
    with open(filename) as f:
        for line in f.readlines():
            if line.find(mode) >= 0:
                # print(line.find(mode))
                if (i+1) %(switch+1) ==0:
                  i = 0
                  # print(line)
                  nums = re.findall(r"\d+\.?\d*", line[line.find(mode)+len(mode) :])
                  ret.append(float(nums[0]))
                else:
                  # print(i)
                  i+=1
    return ret

if __name__ == '__main__':
  # def print_val_acc(mode, datasets, batch_sizes, suffix=None):
  batch_sizes = {
        # 'reddit': (128,512, 1024,2048,4096,  8192, 16384, 32768,65536,  153431,'mix7'),
         'reddit': (1024,),
        # 'ogbn-arxiv': (128,  512,1024,2048,  4096,  8192,16384, 32768, 65536,90941,'mix3'),
        'ogbn-arxiv': (1024,),
        # 'ogbn-products': (512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,'mix4'),
        'ogbn-products': (1024,32768,131072)
    }


  xlabel = 'label class'
  ylabel = 'per'
  yticks = {'reddit':[0.92,0.925,0.93,0.935,0.94],
            'ogbn-arxiv':[0.0,0.5,1.0,1.5,2,3,4,5,8],
            'ogbn-products':[0.85,0.86,0.87,0.88,0.89,0.90,0.91,0.92]
            }

  ylim = {'reddit':[0.92,0.945],
            'ogbn-arxiv':[0.0,8],
            'ogbn-products':[0.80,0.92]
            }
  xticks = [0,13,16,20,24,30,45]
#   labels = batch_sizes['ogbn-arxiv']
  
  myparams = {
    'axes.labelsize': '10',
    'xtick.labelsize': '10',
    'ytick.labelsize': '10',
    'font.family': 'Times New Roman',
    'figure.figsize': '4, 3',  #图片尺寸
    'lines.linewidth': 1,
    'legend.fontsize': '8',
    'legend.loc': 'upper right', #[]"upper right", "upper left"]
    'legend.numpoints': 1,
    # 'lines.ncol': 2,
  }
  # ,'16,16','32,32','mode1-33'
  # suffixs = ['degree-100-5','degree-100-10','degree-100-no','degree-50-5','degree-50-10','degree-50-no','normal']
  # suffixs = ['degree-100-5-sample','degree-100-10-sample','degree-100-no-sample','degree-50-5-sample','degree-50-10-sample','degree-50-no-sample','normal-sample','tmp']
  # labels = ['sketch-100-5','sketch-100-10','sketch-100-no','sketch-50-5','sketch-50-10','sketch-50-no','normal','tmp']

  # 0 upper degree
  suffixs = ['before','after']
  labels = ['before','after']

  # suffixs = ['degree-100-5']
  # labels = ['degree-100-5-sample']
  log_path = './log/label_distribution'

# ['ogbn-arxiv','reddit','ogbn-products']
  # for ds in ['reddit']:
  #   for bs in batch_sizes[ds]:
  #       val_acc = []
  #       run_time = []
  #       best_val_list = []
  #       best_val_x = [1,2,3,4,5,6,7]
  #       for fanout in fanout_list:
  #           log_file = f'{log_path}/{ds}_no_val_sample/{ds}-{bs}-{fanout}-no_val_sample.log'
  #           val_acc.append(parse_num(log_file, 'val_acc'))
  #           run_time.append(parse_num(log_file, 'gcn_run_time'))
  #       run_time_skip,val_acc_skip = split_list(run_time,val_acc,20)
  #       print(len(val_acc[0]))
  #       for val_list in val_acc:
  #         val_best = 0
  #         for i in val_list:
  #           if i > val_best:
  #             val_best = i
  #         best_val_list.append(val_best)
  #       print(best_val_list)
  #       plot_best_val(best_val_x,best_val_list,ylim[ds],yticks[ds],f'line-{ds}-{bs}-max_fanout-noval_sample.pdf',ds+str(bs))


  # strs = 'mix2'
  # print(strs.find('mix'))  int(strs[2])

# show label percentage
  # for ds in ['ogbn-arxiv']:
  #   per = []
  #   label_class = []
  #   node_nums = []
  #   log_file = f'{log_path}/{ds}-before.log'  
  #   node_nums.append(parse_num(log_file, 'origin nums',0))
  #   print(node_nums)
  #   for sf in suffixs:
  #     log_file = f'{log_path}/{ds}-{sf}.log'
  #     label_class.append(parse_num(log_file, 'label class',0))
  #     per.append(parse_num(log_file, 'per',0))
  #     print(per)
  #   plot_line(myparams,'label compare',label_class,per,node_nums,labels,xlabel,ylabel,xticks,yticks[ds],ylim[ds],f'line-{ds}-label-test.pdf')


# show sketch nodes class details
# reddit ogbn-arxiv

for ds in ['reddit']:
  # log_file = f'{log_path}/{ds}-details-6.log'
#   log_file = f'{log_path}/tmp.log'
  log_file = f'./log/{ds}/{ds}-k_core.log'
  class_nid = {}
  class_score = {}
  class_mask = {}
  class_degree = {}
  class_same_per = {}
  nid_list = parse_num(log_file, 'nid:',0)
  class_list = parse_num(log_file, 'label:',0)
  score_list = parse_num(log_file, 'score:',0)
  mask_list = parse_num(log_file, 'mask:',0)
  degree_list = parse_num(log_file,'degree:',0)
  # neighbor_same = []
  # for i in parse_num(log_file, 'same per',0):
  #   neighbor_same.append(int(i)/100)
  #   #  print(i)
  #   #  if i >= 50:
  #   #     neighbor_same.append(int(i)/100)
  #   #  else:
  #   #     neighbor_same.append(0)
  for i in range(len(class_list)):
      # print(class_list[i])
      if mask_list[i]==0:
        tmp_class = int(class_list[i])
        if class_nid.get(tmp_class)==None:
          class_nid[tmp_class] = []
          class_score[tmp_class] = []
          class_mask[tmp_class] = []
          class_degree[tmp_class] = []
          
          class_nid[tmp_class].append(nid_list[i])
          class_score[tmp_class].append(score_list[i])
          class_degree[tmp_class].append(degree_list[i])
          # class_mask[tmp_class].append(mask_list[i])
          # class_same_per[tmp_class].append(neighbor_same[i])     
        else:
          
          class_nid[tmp_class].append(nid_list[i])
          class_score[tmp_class].append(score_list[i])
          class_degree[tmp_class].append(degree_list[i])
        # class_mask[tmp_class].append(mask_list[i])
        # class_same_per[tmp_class].append(neighbor_same[i]) 
  # print(class_nid)
  # class_nid = sorted(class_nid.items())
  # class_degree = sorted(class_degree.items())
  # class_neighbor = sorted(class_neighbor.items())
  # print(class_nid)
  # print(class_same_per[15])
  # result = []
  # for key in class_same_per.keys():
  #   for per in class_same_per[key]:
  #     result.append(per)
  #     # if(per>0.5):
  #     #   result.append(1)
  #     # else:
  #     #   result.append(0)
  # print(len(result), ' ',np.mean(result))
  sum = 0
  for key in class_score.keys():
    #  print(key)
     sum+=len(class_score[key])
     print('label class:{:d} nodes num:{:d} avg degree:{:.2f} var degree:{:.2f} avg score:{:.2f} var score:{:.2f}'.format(key, len(class_score[key]),np.mean(class_degree[key]),np.var(class_degree[key]) ,np.mean(class_score[key]),np.var(class_score[key])))
  
  print(sum)
  # print(class_score[30])
  # print(class_degree[30])
  # print(len(class_score[30]))
  result = []
  class_res = {}
  for i,val in enumerate(class_score):
     for j,score in enumerate(class_score[i]):
      if(score>200):
         tmp_class = int(class_list[int(class_nid[i][j])])
         if class_res.get(tmp_class)==None:
          class_res[tmp_class] = []
          class_res[tmp_class].append(class_nid[i][j])  
         else:     
          class_res[tmp_class].append(class_nid[i][j]) 
         result.append(class_nid[i][j])
  # print(result)
  # with open(f'./tmp.log','a') as file:
  #   for i in result:
  #      strs = str(int(i))+' '
  #      file.write(strs)
     
  print(len(result))
  for key in class_res.keys():
     print('label class:{:d} nodes num:{:d} '.format(key, len(class_res[key])))
  



  # print(degree_list)
  #  print(nid)
  #  print(class_list)
  #  print(len(class_nid[16]))