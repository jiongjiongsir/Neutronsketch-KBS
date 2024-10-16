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
# neg_nid = parse_num('./log/ogbn-products/ogbn-products-1024-mode8-and-mode7-2-degree-200.log')
for ds in ['ogbn-products']:
  # log_file = f'{log_path}/{ds}-details-6.log'
  # log_file = f'{log_path}/{ds}-NCS.log'
  log_file = f'./log/{ds}/{ds}-NCS.log'
  # products inter class sim 0.1
  # log_file_tmp = f'.//log/{ds}/{ds}-1024-mode7-2-inter-class-sim-0.1.log'
  # products inter class sim 0.2
  log_file_tmp = f'./log/{ds}/{ds}-1024-mode8-test-3.log'
  # products inter class sim 0.3
  # log_file_tmp = f'.//log/{ds}/{ds}-1024-mode7-2-inter-class-sim-0.3.log'

  # reddit
  # log_file_tmp = f'./log/{ds}/{ds}-1024-mode8-and-mode7.log'
  # ogbn-arxiv
  # log_file_tmp = f'./log/{ds}/{ds}-1024-step-1.log'
  class_nid = {}
  class_degree = {}
  class_neighbor = {}
  class_same_per = {}
  class_ncs = {}
  nid = parse_num(log_file, 'Sketch nodes',0)
  tmp_nid = parse_num(log_file_tmp,'Sketch nodes',0)
  class_list = parse_num(log_file, 'label class',0)
  print(len(class_list))
  degree_list = parse_num(log_file, 'degree:',0)
  NCS_list = parse_num(log_file,'NCS',0)
  neighbor_class = parse_num(log_file, '1-hop class nums',0)
  neighbor_same = []
  for i in parse_num(log_file, 'same per',0):
    neighbor_same.append(int(i)/100)
    #  print(i)
    #  if i >= 50:
    #     neighbor_same.append(int(i)/100)
    #  else:
    #     neighbor_same.append(0)
  for i in range(len(class_list)):
      # print(class_list[i])
      tmp_class = int(class_list[i])
      if class_nid.get(tmp_class)==None:
        class_nid[tmp_class] = []
        class_degree[tmp_class] = []
        class_neighbor[tmp_class] = []
        class_same_per[tmp_class] = []
        class_ncs[tmp_class] = []
        class_nid[tmp_class].append(nid[i])
        class_degree[tmp_class].append(degree_list[i])
        class_neighbor[tmp_class].append(neighbor_class[i])
        class_same_per[tmp_class].append(neighbor_same[i])
        class_ncs[tmp_class].append(NCS_list[i])     
      else:
        class_nid[tmp_class].append(nid[i])
        class_degree[tmp_class].append(degree_list[i])
        class_neighbor[tmp_class].append(neighbor_class[i])
        class_same_per[tmp_class].append(neighbor_same[i])
        class_ncs[tmp_class].append(NCS_list[i])  
  # print(class_nid)
  # class_nid = sorted(class_nid.items())
  # class_degree = sorted(class_degree.items())
  # class_neighbor = sorted(class_neighbor.items())
  # print(class_nid)
  # print(class_same_per[15])

  result = []
  for key in class_same_per.keys():
    for per in class_same_per[key]:
      result.append(per)
      # if(per>0.5):
      #   result.append(1)
      # else:
      #   result.append(0)
  print(len(result), ' ',np.mean(result))
  tmp_list_score = []
  tmp_list_nbr_sim = []
  class_mean_sim_list = []
  for key in class_degree.keys():
     tmp_dict_nbr_sim = dict(zip(class_nid[key],class_same_per[key]))
     tmp_dict = dict(zip(class_nid[key],class_ncs[key]))
     tmp_list_score.append(sorted(tmp_dict.items(),key=lambda x:x[1],reverse=True))
     tmp_list_nbr_sim.append(sorted(tmp_dict_nbr_sim.items(),key=lambda x:x[1],reverse=True))
     class_mean_sim_list.append(np.mean(class_same_per[key]))
    #  print(key)
     print('label class:{:d} nodes num:{:d} avg degree:{:.2f} avg classes:{:.2f} avg same per{:.2f} avg NCS{:.2f}'.format(key, len(class_degree[key]), np.mean(class_degree[key]),np.mean(class_neighbor[key]),np.mean(class_same_per[key]),np.mean(class_ncs[key])))
  score = dict(zip(list(class_degree),tmp_list_score))
  nbr_sim = dict(zip(list(class_degree),tmp_list_nbr_sim))
  nid_degree = dict(zip(nid,degree_list))
  nid_sim = dict(zip(nid,neighbor_same))
  nid_class = dict(zip(nid,class_list))
  class_mean_sim = dict(zip(list(class_degree.keys()),class_mean_sim_list))
  # print(score[0])
  # file_path = './ogbn-products-tmp.log'
  # for key in class_same_per.keys():
  #     # print(key,simlist[key])
  #     if np.mean(class_same_per[key])<0.5:
  #       # print(class_same_per[key])
  #        with open(file_path,'a') as file:
  #           for x,i in enumerate(class_nid[key]):
  #             if(class_same_per[key][x]>0.3):
  #               strs = str(int(i))+' '
  #               file.write(strs)
  #     # # elif np.mean(class_same_per[key])<0.8:
  #     # #   index_h = int(len(class_same_per[key])*np.mean(class_same_per[key]))
  #     # #   index_l = int(len(class_same_per[key])*(0.5))
  #     # #   # print(class_same_per[key][0:index_l])
  #     # #   with open(file_path,'a') as file:
  #     # #     # for i in class_nid[key][0:index_l]+class_nid[key][index_h:-1]:
  #     # #     # for x,i in enumerate(class_nid[key][0:index_l]):
  #     # #     for x,i in enumerate(class_nid[key][index_h:-1]):
  #     # #         if(class_same_per[key][x+index_h]>0.3):
  #     # #           strs = str(int(i))+' '
  #     # #           file.write(strs)
  #     elif key==2  or key == 8  or key==13:
  #       index_h = int(len(class_same_per[key])*0.60)
  #       index_l = int(len(class_same_per[key])*0.10)
  #       with open(file_path,'a') as file:
  #           for x,i in enumerate(class_nid[key][index_l:index_h]):
  #             if(class_same_per[key][x+index_l]>0.35):
  #               strs = str(int(i))+' '
  #               file.write(strs)
  #     elif key == 6:
  #       index_h = int(len(class_same_per[key])*0.85)
  #       index_l = int(len(class_same_per[key])*0.35)
  #       with open(file_path,'a') as file:
  #           for x,i in enumerate(class_nid[key][index_l:index_h]):
  #             if(class_same_per[key][x+index_l]>0.35):
  #               strs = str(int(i))+' '
  #               file.write(strs)
  #     elif key==3:
  #         index_h = int(len(class_same_per[key])*0.8)
  #         index_l = int(len(class_same_per[key])*0.2)
  #         with open(file_path,'a') as file:
  #           for x,i in enumerate(class_nid[key][index_l:index_h]):
  #             if(class_same_per[key][x]>0.2):
  #               strs = str(int(i))+' '
  #               file.write(strs)
  #     elif key==4:
  #         index_h = int(len(class_same_per[key])*0.52)
  #         index_l = int(len(class_same_per[key])*0.02)
  #         with open(file_path,'a') as file:
  #           for x,i in enumerate(class_nid[key][index_l:index_h]):
  #             if(class_same_per[key][x]>0.2):
  #               strs = str(int(i))+' '
  #               file.write(strs)
  #     else:
  #       index_h = int(len(class_same_per[key])*1.0)
  #       index_l = int(len(class_same_per[key])*0.4)
  #       # print(class_same_per[key][0:index_l])
  #       # print(len(class_nid[key][index_h:-1]))
  #       # print(len(class_same_per[key][index_h:-1]))
  #       with open(file_path,'a') as file:
  #         # for i in class_nid[key][int(len(class_same_per[key])*0.1):index_l]+class_nid[key][index_h:-1]:
  #         for x,i in enumerate(class_nid[key][index_l:index_h]):
  #         # for x,i in enumerate(class_nid[key][0:index_l]):
  #         # for x,i in enumerate(class_nid[key][index_h:-1]):
  #             if(class_same_per[key][x+index_l]>0.3):
  #               strs = str(int(i))+' '
  #               file.write(strs)
  
  sums = 0
  tmp_list_nid = []
  tmp_list_ncs = []
  file_path = f'./{ds}-tmp.log'
  # 1->0  2->1  3->0,1  4-0,2 
  # key_list = [0,1,2,3,4,6,7,8,9,10,13,17]
  node_num = 5000
  # with open(file_path,'w') as file:
    # for key in class_same_per.keys():
      # if len(class_nid[key]) < node_num:
      #   node_num = len(class_nid[key])
      # else:
      #   node_num = 5000
    # if(np.mean(class_ncs[key])<0.05):
    #   with open(file_path,'a') as file:
    #         for x,i in enumerate(class_nid[key]):
    #           if(class_ncs[key][x]<np.mean(class_ncs[key])):
    #             strs = str(int(i))+' '
    #             file.write(strs)
    #             sums+=1
    # elif key==6 or key==8 or key==9 or key==13 or key==2 or key==17:
    #       with open(file_path,'a') as file:
    #         for x,i in enumerate(class_nid[key]):
    #           if(class_ncs[key][x]<0.3):
    #             strs = str(int(i))+' '
    #             file.write(strs)
    #             sums+=1
    # if key==4:
    #       with open(file_path,'a') as file:
    #         for x,i in enumerate(class_nid[key]):
    #           if(class_ncs[key][x]<0.2 or class_ncs[key][x]>0.9):
    #             strs = str(int(i))+' '
    #             file.write(strs)
    #             sums+=1
    # else: or (class_ncs[key][x]>0.05 and class_ncs[key][x]<0.15)
    # or x>0.9*len(score[key]) reddit 4->2 6? 7->6? 9->2,4? 11->4,6 12->9 13->2,4,12 14->6,7,8,9 15->8
    # 16->1 2 3 5 6 8 9 10 13 15  17->6,7  18->0 1 5 6 9 10 11  19->1 4 7 9 14 19  20->2 6 10
     
      # if key == 0 or key==1 or key==2 or key==3 or key==4 or key==5 or key==6 or key==7 or key==8 or key==9 or key==10 or key==11 or key==12 or key==13 or key==14 or key==15 or key==16 or key==18 or key==19\
      #       or key==20:
        # if key_list.count(key)!=0:
        # for x,(i,ncs) in enumerate(score[key]):
        #       if(x < 0.5*len(score[key]) ):
        #         strs = str(int(i))+' '
        #         file.write(strs)
        #         sums+=1
        # elif key==4:
        #     for x,(i,ncs) in enumerate(score[key]):
        #         if(x < 0.2*len(score[key]) ):
        #           strs = str(int(i))+' '
        #           file.write(strs)
        #           sums+=1

        # for x,(i,ncs) in enumerate(nbr_sim[key]):
        #       if(x < 0.9*len(nbr_sim[key]) ):
        #         strs = str(int(i))+' '
        #         file.write(strs)
        #         sums+=1

  print(sums)
  print(len(nbr_sim[20]))
  print(len(tmp_nid))
  mean_degree = np.mean(degree_list)
  print("mean degree:",np.mean(degree_list))
  degree_sum = 0
  index = 0
  hd = 0
  sketch = []
  neg_nid = []
  nid_degree = dict(sorted(nid_degree.items(),key=lambda x:x[1],reverse=True))
  # print(list(nid_degree.items())[0:100])
  # print(nid_degree[0:100])
  print(len(list(nid_degree.keys())))
  with open(file_path,'w') as file:
    for i in list(nid_degree.keys()):
    # for i in tmp_nid:
    #   if (class_mean_sim[nid_class[i]]) <0.5:
    #     if nid_sim[i] >=0.1:
    #       # strs = str(int(i))+' '
    #       # file.write(strs)
    #       index+=1
    #       sketch.append(i)
    #   else:
    #     if nid_sim[i] >=0.1 and nid_sim[i] <=class_mean_sim[nid_class[i]]:
    #       # strs = str(int(i))+' '
    #       # file.write(strs)
    #       index+=1
    #       sketch.append(i)
    #     elif nid_sim[i]>class_mean_sim[nid_class[i]] and nid_degree[i]<200:
    #       # strs = str(int(i))+' '
    #       # file.write(strs)
    #       index+=1
    #       sketch.append(i)

      # if nid_class[i]==8:
      #   if nid_sim[i] >=0.0 and nid_sim[i] <= 0.81:
      #     strs = str(int(i))+' '
      #     file.write(strs)
      #     degree_sum+=nid_sim[i]
      #     index+=1
      #     # print(nid_degree[i],nid_sim[i])
      # if nid_class[i]==13:
      #   if nid_sim[i] >=0.15 and nid_sim[i] <= 0.85:
      #     strs = str(int(i))+' '
      #     file.write(strs)
      #     degree_sum+=nid_sim[i]
      #     index+=1
      #     # print(nid_degree[i],nid_sim[i])
      # if nid_class[i]==19:
      #   if nid_sim[i] >=0.3 and nid_sim[i] <= 1.0:
      #     strs = str(int(i))+' '
      #     file.write(strs)
      #     degree_sum+=nid_sim[i]
      #     index+=1
      #     # print(nid_degree[i],nid_sim[i])
      # if nid_class[i]==7:
      #   if nid_sim[i] >=0.2 and nid_sim[i] <= 0.93:
      #     strs = str(int(i))+' '
      #     file.write(strs)
      #     degree_sum+=nid_sim[i]
      #     index+=1
      #     # print(nid_degree[i],nid_sim[i])
      # if nid_class[i]==4:
      #   if nid_sim[i] >=0.2 and nid_sim[i] <= 0.92:
      #     strs = str(int(i))+' '
      #     file.write(strs)
      #     degree_sum+=nid_sim[i]
      #     index+=1
      #     # print(nid_degree[i],nid_sim[i])
      # else:(nid_sim[i] >=0.0 and nid_sim[i] <0.1)
        if nid_sim[i] >= 0.85:
          if(nid_degree[i]<=mean_degree):
            strs = str(int(i))+' '
            file.write(strs)
            hd+=1
            index+=1
          degree_sum+=nid_degree[i]

        else:
          strs = str(int(i))+' '
          file.write(strs)
          index+=1
          # print(nid_degree[i],nid_sim[i])
  print("sketch nodes:",index)
  # print(nid_degree[70630])
  # print("sketch nodes:",len(tmp_nid))
  neg_nid =  list(set(list(nid_degree.keys())).difference(set(tmp_nid)))
  flag = 0
  # with open(file_path,'w') as file:
  #   for i in sketch:
  #   # for i in list(nid_degree.keys()):
  #     # print(nid_degree[i])
  #     # if(nid_degree[i]>=np.mean(degree_list)):    
  #       strs = str(int(i))+' '
  #       file.write(strs)
  #       flag+=1    
  print("negative nodes:",len(neg_nid))
  print("nodes num:",flag)
  print(degree_sum/index)
  print("hd:",hd)
  # print(tmp_list_nid)
  # print(tmp_list_ncs)
  # tmp = key_list[-1]
  tmp =1
  # print(len(nbr_sim[tmp]),nbr_sim[tmp][int(0.5*len(nbr_sim[tmp])-100):int(0.5*len(nbr_sim[tmp]))])
  # print(class_degree[tmp][0:100])
  # print(class_nid[tmp][0:100])
  # print( np.mean(class_ncs[tmp]),np.var(class_ncs[tmp]))
  # tmp = []
  # for key in class_same_per.keys():
  #   for i in range(len(class_same_per[key])):
  #     if class_same_per[key][i] >np.mean(class_same_per[key]):
  #       tmp.append(class_nid[key][i])
  
  # with open(f'./tmp.log','w') as file:
  #         # for i in class_nid[key][0:index_l]+class_nid[key][index_h:-1]:
  #   for i in tmp:
  #         # for i in class_nid[key][index_h:-1]:
  #     strs = str(int(i))+' '
  #     file.write(strs)
  tmp = 0

  index_h = int(len(class_same_per[tmp])*0.95)
  index_l = int(len(class_same_per[tmp])*0.6)
  # print(index_h,len(class_degree[1]))
  # print(class_same_per[tmp][index_l:index_h])
  # print(class_degree[tmp][index_l:index_h])
  # print(class_degree[0])
  # print(class_ncs[tmp])
  tmp_list = []
  for i in class_ncs[tmp]:
    if i >0.5 :
      tmp_list.append(i)
  print(len(tmp_list))
  # print((class_ncs[tmp]))
  
  # print(degree_list)
  #  print(nid)
  #  print(class_list)
  #  print(len(class_nid[16]))




#experment of High nbr sim nodes 

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import PercentFormatter

# myparams = {
#         'axes.labelsize': '12',
#         'xtick.labelsize': '12',
#         'ytick.labelsize': '12',
#         # 'font.family': 'Times New Roman',
#         'figure.figsize': '5, 4',  #图片尺寸
#         'lines.linewidth': 2,
#         'legend.fontsize': '9',
#         # 'legend.loc': 'best', #[]"upper right", "upper left"]
#         'legend.loc': 'upper left', #[]"upper right", "upper left"]
#         'legend.numpoints': 1,
#         'legend.frameon': False,
#         # 'lines.ncol': 2,
#     }
# plt.rcParams.update(myparams)


# # 创建数据
# categories = ['Ogbn-arxiv', 'Reddit', 'Ogbn-products']
# data_part1 = np.array([37.8, 51.7, 52.3])
# data_part2 = np.array([61.4, 47.2, 45.9])
# data_part3 = np.array([0.8, 1.1, 1.8])

# # 创建柱形图
# fig, ax = plt.subplots()
# ax.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# bar_width = 0.35
# bar1 = ax.bar(categories, data_part1, bar_width, label='Nbr Sim > 0.9')
# bar2 = ax.bar(categories, data_part2, bar_width, bottom=data_part1, label='Others')
# bar3 = ax.bar(categories, data_part3, bar_width, bottom=data_part1+data_part2, label='Nbr Sim < 0.1')

# # 添加标签和图例
# ax.set_ylabel('Percentage')
# # ax.set_title('Stacked Bar Chart with Three Parts')
# ax.legend(loc='upper left',bbox_to_anchor=(1.05, 0.50),ncol=1,frameon=False)

# plt.savefig('./tmp-1.pdf',bbox_inches='tight')

# # 显示图表
# plt.show()



# import matplotlib.pyplot as plt
# import numpy as np

# myparams = {
#     'axes.labelsize': '12',
#     'xtick.labelsize': '12',
#     'ytick.labelsize': '12',
#     'figure.figsize': '8, 4',  # 图片尺寸
#     'lines.linewidth': 2,
#     'legend.fontsize': '12',
#     'legend.loc': 'upper center',
#     'legend.numpoints': 1,
#     'legend.frameon': False,
# }

# plt.rcParams.update(myparams)

# # 创建数据
# x = np.array([1, 2, 3, 4])
# bar1_data = np.array([100, 88, 70, 60])
# line1_data = np.array([93.9, 93.1, 94.0, 92.5])

# plt.figure(figsize=(10, 4))

# # 创建图表和子图
# fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharey=True)

# # 左侧图
# bar_width = 0.35
# bar1 = ax1.bar(x, bar1_data, bar_width, color='teal', alpha=0.7, label='Training Time Percentage')
# ax1.set_ylabel('Relative Training Time')
# ax1.tick_params('y')

# ax1.set_ylim(0, 100)

# # 右侧图
# line1 = ax1.plot(x, line1_data, color='purple', marker='o', linestyle='--', label='Val ACC')
# ax2 = ax1.twinx()
# ax2.set_ylabel('Val ACC')
# ax2.tick_params('y')

# ax2.set_ylim(90, 95)
# ax2.axhline(93.9, color='black', linestyle='--', dashes=(15, 10), linewidth=0.3, label='Origin Val ACC')

# # 右侧图
# bar2_data = np.array([80, 90, 85, 75])
# line2_data = np.array([92.0, 94.2, 93.5, 91.8])

# bar2 = ax2.bar(x, bar2_data, bar_width, color='orange', alpha=0.7, label='Training Time Percentage')
# line2 = ax2.plot(x, line2_data, color='red', marker='x', linestyle='--', label='Val ACC')

# ax2.set_ylabel('Val ACC')
# ax2.tick_params('y')
# ax2.axhline(92.0, color='green', linestyle='--', dashes=(15, 10), linewidth=0.3, label='New Val ACC')

# # 添加标题
# plt.suptitle('Combined Bar and Line Chart')

# # 设置图例
# lines = line1 + line2
# labels = [l.get_label() for l in lines]
# fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, frameon=False)

# # 保存图表到本地文件
# plt.savefig('./tmp.pdf', bbox_inches='tight')

# # 显示图表
# plt.show()