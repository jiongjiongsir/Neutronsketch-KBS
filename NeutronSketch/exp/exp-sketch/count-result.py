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
  plt.axhline(0, color='black')
  axes1 = plt.subplot(111)#figure1的子图1为axes1
  print(zip(X, Y))
  # for i, (x, y) in enumerate(zip(X, Y)):
    # print()
    # plt.plot(x, y, label = labels[i], marker=makrer_list[i], markersize=3,markevery=marker_every[i])
  plt.scatter(X, Y, color=color_list[0])
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
  for ds in ['ogbn-products']:
    result_file = f'./log/label_distribution/{ds}-result-tmp.log'
    label_list = parse_num(result_file,'label',0)
    right_list = parse_num(result_file,'right',0)
    index = (int(len(label_list)/2))
    sums = 0
    res = []
    label = []
    for i in range(index):
        if(right_list[i]>=right_list[index+i]):
            tmp = right_list[i] - right_list[index+i]
            sums += tmp
            strs = 'upper'
            res.append(int(tmp))
        else:
            tmp = right_list[index+i] - right_list[i]
            sums -= tmp
            strs = 'lower'
            res.append(0-int(tmp))
        label.append(i)

        
        print('After full label {:d} {:s} {:d}'.format(i,strs,int(tmp)))
    print('All changes : ',sums)
    print(label,res)
    # print(label_list)
    # print(right_list)

    myparams = {
    'axes.labelsize': '10',
    'xtick.labelsize': '10',
    'ytick.labelsize': '10',
    'figure.figsize': '4, 3',  #图片尺寸
    'lines.linewidth': 1,
    'legend.fontsize': '8',
    'legend.loc': 'lower left', #[]"upper right", "upper left"]
    'legend.numpoints': 1,
    # 'lines.ncol': 2,
  }
  plot_line(myparams,"diff of Sketch and Origin",label,res,None,None,'Class','right val nodes num',[0,10,20,30,40],[-200,-30,0,30,200],[-200,200],'./line-diff-Sketch-Origin-reddit.pdf')