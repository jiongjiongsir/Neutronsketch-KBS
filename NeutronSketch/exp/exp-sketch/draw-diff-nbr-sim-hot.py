
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap, LinearSegmentedColormap,TwoSlopeNorm
from matplotlib.patches import Rectangle,Patch
import numpy as np
import os
import re

def parse_max_acc(file_path):
    assert file_path
    val_acc = []
    if not os.path.exists(file_path):
        print(f'{file_path} not exist')    
    with open(file_path) as f:
        while True:
            line = f.readline()
            if not line:
                break
            if(re.findall(r"val_acc ([0-9\.]+)", line)!=[]):
                val_acc.append(re.findall(r"val_acc ([0-9\.]+)", line)[0])
    return float(max(val_acc))
def cal_max_acc(dataset):
    lists = ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
    max_acc = []
    file_path = f'./log/{dataset}-nbr-sim'
    for i in range(10):
        tmp = []
        index_1 = i
        low_1 = lists[index_1]
        high_1 = lists[index_1+1]
        for j in range(10):
            # print(i,j)
            index_2 = j
            if j<i:
                index_1,index_2 = j,i
                low_1 = lists[index_1]
                high_1 = lists[index_1+1]
            else:
                index_1,index_2 = i,j
                low_1 = lists[index_1]
                high_1 = lists[index_1+1]
            low_2 = lists[index_2]
            high_2 = lists[index_2+1]
            # print('index:',index_1,index_2)
            suffix = f'-{low_1}-{high_1}-and-{low_2}-{high_2}'
            log_path = f'{file_path}/{dataset}{suffix}.log'
            # print(parse_max_acc(log_path))
            tmp.append(parse_max_acc(log_path))
            # print(tmp)
        max_acc.append(tmp)
    return max_acc

if __name__ == '__main__':
        # 设置中文字体
        myparams = {
                'axes.labelsize': '18',
                'xtick.labelsize': '18',
                'ytick.labelsize': '18',
                'font.family': 'Times New Roman',
                'figure.figsize': '5, 4',  #图片尺寸
                'lines.linewidth': 2,
                'legend.fontsize': '18',
                # 'legend.loc': 'best', #[]"upper right", "upper left"]
                'legend.loc': 'upper left', #[]"upper right", "upper left"]
                'legend.numpoints': 1,
                'legend.frameon': False,
                # 'lines.ncol': 2,
                }
        plt.rcParams.update(myparams)
        # plt.rcParams['font.family'] = 'SimHei'
        
        # 数据
        # data = [[91.1, 93.6, 93.8,93.6,94.3],
        #         [91.4, 93.7, 93.6,95.4,92.1],
        #         [92.6, 93.7, 93.6,94.5,96.2],
        #         [93.6, 93.8, 93.5,94.2,92.1],
        #         [93.2, 93.6, 93.4,94.3,96.7]]
        # data = cal_max_acc('reddit')
        dataset = 'reddit'
        vmin_1 = 0.92
        vmax_1 = 0.94

        # dataset = 'ogbn-products'
        vmin_2 = 0.80
        vmax_2 = 0.91
        data1 = np.array(cal_max_acc('reddit'))[::-1]
        data2 = np.array(cal_max_acc('ogbn-products'))[::-1]
        # 创建 DataFrame colums = ['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']
        df_1 = pd.DataFrame(data1, columns=['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'], index=reversed(['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']))
        df_2 = pd.DataFrame(data2, columns=['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'], index=reversed(['0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']))
        # 制作热力图
        # plt.figure(figsize=(10, 7))
        fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(20, 7.5))
        plt.subplots_adjust(wspace=0.1)
        sns.set(font_scale=1.0)  # 设置字体比例
        # custom_colors = sns.diverging_palette(20, 220, n=256, as_cmap=True)  # 创建自定义的调色板
        custom_colors = ["#8B4513", "#FF4500"]  # 棕色和红色
        # original_cmap = plt.cm.YlGnBu
        # colors = original_cmap(np.linspace(0, 1, 256))
        # newcolors = np.vstack(([[0.8, 0.8, 0.8, 1]], colors))  # 添加灰色作为低值颜色
        # new_cmap = ListedColormap(newcolors)
        midpoint = 0.5  # 设定中间值为 50

        # 创建颜色映射
        cmap = 'coolwarm'  # 使用内置的diverging colormap

        # 设置TwoSlopeNorm
        norm_1 = TwoSlopeNorm(vmin=0.0, vcenter=vmin_1, vmax=vmax_1)
        norm_2 = TwoSlopeNorm(vmin=0.0, vcenter=vmin_2, vmax=vmax_2)
        # 设置横轴和纵轴标签
        # heatmap = sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.1f')  # 保留一位小数

        lower_ticks_1 = np.linspace(0.0, vmin_1, 6, endpoint=False)
        upper_ticks_1 = np.linspace(vmin_1, vmax_1, 6, endpoint=True)

        # 合并刻度数组
        ticks_1 = np.concatenate([lower_ticks_1, upper_ticks_1])  

        # heatmap = sns.heatmap(df_1,ax=ax1 ,annot=True, fmt=".3f",cmap='YlGnBu',norm=Normalize(vmin=vmin_1, vmax=vmax_1), cbar_kws={"ticks": np.linspace(vmin_1, vmax_1, 11)})
        # heatmap1 = sns.heatmap(df_1, ax=ax1, annot=True, fmt=".3f", cmap=new_cmap, norm=Normalize(vmin=vmin_1, vmax=vmax_1), cbar_kws={"extend": "min", "ticks": np.linspace(vmin_1, vmax_1, 11)})
        heatmap1 = sns.heatmap(df_1, ax=ax1, annot=True, fmt=".3f", cmap=cmap, norm=norm_1, cbar_kws={"ticks": ticks_1})
        # 设置横轴和纵轴标签
        # plt.xlabel('Neighbor Similarity')
        # plt.ylabel('Neighbor Similarity')
        # 添加标题
        rect_low = Rectangle((0, 6), 4, 4, linewidth=3, edgecolor='red', facecolor='none')
        rect_mod = Rectangle((3, 0.01), 3, 3, linewidth=3, edgecolor='black', facecolor='none')
        rect_high = Rectangle((7, 0.01), 3, 3, linewidth=3, edgecolor='blue', facecolor='none')
        ax1.add_patch(rect_low)
        ax1.add_patch(rect_mod)
        ax1.add_patch(rect_high)
        ax1.set_xlabel('Neighbor Similarity', fontsize=20)  # 设置标签字体大小
        ax1.set_ylabel('Neighbor Similarity', fontsize=20)
        ax1.tick_params(axis='x', labelsize=18,rotation=0,length=0)
        ax1.set_xticks([1,2,3,4,5,6,7,8,9,10])
        ax1.set_xticklabels(df_1.columns,ha='center')
        ax1.tick_params(axis='y', labelsize=18,rotation=0,pad=11,length=0)
        ax1.set_yticks([0,1,2,3,4,5,6,7,8,9])
        ax1.set_yticklabels(df_1.index,ha='center')
        # ax1.set_title('Reddit(Baseline 0.939)', fontsize=18)

        lower_ticks_2 = np.linspace(0.0, vmin_2, 6, endpoint=False)
        upper_ticks_2 = np.linspace(vmin_2, vmax_2, 6, endpoint=True)

        # 合并刻度数组
        ticks_2 = np.concatenate([lower_ticks_2, upper_ticks_2])        

        # heatmap = sns.heatmap(df, annot=True, cmap='YlGnBu', fmt='.1f')  # 保留一位小数
        
        # heatmap = sns.heatmap(df_2,ax=ax3 ,annot=True, fmt=".3f",cmap='YlGnBu',norm=Normalize(vmin=vmin_2, vmax=vmax_2), cbar_kws={"ticks": np.linspace(vmin_2, vmax_2, 11)})
        # heatmap = sns.heatmap(df_2, ax=ax3, annot=True, fmt=".3f", cmap=new_cmap, norm=Normalize(vmin=vmin_2, vmax=vmax_2), cbar_kws={"extend": "min", "ticks": np.linspace(vmin_2, vmax_2, 11)})
        heatmap1 = sns.heatmap(df_2, ax=ax3, annot=True, fmt=".3f", cmap=cmap, norm=norm_2, cbar_kws={"ticks": ticks_2})
        # ax3.set_title('Ogbn-Products(Baseline 0.911)', fontsize=18)
        ax3.tick_params(axis='x', labelsize=18,rotation=0,length=0)
        ax3.set_xticks([1,2,3,4,5,6,7,8,9,10])
        ax3.set_xticklabels(df_2.columns,ha='center')
        ax3.tick_params(axis='y', labelsize=18,rotation=0,pad=11,length=0)
        ax3.set_yticks([0,1,2,3,4,5,6,7,8,9])
        ax3.set_yticklabels(df_2.index,ha='center')
        # plt.title('Ogbn-Products(Baseline 91.1)', fontsize=18)
        # plt.savefig(f'./picture/tmp-hot-{dataset}.pdf',bbox_inches='tight')
        rect_low = Rectangle((0, 6), 4, 4, linewidth=3, edgecolor='red', facecolor='none')
        rect_mod = Rectangle((4, 0.01), 3, 3, linewidth=3, edgecolor='black', facecolor='none')
        rect_high = Rectangle((7, 0.01), 3, 3, linewidth=3, edgecolor='blue', facecolor='none')
        ax3.add_patch(rect_low)
        ax3.add_patch(rect_mod)
        ax3.add_patch(rect_high)
        ax3.set_xlabel('Neighbor Similarity', fontsize=20)  # 设置标签字体大小
        ax3.set_ylabel('Neighbor Similarity', fontsize=20)

        ax1.text(-0.2, 10.2, '0.0', ha='center', va='center', color='black', fontsize=13)
        ax3.text(-0.2, 10.2, '0.0', ha='center', va='center', color='black', fontsize=13)
        low_patch = Patch( label='Low Similarity Region', linewidth=2, edgecolor='red', facecolor='none')
        mod_patch = Patch( label='Moderate Similarity Region', linewidth=2, edgecolor='black', facecolor='none')
        high_patch = Patch( label='High Similarity Region', linewidth=2, edgecolor='blue', facecolor='none')
        plt.legend(handles=[low_patch,mod_patch,high_patch],loc='upper left',bbox_to_anchor=(-1.3, 1.13),ncol=3,frameon=False,fontsize=20)
        ax1.annotate('(a) Reddit(Baseline 0.939)', xy=(0.5, -0.15), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=20)
        ax3.annotate('(b) Ogbn-Products(Baseline 0.911)', xy=(0.5, -0.15), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=20)
        plt.savefig(f'./picture/tmp-hot-test-2.pdf',bbox_inches='tight')
        plt.show()


# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # 创建一个10x10的随机数据矩阵
# data = np.random.rand(10, 10)

# # 为数据设置列名
# columns = [i for i in range(10)]

# # 创建热力图
# fig, ax = plt.subplots(figsize=(12, 10))
# sns.heatmap(data, annot=True, fmt=".2f", cmap='viridis', ax=ax,
#             xticklabels=columns)

# # 设置x轴标签旋转为水平，并确保标签左对齐
# ax.tick_params(axis='x', labelsize=12, rotation=0)
# ax.set_xticks(np.arange(data.shape[1]), minor=False)  # 设置x轴刻度位于方块左边缘
# ax.set_xticklabels(columns, rotation=0, ha="left")  # 水平显示且左对齐

# # 设置y轴标签字体大小
# ax.tick_params(axis='y', labelsize=12)
# plt.savefig(f'./picture/tmp-hot-test-1.pdf',bbox_inches='tight')
# # 显示图形
# plt.show()