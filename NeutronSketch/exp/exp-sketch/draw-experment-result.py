import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import PercentFormatter

myparams = {
        'axes.labelsize': '14',
        'xtick.labelsize': '14',
        'ytick.labelsize': '14',
        'font.family': 'Times New Roman',
        'figure.figsize': '5, 4',  #图片尺寸
        'lines.linewidth': 2,
        'legend.fontsize': '14',
        # 'legend.loc': 'best', #[]"upper right", "upper left"]
        'legend.loc': 'upper left', #[]"upper right", "upper left"]
        'legend.numpoints': 1,
        'legend.frameon': False,
        # 'lines.ncol': 2,
    }
plt.rcParams.update(myparams)

# 创建数据
x = np.array([1, 2, 3])
bar1_data = np.array([9.6, 22.3, 43.8])
bar2_data = np.array([93.3, 63.4, 82])

# 创建图表和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
plt.subplots_adjust(wspace=0.3)
# 绘制第一个柱形图
bar_width = 0.35
bar1 = ax1.bar(x, bar1_data, bar_width, color='teal', alpha=0.7, label='Training Time With NeutronSketch')
ax1.set_ylabel('Norm. Training Time (%)')
ax1.set_xlabel('(a) GCN')
# ax1.legend(loc='upper left', frameon=False)
ax1.set_xticks(x)
ax1.set_xticklabels(['Ogbn-Arxiv', 'Reddit', 'Ogbn-Products'])
ax1.set_ylim(0, 100)
# ax1.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# 绘制第二个柱形图
bar2 = ax2.bar(x, bar2_data, bar_width, color='teal', alpha=0.7)
# ax2.set_ylabel('Relative Training Time')
ax2.set_xlabel('(b) ClusterGCN')
# ax2.legend(loc='upper left', frameon=False)
ax2.set_xticks(x)
ax2.set_xticklabels(['Ogbn-Arxiv', 'Reddit', 'Ogbn-Products'])
# ax2.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
ax2.set_ylim(0, 100)
ax2.set_ylabel('Norm. Training Time (%)')
# ax1.yaxis.set_label_coords(-0.15, 0.5)  # 调整第一个子图的 y 轴标签位置
# ax2.yaxis.set_label_coords(-0.15, 0.5)  # 调整第二个子图的 y 轴标签位置
fig.legend(loc='upper left',bbox_to_anchor=(0.3, 1.0),ncol=3,frameon=False)

plt.savefig('./tmp-4.pdf',bbox_inches='tight')

# 显示图表
plt.show()