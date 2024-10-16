# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import PercentFormatter

# myparams = {
#         'axes.labelsize': '13',
#         'xtick.labelsize': '13',
#         'ytick.labelsize': '13',
#         'font.family': 'Times New Roman',
#         'figure.figsize': '5, 4',  #图片尺寸
#         'lines.linewidth': 2,
#         'legend.fontsize': '13',
#         # 'legend.loc': 'best', #[]"upper right", "upper left"]
#         'legend.loc': 'upper left', #[]"upper right", "upper left"]
#         'legend.numpoints': 1,
#         'legend.frameon': False,
#         # 'lines.ncol': 2,
#     }
# plt.rcParams.update(myparams)


# # 创建数据
# categories = ['Ogbn-Arxiv', 'Reddit', 'Ogbn-Products']
# data_part1 = np.array([37.8, 51.7, 52.3])
# data_part2 = np.array([61.4, 47.2, 45.9])
# data_part3 = np.array([0.8, 1.1, 1.8])

# # 创建柱形图
# fig, ax = plt.subplots(figsize=(5, 4))
# # ax.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# bar_width = 0.35
# line_width=0.5
# bar1 = ax.bar(categories, data_part1, bar_width,color='#0080FF', label='Nbr Sim > 0.9',lw = line_width,ec='black')
# bar2 = ax.bar(categories, data_part2, bar_width,color='#B02417', bottom=data_part1, label='Others',lw = line_width,ec='black')
# bar3 = ax.bar(categories, data_part3, bar_width,color='#F29E38', bottom=data_part1+data_part2, label='Nbr Sim < 0.1',lw = line_width,ec='black')

# # 添加标签和图例
# ax.set_ylabel('Nbr Similarity Distribution(%)')
# # ax.set_title('Stacked Bar Chart with Three Parts')
# ax.legend(loc='upper left',bbox_to_anchor=(-0.15, 1.12),ncol=3,frameon=False)

# plt.savefig('./tmp-1.pdf',bbox_inches='tight')

# # 显示图表
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

myparams = {
        'axes.labelsize': '16',
        'xtick.labelsize': '16',
        'ytick.labelsize': '16',
        'font.family': 'Times New Roman',
        'figure.figsize': '5, 4',  #图片尺寸
        'lines.linewidth': 2,
        'legend.fontsize': '15',
        # 'legend.loc': 'best', #[]"upper right", "upper left"]
        'legend.loc': 'upper left', #[]"upper right", "upper left"]
        'legend.numpoints': 1,
        'legend.frameon': False,
        # 'lines.ncol': 2,
    }
plt.rcParams.update(myparams)


# 创建数据
categories = ['Ogbn-Arxiv', 'Reddit', 'Ogbn-Products']
data_part1 = np.array([0.8, 1.1, 1.8])
data_part2 = np.array([61.4, 47.2, 45.9])
data_part3 = np.array([37.8, 51.7, 52.3])
# 0.8 1.1 1.8
# 创建柱形图
fig, ax = plt.subplots(figsize=(5, 4))
# ax.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
bar_width = 0.25
line_width=0.8
bar1 = ax.bar(categories, data_part1, bar_width,color='#0080FF', label='Neighbor Similarity < 0.1',lw = line_width,ec='black')
bar2 = ax.bar(categories, data_part2, bar_width,color='#B02417', bottom=data_part1, label='0.1< Neighbor Similarity < 0.9',lw = line_width,ec='black')
bar3 = ax.bar(categories, data_part3, bar_width,color='#F29E38', bottom=data_part1+data_part2, label='0.9 < Neighbor Similarity',lw = line_width,ec='black')

# 添加标签和图例
ax.set_ylabel('Vertices Distribution(%)',fontsize=16)
# ax.set_title('Stacked Bar Chart with Three Parts')
ax.legend(loc='center left',bbox_to_anchor=(1.0, 0.5),ncol=1,frameon=False,labelspacing=1.8)

plt.savefig('./tmp-1-2.pdf',bbox_inches='tight')

# 显示图表
plt.show()