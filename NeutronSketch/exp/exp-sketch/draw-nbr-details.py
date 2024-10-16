import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

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
# x = np.array([1, 2, 3,4])
# # x_1 = [0.5,1.5,2.5]
# # x_2 = [1.5,2.5,3.5]
# bar1_data = np.array([0.19, 0.23, 0.67,0.38])
# # bar2_data = np.array([88, 70, 60])
# line1_data = np.array([3,4, 26, 3])
# # line2_data = np.array([91.1,86.6, 88.1, 83.7])

# plt.figure(figsize=(7, 4))

# # 创建图表和子图
# fig, ax1 = plt.subplots()

# # 绘制两组柱形图在左侧y轴
# bar_width = 0.35
# bar1 = ax1.plot(x , bar1_data, color='teal',  marker='*', linestyle='--', label='Inter Class Simliarity With 23')
# # bar2 = ax1.bar(x + bar_width/2, bar2_data, bar_width, color='g', alpha=0.7, label='NeutronSketch')
# # ax1.set_xlabel('X Axis')
# ax1.set_ylabel('Inter Class Similarity')
# ax1.tick_params('y')
# # ax1.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# # 在同一图中创建另一个y轴用于折线图
# ax2 = ax1.twinx()
# # ax2.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# ax2.plot(x, line1_data, color='purple', marker='o', linestyle='--', label='Number of misclassified vertices in validation results')

# # ax2.plot(x + bar_width/2, line2_data, color='black', marker='x', linestyle='--', label='NeutronSketch')
# ax2.set_ylabel('Nodes Number')
# ax2.tick_params('y')

# # ax2.axhline(93.9, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3,label='Origin Val ACC')

# ax1.set_ylim(0, 1)
# ax2.set_ylim(0, 30)

# # 设置图例
# ax1.legend(loc='upper left',bbox_to_anchor=(1.2, 0.50),ncol=1,frameon=False)
# ax2.legend(loc='upper left',bbox_to_anchor=(1.2, 0.62),ncol=1,frameon=False)

# ax1.set_xticks(x)
# ax1.set_xticklabels(['18', '19', '22', '39'])

# # 添加标题
# # plt.title('Combined Bar and Line Chart')
# # fig.text(0.5, -0.05, 'Reddit', ha='center', fontsize=10)
# # 保存图表到本地文件
# plt.savefig('./tmp-3.pdf',bbox_inches='tight')

# # 显示图表
# plt.show()

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


fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(9, 3.5))
plt.subplots_adjust(wspace=0.5)
# 创建数据
x = np.array([1, 2, 3,4])
# x_1 = [0.5,1.5,2.5]
# x_2 = [1.5,2.5,3.5]
line1_data_1 = np.array([0.19, 0.23, 0.67,0.38])
# bar2_data = np.array([88, 70, 60])
line1_data_2 = np.array([3,4, 26, 3])
# line2_data = np.array([91.1,86.6, 88.1, 83.7])
line1_data_3 = np.array([0.16,0.66, 0.53, 0.64])
line1_data_4 = np.array([4,34, 15, 21])

bar1 = ax1.plot(x , line1_data_1, color='#0080FF',  marker='*', linestyle='--', label='Inter Class Simliarity')
# bar2 = ax1.bar(x + bar_width/2, bar2_data, bar_width, color='g', alpha=0.7, label='NeutronSketch')
# ax1.set_xlabel('X Axis')
ax1.set_ylabel('Inter Class Similarity')
ax1.tick_params('y')
# ax1.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# 在同一图中创建另一个y轴用于折线图
ax2 = ax1.twinx()
# ax2.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
ax2.plot(x, line1_data_2, color='#B02417', marker='o', linestyle='--', label='Number of Misclassified Vertices in Validation Results')

# ax2.plot(x + bar_width/2, line2_data, color='black', marker='x', linestyle='--', label='NeutronSketch')
# ax2.set_ylabel('Nodes Number')
# ax2.tick_params('y')

# ax2.axhline(93.9, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3,label='Origin Val ACC')

ax1.set_ylim(0, 1)
ax2.set_ylim(0, 30)
ax2.set_yticks([0,6,12,18,24,30])
ax1.set_xticks(x)
ax1.set_xticklabels(['18', '19', '22', '39'])
ax2.set_ylabel('Nodes Number')
ax3.plot(x , line1_data_3, color='#0080FF',  marker='*', linestyle='--', label='Inter Class Simliarity')
# ax3.bar(x - 0.2, bar_data_2, width=0.4, label='Bar 2', color='g', align='center')
# ax3.set_xticks([1,3,5,7],labels=['Origin', '> 0.9', '> 0.9 with 0.3-0.4', '> 0.9 with 0.0-0.1'])
# 创建第二个子图的第二个y轴
ax4 = ax3.twinx()

# 在第二个子图中绘制折线图
ax4.plot(x, line1_data_4, color='#B02417', marker='o', linestyle='--', label='Number of Misclassified Vertices in Validation Results')
# ax4.plot(x, line_data_2, label='Line 2', color='purple', marker='x')
ax3.set_ylim(0, 1)
ax4.set_ylim(0, 40)
ax4.set_yticks([0,10,20,30,40])
# ax4.axhline(91.1, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3,label='Origin Val ACC')
ax4.set_ylabel('Nodes Number')
ax3.set_xticks(x)
ax3.set_xticklabels(['2', '3', '9', '10'])
ax3.set_ylabel('Inter Class Similarity')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax4.get_legend_handles_labels()
lines += lines2
labels += labels2

# 在第一个子图上显示图例
# ax1.legend(lines, labels, loc='upper left')

# 在第二个子图上显示图例
fig.legend(lines, labels, loc='upper left',bbox_to_anchor=(0.1, 1.0),ncol=2,frameon=False)

ax1.annotate('(a) The Nbr details of label 23 in Reddit', xy=(0.5, -0.15), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=15)
ax3.annotate('(b) The Nbr details of label 0 in Products', xy=(0.5, -0.15), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=15)


plt.savefig('./tmp-3-nbr-details.pdf',bbox_inches='tight')

# 显示图表
plt.show()