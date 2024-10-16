# Motivation 

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
# x = np.array([1, 2, 3,4])
# # x_1 = [0.5,1.5,2.5]
# # x_2 = [1.5,2.5,3.5]
# bar1_data = np.array([100, 19.2, 22.2,9.3])
# # bar2_data = np.array([88, 70, 60])
# line1_data = np.array([93.9,93.1, 94.0, 92.5])
# # line2_data = np.array([91.1,86.6, 88.1, 83.7])

# plt.figure(figsize=(7, 4))

# # 创建图表和子图
# fig, ax1 = plt.subplots()

# # 绘制两组柱形图在左侧y轴
# bar_width = 0.35
# bar1 = ax1.bar(x , bar1_data, bar_width, color='teal', alpha=0.7, label='Training Time Percentage')
# # bar2 = ax1.bar(x + bar_width/2, bar2_data, bar_width, color='g', alpha=0.7, label='NeutronSketch')
# # ax1.set_xlabel('X Axis')
# ax1.set_ylabel('Relative Training Time')
# ax1.tick_params('y')
# ax1.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# # 在同一图中创建另一个y轴用于折线图
# ax2 = ax1.twinx()
# ax2.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# ax2.plot(x, line1_data, color='purple', marker='o', linestyle='--', label='Val ACC')
# # ax2.plot(x + bar_width/2, line2_data, color='black', marker='x', linestyle='--', label='NeutronSketch')
# ax2.set_ylabel('Val ACC')
# ax2.tick_params('y')

# ax2.axhline(93.9, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3,label='Origin Val ACC')

# ax1.set_ylim(0, 100)
# ax2.set_ylim(90, 95)

# # 设置图例
# ax1.legend(loc='upper left',bbox_to_anchor=(1.2, 0.50),ncol=1,frameon=False)
# ax2.legend(loc='upper left',bbox_to_anchor=(1.2, 0.62),ncol=1,frameon=False)

# ax1.set_xticks(x)
# ax1.set_xticklabels(['Origin', 'Setup-1', 'Setup-2', 'Setup-3'])

# # 添加标题
# # plt.title('Combined Bar and Line Chart')
# # fig.text(0.5, -0.05, 'Reddit', ha='center', fontsize=10)
# # 保存图表到本地文件
# plt.savefig('./tmp.pdf',bbox_inches='tight')

# # 显示图表
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

myparams = {
        'axes.labelsize': '18',
        'xtick.labelsize': '18',
        'ytick.labelsize': '18',
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


# 数据准备
# sec3.1
# x = np.array([1, 2, 3, 4])

x = np.array([1.5, 3])
x1 = np.array([0.8,2.2])
# x = ['Origin', '> 0.9', '> 0.9 with 0.3-0.4', '> 0.9 with 0.0-0.1']
# bar_data_1 = np.array([100, 35.5, 27.4,33.5])
# # # bar2_data = np.array([88, 70, 60])
# line_data_1 = np.array([93.9,93.8, 93.9, 93.7])

# nbr details result after delete
bar_data_1 = np.array([100,  62.6])
line_data_1 = np.array([93.9, 93.8])

# # line2_data = np.array([91.1,86.6, 88.1, 83.7])
# bar_data_1 = np.array([10, 15, 7, 12])
# line_data_1 = np.array([30, 40, 20, 25])

# bar_data_2 = np.array([100, 51.0, 53.9, 49.8])
# line_data_2 = np.array([91.1, 90.8, 90.8, 90.6])

# nbr details result after delete
bar_data_2 = np.array([100,  73.0])
line_data_2 = np.array([91.1, 90.9])

# # sec 4
# x_1 = ['origin','100', '200', '300']
# x_2 = ['origin','0.1', '0.2', '0.3']
# bar_data_1 = np.array([100,31.7, 43.7, 65.1])
# # # bar2_data = np.array([88, 70, 60])
# line_data_1 = np.array([91.1,90.3, 90.7, 90.7])
# # # line2_data = np.array([91.1,86.6, 88.1, 83.7])
# # bar_data_1 = np.array([10, 15, 7, 12])
# # line_data_1 = np.array([30, 40, 20, 25])
# bar_data_2 = np.array([100, 40.4, 43.7, 54.3])
# line_data_2 = np.array([91.1, 90.7, 90.7, 90.8])

# 创建画布和两个子图
fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 4))
plt.subplots_adjust(wspace=1)
# bar_width = 0.35
bar_width = 0.30
line_width=1.5
# 在第一个子图中绘制柱形图
ax1.bar(x1-bar_width/2 , bar_data_1, bar_width,  color='#0080FF', label='Training Time',lw = line_width,ec='black')
# ax1.bar(x - 0.2, bar_data_1, width=0.4, label='Bar 1', color='b', align='center')
# ax1.set_xticks([1,2,3,4],labels=['Origin', '> 0.9', '> 0.9 with 0.3-0.4', '> 0.9 with 0.0-0.1'])
ax1.set_xlim(0,3)
# ax1.set_xticks([1,2,3,4],labels=['Origin', '>0.95', '>0.90', '>0.85'])
ax1.set_xticks([0.8,2.2],labels=['Origin', 'Delete'])
# 创建第一个子图的第二个y轴
ax2 = ax1.twinx()

# 在第一个子图中绘制折线图
# ax2.plot(x1, line_data_1, color='#B02417', marker='o', linestyle='--', label='Val ACC')
ax2.bar(x1+bar_width/2 , line_data_1, bar_width,  color='#B02417', label='Val ACC',lw = line_width,ec='black')
# ax2.axhline(93.9, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3,label='Origin Val ACC')
# ax2.plot(x, line_data_1, label='Line 1', color='r', marker='o')

ax1.set_ylim(0, 105)
ax2.set_ylim(90, 95)
# ax2.set_yticks([90,90.5,91,91.5,92])

ax2.axhline(93.9, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3)
ax1.set_ylabel('Norm. Training Time (%)',fontsize=22)
ax2.set_ylabel('Val ACC (%)',fontsize=22)
# ax1.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# ax2.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# ax1.set_xticklabels(['Origin', 'Setup-1', 'Setup-2', 'Setup-3'])
# 在第二个子图中绘制柱形图
ax3.bar(x1-bar_width/2 , bar_data_2, bar_width, color='#0080FF',  label='Training Time',lw = line_width,ec='black')
# ax3.bar(x - 0.2, bar_data_2, width=0.4, label='Bar 2', color='g', align='center')
# ax3.set_xticks([1,3,5,7],labels=['Origin', '> 0.9', '> 0.9 with 0.3-0.4', '> 0.9 with 0.0-0.1'])
ax3.set_xlim(0,3)
# ax3.set_xticks([1,2,3,4],labels=['Origin', '>0.95', '>0.90', '>0.85'])
ax3.set_xticks([0.8,2.2],labels=['Origin','Delete'])
ax3.set_ylabel('Norm. Training Time (%)',fontsize=22)
# 创建第二个子图的第二个y轴
ax4 = ax3.twinx()

# 在第二个子图中绘制折线图
# ax4.plot(x1, line_data_2, color='#B02417', marker='o', linestyle='--', label='Val ACC')
ax4.bar(x1+bar_width/2 , line_data_2, bar_width, color='#B02417',  label='Val ACC',lw = line_width,ec='black')
# ax4.plot(x, line_data_2, label='Line 2', color='purple', marker='x')
ax3.set_ylim(0, 105)
ax4.set_ylim(80, 92)
ax4.set_yticks([80,83,86,89,92])
ax4.axhline(91.1, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3,label='Origin Val ACC')
ax4.set_ylabel('Val ACC (%)',fontsize=22)
# ax3.set_xticklabels(['Origin', 'Setup-1', 'Setup-2', 'Setup-3'])
# 共享图例
# ax3.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# ax4.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# ax1.set_title('(a) Reddit')
# ax3.set_title('(b) Ogbn-Products')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax4.get_legend_handles_labels()
lines += lines2
labels += labels2

# 在第一个子图上显示图例
# ax1.legend(lines, labels, loc='upper left')

# 在第二个子图上显示图例
fig.legend(lines, labels, loc='upper left',bbox_to_anchor=(0.1, 1.1),ncol=3,frameon=False)

# ax1.annotate('(a) TH_Degree', xy=(0.5, -0.15), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=13)
# ax3.annotate('(b) TH_Inter_Class_Sim', xy=(0.5, -0.15), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=13)

ax1.annotate('(a) Reddit', xy=(0.5, -0.17), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=23)
ax3.annotate('(b) Ogbn-Products', xy=(0.5, -0.17), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=23)


# 调整子图布局
plt.tight_layout()

# plt.savefig('./tmp-5-sec3-high-degree-1.pdf',bbox_inches='tight')

plt.savefig('./picture/nbr-after-delete.pdf',bbox_inches='tight')
# 显示图形
plt.show()