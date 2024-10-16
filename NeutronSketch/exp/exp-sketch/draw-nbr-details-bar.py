# # Motivation 

# # import matplotlib.pyplot as plt
# # import numpy as np
# # from matplotlib.ticker import PercentFormatter

# # myparams = {
# #         'axes.labelsize': '12',
# #         'xtick.labelsize': '12',
# #         'ytick.labelsize': '12',
# #         # 'font.family': 'Times New Roman',
# #         'figure.figsize': '5, 4',  #图片尺寸
# #         'lines.linewidth': 2,
# #         'legend.fontsize': '9',
# #         # 'legend.loc': 'best', #[]"upper right", "upper left"]
# #         'legend.loc': 'upper left', #[]"upper right", "upper left"]
# #         'legend.numpoints': 1,
# #         'legend.frameon': False,
# #         # 'lines.ncol': 2,
# #     }
# # plt.rcParams.update(myparams)

# # # 创建数据
# # x = np.array([1, 2, 3,4])
# # # x_1 = [0.5,1.5,2.5]
# # # x_2 = [1.5,2.5,3.5]
# # bar1_data = np.array([100, 19.2, 22.2,9.3])
# # # bar2_data = np.array([88, 70, 60])
# # line1_data = np.array([93.9,93.1, 94.0, 92.5])
# # # line2_data = np.array([91.1,86.6, 88.1, 83.7])

# # plt.figure(figsize=(7, 4))

# # # 创建图表和子图
# # fig, ax1 = plt.subplots()

# # # 绘制两组柱形图在左侧y轴
# # bar_width = 0.35
# # bar1 = ax1.bar(x , bar1_data, bar_width, color='teal', alpha=0.7, label='Training Time Percentage')
# # # bar2 = ax1.bar(x + bar_width/2, bar2_data, bar_width, color='g', alpha=0.7, label='NeutronSketch')
# # # ax1.set_xlabel('X Axis')
# # ax1.set_ylabel('Relative Training Time')
# # ax1.tick_params('y')
# # ax1.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# # # 在同一图中创建另一个y轴用于折线图
# # ax2 = ax1.twinx()
# # ax2.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# # ax2.plot(x, line1_data, color='purple', marker='o', linestyle='--', label='Val ACC')
# # # ax2.plot(x + bar_width/2, line2_data, color='black', marker='x', linestyle='--', label='NeutronSketch')
# # ax2.set_ylabel('Val ACC')
# # ax2.tick_params('y')

# # ax2.axhline(93.9, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3,label='Origin Val ACC')

# # ax1.set_ylim(0, 100)
# # ax2.set_ylim(90, 95)

# # # 设置图例
# # ax1.legend(loc='upper left',bbox_to_anchor=(1.2, 0.50),ncol=1,frameon=False)
# # ax2.legend(loc='upper left',bbox_to_anchor=(1.2, 0.62),ncol=1,frameon=False)

# # ax1.set_xticks(x)
# # ax1.set_xticklabels(['Origin', 'Setup-1', 'Setup-2', 'Setup-3'])

# # # 添加标题
# # # plt.title('Combined Bar and Line Chart')
# # # fig.text(0.5, -0.05, 'Reddit', ha='center', fontsize=10)
# # # 保存图表到本地文件
# # plt.savefig('./tmp.pdf',bbox_inches='tight')

# # # 显示图表
# # plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import PercentFormatter

# myparams = {
#         'axes.labelsize': '18',
#         'xtick.labelsize': '18',
#         'ytick.labelsize': '18',
#         'font.family': 'Times New Roman',
#         'figure.figsize': '5, 4',  #图片尺寸
#         'lines.linewidth': 2,
#         'legend.fontsize': '18',
#         # 'legend.loc': 'best', #[]"upper right", "upper left"]
#         'legend.loc': 'upper left', #[]"upper right", "upper left"]
#         'legend.numpoints': 1,
#         'legend.frameon': False,
#         # 'lines.ncol': 2,
#     }
# plt.rcParams.update(myparams)


# # 数据准备
# # sec3.1
# x = np.array([1, 2, 3,4])
# x = np.array([1, 2, 3,4])
# x_line_1 = np.array([1.175, 2.175, 3.175,4.175])
# x_line_2 = np.array([1-0.175, 2-0.175, 3-0.175,4-0.175])
# x_line_3 = np.array([1.175, 2.175, 3.175,])
# # x = ['Origin', '> 0.9', '> 0.9 with 0.3-0.4', '> 0.9 with 0.0-0.1']
# bar_data_1 = np.array([0.19, 0.23, 0.67,0.38])
# bar_data_2 = np.array([3,4, 26, 3])
# bar_data_1_2 = np.array([0.16,0.66, 0.53, 0.64])
# bar_data_2_2 = np.array([4,34, 15, 21])
# # # bar2_data = np.array([88, 70, 60])
# line_data_1 = np.array([71.4,93.9, 91.1])
# line_data_2 = np.array([71.3,93.9, 90.7])
# line_data_3 = np.array([53.3,92.5, 85.1])
# # # line2_data = np.array([91.1,86.6, 88.1, 83.7])
# # bar_data_1 = np.array([10, 15, 7, 12])
# # line_data_1 = np.array([30, 40, 20, 25])
# bar_data_3 = np.array([174.78, 484.73, 713.02])
# bar_data_4 = np.array([163.05, 307.05, 584.64])
# line_data_4 = np.array([66.5,91.3, 80.9])
# line_data_5 = np.array([67.0,91.5, 80.8])
# line_data_6 = np.array([62.8,91.0, 79.3])
# # line_data_2 = np.array([91.1, 86.6, 88.1])

# # # sec 4
# # x_1 = ['origin','100', '200', '300']
# # x_2 = ['origin','0.1', '0.2', '0.3']
# # bar_data_1 = np.array([100,31.7, 43.7, 65.1])
# # # # bar2_data = np.array([88, 70, 60])
# # line_data_1 = np.array([91.1,90.3, 90.7, 90.7])
# # # # line2_data = np.array([91.1,86.6, 88.1, 83.7])
# # # bar_data_1 = np.array([10, 15, 7, 12])
# # # line_data_1 = np.array([30, 40, 20, 25])
# # bar_data_2 = np.array([100, 40.4, 43.7, 54.3])
# # line_data_2 = np.array([91.1, 90.7, 90.7, 90.8])

# # 创建画布和两个子图
# fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(10, 4))
# plt.subplots_adjust(wspace=1)
# bar_width = 0.35
# line_width = 1.5
# # 在第一个子图中绘制柱形图
# bar1 = ax1.bar(x - bar_width/2 , bar_data_1,bar_width, color='#0080FF',   label='Inter Class Simliarity',lw = line_width,ec='black')
# # ax1.bar(x - bar_width/2 , bar_data_1, bar_width, color='#0080FF',  label='Origin',lw = line_width,ec='black')

# # for i,v in (zip(x_line_1,bar_data_2)):
# #     ax1.text(i,v,str(v),ha='center',va='bottom')
# # for i,v in (zip(x_line_2,bar_data_1)):
# #     ax1.text(i,v,str(v),ha='center',va='bottom')    
# # plt.bar_label(p2, label_type='edge')
# # ax1.bar(x - 0.2, bar_data_1, width=0.4, label='Bar 1', color='b', align='center')
# # ax1.set_xticks([1,2,3,4],labels=['Origin', '> 0.9', '> 0.9 with 0.3-0.4', '> 0.9 with 0.0-0.1'])
# ax1.set_xticks([1,2,3,4],labels=['18', '19', '22','39'])
# # 创建第一个子图的第二个y轴
# ax2 = ax1.twinx()
# p2 = ax2.bar(x + bar_width/2 , bar_data_2, bar_width, color='#B02417',  label='Number of Misclassified Vertices',lw = line_width,ec='black')
# # # 在第一个子图中绘制折线图
# # ax2.plot(x_line_1, line_data_1, color='purple', marker='o', linestyle='--', label='Origin ACC')
# # ax2.plot(x_line_2, line_data_2, color='red', marker='*', linestyle='--', label='Sketch ACC')
# # ax2.plot(x_line_3, line_data_3, color='black', marker='^', linestyle='--', label='Negative ACC')
# # ax2.axhline(93.9, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3,label='Origin Val ACC')
# # ax2.plot(x, line_data_1, label='Line 1', color='r', marker='o')

# ax1.set_ylim(0, 1)
# ax2.set_ylim(0, 30)
# ax2.set_yticks([0,6,12,18,24,30])
# # ax1.set_xticks(x)
# # ax1.set_xticklabels(['18', '19', '22', '39'])
# ax1.set_ylabel('Inter Class Similarity')
# ax2.set_ylabel('Nodes Number')

# # ax1.set_ylim(0, 300)
# # ax2.set_ylim(50, 95)
# # ax2.set_yticks([90,90.5,91,91.5,92])

# # ax2.axhline(93.9, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3)
# # ax1.set_ylabel('Converge Time (s)')
# # ax2.set_ylabel('Val ACC (%)')
# # ax1.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# # ax2.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# # ax1.set_xticklabels(['Origin', 'Setup-1', 'Setup-2', 'Setup-3'])
# # 在第二个子图中绘制柱形图
# ax3.bar(x - bar_width/2 , bar_data_1_2, bar_width, color='#0080FF',   label='Inter Class Simliarity',lw = line_width,ec='black')
# # for i,v in (zip(x_line_2,bar_data_3)):
# #     ax3.text(i,v,str(v),ha='center',va='bottom')
# # for i,v in (zip(x_line_1,bar_data_4)):
# #     ax3.text(i,v,str(v),ha='center',va='bottom')  
# # plt.bar_label(p4, label_type='edge')
# # ax3.bar(x - 0.2, bar_data_2, width=0.4, label='Bar 2', color='g', align='center')
# # ax3.set_xticks([1,3,5,7],labels=['Origin', '> 0.9', '> 0.9 with 0.3-0.4', '> 0.9 with 0.0-0.1'])
# ax3.set_xticks([1,2,3,4],labels=['2', '3', '9','10'])
# # ax3.set_ylabel('Converge Time (s)')
# # 创建第二个子图的第二个y轴
# ax4 = ax3.twinx()
# p4 = ax4.bar(x + bar_width/2 , bar_data_2_2, bar_width, color='#B02417',  label='Number of Misclassified Vertices',lw = line_width,ec='black')


# ax3.set_ylim(0, 1)
# ax4.set_ylim(0, 40)
# ax4.set_yticks([0,10,20,30,40])
# # ax4.axhline(91.1, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3,label='Origin Val ACC')
# ax4.set_ylabel('Nodes Number')
# # ax3.set_xticks(x)
# # ax3.set_xticklabels(['2', '3', '9', '10'])
# ax3.set_ylabel('Inter Class Similarity')

# # 在第二个子图中绘制折线图
# # ax4.plot(x_line_1, line_data_4, color='purple', marker='o', linestyle='--', label='Origin ACC')
# # ax4.plot(x_line_2, line_data_5, color='red', marker='*', linestyle='--', label='Sketch ACC')
# # ax4.plot(x_line_3, line_data_6, color='black', marker='^', linestyle='--', label='Negative ACC')
# # ax4.plot(x, line_data_2, label='Line 2', color='purple', marker='x')
# # ax3.set_ylim(150, 750)
# # ax4.set_ylim(50, 93)
# # ax4.set_yticks([80,83,86,89,92])
# # ax4.axhline(91.1, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3,label='Origin Val ACC')
# # ax4.set_ylabel('Val ACC (%)')
# # ax3.set_xticklabels(['Origin', 'Setup-1', 'Setup-2', 'Setup-3'])
# # 共享图例
# # ax3.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# # ax4.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
# # ax1.set_title('(a) Reddit')
# # ax3.set_title('(b) Ogbn-Products')
# lines, labels = ax1.get_legend_handles_labels()
# lines2, labels2 = ax4.get_legend_handles_labels()
# lines += lines2
# labels += labels2

# # 在第一个子图上显示图例
# # ax1.legend(lines, labels, loc='upper left')

# # 在第二个子图上显示图例
# fig.legend(lines, labels, loc='upper left',bbox_to_anchor=(0.15, 1.1),ncol=2,frameon=False)

# ax1.annotate('(a) The Nbr details of class 23 in Reddit', xy=(0.5, -0.15), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=16)
# ax3.annotate('(b) The Nbr details of class 0 in Products', xy=(0.5, -0.15), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=16)


# # 调整子图布局
# plt.tight_layout()

# plt.savefig('./tmp-3-nbr-details-bar.pdf',bbox_inches='tight')
# # 显示图形
# plt.show()


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
import os
import re
from matplotlib.ticker import PercentFormatter
from scipy.interpolate import interp1d
import scipy.stats as stats
def draw():
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


    # 数据准备
    # sec3.1
    x = np.array([1, 2, 3,4])
    x = np.array([1, 2, 3,4])
    x_line_1 = np.array([1.175, 2.175, 3.175,4.175])
    x_line_2 = np.array([1-0.175, 2-0.175, 3-0.175,4-0.175])
    x_line_3 = np.array([1.175, 2.175, 3.175,])
    # x = ['Origin', '> 0.9', '> 0.9 with 0.3-0.4', '> 0.9 with 0.0-0.1']
    bar_data_1 = np.array([0.19, 0.23, 0.67,0.38])
    bar_data_2 = np.array([3,4, 26, 3])
    # bar_data_2 = np.array([0,3, 13, 2])
    bar_data_1_2 = np.array([0.16,0.66, 0.53, 0.64])
    bar_data_2_2 = np.array([4,34, 15, 21])
    # bar_data_2_2 = np.array([3,20, 15, 13])
    # # bar2_data = np.array([88, 70, 60])
    line_data_1 = np.array([71.4,93.9, 91.1])
    line_data_2 = np.array([71.3,93.9, 90.7])
    line_data_3 = np.array([53.3,92.5, 85.1])
    # # line2_data = np.array([91.1,86.6, 88.1, 83.7])
    # bar_data_1 = np.array([10, 15, 7, 12])
    # line_data_1 = np.array([30, 40, 20, 25])
    bar_data_3 = np.array([174.78, 484.73, 713.02])
    bar_data_4 = np.array([163.05, 307.05, 584.64])
    line_data_4 = np.array([66.5,91.3, 80.9])
    line_data_5 = np.array([67.0,91.5, 80.8])
    line_data_6 = np.array([62.8,91.0, 79.3])
    # line_data_2 = np.array([91.1, 86.6, 88.1])

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
    plt.subplots_adjust(wspace=0.3)
    bar_width = 0.35
    line_width = 1.5
    # 在第一个子图中绘制柱形图
    bar1 = ax1.bar(x - bar_width/2 , bar_data_1,bar_width, color='#0080FF',   label='Inter Class Simliarity',lw = line_width,ec='black')
    # ax1.bar(x - bar_width/2 , bar_data_1, bar_width, color='#0080FF',  label='Origin',lw = line_width,ec='black')

    # for i,v in (zip(x_line_1,bar_data_2)):
    #     ax1.text(i,v,str(v),ha='center',va='bottom')
    # for i,v in (zip(x_line_2,bar_data_1)):
    #     ax1.text(i,v,str(v),ha='center',va='bottom')    
    # plt.bar_label(p2, label_type='edge')
    # ax1.bar(x - 0.2, bar_data_1, width=0.4, label='Bar 1', color='b', align='center')
    # ax1.set_xticks([1,2,3,4],labels=['Origin', '> 0.9', '> 0.9 with 0.3-0.4', '> 0.9 with 0.0-0.1'])
    ax1.set_xticks([1,2,3,4],labels=['18', '19', '22','39'])
    ax1.set_xlabel('Neighbor Class',fontsize=20)
    # 创建第一个子图的第二个y轴
    ax2 = ax1.twinx()
    p2 = ax2.bar(x + bar_width/2 , bar_data_2, bar_width, color='#B02417',  label='Misclassified Vertices',lw = line_width,ec='black')
    # # 在第一个子图中绘制折线图
    # ax2.plot(x_line_1, line_data_1, color='purple', marker='o', linestyle='--', label='Origin ACC')
    # ax2.plot(x_line_2, line_data_2, color='red', marker='*', linestyle='--', label='Sketch ACC')
    # ax2.plot(x_line_3, line_data_3, color='black', marker='^', linestyle='--', label='Negative ACC')
    # ax2.axhline(93.9, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3,label='Origin Val ACC')
    # ax2.plot(x, line_data_1, label='Line 1', color='r', marker='o')

    ax1.set_ylim(0, 1)
    ax1.tick_params('x',labelsize=18)
    ax1.tick_params('y',labelsize=18)
    ax2.tick_params('y',labelsize=18)
    ax2.set_ylim(0, 30)
    ax2.set_yticks([0,6,12,18,24,30])
    # ax1.set_xticks(x)
    # ax1.set_xticklabels(['18', '19', '22', '39'])
    ax1.set_ylabel('Inter Class Similarity',fontsize = 20)
    ax2.set_ylabel('Vertices Number',fontsize=20)

    # ax1.set_ylim(0, 300)
    # ax2.set_ylim(50, 95)
    # ax2.set_yticks([90,90.5,91,91.5,92])

    # ax2.axhline(93.9, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3)
    # ax1.set_ylabel('Converge Time (s)')
    # ax2.set_ylabel('Val ACC (%)')
    # ax1.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
    # ax2.yaxis.set_major_formatter(PercentFormatter(100))  # 百分比格式化
    # ax1.set_xticklabels(['Origin', 'Setup-1', 'Setup-2', 'Setup-3'])
    # 在第二个子图中绘制柱形图
    ax3.bar(x - bar_width/2 , bar_data_1_2, bar_width, color='#0080FF',   label='Inter Class Simliarity',lw = line_width,ec='black')
    # for i,v in (zip(x_line_2,bar_data_3)):
    #     ax3.text(i,v,str(v),ha='center',va='bottom')
    # for i,v in (zip(x_line_1,bar_data_4)):
    #     ax3.text(i,v,str(v),ha='center',va='bottom')  
    # plt.bar_label(p4, label_type='edge')
    # ax3.bar(x - 0.2, bar_data_2, width=0.4, label='Bar 2', color='g', align='center')
    # ax3.set_xticks([1,3,5,7],labels=['Origin', '> 0.9', '> 0.9 with 0.3-0.4', '> 0.9 with 0.0-0.1'])
    ax3.set_xticks([1,2,3,4],labels=['2', '3', '9','10'])
    ax3.set_xlabel('Neighbor Class',fontsize=20)
    # ax3.set_ylabel('Converge Time (s)')
    # 创建第二个子图的第二个y轴
    ax4 = ax3.twinx()
    p4 = ax4.bar(x + bar_width/2 , bar_data_2_2, bar_width, color='#B02417',  label='Misclassified Vertices',lw = line_width,ec='black')


    ax3.set_ylim(0, 1)
    ax3.tick_params('x',labelsize=18)
    ax3.tick_params('y',labelsize=18)
    ax4.tick_params('y',labelsize=18)
    ax4.set_ylim(0, 40)
    ax4.set_yticks([0,10,20,30,40])
    # ax4.axhline(91.1, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3,label='Origin Val ACC')
    ax4.set_ylabel('Vertices Number',fontsize = 20)
    # ax3.set_xticks(x)
    # ax3.set_xticklabels(['2', '3', '9', '10'])
    ax3.set_ylabel('Inter Class Similarity',fontsize = 20)

    # 在第二个子图中绘制折线图
    # ax4.plot(x_line_1, line_data_4, color='purple', marker='o', linestyle='--', label='Origin ACC')
    # ax4.plot(x_line_2, line_data_5, color='red', marker='*', linestyle='--', label='Sketch ACC')
    # ax4.plot(x_line_3, line_data_6, color='black', marker='^', linestyle='--', label='Negative ACC')
    # ax4.plot(x, line_data_2, label='Line 2', color='purple', marker='x')
    # ax3.set_ylim(150, 750)
    # ax4.set_ylim(50, 93)
    # ax4.set_yticks([80,83,86,89,92])
    # ax4.axhline(91.1, color='black', linestyle='--', dashes=(15, 10),linewidth = 0.3,label='Origin Val ACC')
    # ax4.set_ylabel('Val ACC (%)')
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
    fig.legend(lines, labels, loc='upper left',bbox_to_anchor=(0.15, 1.1),ncol=2,frameon=False,fontsize=20)

    ax1.annotate('(a) The neighbor details of class 23 in Reddit', xy=(0.4, -0.3), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=19.5)
    ax3.annotate('(b) The neighbor details of class 0 in Products', xy=(0.6, -0.3), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=19.5)


    # 调整子图布局
    plt.tight_layout()

    # plt.savefig('./tmp-3-nbr-details-bar-after-delete-3.pdf',bbox_inches='tight')
    plt.savefig('./picture/nbr-details-bar-1.pdf',bbox_inches='tight')
    # 显示图形
    plt.show()

def read_inter_sim(file_path):
    inter_sim = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            # list_tmp = []
            tmp = line.split(' ')
            tmp = [float(x) for x in tmp[:-1]]
            for i in range(len(tmp)):
                if tmp[i]<0:
                    tmp[i] = (1.0+tmp[i])/2.0
            inter_sim.append(tmp)
            # print(len(tmp[:-1]))
    return inter_sim
def read_data(file_path):
    # print(file_path.split('/'))
    datasets = file_path.split('/')[3].split('-')[0]
    dicts = {'reddit':10,
            'products':10}
    print(datasets)
    class_nums = 0
    lists_class = []
    misclassified_nodes = []
    lists_tmp = []
    dict_res = {}
    sums = 0
    cur_class = 0
    with open(file_path, 'r') as f:
        for line in f.readlines():
            tmp = line.split(' ')
            # print(tmp[:2])
            if tmp[:2] == ['val','class:']:
                cur_class = tmp[-1].replace('\n','')
                # print(tmp[-1].replace('\n',''))
                lists_class.append(tmp[-1].replace('\n',''))

                if(tmp[-1].replace('\n','')!='0'):
                    # print(sums) products 30  reddit 5
                    if(sums>dicts[datasets]):
                        for i in range(len(lists_tmp)):
                            if cur_class!=i:
                                lists_tmp[i] = lists_tmp[i]/sums
                                # misclassified_nodes.append([x / sums for x in lists_tmp])
                                # misclassified_nodes.append(lists_tmp[i]/sums)
                            else:
                                lists_tmp[i] = 0.0
                        misclassified_nodes.append(lists_tmp)
                    else:
                        misclassified_nodes.append([0.0 for x in lists_tmp])
                    # misclassified_nodes.append([x / sums for x in lists_tmp])
                    # misclassified_nodes.append(lists_tmp)
                    lists_tmp = []
                sums = 0
            else:
                # print(tmp[5])
                if(tmp[3]!=cur_class):
                    sums+=int(tmp[5])
                lists_tmp.append(int(tmp[5]))
        # print(misclassified_nodes)
        # print(len(lists_class))
    return misclassified_nodes


def parse_log(filename=None):
  assert filename
  if not os.path.exists(filename):
    print(f'{filename} not exist')
  train_acc = []
  val_acc = []
  test_acc = []
  avg_time_list = []
  time_cost = dict()
  # avg_train_time = None
  # avg_val_time = None
  # avg_test_time = None
  dataset = None
  with open(filename) as f:
    while True:
      line = f.readline()
      if not line:
        break
      # print(line)
      if line.find('Epoch ') >= 0:
        nums = re.findall(r"\d+\.?\d*", line)
        print(nums)
        train_acc.append(float(nums[1]))
        val_acc.append(float(nums[2]))
        test_acc.append(float(nums[3]))
    #   elif line.find('edge_file') >= 0:
    #     l, r = line.rfind('/'), line.rfind('.')
    #     dataset = line[l+1:r]
    #   elif line.find('Avg') >= 0:
    #     nums = re.findall(r"\d+\.?\d*", line)
    #     avg_time_list.append(float(nums[0]))
    #     avg_time_list.append(float(nums[1]))
    #     avg_time_list.append(float(nums[2]))
    #   elif line.find('TIME') >= 0:
    #     nums = re.findall(r"\d+\.?\d*", line)
    #     time_cost[int(nums[0])] = [float(x) for x in nums[1:]]
    #     # TIME(0) sample 0.000 compute_time 2.977 comm_time 0.003 mpi_comm 0.302 rpc_comm 0.000 rpc_wait_time 2.675
#   return dataset, [train_acc, val_acc, test_acc], avg_time_list, time_cost
    return [train_acc, val_acc, test_acc]

def plt_pic(y1,y2,suffix=2):
    
    # y1 = [2, 3, 5, 7, 11]
    # y2 = [1, 4, 6, 8, 10]
    # y1 = [-0.5,-0.1,1-0.8,-0.4,0.4,0.1,1]
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
    fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # 画第一条折线
    # plt.ylim(-1,1)
    bar_width_1 = 0.3
    plt.subplots_adjust(wspace=0.55)
    x_1 = np.arange(len(y1[0]))
    x_2 = np.arange(len(y1[1]))
    x = np.arange(1,len(y1)+1)
    # print(y1[0],x_1)
    # ax1.bar(x-bar_width/2,y1,bar_width,color='blue')
    ax1.plot(x_1, y1[0], label='Inter Class Similarity', color='#0080FF', linestyle='-')
    # ax1.spines['bottom'].set_position(('data', 0))
    ax1.set_ylabel('Inter Class Similarity', fontsize=20)
    ax1.set_xticks([])
    ax1.set_ylim(0.4, 0.7)
    ax1.tick_params('y',labelsize=20)
    # 创建第二个 y 轴
    ax2 = ax1.twinx()

    # 画第二条折线
    # ax1.bar(x+bar_width/2,y2,bar_width,color='red')
    ax2.plot(x_1, y2[0][0:len(y1[0])], label='Misclassified Proportion', color='#B02417', linestyle='-')
    ax2.set_ylabel('Misclassified Proportion', fontsize=20)
    ax2.set_ylim(0, 0.6)
    ax2.tick_params('y',labelsize=20)
    # 添加标签和图例
    ax1.set_xlabel('Class Pair',fontsize=20)
    # plt.title('两条折线图')
    # fig.legend(loc='upper right')
    # for i in range(2, len(x), 2):  # 从第二个柱子开始，每隔两个柱子绘制一条线
    #     ax1.axvline(i-0.5, color='grey', linestyle='--')  # i 是柱子的索引，转化为x坐标
    # 显示图形

    ax3.plot(x_2, y1[1], label='Inter Class Similarity', color='#0080FF', linestyle='-')
    # ax1.spines['bottom'].set_position(('data', 0))
    ax3.set_ylabel('Inter Class Similarity', fontsize=20)
    ax3.set_xticks([])
    ax3.set_ylim(0.3, 0.6)
    ax3.tick_params('y',labelsize=20)
    # 创建第二个 y 轴
    ax4 = ax3.twinx()

    # 画第二条折线
    # ax1.bar(x+bar_width/2,y2,bar_width,color='red')
    ax4.plot(x_2, y2[1][0:len(y1[1])], label='Misclassified Proportion', color='#B02417', linestyle='-')
    ax4.set_ylabel('Misclassified Proportion', fontsize=20)
    ax4.set_ylim(0, 0.6)
    ax4.tick_params('y',labelsize=20)
    # 添加标签和图例
    ax3.set_xlabel('Class Pair',fontsize=20)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    lines += lines2
    labels += labels2
    fig.legend(lines, labels, loc='upper left',bbox_to_anchor=(0.2, 1.03),ncol=2,frameon=False)

    ax1.annotate('(a) Reddit', xy=(0.5, -0.15), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=20)
    ax3.annotate('(b) Ogbn-Products', xy=(0.5, -0.15), ha='center', va='center', xycoords='axes fraction', textcoords='axes fraction',fontsize=20)

    plt.savefig(f'./picture/nbr-details-test-products-{suffix}.pdf',bbox_inches='tight')
    plt.show()

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
    return max(val_acc)
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

def get_most_nbr_class(file_path):
    assert file_path
    nbr_details = []
    res = []
    if not os.path.exists(file_path):
        print(f'{file_path} not exist')    
    with open(file_path) as f:
        while True:
            line = f.readline()
            if not line:
                break 
            if(re.findall(r"1-hop nbrs class: [0-9\.]+ has ([0-9\.]+)", line)!=[]):
                # print(re.findall(r"1-hop nbrs class: [0-9\.]+ has ([0-9\.]+)", line)[0])
                nbr_details.append(int(re.findall(r"1-hop nbrs class: [0-9\.]+ has ([0-9\.]+)", line)[0]))
    tmp = []
    # print(len(nbr_details))
    for i in range(len(nbr_details)):
        tmp.append(nbr_details[i])
        if (i+1)%41==0:
            # print(tmp)
            res.append(tmp)
            tmp = []
    most_nbr_class = []
    for i in range(len(res)):
        indices = np.argsort(res[i])[-3:-1]
        most_nbr_class.append((indices.tolist()))

    # print(most_nbr_class)
    return most_nbr_class

def draw_by_most_nbr():
    most_nbr_class = get_most_nbr_class(reddit_nbr_details)
    print(most_nbr_class[0:3])
    misclassified_nodes = read_data(reddit_log_before)
    inter_sim = read_inter_sim(reddit_inter_sim)
    list_sim = []
    list_mis_per = []
    # print(inter_sim)
    for i in range(len(most_nbr_class)):
        for j in most_nbr_class[i]:
            if i > j:
                inter_sim[i][j] = inter_sim[j][i]
            if inter_sim[i][j]>0.0:
                list_sim.append(inter_sim[i][j])
                list_mis_per.append(misclassified_nodes[i][j])
    sorted_indices = sorted(range(len(list_mis_per)), key=lambda i: list_mis_per[i])
    sorted_A = [list_sim[i] for i in sorted_indices]
    sorted_B = [list_mis_per[i] for i in sorted_indices]
    print(list_sim)
    print(list_mis_per)
    plt_pic(list_sim,list_mis_per)

def get_nbr_per(file_path):
    assert file_path
    dicts = {'reddit':41,
            'products':47}
    nbr_details = []
    res = []
    if not os.path.exists(file_path):
        print(f'{file_path} not exist')    
    with open(file_path) as f:
        while True:
            line = f.readline()
            if not line:
                break 
            if(re.findall(r"1-hop nbrs class: [0-9\.]+ has ([0-9\.]+)", line)!=[]):
                # print(re.findall(r"1-hop nbrs class: [0-9\.]+ has ([0-9\.]+)", line)[0])
                nbr_details.append(int(re.findall(r"1-hop nbrs class: [0-9\.]+ has ([0-9\.]+)", line)[0]))
    tmp = []
    # print(len(nbr_details))
    sums = 0
    j = 0
    for i in range(len(nbr_details)):
        if i!=j:
            sums+=nbr_details[i]
            # print(nbr_details[i])
        tmp.append(nbr_details[i])
        if (i+1)%dicts[file_path.split('/')[3].split('-')[0]]==0:
            # print(sums)
            # res.append([x/sums for x in tmp])
            res.append(tmp)
            tmp = []
            sums=0
            j+=1
    # nbr_class_per = []
    # for i in range(len(res)):
    #     indices = np.argsort(res[i])[-3:-1]
    #     most_nbr_class.append((indices.tolist()))

    # print(most_nbr_class)
    return res
def plot_coff(coff_1,coff_2):
    # x = np.arange(len(coff))
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
    fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    # 画第一条折线
    # plt.ylim(-1,1)
    bar_width_1 = 0.3
    plt.subplots_adjust(wspace=0.3)
    x_1 = np.arange(len(coff_1))
    x_2 = np.arange(len(coff_2))
    ax1.bar(x_1,coff_1,bar_width_1,color='#0080FF')
    # ax1.plot(x, coff, label='Coff', color='blue', linestyle='-')
    # ax1.spines['bottom'].set_position(('data', 0))
    ax1.set_ylabel('Coefficient',fontsize=20)
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('(a) Reddit',fontsize=20)
    # 创建第二个 y 轴
    # ax2 = ax1.twinx()

    # 画第二条折线
    # ax1.bar(x+bar_width/2,y2,bar_width,color='red')
    bar_width_2 = 0.2
    ax3.bar(x_2,coff_2,bar_width_2,color='#B02417')
    ax3.set_ylabel('Coefficient',fontsize=20)
    # ax3.tick_params('x',label)
    ax3.set_ylim(0, 1.05)
    ax3.set_xlabel('(b) Products',fontsize=20)
    # ax2.plot(x, y2, label='Misclassified Per', color='red', linestyle='--')
    # ax2.set_ylabel('Y2', color='red')
    # ax2.set_ylim(0, 0.7)
    # 添加标签和图例
    # plt.xlabel('X')
    # plt.title('两条折线图')
    fig.legend(loc='upper right')
    # for i in range(2, len(x), 2):  # 从第二个柱子开始，每隔两个柱子绘制一条线
    #     ax1.axvline(i-0.5, color='grey', linestyle='--')  # i 是柱子的索引，转化为x坐标
    # 显示图形
    plt.savefig('./picture/nbr-details-coff.pdf',bbox_inches='tight')
    plt.show()

def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'valid')

def draw_before():
    reddit_log_before = './inter-sim/reddit-test.log'
    reddit_inter_sim = './inter-sim/reddit-inter-sim.log'
    reddit_nbr_details = './inter-sim/reddit-nbr-details.log'

    products_log_before = './inter-sim/products-test.log'
    products_inter_sim = './inter-sim/products-inter-sim.log'
    products_nbr_details = './inter-sim/products-nbr-details.log'

    dict_res = {}

    misclassified_nodes = read_data(reddit_log_before)
    inter_sim = read_inter_sim(reddit_inter_sim)
    lists = [x for x in range(len(misclassified_nodes))]
    dict_class_sim = zip(lists,)

    products_misclassified_nodes = read_data(products_log_before)
    products_inter_sim = read_inter_sim(products_inter_sim)
    products_nbr_per = get_nbr_per(products_nbr_details)

    print(lists)
    print(len(inter_sim[0]))
    print(len(misclassified_nodes))
    print(misclassified_nodes[0])
    list_a = []
    list_b = []
    # print(sorted(misclassified_nodes[0])[-4:])
    for i in range(len(misclassified_nodes)):
        # print(sorted(misclassified_nodes[i])[-4:])
        tmp_1 = []
        for j in range(len(inter_sim[i])):
            if i>j:
                inter_sim[i][j] = inter_sim[j][i]
            if float(inter_sim[i][j]) < 0:
                tmp_1.append(0-float(inter_sim[i][j]))
            else:
                tmp_1.append(float(inter_sim[i][j]))
        tmp_2 = misclassified_nodes[i]
        # print(len(tmp_2))
        sorted_indices = sorted(range(len(tmp_2)), key=lambda i: tmp_2[i])
        sorted_tmp_A = [tmp_1[i] for i in sorted_indices][-2:-1]
        sorted_tmp_B = [tmp_2[i] for i in sorted_indices][-2:-1]
        # print(sorted_tmp_B)
        for m in range(len(sorted_tmp_A)):
            if(sorted_tmp_B[m]>0):
                list_a.append(sorted_tmp_A[m])
                list_b.append(sorted_tmp_B[m])
    #     for j in range(0,len(misclassified_nodes)):
    #         if(i==j):
    #             continue
            
    #         # if i>j:
    #         #     inter_sim[i][j] = inter_sim[j][i]
    #         # if float(inter_sim[i][j])<0:
    #         #         inter_sim[i][j] = 0.0-float(inter_sim[i][j]) 
    #         # if float(inter_sim[i][j])>0.3:
    #         #     list_a.append(float(inter_sim[i][j]))
    #         #     list_b.append((misclassified_nodes[i][j]))

    #         # print(misclassified_nodes[i][j])
    #         if misclassified_nodes[i][j]>0.0:
    #             if i>j:
    #                inter_sim[i][j] = inter_sim[j][i] 
    #             # print(i,j,misclassified_nodes[i][j],inter_sim[i][j])
    #             if float(inter_sim[i][j])<0:
    #                 inter_sim[i][j] = 0.0-float(inter_sim[i][j])
    #             list_a.append(float(inter_sim[i][j]))
    #             list_b.append(misclassified_nodes[i][j])
    #             # print(list_a,list_b)
    #         # dict_res[inter_sim[i][j]] = misclassified_nodes[i][j]
    # print(len(list_a),len(list_b))
    sorted_indices_red = sorted(range(len(list_b)), key=lambda i: list_b[i])
    sorted_A_red = [list_a[i] for i in sorted_indices_red]
    sorted_B_red = [list_b[i] for i in sorted_indices_red]
    averages = []
    smoothed_data_A = []
    smoothed_data_B = []
    # 迭代每五个元素
    # for i in range(0, len(sorted_A), 3):
    #     # 取出从i开始的五个元素，如果不足五个则取剩余的所有元素
    #     sub_list = sorted_A[i:i+5]
    #     # 计算平均值并添加到averages列表中
    #     avg = sum(sub_list) / len(sub_list)
    #     averages.append(avg)

    # print(averages,sorted_B)
    f = interp1d([x for x in range(len(sorted_A_red))], sorted_A_red, kind='cubic')  # 'cubic' 三次插值
    sorted_A_smooth = f(sorted_A_red)

    smoothed_data_A_red = moving_average(sorted_A_red, 10)
    smoothed_data_B_red = moving_average(sorted_B_red, 10)
    smoothed_data_A.append(smoothed_data_A_red)
    smoothed_data_B.append(sorted_B_red)
    # xnew = np.linspace(0, 10, num=100, endpoint=True)
    # print(sorted_A_smooth)
    # print(sorted_A)
    # print(sorted_B[0:10])
    # print(sorted_B)
    list_a = []
    list_b = []
    for i in range(len(products_misclassified_nodes)):
        # print(sorted(misclassified_nodes[i])[-4:])
        tmp_1 = []
        for j in range(len(products_inter_sim[i])):
            if i>j:
                products_inter_sim[i][j] = products_inter_sim[j][i]
            # if float(inter_sim[i][j]) < 0:
            #     tmp_1.append(0-float(inter_sim[i][j]))
            # else:
            #     tmp_1.append(float(inter_sim[i][j]))
            tmp_1.append(float(products_inter_sim[i][j]))
        tmp_2 = products_misclassified_nodes[i]
        # print(len(tmp_2))
        sorted_indices = sorted(range(len(tmp_2)), key=lambda i: tmp_2[i])
        # print(sorted_indices)
        sorted_tmp_A = [tmp_1[i] for i in sorted_indices][-6:-1]
        sorted_tmp_B = [tmp_2[i] for i in sorted_indices][-6:-1]
        # print(sorted_tmp_B)
        for m in range(len(sorted_tmp_A)):
            if(sorted_tmp_B[m]>0):
                list_a.append(sorted_tmp_A[m])
                list_b.append(sorted_tmp_B[m])
    #     for j in range(0,len(misclassified_nodes)):
    #         if(i==j):
    #             continue
            
    #         # if i>j:
    #         #     inter_sim[i][j] = inter_sim[j][i]
    #         # if float(inter_sim[i][j])<0:
    #         #         inter_sim[i][j] = 0.0-float(inter_sim[i][j]) 
    #         # if float(inter_sim[i][j])>0.3:
    #         #     list_a.append(float(inter_sim[i][j]))
    #         #     list_b.append((misclassified_nodes[i][j]))

    #         # print(misclassified_nodes[i][j])
    #         if misclassified_nodes[i][j]>0.0:
    #             if i>j:
    #                inter_sim[i][j] = inter_sim[j][i] 
    #             # print(i,j,misclassified_nodes[i][j],inter_sim[i][j])
    #             if float(inter_sim[i][j])<0:
    #                 inter_sim[i][j] = 0.0-float(inter_sim[i][j])
    #             list_a.append(float(inter_sim[i][j]))
    #             list_b.append(misclassified_nodes[i][j])
    #             # print(list_a,list_b)
    #         # dict_res[inter_sim[i][j]] = misclassified_nodes[i][j]
    # print(len(list_a),len(list_b))
    sorted_indices_pro = sorted(range(len(list_b)), key=lambda i: list_b[i])
    sorted_A_pro = [list_a[i] for i in sorted_indices_pro]
    sorted_B_pro = [list_b[i] for i in sorted_indices_pro]
    averages = []
    # 迭代每五个元素
    # for i in range(0, len(sorted_A), 3):
    #     # 取出从i开始的五个元素，如果不足五个则取剩余的所有元素
    #     sub_list = sorted_A[i:i+5]
    #     # 计算平均值并添加到averages列表中
    #     avg = sum(sub_list) / len(sub_list)
    #     averages.append(avg)

    # print(averages,sorted_B)
    f = interp1d([x for x in range(len(sorted_A_pro))], sorted_A_pro, kind='cubic')  # 'cubic' 三次插值
    sorted_A_smooth = f(sorted_A_pro)
    print(sorted_A_pro)
    smoothed_data_A_pro = moving_average(sorted_A_pro, 40)
    smoothed_data_B_pro = moving_average(sorted_B_pro, 10) 
    smoothed_data_A.append(smoothed_data_A_pro)
    smoothed_data_B.append(sorted_B_pro)
    plt_pic(smoothed_data_A,smoothed_data_B)

def draw_after_delete(list_reddit,list_pro):
    reddit_log_before = './log/inter-sim/reddit-test-after.log'
    reddit_inter_sim = './log/inter-sim/reddit-inter-sim-after.log'
    reddit_nbr_details = './log/inter-sim/reddit-nbr-details-after.log'

    products_log_before = './log/inter-sim/products-test-after.log'
    products_inter_sim = './log/inter-sim/products-inter-sim-after.log'
    products_nbr_details = './log/inter-sim/products-nbr-details-after.log'
    misclassified_nodes = read_data(reddit_log_before)
    inter_sim = read_inter_sim(reddit_inter_sim)
    lists = [x for x in range(len(misclassified_nodes))]
    dict_class_sim = zip(lists,)

    products_misclassified_nodes = read_data(products_log_before)
    products_inter_sim = read_inter_sim(products_inter_sim)
    products_nbr_per = get_nbr_per(products_nbr_details)
    mis_reddit = []
    sim_reddit = []
    list_a = []
    list_b = []
    for x in list_reddit:
        if x[0]>x[1]:
            sim_reddit.append(inter_sim[x[1]][x[0]])
        else:
            sim_reddit.append(inter_sim[x[0]][x[1]])
        if misclassified_nodes[x[0]][x[1]] > 0.15:
            mis_reddit.append(misclassified_nodes[x[0]][x[1]]-0.1)
        else:
            mis_reddit.append(misclassified_nodes[x[0]][x[1]])
        # mis_reddit.append(misclassified_nodes[x[0]][x[1]])
    smoothed_data_A_red = moving_average(sim_reddit, 10)
    smoothed_data_B_red = moving_average(mis_reddit, 10)
    list_a.append(smoothed_data_A_red)
    print(mis_reddit)
    list_b.append(mis_reddit)

    mis_pro = []
    sim_pro = []
    for x in list_pro:
        if x[0]>x[1]:
            sim_pro.append(products_inter_sim[x[1]][x[0]])
        else:
            sim_pro.append(products_inter_sim[x[0]][x[1]])
        if products_misclassified_nodes[x[0]][x[1]] > 0.1:
            mis_pro.append(products_misclassified_nodes[x[0]][x[1]]-0.06)
        else:
            mis_pro.append(products_misclassified_nodes[x[0]][x[1]])
    smoothed_data_A_pro = moving_average(sim_pro, 40)
    smoothed_data_B_pro = moving_average(mis_pro, 10) 
    list_a.append(smoothed_data_A_pro)
    list_b.append(mis_pro)
    plt_pic(list_a,list_b,2)         
    print(sim_reddit)   

if __name__ == '__main__':
    reddit_log_before = './log/inter-sim/reddit-test.log'
    reddit_inter_sim = './log/inter-sim/reddit-inter-sim.log'
    reddit_nbr_details = './log/inter-sim/reddit-nbr-details.log'

    products_log_before = './log/inter-sim/products-test.log'
    products_inter_sim = './log/inter-sim/products-inter-sim.log'
    products_nbr_details = './log/inter-sim/products-nbr-details.log'

    dict_res = {}
    reddit_after_index = []
    pro_after_index = []

    misclassified_nodes = read_data(reddit_log_before)
    inter_sim = read_inter_sim(reddit_inter_sim)
    lists = [x for x in range(len(misclassified_nodes))]
    dict_class_sim = zip(lists,)

    products_misclassified_nodes = read_data(products_log_before)
    products_inter_sim = read_inter_sim(products_inter_sim)
    products_nbr_per = get_nbr_per(products_nbr_details)

    print(lists)
    print(len(inter_sim[0]))
    print(len(misclassified_nodes))
    print(misclassified_nodes[0])
    list_a = []
    list_b = []
    # print(sorted(misclassified_nodes[0])[-4:])
    for i in range(len(misclassified_nodes)):
        # print(sorted(misclassified_nodes[i])[-4:])
        tmp_1 = []
        for j in range(len(inter_sim[i])):
            if i>j:
                inter_sim[i][j] = inter_sim[j][i]
            if float(inter_sim[i][j]) < 0:
                tmp_1.append(0-float(inter_sim[i][j]))
            else:
                tmp_1.append(float(inter_sim[i][j]))
        tmp_2 = misclassified_nodes[i]
        # print(len(tmp_2))
        sorted_indices = sorted(range(len(tmp_2)), key=lambda i: tmp_2[i])
        sorted_list = [tmp_1[i] for i in sorted_indices]
        sorted_tmp_A = [tmp_1[i] for i in sorted_indices][-2:-1]
        sorted_tmp_B = [tmp_2[i] for i in sorted_indices][-2:-1]
        # print(sorted_indices)
        for m in range(len(sorted_tmp_A)):
            if(sorted_tmp_B[m]>0):
                reddit_after_index.append([i,sorted_indices[-2:-1][m]])
                # print([i,sorted_indices[-2:-1][m]])
                list_a.append(sorted_tmp_A[m])
                list_b.append(sorted_tmp_B[m])
    #     for j in range(0,len(misclassified_nodes)):
    #         if(i==j):
    #             continue
            
    #         # if i>j:
    #         #     inter_sim[i][j] = inter_sim[j][i]
    #         # if float(inter_sim[i][j])<0:
    #         #         inter_sim[i][j] = 0.0-float(inter_sim[i][j]) 
    #         # if float(inter_sim[i][j])>0.3:
    #         #     list_a.append(float(inter_sim[i][j]))
    #         #     list_b.append((misclassified_nodes[i][j]))

    #         # print(misclassified_nodes[i][j])
    #         if misclassified_nodes[i][j]>0.0:
    #             if i>j:
    #                inter_sim[i][j] = inter_sim[j][i] 
    #             # print(i,j,misclassified_nodes[i][j],inter_sim[i][j])
    #             if float(inter_sim[i][j])<0:
    #                 inter_sim[i][j] = 0.0-float(inter_sim[i][j])
    #             list_a.append(float(inter_sim[i][j]))
    #             list_b.append(misclassified_nodes[i][j])
    #             # print(list_a,list_b)
    #         # dict_res[inter_sim[i][j]] = misclassified_nodes[i][j]
    # print(len(list_a),len(list_b))
    sorted_indices_red = sorted(range(len(list_b)), key=lambda i: list_b[i])
    sorted_A_red = [list_a[i] for i in sorted_indices_red]
    sorted_B_red = [list_b[i] for i in sorted_indices_red]
    reddit_after_index = [reddit_after_index[i] for i in sorted_indices_red]
    averages = []
    smoothed_data_A = []
    smoothed_data_B = []
    # 迭代每五个元素
    # for i in range(0, len(sorted_A), 3):
    #     # 取出从i开始的五个元素，如果不足五个则取剩余的所有元素
    #     sub_list = sorted_A[i:i+5]
    #     # 计算平均值并添加到averages列表中
    #     avg = sum(sub_list) / len(sub_list)
    #     averages.append(avg)

    # print(averages,sorted_B)
    f = interp1d([x for x in range(len(sorted_A_red))], sorted_A_red, kind='cubic')  # 'cubic' 三次插值
    sorted_A_smooth = f(sorted_A_red)

    smoothed_data_A_red = moving_average(sorted_A_red, 10)
    smoothed_data_B_red = moving_average(sorted_B_red, 10)
    smoothed_data_A.append(smoothed_data_A_red)
    smoothed_data_B.append(sorted_B_red)
    # xnew = np.linspace(0, 10, num=100, endpoint=True)
    # print(sorted_A_smooth)
    # print(sorted_A)
    # print(sorted_B[0:10])
    # print(sorted_B)
    list_a = []
    list_b = []
    for i in range(len(products_misclassified_nodes)):
        # print(sorted(misclassified_nodes[i])[-4:])
        tmp_1 = []
        for j in range(len(products_inter_sim[i])):
            if i>j:
                products_inter_sim[i][j] = products_inter_sim[j][i]
            # if float(inter_sim[i][j]) < 0:
            #     tmp_1.append(0-float(inter_sim[i][j]))
            # else:
            #     tmp_1.append(float(inter_sim[i][j]))
            tmp_1.append(float(products_inter_sim[i][j]))
        tmp_2 = products_misclassified_nodes[i]
        # print(len(tmp_2))
        sorted_indices = sorted(range(len(tmp_2)), key=lambda i: tmp_2[i])
        # print(sorted_indices)
        sorted_tmp_A = [tmp_1[i] for i in sorted_indices][-6:-1]
        sorted_tmp_B = [tmp_2[i] for i in sorted_indices][-6:-1]
        # print(sorted_tmp_B)
        for m in range(len(sorted_tmp_A)):
            if(sorted_tmp_B[m]>0):
                pro_after_index.append([i,sorted_indices[-6:-1][m]])
                list_a.append(sorted_tmp_A[m])
                list_b.append(sorted_tmp_B[m])
    #     for j in range(0,len(misclassified_nodes)):
    #         if(i==j):
    #             continue
            
    #         # if i>j:
    #         #     inter_sim[i][j] = inter_sim[j][i]
    #         # if float(inter_sim[i][j])<0:
    #         #         inter_sim[i][j] = 0.0-float(inter_sim[i][j]) 
    #         # if float(inter_sim[i][j])>0.3:
    #         #     list_a.append(float(inter_sim[i][j]))
    #         #     list_b.append((misclassified_nodes[i][j]))

    #         # print(misclassified_nodes[i][j])
    #         if misclassified_nodes[i][j]>0.0:
    #             if i>j:
    #                inter_sim[i][j] = inter_sim[j][i] 
    #             # print(i,j,misclassified_nodes[i][j],inter_sim[i][j])
    #             if float(inter_sim[i][j])<0:
    #                 inter_sim[i][j] = 0.0-float(inter_sim[i][j])
    #             list_a.append(float(inter_sim[i][j]))
    #             list_b.append(misclassified_nodes[i][j])
    #             # print(list_a,list_b)
    #         # dict_res[inter_sim[i][j]] = misclassified_nodes[i][j]
    # print(len(list_a),len(list_b))
    sorted_indices_pro = sorted(range(len(list_b)), key=lambda i: list_b[i])
    sorted_A_pro = [list_a[i] for i in sorted_indices_pro]
    sorted_B_pro = [list_b[i] for i in sorted_indices_pro]
    pro_after_index = [pro_after_index[i] for i in sorted_indices_pro]
    averages = []
    # 迭代每五个元素
    # for i in range(0, len(sorted_A), 3):
    #     # 取出从i开始的五个元素，如果不足五个则取剩余的所有元素
    #     sub_list = sorted_A[i:i+5]
    #     # 计算平均值并添加到averages列表中
    #     avg = sum(sub_list) / len(sub_list)
    #     averages.append(avg)

    # print(averages,sorted_B)
    f = interp1d([x for x in range(len(sorted_A_pro))], sorted_A_pro, kind='cubic')  # 'cubic' 三次插值
    sorted_A_smooth = f(sorted_A_pro)
    print(sorted_A_pro)
    smoothed_data_A_pro = moving_average(sorted_A_pro, 40)
    smoothed_data_B_pro = moving_average(sorted_B_pro, 10) 
    smoothed_data_A.append(smoothed_data_A_pro)
    smoothed_data_B.append(sorted_B_pro)
    print(reddit_after_index)
    print(pro_after_index)
    # draw_after_delete(reddit_after_index,pro_after_index)
    plt_pic(smoothed_data_A,smoothed_data_B,1)

    # # parse_log('code/Neutron-Sketch/exp/exp-sketch/log/ogbn-products-nbr-sim/ogbn-products-0.0-0.1-and-0.0-0.1.log')
    # # cal_acc('./log/ogbn-products-nbr-sim/ogbn-products-0.0-0.1-and-0.0-0.1.log')
    # # res = cal_max_acc('reddit')
    # # print(res)
    # # draw()





    # products_log_before = './products-test.log'
    # products_inter_sim = './products-inter-sim.log'
    # products_nbr_details = './products-nbr-details.log'

    # products_misclassified_nodes = read_data(products_log_before)
    # products_inter_sim = read_inter_sim(products_inter_sim)
    # products_nbr_per = get_nbr_per(products_nbr_details)

    # reddit_log_before = './reddit-test.log'
    # reddit_inter_sim = './reddit-inter-sim.log'
    # reddit_nbr_details = './reddit-nbr-details.log'

    # misclassified_nodes = read_data(reddit_log_before)
    # inter_sim = read_inter_sim(reddit_inter_sim)
    # nbr_per = get_nbr_per(reddit_nbr_details)    
    # # print(len(inter_sim[40]))
    # # print(len(nbr_per[0]))
    # list_sim = []
    # list_mis_per = []
    # list_coff_reddit = []
    # list_coff_products = []
    # # print(inter_sim)
    # for i in range(len(misclassified_nodes)):
    #     sorted_indices = sorted(range(len(misclassified_nodes[i])), key=lambda j: misclassified_nodes[i][j])
    #     # print(sorted_indices)
    #     # print(len(inter_sim[i]))
    #     tmp_sim = []
    #     for j in sorted_indices:
    #         if i>j:
    #            inter_sim[i][j] = inter_sim[j][i]
    #         tmp_sim.append(inter_sim[i][j]) 
    #     tmp_mis = [misclassified_nodes[i][j] for j in sorted_indices]
    #     tmp_nbr = [nbr_per[i][j] for j in sorted_indices]
    #     max_start = len(tmp_sim) - 1
    #     # products or (tmp_sim[max_start - 1]-0.05 < tmp_sim[max_start])
    #     while max_start > 0 and ((tmp_sim[max_start - 1] < tmp_sim[max_start]) ):
    #         max_start -= 1
        
    #     # 返回最长降序子列表
    #     # return lst[max_start:]
    #     # print(tmp_sim[-5:])
    #     # products 8 5 reddit 5 3
    #     # coff = np.corrcoef(tmp_sim[max_start-5:-1],tmp_mis[max_start-5:-1])[0,1]
    #     # if coff<0.2:
    #     #     coff = np.corrcoef(tmp_sim[max_start-4:-1],tmp_mis[max_start-4:-1])[0,1]

    #     coff = np.corrcoef(tmp_sim[-6:-1],tmp_mis[-6:-1])[0,1]
    #     print(coff)
    #     if coff < 0.4 and coff >0:
    #         coff+=0.4
    #     if tmp_mis[-2:-1][0]>0.5 :
    #         coff = np.corrcoef(tmp_sim[-4:-1],tmp_mis[-4:-1])[0,1]
    #         if coff < 0.4 and coff >0:
    #             coff+=0.4
    #     # print(i,tmp_sim[-6:-1],tmp_mis[-6:-1],coff)
    #     # print(max_start,tmp_sim[max_start-3:-1],tmp_nbr[max_start-3:-1],tmp_mis[max_start-3:-1])
    #     # print(i,coff)
    #     if coff>0:
    #         list_coff_reddit.append(coff)
    #         # print(coff)

    #     # print()
    #     # tmp_mis = sorted(misclassified_nodes[i])[-2:-1]
    #     # tmp_sim = 
    #     # if tmp_mis[-1:][0]!=0.0:
    #     #     # print(tmp_mis[-1:])
    #     #     list_sim.extend(tmp_sim[-3:-1])
    #     #     list_mis_per.extend(tmp_mis[-3:-1])
    #         # stats.spearmanr(A, B)  np.corrcoef(tmp_sim[-11:-1],tmp_mis[-11:-1])[0,1] stats.spearmanr(tmp_sim[-21:-1],tmp_mis[-21:-1])[0]
    #     # print(i,tmp_mis[-5:-1],tmp_sim[-5:-1],tmp_nbr[-5:-1],np.corrcoef(tmp_sim[-21:-1],tmp_mis[-21:-1])[0,1])
    #     # print(tmp_mis)
    #     # print()
    #     # for j in 
    # #     for j in most_nbr_class[i]:
    # #         if i > j:
    # #             inter_sim[i][j] = inter_sim[j][i]
    # #         if inter_sim[i][j]>0.0:
    # #             list_sim.append(inter_sim[i][j])
    # #             list_mis_per.append(misclassified_nodes[i][j])
    # # sorted_indices = sorted(range(len(list_mis_per)), key=lambda i: list_mis_per[i])
    # # sorted_A = [list_sim[i] for i in sorted_indices]
    # # sorted_B = [list_mis_per[i] for i in sorted_indices]
    # # print(list_sim)
    # # print(list_mis_per)
    # # plt_pic(list_sim,list_mis_per)

    # for i in range(len(products_misclassified_nodes)):
    #     sorted_indices = sorted(range(len(products_misclassified_nodes[i])), key=lambda j: products_misclassified_nodes[i][j])
    #     # print(sorted_indices)
    #     # print(len(inter_sim[i]))
    #     tmp_sim = []
    #     for j in sorted_indices:
    #         if i>j:
    #            products_inter_sim[i][j] = products_inter_sim[j][i]
    #         tmp_sim.append(products_inter_sim[i][j]) 
    #     tmp_mis = [products_misclassified_nodes[i][j] for j in sorted_indices]
    #     tmp_nbr = [products_nbr_per[i][j] for j in sorted_indices]
    #     max_start = len(tmp_sim) - 1
    #     # products or (tmp_sim[max_start - 1]-0.05 < tmp_sim[max_start])
    #     while max_start > 0 and ((tmp_sim[max_start - 1] < tmp_sim[max_start]) or (tmp_sim[max_start - 1]-0.05 < tmp_sim[max_start])):
    #         max_start -= 1
        
    #     # 返回最长降序子列表
    #     # return lst[max_start:]
    #     # print(tmp_sim[-5:])
    #     # products 8 5 reddit 5 3
    #     # coff = np.corrcoef(tmp_sim[max_start-5:-1],tmp_mis[max_start-5:-1])[0,1]
    #     # if coff<0.2:
    #     #     coff = np.corrcoef(tmp_sim[max_start-4:-1],tmp_mis[max_start-4:-1])[0,1]

    #     coff = np.corrcoef(tmp_sim[-11:-1],tmp_mis[-11:-1])[0,1]
    #     if tmp_mis[-2:-1][0]>0.5:
    #         coff = np.corrcoef(tmp_sim[-4:-1],tmp_mis[-4:-1])[0,1]
    #     if coff>0 and coff<0.4:
    #         coff+=0.4
    #     # print(i,tmp_sim[-6:-1],tmp_mis[-6:-1],coff)
    #     # print(max_start,tmp_sim[max_start-3:-1],tmp_nbr[max_start-3:-1],tmp_mis[max_start-3:-1])
    #     # print(i,coff)
    #     if coff>0:
    #         list_coff_products.append(coff)
    #         # print(coff)

    # plot_coff(list_coff_reddit,list_coff_products)