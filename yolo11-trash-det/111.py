import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 数据准备
days = [1, 3, 7]
observation_means = [2.25,1.58,0.62]
observation_stds = [0.43,0.12,0.18]

control_means = [2.31,1.83,1.36]
control_stds = [0.37,0.15,0.28]

# 创建图表
plt.figure(figsize=(10, 6))

# 绘制观察组折线及误差线
plt.errorbar(days, observation_means, yerr=observation_stds,
             fmt='o-', color='#1f77b4', linewidth=2, capsize=5,
             label='实验组', markersize=8)

# 绘制对照组折线及误差线
plt.errorbar(days, control_means, yerr=control_stds,
             fmt='s--', color='#ff7f0e', linewidth=2, capsize=5,
             label='对照组', markersize=8)

# 添加数据标签
for i, (mean, std) in enumerate(zip(observation_means, observation_stds)):
    plt.text(days[i], mean + 0.1, f"{mean}±{std}",
             ha='center', va='bottom', fontsize=9, color='#1f77b4')

for i, (mean, std) in enumerate(zip(control_means, control_stds)):
    plt.text(days[i], mean - 0.15, f"{mean}±{std}",
             ha='center', va='top', fontsize=9, color='#ff7f0e')

# 设置图表标题和坐标轴
plt.title('两组患者术后1、3、7 d 渗液评分均值变化趋势', fontsize=15, pad=15)
plt.xlabel('术后天数（天）', fontsize=12)
plt.ylabel('疼痛评分（分）', fontsize=12)
plt.xticks(days)

# 设置网格线和坐标范围
plt.grid(True, linestyle='--', alpha=0.6)
plt.ylim(0, 3)

# 添加图例
plt.legend(loc='upper right')

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('术后疼痛评分变化趋势.png', dpi=300)

# 显示图表
plt.show()