import matplotlib.pyplot as plt
import numpy as np

# 数据准备
# methods = [
#     "ResNet+Concat", "ViT+Concat", "EfficientNet+Concat",
#     "Poolformer+Concat", "nnMamba+Concat", "Diamond",
#     "MDL", "VAPL", "HyperFusionNet", "IMF", "HFBsurv",
#     "ITCFN", "AwesomeNet (Ours)"
# ]
methods = [
    "ResNet+Concat", "ViT+Concat",
     "nnMamba+Concat", "Diamond",
    "MDL", "VAPL", "HyperFusionNet", "IMF", "HFBsurv",
    "ITCFN", "MultimodalADNet", "TriLightNet (Ours)"
]
accuracies = [73.75, 71.75, 73.75, 77.76,
              71.50, 69.25, 75.50, 77.75, 75.00, 75.50, 73.25, 81.25]
errors = [3.79, 3.12, 5.04, 1.72,
          7.39, 4.00, 2.69, 3.98, 3.26, 3.76, 4.23, 0.93]
BalanceAccuracies = [73.26, 59.12, 66.71, 69.19, 66.44, 61.46, 69.20,
                     75.39, 74.10, 74.87, 70.53, 79.06]
BalanceAccuracyErrors = [6.15, 6.03, 2.17, 0.72, 1.98, 5.26, 1.50, 1.41, 1.37, 3.23, 2.62, 2.85]


# 设置颜色和样式
colors = plt.cm.tab20.colors[:len(methods)]  # 使用Tab20调色板
x_positions = np.arange(len(methods))  # X轴位置

# 创建画布
plt.figure(figsize=(12, 8))

# 绘制柱状图和误差条
# bars = plt.bar(x_positions, accuracies, yerr=errors, capsize=5,
#                color=colors, edgecolor='black', alpha=0.8)
bars = plt.bar(x_positions, BalanceAccuracies, yerr=BalanceAccuracyErrors, capsize=5,
               color=colors, edgecolor='black', alpha=0.8)

# 添加数据标签
# for bar in bars:
#     height = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2, height + 1,
#              f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
# for bar, error in zip(bars, errors):
#     height = bar.get_height()
#     # 在误差线顶部之上增加额外的偏移量，这里假设为1.5，您可以根据需要调整
#     plt.text(bar.get_x() + bar.get_width() / 2, height + error + 0.5,
#              f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
# 设置标题和坐标轴标签
plt.title('Accuracy of different methods on ADNI1 and ADNI2.', fontsize=16)
plt.xlabel('Method', fontsize=14)
plt.ylabel('Accuracy (%)', fontsize=14)

# 设置X轴标签
plt.xticks(x_positions, methods, rotation=45, ha='right', fontsize=10)

# 添加图例
plt.legend(bars, methods, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# 调整布局避免重叠
plt.tight_layout()

# 显示图像
plt.savefig('ErrorBar.png', bbox_inches='tight', dpi=300)
plt.show()
