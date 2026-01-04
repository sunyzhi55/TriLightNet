import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 数据准备
data = {
    "Model": ["ResNet+Concat", "ViT+Concat", "nnMamba+Concat", "Diamond", "MDL",
              "VAPL", "HyperFusionNet", "IMF", "HFBSurv", "ITCFN",
              "MultimodalADNet", "TriLightNet (Ours)"],
    "Balanced Accuracy": [73.26, 59.12, 66.71, 69.19, 66.44,
                          61.46, 69.20, 75.39, 74.10, 74.87,
                          70.53, 79.06],
    "Params (M)": [66.952, 20.853, 26.042, 23.504, 10.707,
                   63.504, 15.402, 67.843, 34.123, 71.305,
                   4.320, 17.405],
    "FLOPs (G)": [70.924, 20.853, 48.626, 97.638, 19.243,
                  40.350, 47.750, 70.925, 141.849, 71.098,
                  20.307, 10.517]
}
df = pd.DataFrame(data)

# 创建图形和子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# 避免标签重叠：通过索引设置偏移方向
offsets = [(-10, -10), (10, 10), (0, 5), (0, 5),
           (0, 5), (0, 5), (0, -5), (0, -15),
           (5, 5), (0, 5), (0, 5), (-20, 5)]

# 第一张子图：Balanced Accuracy vs Params
scatter1 = ax1.scatter(
    x=df["Balanced Accuracy"],
    y=df["Params (M)"],
    s=df["Params (M)"] * 7,  # 气泡大小与Params成正比
    c=df["Params (M)"],       # 颜色与Params对应
    cmap="viridis",
    alpha=0.75,
    edgecolors='black'
)

# 添加模型标签
for i, row in df.iterrows():
    dx, dy = offsets[i % len(offsets)]
    ax1.annotate(
        row["Model"],
        (row["Balanced Accuracy"], row["Params (M)"]),
        textcoords="offset points",
        xytext=(dx, dy),
        ha='center',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.5, edgecolor="gray")
    )

# 添加colorbar到左侧
cbar1 = plt.colorbar(scatter1, ax=ax1, location='left')
# cbar1.set_label("Params (M)")

# 设置坐标轴和标题
ax1.set_xlabel("Balanced Accuracy (%)")
ax1.set_ylabel("Params (M)")
# ax1.set_title("Model Comparison: Balanced Accuracy vs. Params (Bubble Size = Params)")
ax1.grid(True, linestyle='--', alpha=0.5)
# 添加标题到子图下方
ax1.text(0.5, -0.2, "",
            fontfamily='Times New Roman',  # 新增此行，设置字体
         transform=ax1.transAxes, ha='center', va='center', fontsize=20)

# 第二张子图：Balanced Accuracy vs FLOPs
scatter2 = ax2.scatter(
    x=df["Balanced Accuracy"],
    y=df["FLOPs (G)"],
    s=df["FLOPs (G)"] * 7,  # 气泡大小与FLOPs成正比
    c=df["FLOPs (G)"],       # 颜色与FLOPs对应
    cmap="viridis",
    alpha=0.75,
    edgecolors='black'
)

# 添加模型标签
for i, row in df.iterrows():
    dx, dy = offsets[i % len(offsets)]
    ax2.annotate(
        row["Model"],
        (row["Balanced Accuracy"], row["FLOPs (G)"]),
        textcoords="offset points",
        xytext=(dx, dy),
        ha='center',
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.5, edgecolor="gray")
    )

# 添加colorbar到左侧
cbar2 = plt.colorbar(scatter2, ax=ax2, location='left')
# cbar2.set_label("FLOPs (G)")

# 设置坐标轴和标题
ax2.set_xlabel("Balanced Accuracy (%)")
ax2.set_ylabel("FLOPs (G)")
# ax2.set_title("Model Comparison: Balanced Accuracy vs. FLOPs (Bubble Size = FLOPs)")
ax2.grid(True, linestyle='--', alpha=0.5)
# 添加标题到子图下方
ax2.text(0.5, -0.2, "",
fontfamily='Times New Roman',  # 新增此行，设置字体
         transform=ax2.transAxes, ha='center', va='center', fontsize=20)

# 调整子图间距，为左侧的colorbar留出空间
plt.subplots_adjust(left=0.15, right=0.95, wspace=0.3)

# 保存图片
plt.savefig("bubble.png", dpi=500, bbox_inches='tight', transparent=True)
plt.savefig("bubble.pdf", bbox_inches='tight', transparent=True)
plt.show()