# -*- coding: utf-8 -*-
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# ==== 1. 图像加载与校验 ====
def load_image(img_path, target_size=(320, 320)):
    """加载并标准化图像尺寸"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"图像加载失败，请检查路径: {img_path}")

    # 强制转换为320x320
    if img.shape != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


# ==== 2. 频域变换 ====
def compute_spectrum(img):
    """计算对数幅度谱"""
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    magnitude = 20 * np.log(np.abs(fft_shift) + 1e-6)  # 避免log(0)，+1e-6
    return magnitude


# ==== 3. 三维可视化 ====
def plot_3d_spectrum(X, Y, Z, step=5):
    """生成三维频谱图"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 安全降采样
    X_sub = X[::step, ::step]
    Y_sub = Y[::step, ::step]
    Z_sub = Z[::step, ::step]

    # 绘制表面
    surf = ax.plot_surface(
        X_sub, Y_sub, Z_sub,
        cmap='jet',
        rstride=1,
        cstride=1,
        linewidth=0.1,
        antialiased=True,
        alpha=0.8
    )

    # 图形标注
    ax.set_title("星空背景三维频谱", fontsize=14, fontfamily='SimHei')
    ax.set_xlabel('水平频率 (cycles/pixel)', fontsize=10)
    ax.set_ylabel('垂直频率 (cycles/pixel)', fontsize=10)
    ax.set_zlabel('幅度 (dB)', fontsize=10)

    # 视角调整
    ax.view_init(elev=25, azim=-45)

    # 添加色标
    cbar = fig.colorbar(surf, shrink=0.6, aspect=20)
    cbar.set_label('幅度 (dB)', fontsize=10)

    return fig, ax


# ==== 主程序 ====
if __name__ == "__main__":
    # 配置中文字体（需系统支持）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    try:
        # 1. 加载图像
        img_path = r'D:\cv\lwb_graduation_project\yolo11-trash-det\Dataset\star-plane.jpg' # 确保路径存在
        img = load_image(img_path)

        # 2. 计算频域
        magnitude = compute_spectrum(img)

        # 3. 生成坐标网格
        x = np.linspace(-160, 159, 320)  # 严格匹配320x320
        y = np.linspace(-160, 159, 320)
        X, Y = np.meshgrid(x, y)

        # 4. 可视化（步长=5）
        plot_3d_spectrum(X, Y, magnitude, step=5)

        # 5. 保存结果
        plt.savefig('spectrum_3d.png', dpi=300, bbox_inches='tight')
        plt.show()

    except Exception as e:
        print(f"程序运行错误: {str(e)}")