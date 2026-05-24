import os
import glob
from PIL import Image
import numpy as np

# 配置路径
image_dir = r'D:\桌面\oaoa\val\images'
label_dir = r'D:\桌面\oaoa\val\labels'


def calculate_pixel_ratios():
    # 支持的图片格式
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif']

    # 收集所有目标占比
    all_ratios = []
    processed_files = 0
    error_files = 0

    # 遍历所有图片文件
    for ext in image_extensions:
        for img_path in glob.glob(os.path.join(image_dir, f'*.{ext}')):
            try:
                # 获取对应标签路径
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(label_dir, f"{base_name}.txt")

                if not os.path.exists(label_path):
                    print(f"⚠️ 标签文件缺失: {label_path}")
                    error_files += 1
                    continue

                # 获取图像尺寸
                with Image.open(img_path) as img:
                    img_w, img_h = img.size

                # 读取标签文件
                with open(label_path, 'r') as f:
                    lines = [line.strip() for line in f.readlines()]

                # 处理每个目标
                for line in lines:
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) < 5:
                        print(f"❌ 格式错误: {label_path} -> {line}")
                        error_files += 1
                        continue

                    try:
                        # 解析YOLO格式 (class, x_center, y_center, width, height)
                        _, _, _, w_norm, h_norm = map(float, parts[:5])
                        ratio = w_norm * h_norm  # 直接计算归一化面积占比
                        all_ratios.append(ratio)
                    except ValueError:
                        print(f"❌ 数值错误: {label_path} -> {line}")
                        error_files += 1

                processed_files += 1

            except Exception as e:
                print(f"🔥 处理失败: {img_path} -> {str(e)}")
                error_files += 1

    # 输出统计结果
    if not all_ratios:
        print("\n❌ 未找到有效目标")
        return

    ratios = np.array(all_ratios)
    print("\n📊 统计结果:")
    print(f"✅ 成功处理文件: {processed_files}")
    print(f"❌ 错误文件数: {error_files}")
    print(f"🎯 总目标数量: {len(ratios)}")
    print(f"📏 平均像素占比: {ratios.mean():.4f}")
    print(f"📈 最大像素占比: {ratios.max():.4f}")
    print(f"📉 最小像素占比: {ratios.min():.4f}")
    print(f"📌 中位数占比: {np.median(ratios):.4f}")
    print(f"🔢 25%-75%分位数: {np.quantile(ratios, 0.25):.4f} - {np.quantile(ratios, 0.75):.4f}")


if __name__ == "__main__":
    calculate_pixel_ratios()