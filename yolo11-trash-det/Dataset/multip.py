# import os
# import shutil
#
# # 配置路径
# src_labels = r"D:\桌面\oaoa\val\labels"  # 源标签目录
# dst_labels = r"D:\桌面\multiply\labels"  # 目标标签目录
#
# # 创建目标目录（如果不存在）
# os.makedirs(dst_labels, exist_ok=True)
#
# # 遍历源目录下所有txt文件
# for filename in os.listdir(src_labels):
#     if filename.endswith(".txt"):
#         filepath = os.path.join(src_labels, filename)
#
#         # 统计有效目标数量（非空行）
#         with open(filepath, "r") as f:
#             lines = [line.strip() for line in f.readlines() if line.strip()]
#
#         # 如果包含两个或更多目标
#         if len(lines) >= 2:
#             # 构建目标路径
#             dst_path = os.path.join(dst_labels, filename)
#
#             # 复制文件
#             shutil.copy(filepath, dst_path)
#             print(f"Copied: {filename} ({len(lines)} targets)")
#
# print("\n操作完成！")
# print(f"源目录文件数：{len(os.listdir(src_labels))}")
# print(f"目标目录文件数：{len(os.listdir(dst_labels))}")

import os
import shutil

# 设置标签文件夹路径和图片源文件夹路径
labels_dir = r'D:\桌面\multiply\labels'
images_src_dir = r'D:\桌面\oaoa\val\images'
images_dst_dir = r'D:\桌面\multiply\images'

# 确保目标图片文件夹存在
os.makedirs(images_dst_dir, exist_ok=True)

# 遍历标签文件夹中的所有文件
for label_file in os.listdir(labels_dir):
    if label_file.endswith('.txt'):  # 假设标签文件是txt格式
        # 获取对应的图片文件名（假设图片格式为jpg，如果不是，请修改为实际格式）
        image_filename = label_file.replace('.txt', '.jpg')
        # 构造图片的源路径和目标路径
        image_src_path = os.path.join(images_src_dir, image_filename)
        image_dst_path = os.path.join(images_dst_dir, image_filename)

        # 检查图片是否存在于源文件夹
        if os.path.exists(image_src_path):
            # 复制图片到目标文件夹
            shutil.copy2(image_src_path, image_dst_path)
            print(f'已复制图片: {image_filename}')
        else:
            print(f'未找到图片: {image_filename}')