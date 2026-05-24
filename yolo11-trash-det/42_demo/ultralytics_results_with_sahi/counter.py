import os
from collections import defaultdict

# 初始化一个字典来统计每类的图片数量
category_image_counts = defaultdict(set)

# 指定标签文件所在的文件夹路径
label_folder = r"D:\桌面\oaoa\train\labels"  # 替换为你的标签文件夹路径

# 遍历文件夹中的所有文件
for filename in os.listdir(label_folder):
    if filename.endswith(".txt"):  # 假设标签文件是 .txt 格式
        file_path = os.path.join(label_folder, filename)

        # 打开并读取标签文件
        with open(file_path, 'r') as f:
            lines = f.readlines()

            # 如果文件为空，跳过
            if not lines:
                continue

            # 统计文件中每行的第一个数字（类别编号）
            categories_in_file = set()
            for line in lines:
                parts = line.strip().split()
                if parts:
                    category = parts[0]
                    if category.isdigit():
                        categories_in_file.add(category)

            # 将文件名添加到对应的类别集合中
            for category in categories_in_file:
                category_image_counts[category].add(filename)

# 输出统计结果
print("统计结果：")
for category, images in category_image_counts.items():
    print(f"类别 {category}: {len(images)} 张图片")