import cv2
import os
import natsort  # 需要安装natsort库

# 设置路径参数
image_folder = r'D:\cv\lwb_graduation_project\yolo11-trash-det\Dataset\video-plane'
output_path = r'D:\桌面\video_plane.mp4'

# 获取所有PNG文件并按自然顺序排序
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images = natsort.natsorted(images)  # 自然排序处理文件名

# 视频参数设置
frame_duration = 0.3  # 每张图片显示0.3秒
fps = 10  # 设置帧率为10（通过重复帧实现精确时长）

# 初始化高度和宽度变量
height = None
width = None

# 创建视频编码器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = None  # 初始化视频对象

# 生成视频
for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 以灰度模式读取

    if frame is None:
        print(f"警告：无法读取 {image}，已跳过")
        continue

    if height is None or width is None:
        height, width = frame.shape
        video = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    if frame.shape != (height, width):
        print(f"警告：{image} 尺寸不一致，已跳过")
        continue

    # 计算需要写入的帧数（0.3秒 * 10fps = 3帧）
    for _ in range(int(fps * frame_duration)):
        video.write(frame)

if video is not None:
    video.release()
print(f"视频已生成：{output_path}")