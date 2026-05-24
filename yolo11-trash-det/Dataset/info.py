import os
import cv2
import torch
import time
from pathlib import Path
import numpy as np

# 配置参数
class Config:
    # 模型路径
    model_path = r'D:\cv\qq_3045834499\yolo11-trash-det\42_demo\runs\yolo11n_pretrained\weights\best.pt'

    # 待检测数据路径
    data_dir = r'D:\桌面\oaoa\val\images'

    # 输出路径
    output_dir = r'D:\桌面\result'
    output_img_dir = os.path.join(output_dir, 'detections')
    output_txt_dir = os.path.join(output_dir, 'labels')

    # 推理参数
    img_size = 640  # 输入图像尺寸
    conf_thres = 0.25  # 置信度阈值
    iou_thres = 0.45  # NMS IoU阈值
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # 自动选择设备


def main():
    # 初始化配置
    cfg = Config()

    # 创建输出目录
    Path(cfg.output_img_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.output_txt_dir).mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f'Loading model from {cfg.model_path}...')
    model = torch.load(cfg.model_path, map_location=cfg.device)['model'].float().eval()  # 假设是完整checkpoint
    model.to(cfg.device)

    # 获取类别名称 (根据实际情况修改)
    class_names = ['trash']  # 替换为你的类别名称

    # 获取图像文件列表
    img_files = [f for f in os.listdir(cfg.data_dir) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
    total_images = len(img_files)
    print(f'Found {total_images} images to process')

    # 预热GPU
    if 'cuda' in cfg.device:
        print('Warming up GPU...')
        dummy_input = torch.randn(1, 3, cfg.img_size, cfg.img_size).to(cfg.device)
        for _ in range(10):
            _ = model(dummy_input)

    # 推理循环
    total_time = 0
    fps_list = []

    for idx, img_file in enumerate(img_files, 1):
        img_path = os.path.join(cfg.data_dir, img_file)

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            print(f'Error reading image: {img_path}')
            continue

        # 记录开始时间
        start_time = time.time()

        # 预处理
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = preprocess(img_rgb, cfg.img_size, cfg.device)

        # 推理
        with torch.no_grad():
            pred = model(img_tensor)

        # 后处理
        detections = non_max_suppression(pred, cfg.conf_thres, cfg.iou_thres)[0]

        # 计算耗时
        inference_time = time.time() - start_time
        total_time += inference_time
        fps = 1 / inference_time
        fps_list.append(fps)

        # 保存结果
        save_path_img = os.path.join(cfg.output_img_dir, img_file)
        save_path_txt = os.path.join(cfg.output_txt_dir, Path(img_file).stem + '.txt')

        # 可视化并保存图像
        plot_results(img, detections, class_names, save_path_img)

        # 保存检测结果文本文件
        save_detection_txt(detections, save_path_txt, cfg.img_size)

        print(f'Processed {idx}/{total_images} | FPS: {fps:.2f} | File: {img_file}')

    # 计算统计信息
    avg_fps = sum(fps_list) / len(fps_list)
    total_fps = total_images / total_time

    print('\n===== Final Statistics =====')
    print(f'Total images processed: {total_images}')
    print(f'Total time: {total_time:.2f}s')
    print(f'Average FPS: {avg_fps:.2f}')
    print(f'Total FPS: {total_fps:.2f}')


def preprocess(img, img_size, device):
    """图像预处理"""
    # 保持长宽比的resize
    h, w = img.shape[:2]
    scale = min(img_size / h, img_size / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # resize并填充
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized

    # 转换为tensor
    tensor = torch.from_numpy(padded).permute(2, 0, 1).float().div(255.0)
    return tensor.unsqueeze(0).to(device)


def non_max_suppression(pred, conf_thres, iou_thres):
    """非极大值抑制"""
    return torch.ops.torchvision.nms(
        pred[..., :4],
        pred[..., 4] * pred[..., 5:].max(1)[0],
        iou_thres
    )


def plot_results(img, detections, class_names, save_path):
    """可视化检测结果"""
    for *xyxy, conf, cls in detections:
        label = f'{class_names[int(cls)]} {conf:.2f}'
        xyxy = [int(x) for x in xyxy]

        # 画框
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
        # 添加标签
        cv2.putText(img, label, (xyxy[0], xyxy[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imwrite(save_path, img)


def save_detection_txt(detections, txt_path, img_size):
    """保存检测结果为YOLO格式文本文件"""
    with open(txt_path, 'w') as f:
        for *xyxy, conf, cls in detections:
            # 转换为相对坐标
            x_center = ((xyxy[0] + xyxy[2]) / 2) / img_size
            y_center = ((xyxy[1] + xyxy[3]) / 2) / img_size
            width = (xyxy[2] - xyxy[0]) / img_size
            height = (xyxy[3] - xyxy[1]) / img_size

            line = f'{int(cls)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n'
            f.write(line)


if __name__ == '__main__':
    main()