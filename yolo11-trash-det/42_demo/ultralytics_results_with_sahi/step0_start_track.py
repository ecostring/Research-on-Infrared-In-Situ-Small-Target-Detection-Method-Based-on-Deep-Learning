import sys
import PySpin  # Spinnaker SDK 的 Python 模块
import cv2
import numpy as np


def capture_video():
    # 初始化系统实例
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        print("未检测到摄像头！")
        cam_list.Clear()
        system.ReleaseInstance()
        return

    # 获取摄像头实例
    cam = cam_list.GetByIndex(0)
    cam.Init()

    # 配置采集模式为连续采集
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    cam.BeginAcquisition()

    try:
        while True:
            # 捕获一帧图像
            image_result = cam.GetNextImage()
            if image_result.IsIncomplete():
                print("图像不完整！")
                continue

            # 将图像数据转换为 OpenCV 格式
            image_data = image_result.GetNDArray()  # 获取 numpy 数组
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BAYER_BG2BGR)  # 转换为 BGR 格式（根据摄像头调整）

            # 在此处添加检测逻辑（示例：边缘检测）
            processed_image = cv2.CvtColor(image_data, cv2.COLOR_BGR2GRAY)
            processed_image = cv2.Canny(processed_image, 100, 200)

            # 显示原始图像和处理后的图像
            cv2.imshow("Original", image_data)
            cv2.imshow("Processed", processed_image)

            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # 释放图像缓冲区
            image_result.Release()

    except Exception as e:
        print(f"错误: {e}")

    finally:
        # 清理资源
        cam.EndAcquisition()
        cam.DeInit()
        cam_list.Clear()
        system.ReleaseInstance()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_video()