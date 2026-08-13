# -*- coding: utf-8 -*-
"""
图像读写与像素操作示例
============================================
本脚本演示 OpenCV 中最基础的图像 I/O 操作，包括：
1. 图像的读取、显示和保存
2. 图像的基本属性（尺寸、通道数、数据类型）
3. 像素级访问与修改
4. 图像 ROI（感兴趣区域）截取
5. 图像通道的拆分与合并

作者：巫伟鑫
日期：2024-11-20
"""

import cv2
import numpy as np


def read_and_display_image(image_path: str) -> None:
    """
    读取并显示图像

    参数
    ----
    image_path : str
        图像文件路径
    """
    # ==================== 图像读取 ====================
    # cv2.imread() 第二个参数指定读取方式：
    #   cv2.IMREAD_COLOR     - 以彩色模式读取（默认，忽略透明度通道）
    #   cv2.IMREAD_GRAYSCALE - 以灰度模式读取
    #   cv2.IMREAD_UNCHANGED - 原样读取（包含 alpha 通道）
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 检查图像是否读取成功（非常重要！）
    if img_color is None:
        print(f"错误：无法读取图像 {image_path}，请检查路径是否正确。")
        return

    # ==================== 图像属性 ====================
    print("=" * 50)
    print("图像基本属性：")
    print(f"  彩色图像形状 (高, 宽, 通道数): {img_color.shape}")
    print(f"  彩色图像数据类型: {img_color.dtype}")
    print(f"  彩色图像总像素数: {img_color.size}")
    print(f"  灰度图像形状 (高, 宽): {img_gray.shape}")
    print("=" * 50)

    # ==================== 图像显示 ====================
    # 创建可调整大小的窗口
    cv2.namedWindow("Color Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Gray Image", cv2.WINDOW_NORMAL)

    # 显示图像
    cv2.imshow("Color Image", img_color)
    cv2.imshow("Gray Image", img_gray)

    # 等待键盘输入，参数为等待毫秒数，0 表示无限等待
    # 返回值为按下的键的 ASCII 码
    key = cv2.waitKey(0)

    # 如果按下 's' 键，保存灰度图像
    if key == ord('s'):
        cv2.imwrite("output_gray.jpg", img_gray)
        print("灰度图像已保存为 output_gray.jpg")

    # 销毁所有窗口
    cv2.destroyAllWindows()


def pixel_operations(image_path: str) -> None:
    """
    像素级操作演示

    参数
    ----
    image_path : str
        图像文件路径
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    # 获取图像尺寸
    height, width = img.shape[:2]

    # ==================== 访问单个像素 ====================
    # 注意：OpenCV 中图像的通道顺序是 BGR，不是 RGB！
    # 访问坐标 (100, 100) 处的像素（行, 列）
    pixel = img[100, 100]
    print(f"\n坐标 (100, 100) 处的像素值 (B, G, R): {pixel}")

    # 单独访问蓝色通道
    blue = img[100, 100, 0]
    green = img[100, 100, 1]
    red = img[100, 100, 2]
    print(f"  B = {blue}, G = {green}, R = {red}")

    # ==================== 修改像素值 ====================
    # 修改图像左上角 100x100 区域为红色（BGR: 0, 0, 255）
    img_copy = img.copy()
    img_copy[0:100, 0:100] = [0, 0, 255]

    # ==================== ROI 感兴趣区域 ====================
    # 截取图像中间的一块区域
    roi_x1, roi_y1 = width // 4, height // 4
    roi_x2, roi_y2 = width * 3 // 4, height * 3 // 4
    roi = img[roi_y1:roi_y2, roi_x1:roi_x2]

    # 将 ROI 复制到图像左上角
    img_copy2 = img.copy()
    roi_h, roi_w = roi.shape[:2]
    img_copy2[0:roi_h, 0:roi_w] = roi

    # ==================== 通道拆分与合并 ====================
    # 拆分 BGR 三个通道
    b, g, r = cv2.split(img)
    print(f"\n蓝色通道形状: {b.shape}")
    print(f"绿色通道形状: {g.shape}")
    print(f"红色通道形状: {r.shape}")

    # 合并通道（顺序调换为 RGB）
    img_rgb = cv2.merge([r, g, b])

    # 显示结果
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Red Rectangle", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ROI Copied", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RGB Order", cv2.WINDOW_NORMAL)

    cv2.imshow("Original", img)
    cv2.imshow("Red Rectangle", img_copy)
    cv2.imshow("ROI", roi)
    cv2.imshow("ROI Copied", img_copy2)
    cv2.imshow("RGB Order", img_rgb)

    print("\n按任意键关闭所有窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    cv2.imwrite("output_roi.jpg", roi)
    print("ROI 图像已保存为 output_roi.jpg")


def create_blank_image() -> None:
    """
    创建空白图像并绘制基本图形
    用于理解图像的像素结构
    """
    # 创建一张 400x400 的黑色图像（3通道，uint8 类型）
    # np.zeros() 创建全零数组，dtype=np.uint8 表示无符号8位整数（0-255）
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # 绘制一条蓝色对角线（BGR: 255, 0, 0）
    for i in range(400):
        img[i, i] = [255, 0, 0]  # BGR 顺序

    # 绘制一个绿色矩形
    img[50:150, 200:350] = [0, 255, 0]

    # 绘制一个红色圆形区域
    center_x, center_y, radius = 200, 300, 50
    y, x = np.ogrid[:400, :400]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    img[mask] = [0, 0, 255]

    cv2.namedWindow("Blank Image", cv2.WINDOW_NORMAL)
    cv2.imshow("Blank Image", img)
    print("\n空白图像绘制完成，按任意键关闭...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("output_blank.jpg", img)
    print("空白图像已保存为 output_blank.jpg")


def main():
    """
    主函数：依次运行各个示例
    """
    # 请将此处的路径替换为你自己的图片路径
    test_image = "test.jpg"

    print("=" * 50)
    print("OpenCV 图像读写与像素操作示例")
    print("=" * 50)

    print("\n[1/3] 图像读取与显示...")
    read_and_display_image(test_image)

    print("\n[2/3] 像素级操作...")
    pixel_operations(test_image)

    print("\n[3/3] 创建空白图像...")
    create_blank_image()

    print("\n所有示例运行完毕！")


if __name__ == "__main__":
    main()
