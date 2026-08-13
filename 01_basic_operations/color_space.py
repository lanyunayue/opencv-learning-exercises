# -*- coding: utf-8 -*-
"""
颜色空间转换示例
============================================
本脚本演示 OpenCV 中各种颜色空间的转换，包括：
1. BGR 与灰度图互转
2. BGR 与 HSV 互转
3. BGR 与 RGB 互转
4. 使用 inRange 进行颜色提取（基于 HSV 空间）
5. 各颜色通道的分离与可视化

作者：巫伟鑫
日期：2024-11-20

============================================
学习心得与踩坑记录：

1. HSV 颜色空间真的很有用
   做颜色提取的时候，HSV 比 BGR 好用太多了！因为 H 通道直接对应颜色种类，
   只要设定 H 的范围就能提取指定颜色。BGR 三个通道互相关联，很难调好阈值。

2. OpenCV 中 H 通道的范围是 0-179
   这一点很重要！很多资料里 HSV 的 H 范围是 0-360 度，但 OpenCV 为了用
   一个字节（0-255）存储，把 H 值除以了 2，所以范围变成了 0-179。
   刚开始查颜色的 HSV 范围时，别直接用网上的 0-360 的数值，记得除以 2。

3. 颜色提取的一般步骤
   原图 → 转 HSV → inRange 生成 mask → 形态学操作去噪 → 按位与提取结果
   形态学操作很关键，可以去除很多小噪点，让提取结果更干净。

4. 红色的 HSV 范围比较特殊
   红色在 HSV 色环的两端（0度和360度是同一个颜色），所以在 OpenCV 中
   红色需要用两段范围：[0, 10] 和 [170, 179]，然后把两个 mask 相加。
   这个我一开始不知道，提取红色总是缺一块...
============================================
"""

import cv2
import numpy as np


def bgr_to_gray(image_path: str) -> None:
    """
    BGR 彩色图像转灰度图

    灰度转换公式（亮度加权）：
        Gray = 0.299 * R + 0.587 * G + 0.114 * B

    参数
    ----
    image_path : str
        图像文件路径
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    # 方法1：使用 cvtColor 转换（推荐）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 方法2：读取时直接以灰度模式读取
    # gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    print(f"原始图像形状: {img.shape}")
    print(f"灰度图像形状: {gray.shape}")
    print(f"灰度图像数据类型: {gray.dtype}")

    # 显示对比
    cv2.namedWindow("BGR Color", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Gray", cv2.WINDOW_NORMAL)
    cv2.imshow("BGR Color", img)
    cv2.imshow("Gray", gray)

    print("\n按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def bgr_to_hsv(image_path: str) -> None:
    """
    BGR 转 HSV 颜色空间

    HSV 三个通道的含义：
        H (Hue)        - 色相，范围 [0, 179]，表示颜色的种类
        S (Saturation) - 饱和度，范围 [0, 255]，表示颜色的鲜艳程度
        V (Value)      - 明度，范围 [0, 255]，表示颜色的明亮程度

    HSV 空间非常适合做颜色分割，因为颜色信息主要集中在 H 通道。

    参数
    ----
    image_path : str
        图像文件路径
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    # BGR 转 HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 拆分 HSV 三个通道
    h, s, v = cv2.split(hsv)

    print("HSV 颜色空间信息：")
    print(f"  H 通道范围: [{h.min()}, {h.max()}] (色相)")
    print(f"  S 通道范围: [{s.min()}, {s.max()}] (饱和度)")
    print(f"  V 通道范围: [{v.min()}, {v.max()}] (明度)")

    # 显示各通道
    cv2.namedWindow("Original BGR", cv2.WINDOW_NORMAL)
    cv2.namedWindow("HSV Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("H - Hue", cv2.WINDOW_NORMAL)
    cv2.namedWindow("S - Saturation", cv2.WINDOW_NORMAL)
    cv2.namedWindow("V - Value", cv2.WINDOW_NORMAL)

    cv2.imshow("Original BGR", img)
    cv2.imshow("HSV Image", hsv)
    cv2.imshow("H - Hue", h)
    cv2.imshow("S - Saturation", s)
    cv2.imshow("V - Value", v)

    print("\n按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def color_extraction(image_path: str) -> None:
    """
    基于 HSV 的颜色提取（以蓝色为例）

    使用 cv2.inRange() 函数提取指定颜色范围内的像素，
    生成二值掩码（mask），再用掩码与原图做按位与运算。

    常见颜色的 HSV 范围（OpenCV 中 H 范围是 0-179）：
        红色：  H: [0, 10] 或 [170, 179], S: [43, 255], V: [46, 255]
        绿色：  H: [35, 77],   S: [43, 255], V: [46, 255]
        蓝色：  H: [100, 124], S: [43, 255], V: [46, 255]
        黄色：  H: [26, 34],   S: [43, 255], V: [46, 255]

    参数
    ----
    image_path : str
        图像文件路径
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    # 转换到 HSV 空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义蓝色的 HSV 范围
    lower_blue = np.array([100, 43, 46])   # 下界
    upper_blue = np.array([124, 255, 255])  # 上界

    # 生成掩码：在范围内的像素为 255（白色），范围外为 0（黑色）
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 对掩码进行形态学操作，去除噪点
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # 开运算：去除小的白色噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # 闭运算：填充小的黑色空洞

    # 用掩码提取原图中的蓝色区域
    result = cv2.bitwise_and(img, img, mask=mask)

    # 显示结果
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Blue Extraction", cv2.WINDOW_NORMAL)

    cv2.imshow("Original", img)
    cv2.imshow("Mask", mask)
    cv2.imshow("Blue Extraction", result)

    # 统计蓝色像素占比
    blue_pixels = cv2.countNonZero(mask)
    total_pixels = img.shape[0] * img.shape[1]
    ratio = blue_pixels / total_pixels * 100
    print(f"\n蓝色像素统计：")
    print(f"  蓝色像素数: {blue_pixels}")
    print(f"  总像素数: {total_pixels}")
    print(f"  蓝色占比: {ratio:.2f}%")

    print("\n按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("output_blue_extraction.jpg", result)
    print("颜色提取结果已保存为 output_blue_extraction.jpg")


def bgr_rgb_conversion(image_path: str) -> None:
    """
    BGR 与 RGB 互转

    注意事项：
    - OpenCV 默认使用 BGR 顺序
    - Matplotlib、PIL 等库使用 RGB 顺序
    - 混用会导致颜色显示异常，需要转换

    参数
    ----
    image_path : str
        图像文件路径
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    # 方法1：使用 cvtColor 转换
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 方法2：使用 numpy 索引翻转通道
    # img_rgb = img_bgr[:, :, ::-1]

    # 验证：比较第一个像素的 BGR 和 RGB 值
    print("BGR vs RGB 对比（左上角第一个像素）：")
    print(f"  BGR 顺序: {img_bgr[0, 0]} (B, G, R)")
    print(f"  RGB 顺序: {img_rgb[0, 0]} (R, G, B)")

    # 显示对比
    cv2.namedWindow("BGR (OpenCV)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RGB (Converted)", cv2.WINDOW_NORMAL)
    cv2.imshow("BGR (OpenCV)", img_bgr)
    cv2.imshow("RGB (Converted)", img_rgb)

    print("\n注意观察两图的颜色差异（BGR 和 RGB 顺序不同）")
    print("按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def split_and_merge_channels(image_path: str) -> None:
    """
    通道拆分与合并

    参数
    ----
    image_path : str
        图像文件路径
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"错误：无法读取图像 {image_path}")
        return

    # 拆分 BGR 通道
    b, g, r = cv2.split(img)

    # 创建三幅单通道彩色图像（用于直观显示各通道的颜色贡献）
    # 蓝色通道图像（只有蓝色分量）
    blue_only = cv2.merge([b, np.zeros_like(b), np.zeros_like(b)])
    # 绿色通道图像（只有绿色分量）
    green_only = cv2.merge([np.zeros_like(g), g, np.zeros_like(g)])
    # 红色通道图像（只有红色分量）
    red_only = cv2.merge([np.zeros_like(r), np.zeros_like(r), r])

    # 显示各通道
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Blue Channel", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Green Channel", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Red Channel", cv2.WINDOW_NORMAL)

    cv2.imshow("Original", img)
    cv2.imshow("Blue Channel", blue_only)
    cv2.imshow("Green Channel", green_only)
    cv2.imshow("Red Channel", red_only)

    print("\n各通道图像显示中，按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """
    主函数：依次运行各个颜色空间转换示例
    """
    test_image = "test.jpg"

    print("=" * 50)
    print("OpenCV 颜色空间转换示例")
    print("=" * 50)

    print("\n[1/5] BGR 转灰度图...")
    bgr_to_gray(test_image)

    print("\n[2/5] BGR 转 HSV...")
    bgr_to_hsv(test_image)

    print("\n[3/5] 基于 HSV 的颜色提取（蓝色）...")
    color_extraction(test_image)

    print("\n[4/5] BGR 与 RGB 互转...")
    bgr_rgb_conversion(test_image)

    print("\n[5/5] 通道拆分与合并...")
    split_and_merge_channels(test_image)

    print("\n所有示例运行完毕！")


if __name__ == "__main__":
    main()
