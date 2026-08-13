# -*- coding: utf-8 -*-
"""
阈值分割与形态学操作示例
============================================
本脚本演示 OpenCV 中的图像阈值分割和形态学操作，包括：

阈值分割：
1. 全局二值化（cv2.threshold）
2. 自适应阈值（cv2.adaptiveThreshold）
3. Otsu 大津法（自动计算最优阈值）

形态学操作：
1. 腐蚀（Erosion）
2. 膨胀（Dilation）
3. 开运算（Opening）：先腐蚀后膨胀
4. 闭运算（Closing）：先膨胀后腐蚀
5. 形态学梯度（Morphological Gradient）
6. 礼帽（Top Hat）与黑帽（Black Hat）

作者：巫伟鑫
日期：2024-11-20
"""

import cv2
import numpy as np


def threshold_demo(image_gray: np.ndarray) -> None:
    """
    全局阈值分割演示

    cv2.threshold() 的参数：
    - src: 输入图像（通常是灰度图）
    - thresh: 阈值
    - maxval: 最大值（超过阈值时赋予的值）
    - type: 阈值类型
        cv2.THRESH_BINARY     - 二值化：大于阈值取 maxval，否则取 0
        cv2.THRESH_BINARY_INV - 反二值化：大于阈值取 0，否则取 maxval
        cv2.THRESH_TRUNC      - 截断：大于阈值取阈值，否则不变
        cv2.THRESH_TOZERO     - 低于阈值置零：大于阈值不变，否则取 0
        cv2.THRESH_TOZERO_INV - 超阈值置零：大于阈值取 0，否则不变

    参数
    ----
    image_gray : np.ndarray
        输入灰度图像
    """
    print("  全局阈值分割：")
    print("    原理：根据一个全局阈值将图像分为前景和背景")

    threshold_value = 127  # 阈值
    max_value = 255        # 最大值

    # 各种阈值类型
    # 普通二值化
    _, binary = cv2.threshold(image_gray, threshold_value, max_value, cv2.THRESH_BINARY)
    # 反二值化
    _, binary_inv = cv2.threshold(image_gray, threshold_value, max_value, cv2.THRESH_BINARY_INV)
    # 截断
    _, trunc = cv2.threshold(image_gray, threshold_value, max_value, cv2.THRESH_TRUNC)
    # 低于阈值置零
    _, tozero = cv2.threshold(image_gray, threshold_value, max_value, cv2.THRESH_TOZERO)
    # 超阈值置零
    _, tozero_inv = cv2.threshold(image_gray, threshold_value, max_value, cv2.THRESH_TOZERO_INV)

    print(f"    阈值: {threshold_value}")
    print(f"    二值化 - 白色像素占比: {cv2.countNonZero(binary) / binary.size * 100:.2f}%")

    # 显示各种阈值类型的对比
    titles = ['Original Gray', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [image_gray, binary, binary_inv, trunc, tozero, tozero_inv]

    # 创建 2x3 的对比图
    h, w = image_gray.shape
    comparison = np.zeros((h * 2, w * 3), dtype=np.uint8)

    for i in range(2):
        for j in range(3):
            idx = i * 3 + j
            comparison[i * h:(i + 1) * h, j * w:(j + 1) * w] = images[idx]

    cv2.namedWindow("Threshold Types", cv2.WINDOW_NORMAL)
    cv2.imshow("Threshold Types", comparison)


def adaptive_threshold_demo(image_gray: np.ndarray) -> None:
    """
    自适应阈值分割演示

    当图像光照不均匀时，全局阈值效果不好。自适应阈值根据像素周围的
    局部邻域计算阈值，每个像素有自己的阈值，因此对光照变化更鲁棒。

    cv2.adaptiveThreshold() 参数：
    - src: 输入图像
    - maxValue: 最大值
    - adaptiveMethod: 自适应方法
        cv2.ADAPTIVE_THRESH_MEAN_C     - 邻域均值
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C - 邻域高斯加权和
    - thresholdType: 阈值类型（只能是 THRESH_BINARY 或 THRESH_BINARY_INV）
    - blockSize: 邻域大小（必须是奇数）
    - C: 从均值或加权均值中减去的常数

    参数
    ----
    image_gray : np.ndarray
        输入灰度图像
    """
    print("\n  自适应阈值分割：")
    print("    原理：每个像素根据其邻域计算局部阈值")
    print("    特点：对光照不均匀的图像效果好")

    # 先添加不均匀光照效果（模拟实际场景）
    # 创建一个渐变光照
    rows, cols = image_gray.shape
    gradient_x = np.linspace(0.3, 1.0, cols).reshape(1, -1)
    gradient_y = np.linspace(0.5, 1.0, rows).reshape(-1, 1)
    illumination = gradient_x * gradient_y
    uneven_img = (image_gray.astype(np.float32) * illumination).astype(np.uint8)

    # 全局阈值（对比用）
    _, global_thresh = cv2.threshold(uneven_img, 127, 255, cv2.THRESH_BINARY)

    # 自适应阈值 - 均值法
    adaptive_mean = cv2.adaptiveThreshold(
        uneven_img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,  # 邻域大小（奇数）
        C=2            # 减去的常数
    )

    # 自适应阈值 - 高斯法
    adaptive_gaussian = cv2.adaptiveThreshold(
        uneven_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=11,
        C=2
    )

    print(f"    邻域大小: 11x11, C=2")
    print(f"    全局阈值 - 白色像素占比: {cv2.countNonZero(global_thresh) / global_thresh.size * 100:.2f}%")
    print(f"    自适应均值 - 白色像素占比: {cv2.countNonZero(adaptive_mean) / adaptive_mean.size * 100:.2f}%")
    print(f"    自适应高斯 - 白色像素占比: {cv2.countNonZero(adaptive_gaussian) / adaptive_gaussian.size * 100:.2f}%")

    # 显示结果
    cv2.namedWindow("Uneven Illumination", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Global Threshold", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Adaptive Mean", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Adaptive Gaussian", cv2.WINDOW_NORMAL)

    cv2.imshow("Uneven Illumination", uneven_img)
    cv2.imshow("Global Threshold", global_thresh)
    cv2.imshow("Adaptive Mean", adaptive_mean)
    cv2.imshow("Adaptive Gaussian", adaptive_gaussian)


def otsu_threshold_demo(image_gray: np.ndarray) -> None:
    """
    Otsu 大津法阈值分割演示

    Otsu 法（大津法）是一种自动确定全局阈值的方法。
    它的思想是：找到一个阈值，使得前景和背景的类间方差最大。

    适用场景：图像有明显的双峰直方图（前景和背景两个峰）。

    使用方法：在 cv2.threshold() 中传入 THRESH_OTSU 标志，
    此时 thresh 参数会被忽略，由算法自动计算最优阈值。

    参数
    ----
    image_gray : np.ndarray
        输入灰度图像
    """
    print("\n  Otsu 大津法阈值分割：")
    print("    原理：自动计算使类间方差最大的最优阈值")
    print("    特点：无需手动设置阈值，适合双峰直方图")

    # Otsu 阈值
    otsu_thresh, otsu_result = cv2.threshold(
        image_gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 与手动设置的阈值对比
    manual_thresh = 127
    _, manual_result = cv2.threshold(image_gray, manual_thresh, 255, cv2.THRESH_BINARY)

    print(f"    Otsu 自动计算的阈值: {otsu_thresh}")
    print(f"    手动设置的阈值: {manual_thresh}")

    # 显示结果
    cv2.namedWindow("Original Gray", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Manual Threshold (127)", cv2.WINDOW_NORMAL)
    cv2.namedWindow(f"Otsu Threshold ({int(otsu_thresh)})", cv2.WINDOW_NORMAL)

    cv2.imshow("Original Gray", image_gray)
    cv2.imshow("Manual Threshold (127)", manual_result)
    cv2.imshow(f"Otsu Threshold ({int(otsu_thresh)})", otsu_result)


def morphological_operations_demo(image_gray: np.ndarray) -> None:
    """
    形态学操作演示

    形态学操作是基于图像形状的一系列操作，通常在二值图像上进行。
    基本操作：腐蚀和膨胀。

    腐蚀（Erosion）：
    - 用结构元素扫描图像，只有当结构元素完全在前景内时，输出像素才为前景
    - 效果：缩小白色区域，去除小的白色噪点，断开细长连接

    膨胀（Dilation）：
    - 用结构元素扫描图像，只要结构元素与前景有交集，输出像素就为前景
    - 效果：扩大白色区域，填充小的黑色空洞，连接邻近区域

    组合操作：
    - 开运算（Opening）：先腐蚀后膨胀 → 去除小的白色噪点，不改变大物体大小
    - 闭运算（Closing）：先膨胀后腐蚀 → 填充小的黑色空洞，不改变大物体大小
    - 形态学梯度：膨胀 - 腐蚀 → 得到物体轮廓
    - 礼帽（Top Hat）：原图 - 开运算 → 得到比周围亮的小区域
    - 黑帽（Black Hat）：闭运算 - 原图 → 得到比周围暗的小区域

    参数
    ----
    image_gray : np.ndarray
        输入灰度图像
    """
    print("\n  形态学操作：")
    print("    原理：基于图像形状的操作，通常在二值图像上进行")

    # 先二值化
    _, binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

    # 创建结构元素（核）
    # MORPH_RECT: 矩形核
    # MORPH_ELLIPSE: 椭圆核
    # MORPH_CROSS: 十字核
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    print(f"    结构元素（5x5 矩形核）:\n{kernel}")

    # 腐蚀
    erosion = cv2.erode(binary, kernel, iterations=1)

    # 膨胀
    dilation = cv2.dilate(binary, kernel, iterations=1)

    # 开运算
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 闭运算
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 形态学梯度
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

    # 礼帽
    tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)

    # 黑帽
    blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)

    print(f"    腐蚀后白色像素数: {cv2.countNonZero(erosion)}")
    print(f"    膨胀后白色像素数: {cv2.countNonZero(dilation)}")
    print(f"    开运算后白色像素数: {cv2.countNonZero(opening)}")
    print(f"    闭运算后白色像素数: {cv2.countNonZero(closing)}")

    # 显示结果
    titles = ['Original Binary', 'Erosion', 'Dilation', 'Opening',
              'Closing', 'Gradient', 'Top Hat', 'Black Hat']
    images = [binary, erosion, dilation, opening,
              closing, gradient, tophat, blackhat]

    # 2x4 布局
    h, w = binary.shape
    comparison = np.zeros((h * 2, w * 4), dtype=np.uint8)

    for i in range(2):
        for j in range(4):
            idx = i * 4 + j
            comparison[i * h:(i + 1) * h, j * w:(j + 1) * w] = images[idx]

    cv2.namedWindow("Morphological Operations", cv2.WINDOW_NORMAL)
    cv2.imshow("Morphological Operations", comparison)

    cv2.imwrite("output_morphology.jpg", comparison)
    print("    形态学操作对比图已保存为 output_morphology.jpg")


def noise_removal_demo(image_gray: np.ndarray) -> None:
    """
    使用形态学操作去除噪点的实际应用

    参数
    ----
    image_gray : np.ndarray
        输入灰度图像
    """
    print("\n  形态学去噪实际应用：")

    # 先二值化
    _, binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)

    # 添加椒盐噪声（模拟噪点）
    noisy = binary.copy()
    # 白色噪点
    num_white = np.ceil(0.01 * binary.size)
    coords = [np.random.randint(0, i - 1, int(num_white)) for i in binary.shape]
    noisy[coords[0], coords[1]] = 255
    # 黑色噪点
    num_black = np.ceil(0.01 * binary.size)
    coords = [np.random.randint(0, i - 1, int(num_black)) for i in binary.shape]
    noisy[coords[0], coords[1]] = 0

    # 开运算去除白色噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(noisy, cv2.MORPH_OPEN, kernel)

    # 闭运算填充黑色空洞
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

    print("    步骤：")
    print("      1. 添加椒盐噪声模拟实际图像")
    print("      2. 开运算去除白色噪点")
    print("      3. 闭运算填充黑色空洞")

    # 显示结果
    cv2.namedWindow("Original Binary", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Noisy Binary", cv2.WINDOW_NORMAL)
    cv2.namedWindow("After Opening", cv2.WINDOW_NORMAL)
    cv2.namedWindow("After Closing", cv2.WINDOW_NORMAL)

    cv2.imshow("Original Binary", binary)
    cv2.imshow("Noisy Binary", noisy)
    cv2.imshow("After Opening", opened)
    cv2.imshow("After Closing", closed)


def main():
    """
    主函数：依次运行阈值和形态学操作示例
    """
    test_image = "test.jpg"
    img = cv2.imread(test_image, cv2.IMREAD_COLOR)

    if img is None:
        print(f"错误：无法读取图像 {test_image}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("=" * 50)
    print("OpenCV 阈值分割与形态学操作示例")
    print("=" * 50)

    print("\n[1/5] 全局阈值分割...")
    threshold_demo(gray)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[2/5] 自适应阈值分割...")
    adaptive_threshold_demo(gray)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[3/5] Otsu 大津法阈值...")
    otsu_threshold_demo(gray)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[4/5] 形态学操作...")
    morphological_operations_demo(gray)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[5/5] 形态学去噪应用...")
    noise_removal_demo(gray)
    print("  按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n所有示例运行完毕！")


if __name__ == "__main__":
    main()
