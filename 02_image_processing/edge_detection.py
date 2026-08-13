# -*- coding: utf-8 -*-
"""
边缘检测示例
============================================
本脚本演示 OpenCV 中常用的边缘检测方法，包括：
1. Sobel 算子（一阶导数，分 x 和 y 方向）
2. Scharr 算子（Sobel 的改进版，核更大更准确）
3. Laplacian 算子（二阶导数，对噪声敏感）
4. Canny 边缘检测（最经典的多阶段算法）

边缘检测原理：
图像边缘是像素值发生突变的位置，对应图像灰度的一阶导数极值点
或二阶导数的零点。

作者：巫伟鑫
日期：2024-11-20

============================================
学习心得与踩坑记录：

1. 为什么边缘检测一般在灰度图上做？
   因为边缘检测本质是找灰度变化剧烈的地方，用单通道的灰度图计算量小，
   而且效果也够用。当然也可以在彩色图上做，每个通道分别检测再合并，
   但通常没必要。

2. Sobel 为什么要用 CV_64F？
   因为 Sobel 算子计算的梯度有正有负（由暗到亮是正，由亮到暗是负），
   如果直接用 uint8，负值会被截断成 0，丢失一半的边缘信息。
   所以先用 CV_64F 保存结果，取绝对值后再转回 uint8。

3. Canny 为什么效果最好？
   Canny 是一套完整的算法流程，不是简单的一个卷积核：
   1) 高斯滤波去噪
   2) Sobel 计算梯度幅值和方向
   3) 非极大值抑制（NMS）- 细化边缘，只保留最陡的地方
   4) 双阈值检测 - 区分强边缘和弱边缘，弱边缘只有和强边缘相连才保留
   所以 Canny 检测出来的边缘细、连续、假边缘少。

4. Canny 阈值怎么调？
   经验值：高阈值是低阈值的 2~3 倍。
   阈值越小，检测到的边缘越多，但也越容易出现假边缘。
   阈值越大，边缘越少，但可能会漏掉真边缘。
   实际用的时候可以用滑动条来调，找到最合适的阈值。

5. Laplacian 对噪声敏感
   因为 Laplacian 是二阶导数，会放大噪声。所以一般用 Laplacian 之前
   都会先做高斯平滑，也就是 LoG（Laplacian of Gaussian）。
============================================
"""

import cv2
import numpy as np


def sobel_demo(image_gray: np.ndarray) -> None:
    """
    Sobel 算子边缘检测演示

    Sobel 算子是一种离散微分算子，计算图像灰度的近似梯度。
    它结合了高斯平滑和微分求导，对噪声有一定的抑制能力。

    Sobel x 方向核（检测垂直边缘）：
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]]

    Sobel y 方向核（检测水平边缘）：
        [[-1, -2, -1],
         [ 0,  0,  0],
         [ 1,  2,  1]]

    参数
    ----
    image_gray : np.ndarray
        输入灰度图像
    """
    print("  Sobel 算子边缘检测：")
    print("    原理：计算图像灰度的一阶导数，响应最大处为边缘")
    print("    特点：对噪声有一定抑制，分 x/y 方向检测")

    # Sobel x 方向（检测垂直边缘）
    # ddepth=cv2.CV_64F 表示输出为 64 位浮点数，避免负值被截断
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, dx=1, dy=0, ksize=3)

    # Sobel y 方向（检测水平边缘）
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, dx=0, dy=1, ksize=3)

    # 取绝对值并转换为 uint8
    # 因为梯度可以是正的（由暗到亮）或负的（由亮到暗）
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)

    # 合并 x 和 y 方向的梯度（近似梯度幅值）
    # 精确计算：sqrt(Gx^2 + Gy^2)
    # 近似计算：|Gx| + |Gy|（计算更快）
    sobel_combined = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)

    # 使用 Scharr 算子（ksize=-1 时自动使用 Scharr 核）
    # Scharr 是 Sobel 的改进，对 3x3 核更准确
    scharr_x = cv2.Sobel(image_gray, cv2.CV_64F, dx=1, dy=0, ksize=-1)
    scharr_y = cv2.Sobel(image_gray, cv2.CV_64F, dx=0, dy=1, ksize=-1)
    scharr_x_abs = cv2.convertScaleAbs(scharr_x)
    scharr_y_abs = cv2.convertScaleAbs(scharr_y)
    scharr_combined = cv2.addWeighted(scharr_x_abs, 0.5, scharr_y_abs, 0.5, 0)

    print(f"    Sobel 边缘像素均值: {sobel_combined.mean():.2f}")
    print(f"    Scharr 边缘像素均值: {scharr_combined.mean():.2f}")

    # 显示结果
    cv2.namedWindow("Gray Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Sobel X (Vertical Edges)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Sobel Y (Horizontal Edges)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Sobel Combined", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Scharr Combined", cv2.WINDOW_NORMAL)

    cv2.imshow("Gray Image", image_gray)
    cv2.imshow("Sobel X (Vertical Edges)", sobel_x_abs)
    cv2.imshow("Sobel Y (Horizontal Edges)", sobel_y_abs)
    cv2.imshow("Sobel Combined", sobel_combined)
    cv2.imshow("Scharr Combined", scharr_combined)


def laplacian_demo(image_gray: np.ndarray) -> None:
    """
    Laplacian 算子边缘检测演示

    Laplacian 算子是二阶导数算子，检测图像灰度变化率的变化。
    边缘处二阶导数为零（零交叉点）。

    常见的 Laplacian 核：
        [[0,  1, 0],
         [1, -4, 1],
         [0,  1, 0]]

    或包含对角线的版本：
        [[1,  1, 1],
         [1, -8, 1],
         [1,  1, 1]]

    特点：
    - 对噪声非常敏感（二阶导数放大噪声）
    - 通常需要先做高斯滤波去噪
    - 可以检测各个方向的边缘

    参数
    ----
    image_gray : np.ndarray
        输入灰度图像
    """
    print("\n  Laplacian 算子边缘检测：")
    print("    原理：计算图像灰度的二阶导数，零交叉点为边缘")
    print("    特点：对噪声敏感，通常需先高斯平滑")

    # 直接对原图做 Laplacian
    laplacian_raw = cv2.Laplacian(image_gray, cv2.CV_64F, ksize=3)
    laplacian_raw_abs = cv2.convertScaleAbs(laplacian_raw)

    # 先高斯模糊再做 Laplacian（LoG - Laplacian of Gaussian）
    blurred = cv2.GaussianBlur(image_gray, (5, 5), sigmaX=0)
    laplacian_log = cv2.Laplacian(blurred, cv2.CV_64F, ksize=3)
    laplacian_log_abs = cv2.convertScaleAbs(laplacian_log)

    print(f"    直接 Laplacian 边缘均值: {laplacian_raw_abs.mean():.2f}")
    print(f"    LoG (先高斯再拉普拉斯) 边缘均值: {laplacian_log_abs.mean():.2f}")

    # 显示结果
    cv2.namedWindow("Original Gray", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Laplacian (Raw)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Laplacian (LoG)", cv2.WINDOW_NORMAL)

    cv2.imshow("Original Gray", image_gray)
    cv2.imshow("Laplacian (Raw)", laplacian_raw_abs)
    cv2.imshow("Laplacian (LoG)", laplacian_log_abs)


def canny_demo(image_gray: np.ndarray) -> None:
    """
    Canny 边缘检测演示

    Canny 边缘检测是 John F. Canny 于 1986 年提出的多阶段边缘检测算法，
    是目前最经典、效果最好的边缘检测算法之一。

    Canny 算法的四个步骤：
    1. 高斯滤波：去除噪声
    2. 计算梯度幅值和方向：使用 Sobel 算子
    3. 非极大值抑制（NMS）：细化边缘，只保留局部最大值
    4. 双阈值检测：用高低两个阈值确定强边缘和弱边缘
       - 高于高阈值：强边缘，保留
       - 低于低阈值：非边缘，丢弃
       - 在两阈值之间：弱边缘，仅当与强边缘相连时保留

    参数：
    - threshold1: 低阈值
    - threshold2: 高阈值（推荐为低阈值的 2-3 倍）
    - apertureSize: Sobel 核大小，默认为 3
    - L2gradient: 是否使用更精确的 L2 范数计算梯度

    参数
    ----
    image_gray : np.ndarray
        输入灰度图像
    """
    print("\n  Canny 边缘检测：")
    print("    原理：多阶段算法（高斯滤波+梯度计算+NMS+双阈值）")
    print("    特点：检测精度高，边缘连续，假边缘少")

    # 不同阈值的对比
    # 阈值越小，检测到的边缘越多
    canny_low = cv2.Canny(image_gray, threshold1=50, threshold2=100)
    canny_mid = cv2.Canny(image_gray, threshold1=100, threshold2=200)
    canny_high = cv2.Canny(image_gray, threshold1=150, threshold2=300)

    # 使用 L2 梯度（更精确，但计算稍慢）
    canny_l2 = cv2.Canny(image_gray, threshold1=100, threshold2=200, L2gradient=True)

    # 统计边缘像素数
    print(f"    低阈值(50/100)  边缘像素数: {cv2.countNonZero(canny_low)}")
    print(f"    中阈值(100/200) 边缘像素数: {cv2.countNonZero(canny_mid)}")
    print(f"    高阈值(150/300) 边缘像素数: {cv2.countNonZero(canny_high)}")
    print(f"    L2梯度(100/200) 边缘像素数: {cv2.countNonZero(canny_l2)}")

    # 显示结果
    cv2.namedWindow("Gray Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Canny (50/100)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Canny (100/200)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Canny (150/300)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Canny (L2 gradient)", cv2.WINDOW_NORMAL)

    cv2.imshow("Gray Image", image_gray)
    cv2.imshow("Canny (50/100)", canny_low)
    cv2.imshow("Canny (100/200)", canny_mid)
    cv2.imshow("Canny (150/300)", canny_high)
    cv2.imshow("Canny (L2 gradient)", canny_l2)


def edge_detection_comparison(image_gray: np.ndarray) -> None:
    """
    各种边缘检测方法的对比

    参数
    ----
    image_gray : np.ndarray
        输入灰度图像
    """
    print("\n  各种边缘检测方法对比：")

    # Sobel
    sobel_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_x_abs = cv2.convertScaleAbs(sobel_x)
    sobel_y_abs = cv2.convertScaleAbs(sobel_y)
    sobel_result = cv2.addWeighted(sobel_x_abs, 0.5, sobel_y_abs, 0.5, 0)

    # Laplacian
    laplacian = cv2.Laplacian(image_gray, cv2.CV_64F, ksize=3)
    laplacian_result = cv2.convertScaleAbs(laplacian)

    # Canny
    canny_result = cv2.Canny(image_gray, 100, 200)

    # 四图合一显示
    # 将灰度图也转为三通道以便拼接
    gray_3ch = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    sobel_3ch = cv2.cvtColor(sobel_result, cv2.COLOR_GRAY2BGR)
    laplacian_3ch = cv2.cvtColor(laplacian_result, cv2.COLOR_GRAY2BGR)
    canny_3ch = cv2.cvtColor(canny_result, cv2.COLOR_GRAY2BGR)

    # 添加文字标签
    cv2.putText(gray_3ch, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(sobel_3ch, "Sobel", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(laplacian_3ch, "Laplacian", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(canny_3ch, "Canny", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 2x2 拼接
    top_row = np.hstack([gray_3ch, sobel_3ch])
    bottom_row = np.hstack([laplacian_3ch, canny_3ch])
    comparison = np.vstack([top_row, bottom_row])

    cv2.namedWindow("Edge Detection Comparison", cv2.WINDOW_NORMAL)
    cv2.imshow("Edge Detection Comparison", comparison)

    cv2.imwrite("output_edge_comparison.jpg", comparison)
    print("  对比图已保存为 output_edge_comparison.jpg")


def main():
    """
    主函数：依次运行各种边缘检测示例
    """
    test_image = "test.jpg"
    img = cv2.imread(test_image, cv2.IMREAD_COLOR)

    if img is None:
        print(f"错误：无法读取图像 {test_image}")
        return

    # 转为灰度图（边缘检测一般在灰度图上进行）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("=" * 50)
    print("OpenCV 边缘检测示例")
    print("=" * 50)

    print("\n[1/4] Sobel 算子边缘检测...")
    sobel_demo(gray)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[2/4] Laplacian 算子边缘检测...")
    laplacian_demo(gray)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[3/4] Canny 边缘检测...")
    canny_demo(gray)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[4/4] 各种方法对比...")
    edge_detection_comparison(gray)
    print("  按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n所有示例运行完毕！")


if __name__ == "__main__":
    main()
