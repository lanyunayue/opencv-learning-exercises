# -*- coding: utf-8 -*-
"""
角点检测示例
============================================
本脚本演示 OpenCV 中常用的角点检测方法，包括：
1. Harris 角点检测（cv2.cornerHarris）
2. Shi-Tomasi 角点检测（cv2.goodFeaturesToTrack）
3. 亚像素级角点检测（cv2.cornerSubPix）

角点的定义：
- 图像中灰度梯度在两个方向上都有较大变化的点
- 边缘的交点
- 局部邻域内具有两个不同方向的边缘的点

Harris 角点检测原理：
- 计算图像在 x 和 y 方向的梯度
- 构造结构矩阵 M = [[Ixx, Ixy], [Ixy, Iyy]]
- 计算响应值 R = det(M) - k * trace(M)^2
- R 大于阈值则认为是角点

作者：巫伟鑫
日期：2025-03

============================================
学习心得与踩坑记录：

1. 什么是角点？
   角点就是两条边的交点，或者说在两个方向上灰度变化都很大的点。
   直观理解：把一个小窗口在图像上移动，如果往哪个方向移变化都很大，
   那这个地方大概率就是角点。

2. Harris 角点检测的核心思想
   用结构矩阵 M 的两个特征值来判断：
   - 两个特征值都大 → 角点
   - 一个大一个小 → 边缘
   - 两个都小 → 平坦区域
   响应值 R = det(M) - k * trace(M)^2，R 很大就是角点。

3. Shi-Tomasi 比 Harris 好在哪里？
   Shi-Tomasi 直接用两个特征值中较小的那个作为响应值（R = min(λ1, λ2)），
   只要较小的特征值大于阈值，就是角点。这样更稳健，检测效果更好。
   实际使用中，goodFeaturesToTrack 用得比 cornerHarris 多得多。

4. 阈值怎么选？
   Harris 的阈值通常设为最大响应值的 1%~10%，根据实际图片调整。
   Shi-Tomasi 的 qualityLevel 一般设 0.01 或 0.001，值越小检测到的角点越多。

5. 亚像素级角点有什么用？
   普通角点检测得到的是整数坐标，但实际角点位置可能在两个像素之间。
   在需要高精度的场景（比如相机标定、三维重建），亚像素精度就很重要了。
   cornerSubPix 可以把角点精度优化到小数点后几位。
============================================
"""

import cv2
import numpy as np


def harris_corner_demo(image: np.ndarray) -> None:
    """
    Harris 角点检测演示

    cv2.cornerHarris() 参数：
    - src: 输入图像（灰度图，float32 类型）
    - blockSize: 角点检测中考虑的邻域大小
    - ksize: Sobel 核的大小
    - k: Harris 检测自由参数，通常取 0.04 ~ 0.06

    响应值 R 的含义：
    - R 很大正值：角点
    - |R| 很小：平坦区域
    - R 很大负值：边缘

    参数
    ----
    image : np.ndarray
        输入彩色图像
    """
    print("  Harris 角点检测：")
    print("    原理：基于结构矩阵的特征值判断角点")

    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)  # 必须转换为 float32

    # Harris 角点检测
    # blockSize=2: 2x2 邻域
    # ksize=3: Sobel 核大小
    # k=0.04: Harris 参数
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    print(f"    响应值范围: [{dst.min():.4f}, {dst.max():.4f}]")
    print(f"    响应值均值: {dst.mean():.6f}")

    # 膨胀结果，使角点更明显（可选）
    dst_dilated = cv2.dilate(dst, None)

    # 设置阈值，标记角点
    # 阈值通常设为最大值的某个百分比
    threshold_ratio = 0.01
    result = image.copy()
    result[dst_dilated > threshold_ratio * dst_dilated.max()] = [0, 0, 255]  # 红色标记

    # 统计角点数量
    corner_count = np.sum(dst_dilated > threshold_ratio * dst_dilated.max())
    print(f"    检测到的角点数量（阈值={threshold_ratio}）: {corner_count}")

    # 不同阈值的对比
    result_001 = image.copy()
    result_001[dst_dilated > 0.01 * dst_dilated.max()] = [0, 0, 255]

    result_005 = image.copy()
    result_005[dst_dilated > 0.05 * dst_dilated.max()] = [0, 0, 255]

    result_01 = image.copy()
    result_01[dst_dilated > 0.1 * dst_dilated.max()] = [0, 0, 255]

    print(f"    阈值 1%  时角点数: {np.sum(dst_dilated > 0.01 * dst_dilated.max())}")
    print(f"    阈值 5%  时角点数: {np.sum(dst_dilated > 0.05 * dst_dilated.max())}")
    print(f"    阈值 10% 时角点数: {np.sum(dst_dilated > 0.1 * dst_dilated.max())}")

    # 显示结果
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Harris Response", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Harris Corners (1%)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Harris Corners (5%)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Harris Corners (10%)", cv2.WINDOW_NORMAL)

    # 归一化响应值以便显示
    dst_display = cv2.normalize(dst, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.imshow("Original", image)
    cv2.imshow("Harris Response", dst_display)
    cv2.imshow("Harris Corners (1%)", result_001)
    cv2.imshow("Harris Corners (5%)", result_005)
    cv2.imshow("Harris Corners (10%)", result_01)


def shi_tomasi_corner_demo(image: np.ndarray) -> None:
    """
    Shi-Tomasi 角点检测演示

    Shi-Tomasi 是 Harris 角点检测的改进版本，也叫 "Good Features to Track"。

    与 Harris 的区别：
    - Harris: R = min(lambda1, lambda2) - k * (lambda1 + lambda2)^2
    - Shi-Tomasi: R = min(lambda1, lambda2)
    即 Shi-Tomasi 直接取两个特征值中较小的那个作为响应值，
    只要较小的特征值大于阈值，就认为是角点。

    cv2.goodFeaturesToTrack() 参数：
    - image: 输入图像（8位或32位浮点单通道）
    - maxCorners: 返回的最大角点数，0 表示不限制
    - qualityLevel: 质量等级阈值，角点响应值小于 qualityLevel*max_response 会被丢弃
    - minDistance: 角点之间的最小距离（像素）
    - blockSize: 计算梯度协方差矩阵的邻域大小
    - useHarrisDetector: 是否使用 Harris 检测器（默认 False，即 Shi-Tomasi）
    - k: Harris 参数（只有 useHarrisDetector=True 时有效）

    参数
    ----
    image : np.ndarray
        输入彩色图像
    """
    print("\n  Shi-Tomasi 角点检测：")
    print("    原理：取特征值的较小值作为角点响应，比 Harris 更稳健")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Shi-Tomasi 角点检测
    # 最多返回 100 个角点，质量等级 0.01，角点间距 10 像素
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=10,
        blockSize=3
    )

    print(f"    检测到的角点数量: {len(corners)}")
    print(f"    前 5 个角点坐标：")
    for i, corner in enumerate(corners[:5]):
        x, y = corner.ravel()
        print(f"      角点 {i+1}: ({x:.1f}, {y:.1f})")

    # 绘制角点
    result = image.copy()
    if corners is not None:
        corners = np.int0(corners)
        for i, corner in enumerate(corners):
            x, y = corner.ravel()
            # 画圆标记角点
            cv2.circle(result, (x, y), 3, (0, 255, 0), -1)  # 绿色实心圆
            # 可选：画外接圆
            cv2.circle(result, (x, y), 8, (0, 0, 255), 1)  # 红色空心圆

    # 不同参数的对比
    # 更多角点
    corners_more = cv2.goodFeaturesToTrack(gray, 500, 0.001, 5, blockSize=3)
    result_more = image.copy()
    if corners_more is not None:
        corners_more = np.int0(corners_more)
        for corner in corners_more:
            x, y = corner.ravel()
            cv2.circle(result_more, (x, y), 2, (0, 255, 0), -1)

    print(f"    宽松参数下角点数量: {len(corners_more) if corners_more is not None else 0}")

    # 显示结果
    cv2.namedWindow("Shi-Tomasi Corners (100)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Shi-Tomasi Corners (More)", cv2.WINDOW_NORMAL)

    cv2.imshow("Shi-Tomasi Corners (100)", result)
    cv2.imshow("Shi-Tomasi Corners (More)", result_more)


def subpixel_corner_demo(image: np.ndarray) -> None:
    """
    亚像素级角点检测演示

    前面的角点检测得到的是整数坐标，但实际角点位置可能在像素之间。
    亚像素级角点检测可以得到浮点精度的角点坐标。

    原理：
    基于角点附近的灰度梯度信息，通过迭代或最小二乘法
    计算出亚像素精度的角点位置。

    cv2.cornerSubPix() 参数：
    - image: 输入图像
    - corners: 初始角点坐标（浮点型）
    - winSize: 搜索窗口大小的一半
    - zeroZone: 死区大小的一半（-1, -1 表示没有死区）
    - criteria: 迭代终止条件

    参数
    ----
    image : np.ndarray
        输入彩色图像
    """
    print("\n  亚像素级角点检测：")
    print("    原理：基于梯度信息迭代优化，获得亚像素精度")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 先用 Shi-Tomasi 检测整数级角点
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=50,
        qualityLevel=0.01,
        minDistance=20,
        blockSize=3
    )

    if corners is None:
        print("    未检测到角点，跳过亚像素检测")
        return

    print(f"    初始角点数量: {len(corners)}")

    # 定义迭代终止条件
    # 最大迭代次数 100 或精度达到 0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # 亚像素级角点检测
    # winSize=(5, 5): 搜索窗口为 11x11（2*5+1）
    # zeroZone=(-1, -1): 无死区
    corners_subpix = cv2.cornerSubPix(
        gray,
        corners.astype(np.float32),
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=criteria
    )

    # 对比整数和亚像素坐标
    print(f"    前 5 个角点的整数坐标 vs 亚像素坐标：")
    for i in range(min(5, len(corners))):
        int_x, int_y = corners[i].ravel()
        sub_x, sub_y = corners_subpix[i].ravel()
        print(f"      角点 {i+1}: 整数({int_x:.0f}, {int_y:.0f}) "
              f"-> 亚像素({sub_x:.3f}, {sub_y:.3f}) "
              f"偏移: ({abs(sub_x - int_x):.3f}, {abs(sub_y - int_y):.3f})")

    # 绘制结果
    result = image.copy()
    corners_int = np.int0(corners)
    for corner in corners_int:
        x, y = corner.ravel()
        cv2.circle(result, (x, y), 5, (0, 0, 255), 1)  # 红色：整数角点

    # 亚像素角点用绿色十字标记
    for corner in corners_subpix:
        x, y = corner.ravel()
        # 放大显示偏移（放大 10 倍以便观察）
        offset_x = int((x - int(x)) * 10)
        offset_y = int((y - int(y)) * 10)
        cv2.drawMarker(
            result,
            (int(x) * 1 if True else int(x)),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            10,
            1
        )

    cv2.namedWindow("Subpixel Corners", cv2.WINDOW_NORMAL)
    cv2.imshow("Subpixel Corners", result)


def create_corner_test_image() -> np.ndarray:
    """
    创建一张用于测试角点检测的图像
    包含棋盘格、矩形、圆形等多种形状

    返回
    ----
    np.ndarray
        测试图像
    """
    # 创建 400x400 的黑色图像
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    # 白色背景
    img[:] = [255, 255, 255]

    # 绘制一个黑色矩形（4 个角点）
    cv2.rectangle(img, (50, 50), (150, 150), (0, 0, 0), -1)

    # 绘制一个黑色三角形（3 个角点）
    pts = np.array([[300, 50], [350, 150], [250, 150]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (0, 0, 0))

    # 绘制棋盘格（多个角点）
    square_size = 30
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                x1 = 50 + j * square_size
                y1 = 200 + i * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)

    # 绘制一些线条（交叉点也是角点）
    cv2.line(img, (250, 250), (380, 380), (0, 0, 0), 2)
    cv2.line(img, (380, 250), (250, 380), (0, 0, 0), 2)

    return img


def main():
    """
    主函数：依次运行各种角点检测示例
    """
    # 先创建一张测试图像（有明显角点）
    test_img = create_corner_test_image()
    cv2.imwrite("test_corners.jpg", test_img)
    print("已创建测试图像 test_corners.jpg")

    print("=" * 50)
    print("OpenCV 角点检测示例")
    print("=" * 50)

    print("\n[1/3] Harris 角点检测...")
    harris_corner_demo(test_img)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[2/3] Shi-Tomasi 角点检测...")
    shi_tomasi_corner_demo(test_img)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[3/3] 亚像素级角点检测...")
    subpixel_corner_demo(test_img)
    print("  按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 也可以用真实图片测试
    real_image = "test.jpg"
    import os
    if os.path.exists(real_image):
        print("\n检测到 test.jpg，用真实图片再演示一次 Shi-Tomasi...")
        real_img = cv2.imread(real_image)
        if real_img is not None:
            shi_tomasi_corner_demo(real_img)
            print("  按任意键退出...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print("\n所有示例运行完毕！")


if __name__ == "__main__":
    main()
