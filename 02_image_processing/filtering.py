# -*- coding: utf-8 -*-
"""
图像滤波示例
============================================
本脚本演示 OpenCV 中常用的图像滤波方法，包括：
1. 均值滤波（cv2.blur / boxFilter）
2. 高斯滤波（cv2.GaussianBlur）
3. 中值滤波（cv2.medianBlur）
4. 双边滤波（cv2.bilateralFilter）
5. 自定义卷积核滤波（cv2.filter2D）

各种滤波的适用场景：
- 均值滤波：简单快速，适合轻微的均匀噪声
- 高斯滤波：保留边缘信息的平滑，最常用的去噪方法
- 中值滤波：对椒盐噪声（salt-and-pepper）效果最好
- 双边滤波：在平滑的同时保留边缘，用于美颜等场景

作者：巫伟鑫
日期：2024-11-20

============================================
学习心得与踩坑记录：

1. 各种滤波的适用场景
   - 高斯滤波：最常用，去噪的同时保留边缘信息，一般作为预处理第一步
   - 中值滤波：对椒盐噪声效果最好（就是那种黑白点噪点）
   - 均值滤波：最简单最快，但会模糊边缘，用得不多
   - 双边滤波：去噪的同时保留边缘，但速度慢，常用于美颜磨皮

2. 核大小的选择
   核越大，平滑效果越强，但图像也越模糊，计算量也越大。
   一般从 3x3、5x5 开始试，根据实际效果调整。
   注意：高斯滤波的核大小必须是奇数！

3. sigma 和核大小的关系
   在 GaussianBlur 中，如果指定了 ksize，sigmaX=0 的话，sigma 会根据
   核大小自动计算。公式大概是：sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8

4. 关于椒盐噪声
   椒盐噪声就是图像中随机出现的纯黑或纯白的点，像撒了盐和胡椒一样。
   这种噪声用中值滤波效果最好，因为噪声点是极值，取中值直接就过滤掉了。
   用高斯滤波处理椒盐噪声效果就很差。

5. 双边滤波为什么能保留边缘？
   因为它不仅考虑了空间距离（空间域高斯），还考虑了像素值差异（值域高斯）。
   在边缘处，两边的像素值差异很大，值域高斯的权重很小，所以不会把边缘
   两边的像素平均到一起，边缘就保留下来了。
============================================
"""

import cv2
import numpy as np


def add_noise(image: np.ndarray, noise_type: str = "gaussian") -> np.ndarray:
    """
    给图像添加噪声，用于测试滤波效果

    参数
    ----
    image : np.ndarray
        输入图像
    noise_type : str
        噪声类型："gaussian"（高斯噪声）或 "salt_pepper"（椒盐噪声）

    返回
    ----
    np.ndarray
        添加噪声后的图像
    """
    if noise_type == "gaussian":
        # 添加高斯噪声
        mean = 0
        var = 25  # 方差，控制噪声强度
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma, image.shape)
        noisy = image.astype(np.float32) + gaussian
        # 确保像素值在 [0, 255] 范围内
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy

    elif noise_type == "salt_pepper":
        # 添加椒盐噪声
        noisy = image.copy()
        amount = 0.02  # 噪声点比例
        # 盐噪声（白色点）
        num_salt = np.ceil(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 255
        # 椒噪声（黑色点）
        num_pepper = np.ceil(amount * image.size * 0.5)
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2]]
        noisy[coords[0], coords[1]] = 0
        return noisy

    else:
        raise ValueError(f"不支持的噪声类型: {noise_type}")


def mean_blur_demo(image: np.ndarray) -> None:
    """
    均值滤波演示

    均值滤波用一个区域内所有像素的平均值代替中心像素值。
    核越大，平滑效果越强，但图像也越模糊。

    公式：
        dst(x, y) = (1 / ksize.width * ksize.height) * sum(neighborhood)

    参数
    ----
    image : np.ndarray
        输入图像（带噪声）
    """
    print("  均值滤波（Mean Blur）：")
    print("    原理：用邻域像素的平均值替代中心像素")
    print("    特点：简单快速，但会模糊边缘")

    # 不同核大小的对比
    blur_3 = cv2.blur(image, (3, 3))    # 3x3 核
    blur_5 = cv2.blur(image, (5, 5))    # 5x5 核
    blur_9 = cv2.blur(image, (9, 9))    # 9x9 核

    print(f"    3x3 核处理后图像方差: {np.var(blur_3):.2f}")
    print(f"    5x5 核处理后图像方差: {np.var(blur_5):.2f}")
    print(f"    9x9 核处理后图像方差: {np.var(blur_9):.2f}")

    # 显示结果
    cv2.namedWindow("Noisy Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mean Blur 3x3", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mean Blur 5x5", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mean Blur 9x9", cv2.WINDOW_NORMAL)

    cv2.imshow("Noisy Image", image)
    cv2.imshow("Mean Blur 3x3", blur_3)
    cv2.imshow("Mean Blur 5x5", blur_5)
    cv2.imshow("Mean Blur 9x9", blur_9)


def gaussian_blur_demo(image: np.ndarray) -> None:
    """
    高斯滤波演示

    高斯滤波是最常用的图像平滑方法，它对中心附近的像素赋予更高的权重。
    相比均值滤波，高斯滤波在平滑的同时能更好地保留边缘信息。

    二维高斯函数：
        G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))

    参数
    ----
    image : np.ndarray
        输入图像（带噪声）
    """
    print("\n  高斯滤波（Gaussian Blur）：")
    print("    原理：用高斯加权的邻域平均值替代中心像素")
    print("    特点：平滑效果自然，边缘保留较好，最常用的去噪方法")

    # 不同 sigma 的对比
    # ksize 设为 (0, 0) 时，核大小由 sigma 自动计算
    gauss_1 = cv2.GaussianBlur(image, (0, 0), sigmaX=1)
    gauss_3 = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    gauss_5 = cv2.GaussianBlur(image, (0, 0), sigmaX=5)

    # 也可以指定核大小（必须是奇数）
    gauss_5x5 = cv2.GaussianBlur(image, (5, 5), sigmaX=0)

    print(f"    sigma=1  处理后图像方差: {np.var(gauss_1):.2f}")
    print(f"    sigma=3  处理后图像方差: {np.var(gauss_3):.2f}")
    print(f"    sigma=5  处理后图像方差: {np.var(gauss_5):.2f}")

    # 显示结果
    cv2.namedWindow("Gaussian sigma=1", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Gaussian sigma=3", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Gaussian sigma=5", cv2.WINDOW_NORMAL)

    cv2.imshow("Gaussian sigma=1", gauss_1)
    cv2.imshow("Gaussian sigma=3", gauss_3)
    cv2.imshow("Gaussian sigma=5", gauss_5)


def median_blur_demo(image: np.ndarray) -> None:
    """
    中值滤波演示

    中值滤波用邻域像素的中值（排序后的中间值）代替中心像素值。
    对椒盐噪声（随机的黑白点）效果非常好，因为噪声点是极值，
    取中值可以直接过滤掉。

    特点：
    - 对椒盐噪声效果最好
    - 能保留边缘（不像均值滤波那样模糊边缘）
    - 计算速度相对较慢（需要排序）

    参数
    ----
    image : np.ndarray
        输入图像（带椒盐噪声）
    """
    print("\n  中值滤波（Median Blur）：")
    print("    原理：用邻域像素的中值替代中心像素")
    print("    特点：对椒盐噪声效果极好，能保留边缘")

    # 不同核大小的对比（核大小必须是大于 1 的奇数）
    median_3 = cv2.medianBlur(image, 3)
    median_5 = cv2.medianBlur(image, 5)
    median_7 = cv2.medianBlur(image, 7)

    print(f"    3x3 核处理后图像方差: {np.var(median_3):.2f}")
    print(f"    5x5 核处理后图像方差: {np.var(median_5):.2f}")
    print(f"    7x7 核处理后图像方差: {np.var(median_7):.2f}")

    # 显示结果
    cv2.namedWindow("Salt & Pepper Noise", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Median Blur 3x3", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Median Blur 5x5", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Median Blur 7x7", cv2.WINDOW_NORMAL)

    cv2.imshow("Salt & Pepper Noise", image)
    cv2.imshow("Median Blur 3x3", median_3)
    cv2.imshow("Median Blur 5x5", median_5)
    cv2.imshow("Median Blur 7x7", median_7)


def bilateral_filter_demo(image: np.ndarray) -> None:
    """
    双边滤波演示

    双边滤波（Bilateral Filter）同时考虑空间距离和像素值差异，
    因此在平滑图像的同时能很好地保留边缘。

    两个高斯函数的乘积：
    - 空间域高斯：距离越近权重越大（和高斯滤波一样）
    - 值域高斯：像素值越接近权重越大（边缘处像素差异大，权重小）

    参数：
    - d: 滤波时考虑的邻域直径
    - sigmaColor: 颜色空间的 sigma（值域）
    - sigmaSpace: 坐标空间的 sigma（空间域）

    参数
    ----
    image : np.ndarray
        输入图像
    """
    print("\n  双边滤波（Bilateral Filter）：")
    print("    原理：同时考虑空间距离和像素值差异的加权滤波")
    print("    特点：平滑的同时保留边缘，常用于美颜、磨皮等场景")

    # 双边滤波参数
    # d 越大，考虑的邻域范围越大
    # sigmaColor 越大，颜色差异的容忍度越高
    # sigmaSpace 越大，空间范围越大
    bilateral_1 = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    bilateral_2 = cv2.bilateralFilter(image, d=15, sigmaColor=100, sigmaSpace=100)

    # 与高斯滤波对比（同样核大小）
    gaussian_compare = cv2.GaussianBlur(image, (15, 15), sigmaX=0)

    print(f"    双边滤波(d=9)  后图像方差: {np.var(bilateral_1):.2f}")
    print(f"    双边滤波(d=15) 后图像方差: {np.var(bilateral_2):.2f}")
    print(f"    高斯滤波(15x15)后图像方差: {np.var(gaussian_compare):.2f}")

    # 显示结果
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Bilateral d=9", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Bilateral d=15", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Gaussian 15x15", cv2.WINDOW_NORMAL)

    cv2.imshow("Original", image)
    cv2.imshow("Bilateral d=9", bilateral_1)
    cv2.imshow("Bilateral d=15", bilateral_2)
    cv2.imshow("Gaussian 15x15", gaussian_compare)


def custom_filter_demo(image: np.ndarray) -> None:
    """
    自定义卷积核滤波演示

    使用 cv2.filter2D() 可以自定义卷积核，实现各种特殊效果。

    常见的卷积核：
    - 锐化核：增强边缘和细节
    - 浮雕核：产生立体浮雕效果
    - Sobel 核：边缘检测
    - 拉普拉斯核：检测图像变化率

    参数
    ----
    image : np.ndarray
        输入图像
    """
    print("\n  自定义卷积核（Custom Filter）：")
    print("    使用 cv2.filter2D() 实现自定义卷积操作")

    # 锐化核（Sharpen）
    # 增强图像的高频成分，使边缘更清晰
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float32)

    # 更强的锐化核
    sharpen_kernel_strong = np.array([
        [-1, -1, -1],
        [-1, 9, -1],
        [-1, -1, -1]
    ], dtype=np.float32)

    # 浮雕核（Emboss）
    emboss_kernel = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ], dtype=np.float32)

    # 应用卷积
    # ddepth=-1 表示输出图像与输入图像深度相同
    sharpen = cv2.filter2D(image, ddepth=-1, kernel=sharpen_kernel)
    sharpen_strong = cv2.filter2D(image, ddepth=-1, kernel=sharpen_kernel_strong)
    emboss = cv2.filter2D(image, ddepth=-1, kernel=emboss_kernel)

    # 显示结果
    cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Sharpen (Weak)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Sharpen (Strong)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Emboss Effect", cv2.WINDOW_NORMAL)

    cv2.imshow("Original Image", image)
    cv2.imshow("Sharpen (Weak)", sharpen)
    cv2.imshow("Sharpen (Strong)", sharpen_strong)
    cv2.imshow("Emboss Effect", emboss)


def main():
    """
    主函数：依次运行各种滤波示例
    """
    test_image = "test.jpg"
    img = cv2.imread(test_image, cv2.IMREAD_COLOR)

    if img is None:
        print(f"错误：无法读取图像 {test_image}")
        return

    print("=" * 50)
    print("OpenCV 图像滤波示例")
    print("=" * 50)

    # 添加高斯噪声，用于测试去噪效果
    noisy_gaussian = add_noise(img, "gaussian")
    # 添加椒盐噪声，用于测试中值滤波
    noisy_sp = add_noise(img, "salt_pepper")

    print("\n[1/5] 均值滤波演示...")
    mean_blur_demo(noisy_gaussian)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[2/5] 高斯滤波演示...")
    gaussian_blur_demo(noisy_gaussian)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[3/5] 中值滤波演示（椒盐噪声）...")
    median_blur_demo(noisy_sp)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[4/5] 双边滤波演示...")
    bilateral_filter_demo(img)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[5/5] 自定义卷积核演示...")
    custom_filter_demo(img)
    print("  按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存噪声图像，方便后续使用
    cv2.imwrite("output_noisy_gaussian.jpg", noisy_gaussian)
    cv2.imwrite("output_noisy_saltpepper.jpg", noisy_sp)
    print("\n噪声图像已保存，方便后续练习使用。")
    print("\n所有示例运行完毕！")


if __name__ == "__main__":
    main()
