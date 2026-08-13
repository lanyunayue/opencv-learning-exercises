# -*- coding: utf-8 -*-
"""
轮廓检测与霍夫变换示例
============================================
本脚本演示 OpenCV 中的轮廓检测和霍夫变换，包括：

轮廓检测：
1. 轮廓查找（cv2.findContours）
2. 轮廓绘制（cv2.drawContours）
3. 轮廓特征（面积、周长、外接矩形、最小外接圆等）
4. 轮廓近似（cv2.approxPolyDP）

霍夫变换：
1. 霍夫直线变换（cv2.HoughLines）
2. 概率霍夫直线变换（cv2.HoughLinesP）
3. 霍夫圆变换（cv2.HoughCircles）

作者：巫伟鑫
日期：2024-11-20
"""

import cv2
import numpy as np


def find_and_draw_contours(image: np.ndarray) -> tuple:
    """
    查找并绘制轮廓

    cv2.findContours() 参数：
    - image: 输入二值图像（通常是边缘检测的结果）
    - mode: 轮廓检索模式
        cv2.RETR_EXTERNAL   - 只检测最外层轮廓
        cv2.RETR_LIST       - 检测所有轮廓，不建立等级关系
        cv2.RETR_CCOMP      - 检测所有轮廓，建立两级等级关系
        cv2.RETR_TREE       - 检测所有轮廓，建立完整的等级树
    - method: 轮廓近似方法
        cv2.CHAIN_APPROX_NONE      - 存储所有轮廓点
        cv2.CHAIN_APPROX_SIMPLE    - 压缩水平、垂直、对角线方向的点，只保留端点
        cv2.CHAIN_APPROX_TC89_L1   - Teh-Chin 链码近似算法
        cv2.CHAIN_APPROX_TC89_KCOS - Teh-Chin 链码近似算法

    返回值：
    - contours: 轮廓列表，每个轮廓是一个 numpy 数组
    - hierarchy: 轮廓的层次结构信息

    参数
    ----
    image : np.ndarray
        输入彩色图像

    返回
    ----
    tuple
        (contours, hierarchy, result_image)
    """
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二值化（Canny 边缘检测或阈值处理）
    edges = cv2.Canny(gray, 50, 150)

    # 查找轮廓
    # 注意：OpenCV 4.x 中 findContours 返回 (contours, hierarchy)
    # OpenCV 3.x 中返回 (image, contours, hierarchy)
    contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,      # 只检测最外层轮廓
        cv2.CHAIN_APPROX_SIMPLE  # 压缩冗余点
    )

    print(f"    检测到的轮廓数量: {len(contours)}")

    # 绘制轮廓
    # cv2.drawContours() 参数：
    #   - image: 要绘制的图像
    #   - contours: 轮廓列表
    #   - contourIdx: 要绘制的轮廓索引，-1 表示绘制所有
    #   - color: 轮廓颜色
    #   - thickness: 线宽
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # 用不同颜色绘制每个轮廓
    result_colorful = image.copy()
    for i, contour in enumerate(contours):
        # 生成随机颜色
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.drawContours(result_colorful, contours, i, color, 2)

    # 显示结果
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Edges (Canny)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Contours (Green)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Contours (Colorful)", cv2.WINDOW_NORMAL)

    cv2.imshow("Original", image)
    cv2.imshow("Edges (Canny)", edges)
    cv2.imshow("Contours (Green)", result)
    cv2.imshow("Contours (Colorful)", result_colorful)

    return contours, hierarchy, result


def contour_features(image: np.ndarray, contours: list) -> None:
    """
    轮廓特征计算

    常用的轮廓特征：
    - 面积：cv2.contourArea()
    - 周长：cv2.arcLength()
    - 外接矩形：cv2.boundingRect()
    - 最小外接矩形：cv2.minAreaRect()
    - 最小外接圆：cv2.minEnclosingCircle()
    - 椭圆拟合：cv2.fitEllipse()
    - 轮廓矩：cv2.moments()
    - 质心：由矩计算得到
    - 近似多边形：cv2.approxPolyDP()

    参数
    ----
    image : np.ndarray
        输入彩色图像
    contours : list
        轮廓列表
    """
    print("\n  轮廓特征：")

    result = image.copy()

    for i, contour in enumerate(contours):
        # 计算轮廓面积
        area = cv2.contourArea(contour)

        # 跳过太小的轮廓
        if area < 100:
            continue

        # 计算轮廓周长
        # 第二个参数 True 表示轮廓是闭合的
        perimeter = cv2.arcLength(contour, True)

        # 轮廓矩（Moments）
        M = cv2.moments(contour)

        # 计算质心
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx, cy = 0, 0

        # 外接矩形（正矩形，不考虑旋转）
        x, y, w, h = cv2.boundingRect(contour)

        # 最小外接矩形（考虑旋转）
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 最小外接圆
        (circle_x, circle_y), radius = cv2.minEnclosingCircle(contour)
        center = (int(circle_x), int(circle_y))
        radius = int(radius)

        # 轮廓近似（Douglas-Peucker 算法）
        # 第二个参数是 epsilon，控制近似精度（通常取周长的 1%-5%）
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)

        print(f"\n    轮廓 {i}:")
        print(f"      面积: {area:.2f}")
        print(f"      周长: {perimeter:.2f}")
        print(f"      质心: ({cx}, {cy})")
        print(f"      外接矩形: x={x}, y={y}, w={w}, h={h}")
        print(f"      外接矩形宽高比: {w/h:.2f}" if h > 0 else "      外接矩形宽高比: N/A")
        print(f"      最小外接圆半径: {radius}")
        print(f"      近似多边形顶点数: {len(approx)}")

        # 绘制外接矩形（绿色）
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 绘制最小外接矩形（红色）
        cv2.drawContours(result, [box], 0, (0, 0, 255), 2)

        # 绘制最小外接圆（蓝色）
        cv2.circle(result, center, radius, (255, 0, 0), 2)

        # 绘制质心（黄色十字）
        cv2.drawMarker(result, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 10, 2)

        # 绘制近似多边形（品红色）
        cv2.drawContours(result, [approx], 0, (255, 0, 255), 2)

    cv2.namedWindow("Contour Features", cv2.WINDOW_NORMAL)
    cv2.imshow("Contour Features", result)

    cv2.imwrite("output_contour_features.jpg", result)
    print("\n  轮廓特征图已保存为 output_contour_features.jpg")


def contour_approximation(image: np.ndarray, contours: list) -> None:
    """
    轮廓近似演示（多边形拟合）

    使用 Douglas-Peucker 算法将轮廓近似为更简单的多边形。
    通过调整 epsilon 参数，可以控制近似的精度。

    应用场景：
    - 形状识别（根据多边形顶点数判断形状）
    - 轮廓简化（减少数据量）

    参数
    ----
    image : np.ndarray
        输入彩色图像
    contours : list
        轮廓列表
    """
    print("\n  轮廓近似（多边形拟合）：")
    print("    原理：Douglas-Peucker 算法，用更少的点近似轮廓")

    result = image.copy()

    # 筛选较大的轮廓
    large_contours = [c for c in contours if cv2.contourArea(c) > 500]
    print(f"    大轮廓数量: {len(large_contours)}")

    for contour in large_contours:
        perimeter = cv2.arcLength(contour, True)

        # 不同 epsilon 值的近似效果
        epsilons = [0.01, 0.02, 0.05, 0.1]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

        for eps, color in zip(epsilons, colors):
            approx = cv2.approxPolyDP(contour, eps * perimeter, True)
            print(f"      epsilon={eps:.0%}: {len(approx)} 个顶点")

        # 用 epsilon=0.02 的近似做形状判断
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        num_vertices = len(approx)

        # 绘制近似多边形
        cv2.drawContours(result, [approx], 0, (0, 255, 0), 2)

        # 根据顶点数判断形状
        shape = "Unknown"
        if num_vertices == 3:
            shape = "Triangle"
        elif num_vertices == 4:
            # 判断是矩形还是正方形
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"
        elif num_vertices == 5:
            shape = "Pentagon"
        elif num_vertices == 6:
            shape = "Hexagon"
        else:
            shape = f"Polygon({num_vertices})"

        # 在质心位置标注形状
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            cv2.putText(result, shape, (cx - 30, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        print(f"    检测形状: {shape} ({num_vertices} 个顶点)")

    cv2.namedWindow("Contour Approximation", cv2.WINDOW_NORMAL)
    cv2.imshow("Contour Approximation", result)


def hough_lines_demo(image: np.ndarray) -> None:
    """
    霍夫直线变换演示

    霍夫变换（Hough Transform）是一种检测几何形状的方法，
    它通过将图像空间的点映射到参数空间，
    然后在参数空间中寻找峰值来检测形状。

    直线的极坐标表示：
        rho = x * cos(theta) + y * sin(theta)
    其中 rho 是原点到直线的距离，theta 是直线法线与 x 轴的夹角。

    cv2.HoughLines() 参数：
    - image: 输入二值图像（通常是边缘检测结果）
    - rho: 距离分辨率（像素）
    - theta: 角度分辨率（弧度）
    - threshold: 累加器阈值，只有大于阈值的才被认为是直线

    cv2.HoughLinesP() - 概率霍夫变换，速度更快，返回线段端点

    参数
    ----
    image : np.ndarray
        输入彩色图像
    """
    print("\n  霍夫直线变换：")
    print("    原理：将图像空间的点映射到参数空间，通过峰值检测直线")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # 标准霍夫变换
    # 返回值：[[rho, theta], ...]
    lines = cv2.HoughLines(edges, rho=1, theta=np.pi / 180, threshold=100)

    result_standard = image.copy()
    if lines is not None:
        print(f"    标准霍夫变换检测到 {len(lines)} 条直线")
        for i, line in enumerate(lines[:20]):  # 只画前 20 条
            rho, theta = line[0]
            # 将极坐标转换为直角坐标的两个点
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # 延长线段
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(result_standard, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 概率霍夫变换（更常用）
    # 返回值：[[x1, y1, x2, y2], ...]
    lines_p = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,   # 最小线段长度
        maxLineGap=10       # 线段之间的最大间隙
    )

    result_prob = image.copy()
    if lines_p is not None:
        print(f"    概率霍夫变换检测到 {len(lines_p)} 条线段")
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_prob, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 显示结果
    cv2.namedWindow("Edges", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Hough Lines (Standard)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Hough Lines (Probabilistic)", cv2.WINDOW_NORMAL)

    cv2.imshow("Edges", edges)
    cv2.imshow("Hough Lines (Standard)", result_standard)
    cv2.imshow("Hough Lines (Probabilistic)", result_prob)


def hough_circles_demo(image: np.ndarray) -> None:
    """
    霍夫圆变换演示

    霍夫圆变换使用霍夫梯度法（Hough Gradient Method）检测圆形。

    cv2.HoughCircles() 参数：
    - image: 输入灰度图像
    - method: 检测方法，目前只有 HOUGH_GRADIENT
    - dp: 累加器分辨率与图像分辨率的反比（1 表示相同，2 表示一半）
    - minDist: 检测到的圆的圆心之间的最小距离
    - param1: Canny 边缘检测的高阈值（低阈值为其一半）
    - param2: 累加器阈值，越小检测到的圆越多
    - minRadius: 最小圆半径
    - maxRadius: 最大圆半径

    参数
    ----
    image : np.ndarray
        输入彩色图像
    """
    print("\n  霍夫圆变换：")
    print("    原理：霍夫梯度法，检测圆形目标")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊去噪（霍夫圆变换对噪声敏感）
    gray_blurred = cv2.medianBlur(gray, 5)

    # 霍夫圆变换
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,        # 圆心之间的最小距离
        param1=50,         # Canny 高阈值
        param2=30,         # 累加器阈值（越小检测到的圆越多）
        minRadius=10,      # 最小半径
        maxRadius=100      # 最大半径
    )

    result = image.copy()

    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"    检测到 {len(circles[0])} 个圆")

        for i, circle in enumerate(circles[0]):
            x, y, r = circle
            print(f"      圆 {i+1}: 圆心=({x}, {y}), 半径={r}")

            # 绘制圆心（红色）
            cv2.circle(result, (x, y), 2, (0, 0, 255), 3)
            # 绘制圆周（绿色）
            cv2.circle(result, (x, y), r, (0, 255, 0), 2)
    else:
        print("    未检测到圆")

    cv2.namedWindow("Hough Circles", cv2.WINDOW_NORMAL)
    cv2.imshow("Hough Circles", result)


def create_shapes_image() -> np.ndarray:
    """
    创建一张包含各种几何形状的测试图像

    返回
    ----
    np.ndarray
        测试图像
    """
    img = np.ones((400, 500, 3), dtype=np.uint8) * 255  # 白色背景

    # 黑色矩形
    cv2.rectangle(img, (30, 30), (130, 130), (0, 0, 0), 2)

    # 黑色正方形
    cv2.rectangle(img, (170, 30), (270, 130), (0, 0, 0), 2)

    # 三角形
    pts = np.array([[350, 30], [420, 130], [280, 130]], np.int32)
    cv2.polylines(img, [pts], True, (0, 0, 0), 2)

    # 圆形
    cv2.circle(img, (80, 220), 50, (0, 0, 0), 2)

    # 椭圆
    cv2.ellipse(img, (220, 220), (60, 40), 0, 0, 360, (0, 0, 0), 2)

    # 五边形
    pts_pentagon = cv2.ellipse2Poly((370, 220), (50, 50), 0, 0, 360, 72)
    pts_pentagon = np.array(pts_pentagon[:5], np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts_pentagon], True, (0, 0, 0), 2)

    # 六边形
    pts_hex = cv2.ellipse2Poly((80, 340), (50, 50), 0, 0, 360, 60)
    pts_hex = np.array(pts_hex[:6], np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts_hex], True, (0, 0, 0), 2)

    # 一些随机直线
    cv2.line(img, (150, 300), (300, 380), (0, 0, 0), 2)
    cv2.line(img, (200, 350), (400, 320), (0, 0, 0), 2)
    cv2.line(img, (350, 280), (480, 370), (0, 0, 0), 2)

    # 几个实心圆（用于霍夫圆检测测试）
    cv2.circle(img, (420, 340), 25, (0, 0, 0), -1)
    cv2.circle(img, (470, 370), 15, (0, 0, 0), -1)

    return img


def main():
    """
    主函数：依次运行轮廓检测和霍夫变换示例
    """
    # 创建测试图像
    test_img = create_shapes_image()
    cv2.imwrite("test_shapes.jpg", test_img)
    print("已创建测试图像 test_shapes.jpg")

    print("=" * 50)
    print("OpenCV 轮廓检测与霍夫变换示例")
    print("=" * 50)

    print("\n[1/5] 轮廓查找与绘制...")
    contours, hierarchy, _ = find_and_draw_contours(test_img)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[2/5] 轮廓特征计算...")
    contour_features(test_img, contours)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[3/5] 轮廓近似与形状识别...")
    contour_approximation(test_img, contours)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[4/5] 霍夫直线变换...")
    hough_lines_demo(test_img)
    print("  按任意键继续...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n[5/5] 霍夫圆变换...")
    hough_circles_demo(test_img)
    print("  按任意键退出...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\n所有示例运行完毕！")


if __name__ == "__main__":
    main()
