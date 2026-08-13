# OpenCV 学习练习题与示例代码

> 作者：巫伟鑫
> 日期：2024-11-20

## 项目简介

本仓库是 OpenCV（Open Source Computer Vision Library）的 Python 版本学习练习题和示例代码集合。项目按照学习路径分为三个模块：基本操作、图像处理、特征检测，每个模块都包含可直接运行的示例脚本和详细的中文注释，适合初学者系统性地学习计算机视觉基础。

## 功能特性

- **零基础入门**：从图像读写开始，循序渐进
- **代码即文档**：每个示例都有详细的中文注释说明原理
- **模块化组织**：按知识点分类，便于查找和复习
- **可直接运行**：所有脚本均可独立运行，快速验证效果
- **覆盖核心 API**：涵盖 OpenCV 最常用的 20+ 个核心函数

## 目录结构

```
opencv-learning-exercises/
├── 01_basic_operations/      # 基本操作模块
│   ├── image_io.py           # 图像读写与像素操作
│   └── color_space.py        # 颜色空间转换
├── 02_image_processing/      # 图像处理模块
│   ├── filtering.py          # 图像滤波（均值、高斯、中值）
│   ├── edge_detection.py     # 边缘检测（Sobel、Canny、Laplacian）
│   └── threshold.py          # 阈值分割与形态学操作
├── 03_feature_detection/     # 特征检测模块
│   ├── corners.py            # 角点检测（Harris、Shi-Tomasi）
│   └── contours.py           # 轮廓检测与霍夫变换
├── requirements.txt          # 依赖包列表
├── LICENSE                   # MIT 许可证
└── README.md                 # 项目说明
```

## 环境要求

- Python 3.8+
- OpenCV 4.x
- NumPy
- Matplotlib（用于显示图像）

## 安装使用

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/opencv-learning-exercises.git
cd opencv-learning-exercises
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行示例

每个 Python 脚本都可以直接运行，例如：

```bash
# 运行图像读写示例
python 01_basic_operations/image_io.py

# 运行边缘检测示例
python 02_image_processing/edge_detection.py

# 运行角点检测示例
python 03_feature_detection/corners.py
```

> 注意：脚本默认会读取 `test.jpg` 作为输入图片，请在运行前准备一张测试图片放在脚本同级目录，或修改脚本中的图片路径。

## 模块详解

### 01 - 基本操作

| 文件 | 知识点 | 核心函数 |
|------|--------|----------|
| image_io.py | 图像读取、显示、保存，像素级操作 | `cv2.imread()`, `cv2.imshow()`, `cv2.imwrite()` |
| color_space.py | BGR/RGB/HSV/GRAY 颜色空间转换 | `cv2.cvtColor()`, `cv2.inRange()` |

### 02 - 图像处理

| 文件 | 知识点 | 核心函数 |
|------|--------|----------|
| filtering.py | 均值滤波、高斯滤波、中值滤波、双边滤波 | `cv2.blur()`, `cv2.GaussianBlur()`, `cv2.medianBlur()` |
| edge_detection.py | Sobel、Laplacian、Canny 边缘检测 | `cv2.Sobel()`, `cv2.Laplacian()`, `cv2.Canny()` |
| threshold.py | 二值化、自适应阈值、形态学操作 | `cv2.threshold()`, `cv2.adaptiveThreshold()`, `cv2.morphologyEx()` |

### 03 - 特征检测

| 文件 | 知识点 | 核心函数 |
|------|--------|----------|
| corners.py | Harris 角点、Shi-Tomasi 角点检测 | `cv2.cornerHarris()`, `cv2.goodFeaturesToTrack()` |
| contours.py | 轮廓查找、轮廓绘制、霍夫直线/圆检测 | `cv2.findContours()`, `cv2.drawContours()`, `cv2.HoughLines()`, `cv2.HoughCircles()` |

## 示例代码

以下是一个简单的图像读取与显示示例：

```python
import cv2

# 读取图像（第二个参数：cv2.IMREAD_COLOR 彩色，cv2.IMREAD_GRAYSCALE 灰度）
img = cv2.imread('test.jpg', cv2.IMREAD_COLOR)

# 检查是否读取成功
if img is None:
    print('无法读取图片，请检查路径是否正确')
    exit()

# 创建窗口并显示图像
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', img)

# 等待按键，按下任意键后关闭窗口
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 学习建议

1. **按顺序学习**：建议从 `01_basic_operations` 开始，逐步深入
2. **动手修改**：尝试修改参数，观察输出变化，加深理解
3. **结合文档**：遇到不熟悉的函数，查阅 [OpenCV 官方文档](https://docs.opencv.org/)
4. **实际应用**：用自己的图片做测试，将知识点应用到实际场景

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件
