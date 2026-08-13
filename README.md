# OpenCV 学习练习题与示例代码

> 作者：巫伟鑫
> 开始时间：2024年11月
> 最后更新：2025年4月
> 状态：持续更新中 🚧

---

## 项目简介

本仓库是我学习 OpenCV Python 版本时的练习题和示例代码集合。按照学习路径分为多个模块，每个模块都有可直接运行的示例脚本和详细的中文注释。

我是智能网联汽车专业的学生，学 OpenCV 主要是为了以后做车载视觉感知打基础。仓库里的代码都是我跟着教程一边学一边敲的，每个示例我都自己跑过，注释里也记录了我自己的理解和踩过的坑。

如果你也是初学者，希望这个仓库能帮到你~

---

## 学习路线

完整的学习路线图请看：[docs/learning-path.md](docs/learning-path.md)

简单概括一下我的学习路径：

```
入门基础 → 图像处理 → 特征检测 → 进阶内容 → 项目实战
  ↓          ↓          ↓          ↓          ↓
图像读写    图像滤波    角点检测    直方图      答题卡识别
像素操作    边缘检测    轮廓检测    图像变换    车道线检测
颜色空间    阈值分割    霍夫变换    视频处理    人脸检测
          形态学操作    特征匹配    目标追踪    手势识别
```

---

## 目录结构

```
opencv-learning-exercises/
├── 01_basic_operations/      # 第一阶段：基本操作
│   ├── image_io.py           # 图像读写与像素操作
│   └── color_space.py        # 颜色空间转换
├── 02_image_processing/      # 第二阶段：图像处理
│   ├── filtering.py          # 图像滤波（均值、高斯、中值、双边）
│   ├── edge_detection.py     # 边缘检测（Sobel、Laplacian、Canny）
│   └── threshold.py          # 阈值分割与形态学操作
├── 03_feature_detection/     # 第三阶段：特征检测
│   ├── corners.py            # 角点检测（Harris、Shi-Tomasi）
│   └── contours.py           # 轮廓检测与霍夫变换
├── docs/
│   └── learning-path.md      # 详细学习路线图
├── requirements.txt          # 依赖包列表
├── LICENSE                   # MIT 许可证
└── README.md                 # 项目说明
```

---

## 各模块学习进度

| 模块 | 状态 | 完成时间 | 练习数 |
|------|------|---------|--------|
| 01 - 基本操作 | ✅ 已完成 | 2024-11 | 2 |
| 02 - 图像处理 | ✅ 已完成 | 2024-12 | 3 |
| 03 - 特征检测 | 🔄 进行中 | 2025-03 | 2 |
| 04 - 直方图与变换 | 📌 待开始 | - | 0 |
| 05 - 视频与追踪 | 📌 待开始 | - | 0 |
| 06 - 项目实战 | 📌 规划中 | - | 0 |

---

## 环境要求

- Python 3.8+
- OpenCV 4.x
- NumPy
- Matplotlib（用于显示图像，可选）

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

> 注意：脚本默认会读取 `test.jpg` 作为输入图片，请在运行前准备一张测试图片放在对应目录，或修改脚本中的图片路径。
>
> 有些脚本（如角点检测、轮廓检测）会自动创建测试图像，不需要额外准备。

---

## 模块详解

### 01 - 基本操作

| 文件 | 知识点 | 核心函数 |
|------|--------|----------|
| image_io.py | 图像读取、显示、保存，像素级操作，ROI，通道拆分合并 | `cv2.imread()`, `cv2.imshow()`, `cv2.imwrite()`, `cv2.split()`, `cv2.merge()` |
| color_space.py | BGR/RGB/HSV/GRAY 颜色空间转换，颜色提取 | `cv2.cvtColor()`, `cv2.inRange()` |

### 02 - 图像处理

| 文件 | 知识点 | 核心函数 |
|------|--------|----------|
| filtering.py | 均值滤波、高斯滤波、中值滤波、双边滤波、自定义卷积核 | `cv2.blur()`, `cv2.GaussianBlur()`, `cv2.medianBlur()`, `cv2.bilateralFilter()`, `cv2.filter2D()` |
| edge_detection.py | Sobel、Laplacian、Canny 边缘检测 | `cv2.Sobel()`, `cv2.Laplacian()`, `cv2.Canny()` |
| threshold.py | 二值化、自适应阈值、Otsu、形态学操作 | `cv2.threshold()`, `cv2.adaptiveThreshold()`, `cv2.morphologyEx()` |

### 03 - 特征检测

| 文件 | 知识点 | 核心函数 |
|------|--------|----------|
| corners.py | Harris 角点、Shi-Tomasi 角点、亚像素级角点 | `cv2.cornerHarris()`, `cv2.goodFeaturesToTrack()`, `cv2.cornerSubPix()` |
| contours.py | 轮廓查找、轮廓特征、轮廓近似、霍夫直线/圆检测 | `cv2.findContours()`, `cv2.drawContours()`, `cv2.HoughLines()`, `cv2.HoughCircles()` |

---

## 学习心得

### 给初学者的几点建议

1. **先跑起来，再理解原理**

   一开始不用死磕每个函数的底层实现，先把代码跑起来，看到效果，有了直观认识再去深入原理。这样不容易劝退。

2. **改参数，多观察**

   每个函数的参数都试着改一改，看看输出有什么变化。比如 Canny 的两个阈值，调高调低对比一下效果，理解就深刻了。

3. **BGR 还是 RGB？这是个问题**

   OpenCV 默认是 BGR 顺序，Matplotlib、PIL 等库是 RGB 顺序，混用会导致颜色显示异常。踩过几次坑就记住了...

4. **数据类型很重要**

   OpenCV 里很多函数对输入图像的数据类型有要求，比如 Sobel 算子推荐用 CV_64F 来避免负值被截断。类型不对可能得到奇怪的结果。

5. **多做小项目**

   学完一个阶段，找个小项目练练手。比如学完轮廓检测，可以做个硬币计数的小程序，把学过的知识点串起来。

---

## 参考资料

### 官方文档
- [OpenCV 官方文档](https://docs.opencv.org/) - 最权威的参考
- [OpenCV Python 官方教程](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### 视频教程
- **于仕琪** OpenCV 入门教程（B站）- 强烈推荐入门看
- **同济子豪兄** OpenCV 实战教程（B站）- 有很多有趣的小项目
- **唐宇迪** OpenCV 课程 - 偏实战应用

### 书籍
- 《学习 OpenCV 3》- 经典教材
- 《OpenCV 4 快速入门》- 于仕琪著，适合入门
- 《数字图像处理》（冈萨雷斯）- 原理必看

### 网站
- [PyImageSearch](https://pyimagesearch.com/) - 很多实战教程
- [LearnOpenCV](https://learnopencv.com/) - 质量很高的 OpenCV 教程

---

## 待完成 / 计划中

- [ ] 第四阶段：直方图与图像变换
- [ ] 第四阶段：几何变换（缩放、平移、旋转、仿射、透视）
- [ ] 第五阶段：视频处理与背景建模
- [ ] 第五阶段：光流法与对象追踪
- [ ] 第六阶段：SIFT / SURF 特征点检测与匹配
- [ ] 第六阶段：图像拼接（全景图）
- [ ] 项目实战：答题卡识别
- [ ] 项目实战：车道线检测
- [ ] 项目实战：人脸检测与识别
- [ ] 项目实战：手势识别

---

## 关于作者

巫伟鑫 | 深圳信息职业技术大学 | 智能网联汽车工程技术（鸿蒙班）

📧 wwxwuweixin@qq.com | 🌐 github.com/lanyunayue

---

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

**如果这个仓库对你有帮助，欢迎 Star 支持一下~ 有问题也欢迎提 Issue 交流！** 😊
