# 🎨 Digital Image Processing Labs

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](CONTRIBUTING.md)

*A comprehensive collection of Digital Image Processing laboratory assignments covering fundamental to advanced topics in computer vision and image analysis.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Repository Structure](#-repository-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Lab Descriptions](#-lab-descriptions)
- [Examples](#-examples)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🔍 Overview

This repository contains a comprehensive series of **13+ laboratory assignments** designed for a Digital Image Processing course. Each lab provides hands-on experience with essential image processing techniques, from basic Python programming to advanced topics like frequency domain filtering, morphological operations, and machine learning classification.

The labs use industry-standard libraries like **OpenCV**, **NumPy**, and **scikit-image** to implement real-world image processing algorithms used in computer vision applications.

---

## ✨ Features

- 📚 **13+ Complete Labs** covering core DIP concepts
- 🎯 **Well-Documented Code** with clear explanations
- 🖼️ **Sample Images** included for testing and learning
- 🔧 **Modular Structure** for easy navigation
- 📊 **Visualization Tools** using Matplotlib
- 🧪 **Practical Applications** from theory to implementation
- 🚀 **Beginner-Friendly** with progressive complexity

---

## 📁 Repository Structure

```
Digital-Image-Processing/
│
├── src/
│   └── labs/                          # All lab Python scripts
│       ├── lab01_python_basics.py
│       ├── lab02_image_creation.py
│       ├── lab03_connected_components.py
│       ├── lab04_image_enhancement.py
│       ├── lab05_histogram_equalization.py
│       ├── lab06_spatial_filtering.py
│       ├── lab07_edge_detection.py
│       ├── lab09_morphological_operations.py
│       ├── lab10_color_processing.py
│       ├── lab11_frequency_domain.py
│       ├── lab12_texture_analysis.py
│       ├── lab13_classification.py
│       └── lab_final_license_plate.py
│
├── data/
│   └── images/                        # Sample images and datasets
│
├── docs/                              # Additional documentation
├── examples/                          # Example outputs and demos
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── LICENSE                           # MIT License
├── CONTRIBUTING.md                   # Contribution guidelines
└── README.md                         # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/muzammil5539/Digital-Image-Processing.git
cd Digital-Image-Processing
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required Packages:**
- `opencv-python` - Computer vision library
- `numpy` - Numerical computing
- `matplotlib` - Data visualization
- `scikit-image` - Image processing algorithms
- `pandas` - Data manipulation (for classification lab)
- `scipy` - Scientific computing (optional)
- `pillow` - Image handling (optional)

---

## 💻 Usage

### Running a Lab

Navigate to the repository directory and run any lab script:

```bash
# Example: Run Lab 04 (Image Enhancement)
python src/labs/lab04_image_enhancement.py
```

### Using Sample Images

All sample images are located in the `data/images/` directory. Update the file paths in the lab scripts if needed:

```python
# Example in lab script
filename = 'data/images/Fig0241(a)(einstein low contrast).tif'
image = cv.imread(filename, cv.IMREAD_GRAYSCALE)
```

### Quick Start Example

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load an image
image = cv.imread('data/images/einstein.tif', cv.IMREAD_GRAYSCALE)

# Apply a transformation
enhanced = cv.equalizeHist(image)

# Display results
plt.subplot(1, 2, 1), plt.imshow(image, cmap='gray')
plt.title('Original'), plt.axis('off')
plt.subplot(1, 2, 2), plt.imshow(enhanced, cmap='gray')
plt.title('Enhanced'), plt.axis('off')
plt.show()
```

---

## 📖 Lab Descriptions

### Lab 01: Python Basics
**File:** `lab01_python_basics.py`  
**Topics:** Tuple sorting, dictionary operations, number system conversion, normalization functions  
**Difficulty:** ⭐ Beginner

### Lab 02: Image Creation & Gradients
**File:** `lab02_image_creation.py`  
**Topics:** Creating blank images, generating linear gradients, quantization levels  
**Difficulty:** ⭐ Beginner

### Lab 03: Connected Components
**File:** `lab03_connected_components.py`  
**Topics:** Object counting in binary images, distance metrics (Euclidean, Manhattan, Chessboard)  
**Difficulty:** ⭐⭐ Intermediate

### Lab 04: Image Enhancement
**File:** `lab04_image_enhancement.py`  
**Topics:** Negative transformation, logarithmic transformation, power-law transformation, gray-level slicing  
**Difficulty:** ⭐⭐ Intermediate

### Lab 05: Histogram Processing
**File:** `lab05_histogram_equalization.py`  
**Topics:** Histogram equalization, contrast stretching, histogram computation and plotting  
**Difficulty:** ⭐⭐ Intermediate

### Lab 06: Spatial Domain Filtering
**File:** `lab06_spatial_filtering.py`  
**Topics:** Mean/averaging filters (3×3, 5×5), noise reduction  
**Difficulty:** ⭐⭐ Intermediate

### Lab 07: Edge Detection
**File:** `lab07_edge_detection.py`  
**Topics:** Sobel operator, gradient computation, phase thresholding, mean filter with padding  
**Difficulty:** ⭐⭐⭐ Advanced

### Lab 09: Morphological Operations
**File:** `lab09_morphological_operations.py`  
**Topics:** Erosion, dilation, opening, closing, morphological gradient  
**Difficulty:** ⭐⭐⭐ Advanced

### Lab 10: Color Image Processing
**File:** `lab10_color_processing.py`  
**Topics:** RGB to HSV conversion, Gaussian smoothing, Sobel on color channels  
**Difficulty:** ⭐⭐⭐ Advanced

### Lab 11: Frequency Domain Processing
**File:** `lab11_frequency_domain.py`  
**Topics:** FFT, magnitude spectrum visualization, low-pass filtering with rectangular mask  
**Difficulty:** ⭐⭐⭐ Advanced

### Lab 12: Texture Analysis
**File:** `lab12_texture_analysis.py`  
**Topics:** GLCM (Gray Level Co-occurrence Matrix), texture features, spectral profiles  
**Difficulty:** ⭐⭐⭐ Advanced

### Lab 13: Classification
**File:** `lab13_classification.py`  
**Topics:** Minimum distance classifier, Iris dataset, confusion matrix, precision, recall, accuracy  
**Difficulty:** ⭐⭐⭐ Advanced

### Lab Final: License Plate Detection
**File:** `lab_final_license_plate.py`  
**Topics:** Connected components analysis, rectangle detection, practical application  
**Difficulty:** ⭐⭐⭐⭐ Expert

---

## 🎯 Examples

### Example 1: Image Enhancement

```python
# Enhance low-contrast images using histogram equalization
import cv2 as cv

image = cv.imread('data/images/Fig0241(a)(einstein low contrast).tif', cv.IMREAD_GRAYSCALE)
enhanced = cv.equalizeHist(image)
cv.imwrite('examples/enhanced_einstein.png', enhanced)
```

### Example 2: Edge Detection

```python
# Detect edges using Sobel operator
import cv2 as cv
import numpy as np

image = cv.imread('data/images/your_image.png', cv.IMREAD_GRAYSCALE)
sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(sobelx**2 + sobely**2)
```

### Example 3: Morphological Operations

```python
# Apply opening to remove noise
import cv2 as cv
import numpy as np

image = cv.imread('data/images/noisy_image.png', cv.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)
opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
```

---

## 🤝 Contributing

Contributions are welcome! Whether you want to:
- 🐛 Report bugs
- 💡 Suggest new features
- 📝 Improve documentation
- ➕ Add new labs or examples

Please read our [Contributing Guidelines](CONTRIBUTING.md) to get started.

### Quick Contribution Steps

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use the code for personal and educational purposes
- ✅ Modify and distribute the code
- ✅ Use in commercial projects

---

## 📞 Contact

**Repository Maintainer:** Muzammil  
**GitHub:** [@muzammil5539](https://github.com/muzammil5539)

For questions, suggestions, or issues:
- 📧 Open an issue on GitHub
- 💬 Start a discussion in the repository

---

## 🙏 Acknowledgments

- Thanks to all contributors who have helped improve this repository
- Sample images courtesy of various digital image processing resources
- Built with ❤️ for the image processing community

---

<div align="center">

**⭐ If you find this repository helpful, please consider giving it a star! ⭐**

Made with 🎨 for Digital Image Processing enthusiasts

</div>
