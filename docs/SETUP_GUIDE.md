# Setup Guide

This guide will help you set up your development environment for the Digital Image Processing labs.

## System Requirements

### Minimum Requirements
- **OS:** Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python:** Version 3.7 or higher
- **RAM:** 4 GB minimum (8 GB recommended)
- **Disk Space:** 2 GB for Python, libraries, and sample images

### Recommended Setup
- Python 3.9 or 3.10
- 8 GB RAM or more
- SSD for faster image processing
- Display with at least 1920x1080 resolution

## Installation Steps

### 1. Install Python

#### Windows
1. Download Python from [python.org](https://www.python.org/downloads/)
2. Run the installer
3. ⚠️ **Important:** Check "Add Python to PATH"
4. Click "Install Now"
5. Verify installation:
   ```bash
   python --version
   ```

#### macOS
Using Homebrew:
```bash
brew install python3
python3 --version
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

### 2. Clone the Repository

```bash
# Using HTTPS
git clone https://github.com/muzammil5539/Digital-Image-Processing.git

# Or using SSH
git clone git@github.com:muzammil5539/Digital-Image-Processing.git

# Navigate to the directory
cd Digital-Image-Processing
```

### 3. Create Virtual Environment

Creating a virtual environment is **highly recommended** to avoid dependency conflicts.

#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt when activated.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- opencv-python
- numpy
- matplotlib
- scikit-image
- pandas
- scipy
- pillow

#### Verify Installation

```python
python -c "import cv2; import numpy; import matplotlib; print('All packages installed successfully!')"
```

### 5. Test Your Setup

Run a simple test script:

```bash
python src/labs/test.py
```

Or create a quick test:

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Create a simple gradient image
img = np.linspace(0, 255, 256*256).reshape(256, 256).astype(np.uint8)
plt.imshow(img, cmap='gray')
plt.title('Test Image - Gradient')
plt.axis('off')
plt.show()

print("✅ Setup successful!")
```

## IDE/Editor Setup

### Visual Studio Code (Recommended)

1. Install VS Code from [code.visualstudio.com](https://code.visualstudio.com/)
2. Install Python extension:
   - Open VS Code
   - Go to Extensions (Ctrl+Shift+X)
   - Search for "Python" by Microsoft
   - Click Install

3. Configure Python interpreter:
   - Press Ctrl+Shift+P
   - Type "Python: Select Interpreter"
   - Choose your virtual environment

### PyCharm

1. Install PyCharm from [jetbrains.com/pycharm](https://www.jetbrains.com/pycharm/)
2. Open the project folder
3. Configure interpreter:
   - File → Settings → Project → Python Interpreter
   - Add → Existing environment
   - Select your venv/bin/python

### Jupyter Notebook

For interactive exploration:

```bash
pip install jupyter
jupyter notebook
```

## Troubleshooting

### Common Issues

#### 1. OpenCV Import Error

**Problem:** `ImportError: No module named 'cv2'`

**Solution:**
```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python
```

#### 2. NumPy Version Conflict

**Problem:** `ImportError: numpy.core.multiarray failed to import`

**Solution:**
```bash
pip install --upgrade numpy
```

#### 3. Matplotlib Display Issues on Linux

**Problem:** Matplotlib windows don't display

**Solution:**
```bash
sudo apt-get install python3-tk
```

#### 4. Permission Denied (Linux/macOS)

**Problem:** Permission errors during pip install

**Solution:**
```bash
pip install --user -r requirements.txt
```

#### 5. Virtual Environment Not Activating

**Windows PowerShell:**
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Getting More Help

- Check [Stack Overflow](https://stackoverflow.com/questions/tagged/opencv+python)
- OpenCV Forum: https://forum.opencv.org/
- Create an issue on GitHub

## Next Steps

Once your setup is complete:

1. 📖 Read the [Lab Documentation](README.md)
2. 🚀 Start with Lab 01: `python src/labs/lab01_python_basics.py`
3. 📸 Explore sample images in `data/images/`
4. 💻 Try modifying examples to learn

## Updating Dependencies

To update all packages to the latest versions:

```bash
pip install --upgrade -r requirements.txt
```

To update a specific package:

```bash
pip install --upgrade opencv-python
```

---

**Happy Coding! 🎨**
