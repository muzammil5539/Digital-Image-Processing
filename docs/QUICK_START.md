# Quick Start Guide

Get up and running with Digital Image Processing labs in under 5 minutes!

## 🚀 Quick Setup (5 minutes)

### Step 1: Clone Repository (1 min)
```bash
git clone https://github.com/muzammil5539/Digital-Image-Processing.git
cd Digital-Image-Processing
```

### Step 2: Install Dependencies (2 min)
```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### Step 3: Run Your First Lab (2 min)
```bash
# Run histogram computation example
python src/labs/lab04_image_enhancement.py
```

**Note:** If the script doesn't run, you may need to update the image paths in the script:
```python
# Change this:
filename = 'Fig0241(a)(einstein low contrast).tif'

# To this:
filename = 'data/images/Fig0241(a)(einstein low contrast).tif'
```

---

## 🎯 Try These Examples

### Example 1: View an Image (30 seconds)

Create a file `quick_test.py`:

```python
import cv2 as cv

# Load and display an image
image = cv.imread('data/images/Fig0241(a)(einstein low contrast).tif', cv.IMREAD_GRAYSCALE)
cv.imshow('Einstein', image)
cv.waitKey(0)
cv.destroyAllWindows()
```

Run it:
```bash
python quick_test.py
```

Press any key to close the window.

---

### Example 2: Enhance Image Contrast (1 minute)

```python
import cv2 as cv
import matplotlib.pyplot as plt

# Load image
image = cv.imread('data/images/Fig0241(a)(einstein low contrast).tif', cv.IMREAD_GRAYSCALE)

# Enhance with histogram equalization
enhanced = cv.equalizeHist(image)

# Display side-by-side
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original - Low Contrast')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(enhanced, cmap='gray')
plt.title('Enhanced - Better Contrast')
plt.axis('off')

plt.tight_layout()
plt.show()
```

**Result:** You'll see how histogram equalization dramatically improves image contrast!

---

### Example 3: Detect Edges (1 minute)

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv.imread('data/images/Fig0241(a)(einstein low contrast).tif', cv.IMREAD_GRAYSCALE)

# Detect edges with Canny
edges = cv.Canny(image, 50, 150)

# Display
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Detected Edges')
plt.axis('off')

plt.show()
```

---

### Example 4: Apply Blur Filter (1 minute)

```python
import cv2 as cv
import matplotlib.pyplot as plt

# Load image
image = cv.imread('data/images/Fig0241(a)(einstein low contrast).tif', cv.IMREAD_GRAYSCALE)

# Apply Gaussian blur
blurred = cv.GaussianBlur(image, (15, 15), 0)

# Display
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(blurred, cmap='gray')
axes[1].set_title('Gaussian Blur (15x15)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

---

## 📚 What to Learn Next

### Beginner Track (Week 1-2)
1. **Lab 01**: Python basics and data structures
2. **Lab 02**: Create images and gradients from scratch
3. **Lab 03**: Count objects in images

### Intermediate Track (Week 3-4)
4. **Lab 04**: Image enhancement techniques
5. **Lab 05**: Histogram processing
6. **Lab 06**: Spatial filtering (blur, sharpen)

### Advanced Track (Week 5-8)
7. **Lab 07**: Edge detection (Sobel, Canny)
8. **Lab 09**: Morphological operations
9. **Lab 10**: Color image processing
10. **Lab 11**: Frequency domain (FFT)
11. **Lab 12**: Texture analysis
12. **Lab 13**: Machine learning classification

### Expert Track (Week 9+)
13. **Lab Final**: License plate detection project

---

## 🔧 Common Tasks

### Change Image Paths
Most lab files reference images like this:
```python
filename = 'Fig0241(a)(einstein low contrast).tif'
```

Update to use the organized structure:
```python
filename = 'data/images/Fig0241(a)(einstein low contrast).tif'
```

### Save Output Images
```python
# After processing
cv.imwrite('output/my_result.png', processed_image)
```

### Display Multiple Images
```python
# Use subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# axes[row, col].imshow(image, cmap='gray')
```

---

## ❓ Troubleshooting

### Issue: "No module named 'cv2'"
**Solution:** Install OpenCV
```bash
pip install opencv-python
```

### Issue: "Image not found"
**Solution:** Check your path
```python
import os
print(os.getcwd())  # Print current directory
# Update path accordingly
```

### Issue: "Display window not showing"
**Solution:** Add proper wait and destroy
```python
cv.waitKey(0)  # Wait for key press
cv.destroyAllWindows()  # Close windows
```

### Issue: "Matplotlib doesn't show plot"
**Solution:** Add plt.show()
```python
plt.imshow(image, cmap='gray')
plt.show()  # This line is required!
```

---

## 🎓 Learning Tips

1. **Start Simple**: Begin with Lab 01 even if you know Python
2. **Visualize Everything**: Use `plt.imshow()` to see intermediate results
3. **Experiment**: Change parameters (kernel sizes, thresholds) to understand effects
4. **Read Comments**: Labs have helpful comments explaining the code
5. **Compare Results**: Always show before/after images
6. **Ask Questions**: Open issues on GitHub if you're stuck

---

## 📖 Next Steps

1. ✅ Complete the Quick Setup above
2. 📖 Read the [Setup Guide](SETUP_GUIDE.md) for detailed instructions
3. 🔍 Explore [API Reference](API_REFERENCE.md) for function details
4. 🚀 Start with [Lab 01](../src/labs/lab01_python_basics.py)
5. 💡 Try the [Examples](../examples/README.md)
6. 🤝 Read [Contributing Guidelines](../CONTRIBUTING.md) to contribute

---

## 🎉 Ready to Start!

You're all set! Pick a lab and start learning:

```bash
# Lab 04: Image Enhancement (Recommended first lab)
python src/labs/lab04_image_enhancement.py

# Lab 07: Edge Detection (Popular!)
python src/labs/lab07_edge_detection.py

# Lab 13: Classification (ML with images)
python src/labs/lab13_classification.py
```

**Happy Learning! 🎨📸**
