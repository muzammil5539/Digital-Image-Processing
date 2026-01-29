# API Reference

This document provides a reference for common functions and operations used throughout the Digital Image Processing labs.

## Table of Contents
- [OpenCV Functions](#opencv-functions)
- [NumPy Operations](#numpy-operations)
- [Matplotlib Visualization](#matplotlib-visualization)
- [Custom Functions](#custom-functions)

---

## OpenCV Functions

### Image I/O

#### `cv.imread(filename, flag)`
Loads an image from file.

**Parameters:**
- `filename` (str): Path to image file
- `flag` (int): Image reading mode
  - `cv.IMREAD_COLOR` or `1`: Color image (default)
  - `cv.IMREAD_GRAYSCALE` or `0`: Grayscale image
  - `cv.IMREAD_UNCHANGED` or `-1`: Image with alpha channel

**Returns:** NumPy array representing the image

**Example:**
```python
image = cv.imread('data/images/sample.png', cv.IMREAD_GRAYSCALE)
```

#### `cv.imwrite(filename, image)`
Saves an image to file.

**Parameters:**
- `filename` (str): Output file path
- `image` (ndarray): Image to save

**Example:**
```python
cv.imwrite('output.png', processed_image)
```

#### `cv.imshow(window_name, image)`
Displays an image in a window.

**Parameters:**
- `window_name` (str): Window title
- `image` (ndarray): Image to display

**Note:** Must be followed by `cv.waitKey()` to display properly.

---

### Image Processing

#### `cv.equalizeHist(src)`
Equalizes the histogram of a grayscale image.

**Parameters:**
- `src` (ndarray): Source 8-bit single channel image

**Returns:** Equalized image

**Example:**
```python
equalized = cv.equalizeHist(gray_image)
```

#### `cv.Sobel(src, ddepth, dx, dy, ksize)`
Applies Sobel operator for edge detection.

**Parameters:**
- `src` (ndarray): Source image
- `ddepth` (int): Output image depth (e.g., `cv.CV_64F`)
- `dx` (int): Order of derivative in x direction
- `dy` (int): Order of derivative in y direction
- `ksize` (int): Size of Sobel kernel (1, 3, 5, or 7)

**Returns:** Gradient image

**Example:**
```python
sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)
```

#### `cv.GaussianBlur(src, ksize, sigmaX)`
Applies Gaussian blur to an image.

**Parameters:**
- `src` (ndarray): Source image
- `ksize` (tuple): Kernel size (width, height) - must be odd
- `sigmaX` (float): Standard deviation in X direction

**Returns:** Blurred image

**Example:**
```python
blurred = cv.GaussianBlur(image, (5, 5), 0)
```

---

### Morphological Operations

#### `cv.erode(src, kernel, iterations)`
Erodes an image using a structuring element.

**Parameters:**
- `src` (ndarray): Source image
- `kernel` (ndarray): Structuring element
- `iterations` (int): Number of times erosion is applied

**Returns:** Eroded image

**Example:**
```python
kernel = np.ones((5, 5), np.uint8)
eroded = cv.erode(image, kernel, iterations=1)
```

#### `cv.dilate(src, kernel, iterations)`
Dilates an image using a structuring element.

**Returns:** Dilated image

#### `cv.morphologyEx(src, op, kernel)`
Performs advanced morphological transformations.

**Parameters:**
- `op` (int): Type of operation
  - `cv.MORPH_OPEN`: Opening
  - `cv.MORPH_CLOSE`: Closing
  - `cv.MORPH_GRADIENT`: Morphological gradient
  - `cv.MORPH_TOPHAT`: Top hat
  - `cv.MORPH_BLACKHAT`: Black hat

**Example:**
```python
opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
```

---

### Color Space Conversions

#### `cv.cvtColor(src, code)`
Converts image from one color space to another.

**Parameters:**
- `src` (ndarray): Source image
- `code` (int): Color conversion code
  - `cv.COLOR_BGR2GRAY`: BGR to grayscale
  - `cv.COLOR_BGR2HSV`: BGR to HSV
  - `cv.COLOR_BGR2RGB`: BGR to RGB

**Example:**
```python
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
```

---

## NumPy Operations

### Array Creation

```python
# Create zeros array
zeros = np.zeros((height, width), dtype=np.uint8)

# Create ones array
ones = np.ones((height, width), dtype=np.uint8)

# Create array from shape
like_array = np.zeros_like(image)
```

### Mathematical Operations

```python
# Negative transformation
negative = 255 - image

# Logarithmic transformation
c = 255 / np.log(1 + np.max(image))
log_transform = np.uint8(c * np.log(1 + image))

# Power-law transformation
gamma = 0.5
power_law = np.uint8(255 * ((image / 255) ** gamma))
```

### Statistical Functions

```python
# Mean
mean_value = np.mean(image)

# Standard deviation
std_dev = np.std(image)

# Min and max
min_val = np.min(image)
max_val = np.max(image)
```

### FFT Operations

```python
# Forward FFT
f_transform = np.fft.fft2(image)
f_shift = np.fft.fftshift(f_transform)

# Inverse FFT
f_ishift = np.fft.ifftshift(f_shift)
image_back = np.fft.ifft2(f_ishift)
image_back = np.abs(image_back)
```

---

## Matplotlib Visualization

### Basic Plotting

```python
import matplotlib.pyplot as plt

# Display single image
plt.imshow(image, cmap='gray')
plt.title('Image Title')
plt.axis('off')
plt.show()
```

### Subplots

```python
# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Display images in grid
axes[0, 0].imshow(image1, cmap='gray')
axes[0, 0].set_title('Image 1')
axes[0, 0].axis('off')

axes[0, 1].imshow(image2, cmap='gray')
axes[0, 1].set_title('Image 2')
axes[0, 1].axis('off')

plt.tight_layout()
plt.show()
```

### Histogram Plotting

```python
# Plot histogram
plt.hist(image.ravel(), bins=256, range=[0, 256])
plt.title('Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.show()
```

### Saving Figures

```python
plt.savefig('output.png', dpi=300, bbox_inches='tight')
```

---

## Custom Functions

### Distance Metrics

#### Euclidean Distance
```python
def euclidean_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
```

#### Manhattan Distance
```python
def manhattan_distance(x1, y1, x2, y2):
    """Calculate Manhattan (city block) distance."""
    return abs(x2 - x1) + abs(y2 - y1)
```

#### Chessboard Distance
```python
def chessboard_distance(x1, y1, x2, y2):
    """Calculate Chessboard (Chebyshev) distance."""
    return max(abs(x2 - x1), abs(y2 - y1))
```

### Image Transformations

#### Negative Transformation
```python
def negative_transform(image):
    """Create negative of an image."""
    return 255 - image
```

#### Contrast Stretching
```python
def contrast_stretch(image, r1, r2, s1, s2):
    """
    Apply contrast stretching transformation.
    
    Parameters:
        image: Input image
        r1, r2: Input intensity range
        s1, s2: Output intensity range
    """
    result = np.zeros_like(image, dtype=np.float32)
    
    # Apply piecewise linear transformation
    mask1 = image < r1
    mask2 = (image >= r1) & (image <= r2)
    mask3 = image > r2
    
    result[mask1] = (s1 / r1) * image[mask1]
    result[mask2] = ((s2 - s1) / (r2 - r1)) * (image[mask2] - r1) + s1
    result[mask3] = ((255 - s2) / (255 - r2)) * (image[mask3] - r2) + s2
    
    return np.uint8(result)
```

---

## Usage Examples

### Complete Image Enhancement Pipeline

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv.imread('data/images/sample.png', cv.IMREAD_GRAYSCALE)

# Apply histogram equalization
equalized = cv.equalizeHist(image)

# Apply Gaussian blur
blurred = cv.GaussianBlur(equalized, (5, 5), 0)

# Detect edges
edges = cv.Canny(blurred, 50, 150)

# Display results
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original')
axes[0, 1].imshow(equalized, cmap='gray')
axes[0, 1].set_title('Equalized')
axes[1, 0].imshow(blurred, cmap='gray')
axes[1, 0].set_title('Blurred')
axes[1, 1].imshow(edges, cmap='gray')
axes[1, 1].set_title('Edges')

for ax in axes.flat:
    ax.axis('off')

plt.tight_layout()
plt.show()
```

---

## Additional Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/)
- [scikit-image Documentation](https://scikit-image.org/)

---

*For lab-specific implementations, refer to individual lab files in `src/labs/`*
