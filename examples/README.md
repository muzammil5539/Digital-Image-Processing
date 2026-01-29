# Examples

This directory contains example outputs and demonstrations of the Digital Image Processing labs.

## Quick Examples

### Example 1: Basic Image Enhancement

Enhance a low-contrast image using histogram equalization:

```python
import cv2 as cv
import matplotlib.pyplot as plt

# Load image
image = cv.imread('data/images/Fig0241(a)(einstein low contrast).tif', cv.IMREAD_GRAYSCALE)

# Apply histogram equalization
enhanced = cv.equalizeHist(image)

# Display results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(enhanced, cmap='gray')
axes[1].set_title('Enhanced Image')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('examples/histogram_equalization_example.png')
plt.show()
```

### Example 2: Edge Detection with Sobel

Detect edges in an image using the Sobel operator:

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv.imread('data/images/your_image.png', cv.IMREAD_GRAYSCALE)

# Apply Sobel operator
sobelx = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=3)
sobely = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=3)

# Calculate magnitude
magnitude = np.sqrt(sobelx**2 + sobely**2)
magnitude = np.uint8(magnitude)

# Display
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(image, cmap='gray')
plt.title('Original'), plt.axis('off')
plt.subplot(132), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.axis('off')
plt.subplot(133), plt.imshow(magnitude, cmap='gray')
plt.title('Edge Magnitude'), plt.axis('off')
plt.tight_layout()
plt.show()
```

### Example 3: Morphological Operations

Remove noise using morphological opening:

```python
import cv2 as cv
import numpy as np

# Load binary/noisy image
image = cv.imread('data/images/noisy_image.png', cv.IMREAD_GRAYSCALE)

# Define kernel
kernel = np.ones((5, 5), np.uint8)

# Apply morphological operations
erosion = cv.erode(image, kernel, iterations=1)
dilation = cv.dilate(image, kernel, iterations=1)
opening = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel)

# Save results
cv.imwrite('examples/morphology_opening.png', opening)
```

### Example 4: Color Space Conversion

Convert RGB image to HSV and extract specific colors:

```python
import cv2 as cv
import numpy as np

# Load color image
image = cv.imread('data/images/colored_image.png')

# Convert to HSV
hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

# Define range for red color
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

# Create mask
mask = cv.inRange(hsv, lower_red, upper_red)

# Extract red regions
result = cv.bitwise_and(image, image, mask=mask)

cv.imshow('Original', image)
cv.imshow('Red Mask', mask)
cv.imshow('Result', result)
cv.waitKey(0)
cv.destroyAllWindows()
```

### Example 5: Frequency Domain Filtering

Apply low-pass filter in frequency domain:

```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv.imread('data/images/sample.png', cv.IMREAD_GRAYSCALE)

# Compute FFT
f_transform = np.fft.fft2(image)
f_shift = np.fft.fftshift(f_transform)

# Create low-pass filter mask
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.uint8)
r = 30  # radius
cv.circle(mask, (ccol, crow), r, 1, -1)

# Apply mask
f_shift_filtered = f_shift * mask

# Inverse FFT
f_ishift = np.fft.ifftshift(f_shift_filtered)
img_filtered = np.fft.ifft2(f_ishift)
img_filtered = np.abs(img_filtered)

# Display
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Original'), plt.axis('off')
plt.subplot(122), plt.imshow(img_filtered, cmap='gray')
plt.title('Low-Pass Filtered'), plt.axis('off')
plt.show()
```

## Running Examples

To run examples from this directory:

```bash
# Create a new Python file with the example code
python example_script.py
```

## Example Outputs

Place your output images here for reference:
- `histogram_equalization_example.png`
- `edge_detection_example.png`
- `morphology_opening.png`
- `color_extraction_example.png`
- `frequency_filtering_example.png`

## Tips for Creating Examples

1. **Use meaningful filenames:** `before_after_operation.png`
2. **Add comments:** Explain what each step does
3. **Visualize results:** Use matplotlib for side-by-side comparisons
4. **Save outputs:** Store intermediate and final results
5. **Document parameters:** Note kernel sizes, threshold values, etc.

## Contributing Examples

Have a cool example? Add it here!

1. Create a clear, documented code snippet
2. Test it thoroughly
3. Add to this README
4. Submit a pull request

---

**Explore, experiment, and enjoy learning Digital Image Processing! 🎨**
