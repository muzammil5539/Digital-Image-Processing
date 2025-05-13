# Digital Image Processing (DIP) Labs

This repository contains a series of laboratory assignments for a Digital Image Processing course. Each lab covers a specific topic, from basic Python programming and image creation to advanced frequency-domain filtering and classification.

---

## Repository Structure

- **lab01001.py** – Lab 01: Python Basics (tuple sorting, minimum value in a dict, number system conversion, normalization function).
- **lab002.py** – Lab 02: Image Creation & Gradient Generation (blank images, linear gradients at different quantization levels).
- **lab03.py** – Lab 03: Connected Components & Distance Metrics (count objects in a binary image, compute Euclidean, Manhattan, Chessboard distances).
- **lab 04.py** – Lab 04: Image Enhancement (negative, logarithmic, power-law transforms, gray-level slicing, histogram computation).
- **lab05main.py** – Lab 05: Histogram Equalization & Contrast Stretching (contrast adjustment, global histogram equalization, plotting histograms).
- **lab 06 main.py** – Lab 06: Spatial Domain Filtering (mean/averaging filters of size 3×3 and 5×5).
- **lab07.py** – Lab 07: Edge Detection & Local Filters (Sobel gradient, thresholding by phase, mean filter with padding).
- **lab09.py** – Lab 09: Morphological Operations (erosion, dilation, opening, closing, morphological gradient).
- **lab10.py** – Lab 10: Color Image Processing (RGB→HSV conversion, Gaussian smoothing, Sobel on color channels).
- **lab11.py** – Lab 11: Frequency Domain Processing (FFT magnitude spectrum visualization, low-pass filtering using rectangular mask).
- **lab12.py** – Lab 12: Texture Analysis (GLCM feature extraction, spectral profiles S(r) and S(θ)).
- **lab13.py** – Lab 13: Classification (minimum distance classifier on the Iris dataset, confusion matrix, precision, recall, accuracy).
- **labfinal/labfinal.py** – Final Lab: License Plate Detection (connected-components-based rectangle detection suitable for license plates).

Additional image files used by various labs are included in the repository root (e.g. `Fig0241(a)(einstein low contrast).tif`, `lab009coin.png`, etc.).

---

## Dependencies

Ensure you have the following Python packages installed:

```powershell
pip install opencv-python numpy matplotlib scikit-image pandas
```

- **OpenCV** (`opencv-python`)
- **NumPy**
- **Matplotlib**
- **scikit-image**
- **pandas**

---

## Usage

To run a lab script, open a terminal in the `d:\CE\blah\dip` folder and execute:

```powershell
python labXX.py
```

Replace `labXX.py` with the desired lab filename (for example, `lab07.py`).

---

## License

This repository is released under the MIT License. Feel free to use and modify the code for educational purposes.
