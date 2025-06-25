# DocScanner_Replica
Inspired by mobile applications like CamScanner, this project implements a document scanner using OpenCV in Python. It includes perspective correction, document boundary detection, and a final enhancement filter called **Magic Color** that improves readability by increasing contrast and saturation intelligently.

## Table of Contents

* [Edge Detection and Perspective Correction](https://github.com/KimiyaVahidMotlagh/DocScanner_Replica?tab=readme-ov-file#edge-detection-and-perspective-correction)
* [Filters and Enhancement](https://github.com/KimiyaVahidMotlagh/DocScanner_Replica?tab=readme-ov-file#filters-and-enhancement)
* [Real-Time Controls](https://github.com/KimiyaVahidMotlagh/DocScanner_Replica?tab=readme-ov-file#real-time-controls
)
* [Result Examples](https://github.com/KimiyaVahidMotlagh/DocScanner_Replica?tab=readme-ov-file#result-examples)


### Edge Detection and Perspective Correction
The project begins by detecting the document edges in the image and applying perspective correction to flatten the image. This mimics the way scanning apps like CamScanner transform angled shots into clean, readable documents.

Key functions:
* order_points: Sorts the contour points to top-left, top-right, bottom-right, bottom-left.
* four_point_transform: Applies a warp transformation to generate a top-down view of the document.

### Filters and Enhancement

Once the image is transformed, several filters can be applied:

* **Grayscale**: Converts image to standard grayscale.
* **Black & White**: Increases readability using adaptive thresholding.
* **Magic Color**: Increases brightness, contrast, and sharpness to make documents appear vivid and more like scanned copies.

### Real-Time Controls

A simple OpenCV-based UI allows you to:

* Press buttons to apply different filters.
* Press save button to save the currently displayed image.
* Press `q` to quit the window.

The UI runs in a `while` loop until the user chooses to exit, creating a live and interactive experience.


### Result Examples

Here’s an example of how the sample image can be enhanced with color magic filter document clarity:

<picture>
 <source media="(prefers-color-scheme: dark)" srcset="https://github.com/KimiyaVahidMotlagh/ANN_manualcoding/blob/main/Pictures/CostPlot.jpg">
 <img alt="Shows an illustrated sun in light color mode and a moon with stars in dark color mode." src="https://github.com/KimiyaVahidMotlagh/ANN_manualcoding/blob/main/Pictures/CostPlot.jpg">
</picture>

