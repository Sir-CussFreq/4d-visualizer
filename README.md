# 4d-visualizer

**Explore four-dimensional polytopes in real time with interactive projections, stable exterior surface coloring, and manual 3D rotation.**

## Overview

`tesseract.py` is a Python script by [Sir-CussFreq](https://github.com/Sir-CussFreq) that renders 4D polytopes by projecting them into a 3D environment. The script computes and groups the exterior facets of the 3D projection into flat faces, assigning each a persistent random color that remains stable as long as the facet is visible. Additionally, each vertex is optionally labeled with its index, and the 3D view is fully manual — allowing you to click-and-drag to rotate the view while the 4D object rotates in its own space (click and drag rotation is currently broken).

## Features

- **4D Rotation & Projection:**  
  The object rotates in its native 4D space before being projected into 3D.
  
- **(Currently un)Stable Exterior Surface Coloring:**  
  The exterior facets of the 3D projection are computed by grouping the convex hull’s triangulated facets into flat faces (based on their corner vertices). Each facet is assigned a persistent color that remains constant as long as that facet is detected.
  
- **Interactive Controls:**  
  - **1-6:** Select different polytopes.  
  - **R:** Toggle between perspective and orthographic projection.  
  - **+ / -:** Increase or decrease the 4D rotation speed.  
  - **T:** Cycle through visual themes.  
  - **M:** Toggle the on-screen menu.  
  - **P:** Toggle 4D plane overlays.  
  - **[ / ]:** Adjust plane overlay opacity.  
  - **H:** Toggle filled exterior surfaces.  
  - **O / L:** Increase or decrease surface opacity.  
  - **V:** Toggle vertex labels on/off.  
  - **Scroll:** Zoom in and out.
  
- **Manual 3D Rotation:**  
  The 3D view is manually rotatable via click-and-drag, so you can adjust the camera perspective while the 4D object rotates in its own space.
  
- **Optional GPU Acceleration:**  
  If available, CuPy is used to accelerate computations; otherwise, the script falls back to NumPy.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Sir-CussFreq/4d-visualizer.git
   cd 4d-visualizer
   ```

2. **Install Dependencies:**

   Make sure you have Python 3.10+ installed. Install the required packages using the provided `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

   *Optional:* For GPU acceleration, ensure that [CuPy](https://cupy.dev/) is installed (this should already be included in `requirements.txt`).

## Usage

Run the script with:

  ```bash
  python tesseract.py
  ```

Use the on-screen hotkeys to interact with the visualization.

## License

This project is licensed under the [MIT License](LICENSE).
