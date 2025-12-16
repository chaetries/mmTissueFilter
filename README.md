# mmTissueFilter - Tissue Annotation & Analysis

This project provides tools for annotating tissue regions in Mueller Matrix images, training a U-Net model for segmentation, and running inference using the trained model.

## Prerequisites

Ensure you have Python installed (tested with Python 3.8+). You can install the required dependencies using:

```bash
pip install -r requirements.txt
```

Common dependencies include:
- `numpy`
- `torch`, `torchvision` (for model training/inference)
- `PyQt5` (for the annotation GUI)
- `matplotlib`, `Pillow` (for visualization)
- `jupyter` (to run notebooks)

---

## 1. Initial GUI Annotation

The **Tissue Annotation Tool** allows users to manually annotate tissue regions on images to create ground truth masks for training.

**How to run:**
Navigate to the root directory and run:

```bash
python annotator_gui/main.py
```

**Usage:**
1.  Launch the application.
2.  Open a Mueller Matrix image (or composite).
3.  Use the drawing tools to label regions (e.g., Tissue, Background, etc.).
4.  Save the generated masks.

---

## 2. Model Training

The model training is handled via a Jupyter Notebook, which trains a U-Net architecture to segment the tissue regions based on your annotations.

**How to run:**
Start Jupyter Notebook:

```bash
jupyter notebook notebooks/m11_unet/m11_unet_training.ipynb
```

**Steps:**
1.  Open the `m11_unet_training.ipynb` notebook.
2.  Configure the dataset paths if necessary.
3.  Run the cells to train the model.
4.  The best model weights will be saved to `models/best_model.pth`.

---

## 3. Running Trained Model

You can run the trained model to perform segmentation on new images using either Python or MATLAB.

### Option A: Python (Recommended)

The Python script loads the model and processes a sample image.

**How to run:**
```bash
cd run_trained
python run_demo.py
```

**Configuration:**
- Edit `run_trained/run_demo.py` to change `INPUT_IMAGE_PATH` or `MODEL_PATH` as needed.
- By default, it expects the model at `../models/best_model.pth`.
- Output is saved as `prediction_result.png`.

### Option B: MATLAB

The MATLAB script `run_demo.m` acts as a wrapper that calls the Python inference script.

**How to run:**
1.  Open MATLAB.
2.  Navigate to the `run_trained` folder.
3.  Run the script:
    ```matlab
    run_demo
    ```

**Note for MATLAB:**
- Ensure `python` is in your system path.
- The script executes `python run_demo.py` via the `system` command and displays the resulting `prediction_result.png`.
