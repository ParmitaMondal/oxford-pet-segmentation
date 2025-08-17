# Oxford-IIIT Pet Segmentation

This repository contains a TensorFlow 2 implementation of image segmentation on the **Oxford-IIIT Pet Dataset**.  
The model is a CNN-based encoder-decoder built with `Conv2D`, `MaxPooling2D`, and `Conv2DTranspose` layers.

## 📂 Project Structure
- `pet_segmentation.py` — Python script for training, evaluation, and visualization.
- `Utility.py` — Helper functions for preprocessing, displaying masks, and callbacks.
- `requirements.txt` — Python dependencies.
- `README.md` — Project overview.

## 🚀 Features
- Loads Oxford-IIIT Pet dataset via **TensorFlow Datasets (TFDS)**.
- Data preprocessing for training and testing using `Utility.py`.
- CNN model with convolutional and transpose convolution layers for segmentation.
- Training loop with validation monitoring.
- Visualization of predictions (input, true mask, predicted mask).
- Loss/accuracy plots.

## 📦 Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/<your-username>/oxford-pet-segmentation.git
cd oxford-pet-segmentation
pip install -r requirements.txt
