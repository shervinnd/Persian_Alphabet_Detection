# **Persian_Alphabet_Detection**📚✨

Welcome to the **Alphabet Recognition** project! This repository
implements a deep learning-based solution for recognizing Persian
alphabet characters using both **Fully Connected Neural Networks** and
**Convolutional Neural Networks (CNNs)**. Built with Python and Keras,
this project showcases the power of machine learning in image
classification tasks. 🚀

------------------------------------------------------------------------

## 📖 Overview

This project focuses on classifying Persian alphabet characters from
images using two distinct neural network architectures:

-   **Model 1**: A fully connected neural network with multiple dense
    layers.
-   **Model 2**: A convolutional neural network (CNN) with
    convolutional, batch normalization, max-pooling, and dropout layers.

The models are trained and evaluated on custom datasets, with additional
testing on real-world images. The project includes data preprocessing,
model training, performance evaluation, and visualization of results
like ROC curves and sample predictions. 📊

------------------------------------------------------------------------

## 🛠️ Features

-   **Data Preprocessing**: Custom `DataLoader` class for loading,
    resizing, normalizing, and augmenting images (zoom, invert, etc.).
    🖼️
-   **Model Architectures**:
    -   **Model 1**: Sequential dense layers with ReLU activation and
        softmax output for 43 classes.
    -   **Model 2**: CNN with Conv2D, BatchNormalization, MaxPooling,
        and Dropout for robust feature extraction.
-   **Training & Evaluation**:
    -   Trained on three datasets (`DS-01`, `DS-02`, `DS-03`) with
        validation splits.
    -   Performance metrics: Accuracy, Loss, ROC curves, and AUC for
        each class.
-   **Visualization**:
    -   Sample images from training and test sets.
    -   Training/validation accuracy and loss plots.
    -   ROC curves for each class in a 7x7 grid.
    -   Real-world image predictions with side-by-side model
        comparisons.
-   **Real-World Testing**: Preprocesses and predicts on real-world
    images using both models. 🌍

------------------------------------------------------------------------

## 📂 Project Structure

``` plaintext
Alphabet-Recognition/
├── Datasets/
│   ├── DS-01/                # Dataset 1
│   ├── DS-02/                # Dataset 2
│   ├── DS-03/                # Dataset 3
│   └── Real Data/            # Real-world test images
├── Models/
│   └── Neural Network/       # Saved model files
├── Datasets/
│   └── DataLoader/           # DataLoader class
├── alphabet_recognition.py    # Main script
└── README.md                 # Project documentation
```

------------------------------------------------------------------------

## 🚀 Getting Started

### Prerequisites

-   Python 3.8+ 🐍

-   Required libraries:

    ``` bash
    pip install keras matplotlib numpy pandas scikit-learn opencv-python
    ```

### Installation

1.  Clone the repository:

    ``` bash
    git clone https://github.com/shahin-ro/Alphabet-Recognition.git
    cd Alphabet-Recognition
    ```

2.  Install dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

3.  Download or prepare the datasets (`DS-01`, `DS-02`, `DS-03`, and
    `Real Data`) and place them in the `Datasets/` folder.

### Running the Project

1.  Update the dataset paths in `alphabet_recognition.py` to match your
    local setup:

    ``` python
    DATASET1 = "path/to/DS-01"
    DATASET2 = "path/to/DS-02"
    DATASET3 = "path/to/DS-03"
    REAL_DATA = "path/to/Real Data"
    ```

2.  Run the main script:

    ``` bash
    python alphabet_recognition.py
    ```

------------------------------------------------------------------------

## 📈 Model Details

### Model 1: Fully Connected Neural Network

-   **Architecture**:
    -   Input: 64x64 grayscale images
    -   Layers: Flatten → Dense (2048, 1024, 512, 256, 64, 43) with ReLU
        and softmax
-   **Training**: Adam optimizer, sparse categorical crossentropy loss,
    20 epochs
-   **Performance**: Evaluated with accuracy, loss, and ROC curves

### Model 2: Convolutional Neural Network

-   **Architecture**:
    -   Input: 64x64 grayscale images (reshaped to 1x64x64)
    -   Layers: Conv2D → BatchNorm → MaxPooling → Dropout (x2) → Flatten
        → Dense (512, 256, 43)
-   **Training**: Adam optimizer, sparse categorical crossentropy loss,
    20 epochs
-   **Performance**: Higher accuracy due to convolutional feature
    extraction

------------------------------------------------------------------------

## 📊 Results

-   **Training/Validation Accuracy & Loss**:
    -   Visualized for both models using Matplotlib.
-   **Test Accuracy**:
    -   Model 1: \~\[Insert test accuracy from script\]
    -   Model 2: \~\[Insert test accuracy from script\]
-   **ROC Curves**:
    -   Plotted for all 43 classes, showing AUC for each.
-   **Real-World Predictions**:
    -   Both models predict on real-world images, displayed side-by-side
        for comparison.

------------------------------------------------------------------------

## 🖼️ Visualizations

### Sample Images

The script visualizes one image per class from the training set and 10
test images with true and predicted labels.

### Training Plots

Training and validation accuracy/loss are plotted to analyze model
performance over epochs.

### ROC Curves

A 7x7 grid of ROC curves shows the performance of each class, with AUC
values for detailed insights.

------------------------------------------------------------------------

## 🔧 Usage

1.  **Training**: Run `alphabet_recognition.py` to train both models and
    generate visualizations.
2.  **Testing**: The script evaluates models on test data and real-world
    images.
3.  **Customization**:
    -   Adjust `EPOCHS`, `IMAGE_SIZE`, or `SHRINK` in the script for
        experimentation.
    -   Modify the `DataLoader` parameters (e.g., `zoom`, `contrast`)
        for different preprocessing.

------------------------------------------------------------------------

## 🤝 Contributing

Contributions are welcome! 🙌 To contribute:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature`).
3.  Commit your changes (`git commit -m "Add your feature"`).
4.  Push to the branch (`git push origin feature/your-feature`).
5.  Open a Pull Request.

------------------------------------------------------------------------

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for
details.

------------------------------------------------------------------------

## 🙏 Acknowledgments

-   Inspired by Persian alphabet recognition challenges.
-   Thanks to the open-source community for libraries like Keras,
    Matplotlib, and OpenCV.
-   Dataset credits: \[Insert dataset source or credit if applicable\].

------------------------------------------------------------------------

## 📬 Contact

For questions or feedback, reach out via GitHub Issues or connect with
the project maintainer at \[your-email@example.com\].

Happy coding! 🎉
