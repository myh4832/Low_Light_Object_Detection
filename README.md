
# **Low Light Object Detection**

## **Project Results and Overview**
This project focuses on improving object detection performance under low-light conditions. The primary objectives of this project were:
- To simulate realistic low-light environments and evaluate object detection algorithms.
- To enhance detection accuracy using data augmentation and model optimization techniques.

### **Key Results**
- Improved detection accuracy by **X%** in low-light conditions.
- Achieved real-time processing speed of **Y FPS** on NVIDIA A6000 GPU.
- Memory usage optimized to **Z MB** per image.

The motivation for this project stems from the increasing demand for robust computer vision systems that can operate reliably in challenging environments, such as nighttime surveillance and autonomous driving.

---

## **Source Code**
The source code is organized as follows:

```
├── data/
│   ├── raw/               # Raw datasets
│   ├── processed/         # Preprocessed datasets
├── src/
│   ├── models/            # Model architectures
│   ├── transforms/        # Data augmentation techniques
│   ├── train.py           # Training pipeline
│   ├── test.py            # Testing and evaluation
├── results/
│   ├── metrics.csv        # Performance metrics
│   ├── graphs/            # Graphs and visualizations
├── README.md              # Project overview and documentation
├── requirements.txt       # Python dependencies
```

### **Setup Instructions**
To set up the project locally:
1. Clone this repository:
   ```bash
   git clone https://github.com/myh4832/Low_Light_Object_Detection.git
   cd Low_Light_Object_Detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Performance Metrics**
Below are the performance metrics for this project:

| Metric             | Value            |
|--------------------|------------------|
| **Detection Accuracy** | 85.2%          |
| **Inference Speed**    | 25 FPS         |
| **Memory Usage**       | 1.2 GB per image |

### **Performance Graphs**
![Accuracy vs Epoch](results/graphs/accuracy_vs_epoch.png)
![Inference Speed](results/graphs/inference_speed.png)

---

## **Installation and Usage**

### **Installation**
Follow these steps to set up the environment:
1. Ensure Python 3.8+ is installed.
2. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### **Usage**
To train the model:
```bash
python src/train.py --config config/train_config.yaml
```

To evaluate the model:
```bash
python src/test.py --weights checkpoints/model_best.pth --data data/processed/test
```

---

## **References and Documentation**
### **References**
- [YOLOv5 Official Repository](https://github.com/ultralytics/yolov5)
- Papers:
  - "Low-Light Object Detection Using Simulated Environments" (Link)

### **Documentation**
- Key algorithms:
  - **Gamma Correction** for simulating low-light conditions.
  - **Data Augmentation** using Gaussian noise and color jitter.

---

## **Issues and Contributions**
### **Known Issues**
- Limited performance on highly occluded objects in extremely dark scenarios.
- Inference speed decreases with larger input image sizes.

### **Contributing**
We welcome contributions! Here’s how you can help:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and submit a pull request.

To report bugs or suggest features, please open an issue.

---

## **Future Work**
- Explore transformer-based architectures for low-light object detection.
- Implement unsupervised domain adaptation techniques to improve generalization.
- Integrate additional datasets for further testing and validation.