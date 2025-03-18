#  Wonders of the World Classification using Deep Learning  

##  Overview  
This project implements a **deep learning-based image classification model** to accurately identify and classify images of the **Wonders of the World**.  
Using **Convolutional Neural Networks (CNNs)**, **transfer learning**, and **ClearML for experiment tracking**, the model is trained on a dataset containing images of **ancient and modern wonders** to achieve high classification accuracy.  

---

##  Objectives  
- ✅ Develop a **deep learning model** for **automated image classification** of Wonders of the World.  
- ✅ Implement **CNN architectures** and **transfer learning** for improved performance.  
- ✅ Utilize **ClearML for hyperparameter tuning and experiment tracking**.  
- ✅ Evaluate model accuracy using **precision, recall, and F1-score**.  

---

##  Dataset  
- **Total Images:**  **3,846 images**  
- **Classes:**  **Seven Ancient Wonders & Seven Modern Wonders**  
- **Dataset Source:**  
   [Download Wonders of the World Dataset](https://www.kaggle.com/datasets/balabaskar/wonders-of-the-world-image-classification/code)
- **Preprocessing:**  
  - **Resizing & Normalization** for consistent input  
  - **Data Augmentation** (Random Horizontal Flip, Rotation) to improve generalization  
  - **Train-Test Split:** **80% Training, 20% Testing**  

---

##  Implemented Models & Techniques  
| Model | Key Features | Accuracy | Precision | Recall | F1-Score |  
|------------|--------------------------------|------------|------------|------------|------------|  
| **Custom CNN** | Built from scratch with multiple convolutional layers | **85.3%** | **86.1%** | **85.4%** | **85.7%** |  
| **VGG16 (Transfer Learning)** | Pre-trained on ImageNet, fine-tuned for classification | **88.2%** | **88.9%** | **88.1%** | **88.3%** |  
| **ResNet18 (Transfer Learning)** | Deep residual learning framework | **89.4%** | **89.8%** | **89.3%** | **89.5%** |  

 **Best Performing Model:** **ResNet18 achieved 89.4% accuracy** with the best overall balance of precision and recall.  

---

##  ClearML Integration  
ClearML was used for:  
✅ **Automated experiment tracking** – Logging model performance and hyperparameters.  
✅ **Hyperparameter tuning** – Optimizing learning rate, batch size, and model architecture.  
✅ **Parallel execution & visualization** – Comparing multiple experiments using ClearML dashboard.  

---

##  Technologies Used  
- **Python** (TensorFlow, PyTorch, OpenCV)  
- **Deep Learning Models** (CNNs, Transfer Learning with ResNet18 & VGG16)  
- **Experiment Tracking** (ClearML for logging & hyperparameter tuning)  
- **Jupyter Notebook** for development and experimentation  

---

##  References  
-  [Deep Learning for Image Classification](https://www.tensorflow.org/tutorials/images/classification)  
-  [Transfer Learning with CNNs](https://www.tensorflow.org/tutorials/images/transfer_learning)  
-  [ClearML: Experiment Tracking & Optimization](https://clear.ml/)  

---

##  Contribution  
Contributions are welcome! Fork the repository and submit a pull request to improve the project.
