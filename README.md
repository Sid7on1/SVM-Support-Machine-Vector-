# *****SVM-Support-Machine-Vector*****
This project uses Support Vector Machines (SVM) to classify handwritten digits (0–9) from the sklearn digits dataset. By converting image pixel values into numerical features, the model learns to identify digits based on their structure. It demonstrates the power of SVMs in image-based pattern recognition.
# 🧠 Handwritten Digit Classification using SVM

This project demonstrates the use of **Support Vector Machine (SVM)** for classifying handwritten digits (0–9) using the **Digits Dataset** from scikit-learn. It transforms pixel intensity values of 8x8 images into feature vectors and trains an SVM classifier to accurately predict digits.

## 📌 Project Highlights

- Uses **SVM** for multi-class classification
- Trained on **1797 handwritten digit images**
- Evaluates model using **confusion matrix** and **accuracy score**
- Visualizes predictions and performance

## 📊 Dataset

- **Source**: `sklearn.datasets.load_digits()`
- **Total Samples**: 1797
- **Image Shape**: 8×8 grayscale
- **Classes**: Digits from 0 to 9

## 📈 Algorithm – Support Vector Machine (SVM)

SVM finds the optimal hyperplane that separates data into classes by maximizing the margin between support vectors. For multi-class classification, SVM applies a "one-vs-rest" strategy. It performs well with high-dimensional and structured data like images.

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/svm-digit-classifier.git
   cd svm-digit-classifier

2.	Install required libraries:
   ```bash
   pip install scikit-learn matplotlib
```
3.	Run the Python script:
   ```bash
   python svm_digit_classifier.py
```
👨‍💻 Author
	•	Code and logic written by Siddharth Vishwanath
	•	Refined, structured, and documented with the help of AI assistant

📷 Output Sample
	•	Confusion Matrix
	•	Accuracy Score
	•	Sample Predictions with true and predicted labels

🧠 SVM Visualized

SVM maps features to higher-dimensional space and constructs hyperplanes to separate classes. For digits, pixel data is flattened and classified based on patterns learned during training.
