# 🖋️ Web Application for Converting Handwriting to Digital Text Using Deep Learning

This project aims to develop a system that converts handwritten text from paper into digital format. A Convolutional Neural Network (CNN)-based deep learning model is used to classify handwritten characters and digits, and a user-friendly web interface allows for easy interaction with the system.

---

## 🛠️ Design and Methodology

### 🔄 Overall System Workflow

The system works through the following stages:

1. **Data Acquisition:** Uploading an image that contains handwritten text.
2. **Segmentation:** Splitting the image into lines and individual characters.
3. **Classification:** Assigning each character to the corresponding letter or digit class.
4. **Reconstruction:** Merging characters into words and sentences.
5. **Output Display:** Displaying the extracted text and allowing the user to download it as a PDF.

![System Overview Diagram](yontem_sema.png)  
_Figure 1: System Workflow_

---

### 🔍 Methodology Details

#### 1. Data Preparation
- **MNIST** dataset was used for digits, and **EMNIST** for letters.
- The datasets were preprocessed and normalized before training.

#### 2. Segmentation Process
- **Line Segmentation:** The image is divided into horizontal lines.
- **Character Segmentation:** Characters are isolated and resized to 28x28 pixels.
- **Otsu Binarization** was used to reduce background interference and enhance clarity.

#### 3. CNN-Based Model Training
- A common CNN architecture was applied for both MNIST and EMNIST datasets.

**CNN Architecture Overview:**
- 3 Conv2D + MaxPooling layers
- 3 Dense (fully connected) layers
- Dropout layers for overfitting prevention
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy

**Training Accuracy:**
- MNIST: **99%**
- EMNIST: **93%**

#### 4. Web Application
- Developed using the **Flask** web framework.
- The interface accepts handwriting image uploads, processes them, and displays the recognized text both on-screen and as a downloadable PDF.

---

## 🧪 Usage Steps

### 1. Login Page
- Users input basic information (e.g., name and surname).
- A template form to be digitalized is selected.

### 2. Image Upload
- The user uploads a photo of the handwritten form.
- The system allows the user to crop and isolate the handwriting region.

### 3. Displaying Results
- The processed text is shown in a table format on the screen.
- The user can also download the output as a **PDF** file.




[📹 Click to Watch a Video of a Sample Use of the Web Application](https://drive.google.com/file/d/1DPAPBaXmMQwd0uG6VMSnR4J3NKavsu2Z/view?usp=drive_link)

<video width="600" controls>
  <source src="path/to/your-video.mp4" type="video/mp4">
  Tarayıcınız bu videoyu desteklemiyor.
</video>

## 🔧 Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Clone the Project:**
   ```bash
   git clone https://github.com/your_username/project_name.git
   cd project_name
   ```

3. **Run the Application:**
   ```bash
   python app.py
   ```

---

## 📊 Performance and Results

- **MNIST Model Performance:**
  - Precision: **0.99**
  - Recall: **0.99**
  - F1 Score: **0.99**

- **EMNIST Model Performance:**
  - Precision: **0.92**
  - Recall: **0.91**
  - F1 Score: **0.91**

---

## 📬 Contact

For more information, please contact:
- Ayşegül Toptaş: [aysegulltoptass@gmail.com](mailto:aysegulltoptass@gmail.com)
- Havvanur Bozkurt: [havvabzkrt35@gmail.com](mailto:havvabzkrt35@gmail.com)
