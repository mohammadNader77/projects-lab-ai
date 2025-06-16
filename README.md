# Intelligent Systems Lab - ENCS5141
## Case Study 1: Bike Sharing Dataset - Data Cleaning, Feature Engineering, and Comparative Classification
## Case Study 2: Subjectivity Detection in Tweets

### Table of Contents
- [Project Overview](#project-overview)
- [Part 1: Bike Sharing Dataset](#part-1-bike-sharing-dataset)
  - [Objectives](#objectives)
  - [Procedure](#procedure)
  - [Experiments](#experiments)
  - [Results](#results)
- [Part 2: Comparative Analysis of Classification Models](#part-2-comparative-analysis-of-classification-models)
  - [Objectives](#objectives-1)
  - [Procedure](#procedure-1)
  - [Experiments](#experiments-1)
  - [Results](#results-1)
- [Part 3: Subjectivity Detection in Tweets](#part-3-subjectivity-detection-in-tweets)
  - [Objectives](#objectives-2)
  - [Procedure](#procedure-2)
  - [Results](#results-2)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This repository contains the implementation for **ENCS5141 Intelligent Systems Lab**. The project is split into two case studies:
1. **Case Study 1**: Data Cleaning, Feature Engineering, and Comparative Classification on a Bike Sharing Dataset.
2. **Case Study 2**: Subjectivity Detection for Tweets with a deep learning approach.

### Part 1: Bike Sharing Dataset

#### Objectives
- Explore and preprocess the **Bike Sharing Dataset** by addressing missing values, encoding categorical variables, and scaling numerical features.
- Apply feature selection and dimensionality reduction techniques to improve model performance.
- Compare the performance of a Random Forest model trained on preprocessed data vs. raw data.

#### Procedure
1. **Data Exploration**: Understand the dataset’s structure, features, and identify missing values.
2. **Data Cleaning**: Handle missing values and outliers, impute or remove irrelevant data.
3. **Feature Engineering**: Encode categorical variables, scale numerical features, and apply dimensionality reduction.
4. **Model Training**: Train a **Random Forest** model on the preprocessed data and compare it with the raw data.

#### Experiments
- Compare the performance of the Random Forest model on both the preprocessed and raw data.
- Evaluate the impact of data preprocessing on metrics like accuracy, precision, and recall.

#### Results
- **Summary**: Evaluation of model performance on preprocessed vs. raw data, showing improvements in accuracy, consistency, and training speed.

### Part 2: Comparative Analysis of Classification Models

#### Objectives
- Compare the effectiveness of **Random Forest (RF)**, **Support Vector Machine (SVM)**, and **Multilayer Perceptron (MLP)** on the dataset.
- Analyze the impact of parameter tuning for each model.

#### Procedure
1. **Model Training**: Train RF, SVM, and MLP on the preprocessed dataset.
2. **Model Comparison**: Evaluate models based on accuracy, precision, recall, and computational efficiency (training time and memory used).

#### Experiments
- Measure classification accuracy, precision, recall, and F1-score for each model.
- Explore how parameter tuning affects the models’ performance.

#### Results
- **Summary**: Comparison of the strengths and weaknesses of each model, concluding which one provides the best balance between prediction accuracy and computational efficiency.

### Part 3: Subjectivity Detection in Tweets

#### Objectives
- Implement a **subjectivity classification model** to classify tweets into subjective (SUBJ) or objective (OBJ) categories.
- Use CNNs and pre-trained models for text extraction from images.

#### Procedure
1. **Data Extraction and Preprocessing**: Extract sentences from character images and preprocess the text data.
2. **Model Implementation**: Use **Long Short-Term Memory (LSTM)** and **Transfer Learning** (pre-trained transformers) for subjectivity classification.
3. **Hyperparameter Tuning**: Experiment with learning rate, batch size, and number of layers to improve performance.

#### Results
- **Evaluation**: Report accuracy, precision, recall, and F1-score for each model, and discuss the strengths and weaknesses.

## Requirements
- Python 3.7 or later
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `tensorflow` / `pytorch` (depending on the model used)
  - `torchvision` (for CNN-based models)
  
## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/ENCS5141-Intelligent-Systems-Lab.git
    cd ENCS5141-Intelligent-Systems-Lab
    ```

2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Part 1: Bike Sharing Dataset**
    - Run the Jupyter notebook `bike_sharing.ipynb` for data exploration, preprocessing, and model training.
    - Visualize and compare the performance of the Random Forest model on preprocessed and raw data.

2. **Part 2: Classification Models Comparison**
    - Run the Jupyter notebook `classification_comparison.ipynb` to train and compare Random Forest, SVM, and MLP.
    - Evaluate the models and experiment with parameter tuning.

3. **Part 3: Subjectivity Detection**
    - Run `subjectivity_detection.ipynb` for text extraction, preprocessing, and classification.
    - Tune the LSTM and pre-trained transformer models and evaluate performance.

## Contributing
Feel free to fork this project, submit pull requests, and contribute to the development. For any issues or enhancements, please create an issue in the GitHub repository.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

