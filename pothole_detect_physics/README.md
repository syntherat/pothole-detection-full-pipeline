# AI-Based Pothole Detection System

## Overview

This project is part of the EPICS program and focuses on detecting potholes using a machine learning approach. The system uses a synthetic dataset representing road surface conditions and trains a model to classify whether a road segment contains a pothole or not.

The goal of this project is to simulate how sensor data from vehicles could be used to automatically identify potholes and help improve road maintenance and safety.

---

## Project Structure

```
EPICS/
│
├── Data/
│   ├── pothole_ai_model.pkl
│   └── synthetic_pothole_dataset.csv
│
├── detector_py/
│   ├── generate_dataset.py
│   ├── pothole_detection.py
│   └── run_detector_on_dataset.py
│
├── Model/
│   └── train_ai_model.py
│
├── .gitignore
└── README.md
```

### Folder Description

**Data/**

* Stores the generated dataset and trained machine learning model.

**detector_py/**

* Contains scripts for dataset generation and pothole detection.

**Model/**

* Contains the script used to train the AI model.

---

## Requirements

Install the required Python libraries before running the project.

```
pip install -r requirements.txt
```

Example libraries used in this project:

```
pandas
numpy
scikit-learn
joblib
```

---

## How to Run the Project

### 1. Generate the Dataset

Run the following script to generate the synthetic pothole dataset.

```
python detector_py/generate_dataset.py
```

This will create:

```
Data/synthetic_pothole_dataset.csv
```

---

### 2. Train the Machine Learning Model

```
python Model/train_ai_model.py
```

This will train the model and save it as:

```
Data/pothole_ai_model.pkl
```

---

### 3. Run the Pothole Detection System

```
python detector_py/run_detector_on_dataset.py
```

This script loads the trained model and runs pothole detection on the dataset.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Joblib

---

## Future Improvements

Some possible future enhancements for this project include:

* Using real sensor data instead of synthetic data
* Integrating GPS data to map pothole locations
* Building a mobile application for real-time pothole reporting
* Using deep learning with road images for visual pothole detection
* Deploying the model as a web or cloud-based service

---

## Author

Developed as part of the EPICS project.
