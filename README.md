# Implementing-Support-Vector-Machines-for-Income-and-Age-Cluster-Analysis

#Project Description
This project demonstrates the use of Support Vector Machines (SVM) for clustering income and age data. It includes generating synthetic data, scaling the data, training an SVM model, and visualizing the results.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Examples](#examples)
  - [Data Generation](#data-generation)
  - [Scaling Data](#scaling-data)
  - [Training the SVM Model](#training-the-svm-model)
  - [Plotting Predictions](#plotting-predictions)
  - [Making Predictions](#making-predictions)
- [License](#license)

## Installation
To run this project, you need to have Python installed along with the following libraries:
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install these libraries using pip:
```bash
pip install numpy matplotlib scikit-learn
```

## Usage
To run the project, simply execute the Python script. The script will generate synthetic data, scale it, train an SVM model, and plot the results.

## Project Structure
```bash
svm-income-age-clustering/
├── notebooks/
    └── svm_income_age_clustering.ipynb             # Jupyter notebook with the code
    ├── data_generation                             # Data generation scripts
    ├── data_scaling                                # Data scaling scripts
    ├── svm_training                                # SVM training scripts
    ├── plotting                                    # Plotting utility scripts
    └── predictions                                 # Prediction scripts                         
├── README.md                                       # Project README file
└── requirements.txt                                # List of dependencies
```

## Examples

- Data Generation
This section generates synthetic income and age data for clustering.
```python
import numpy as np

def createClusteredData(N, k):
    np.random.seed(1234)
    pointsPerCluster = float(N) / k
    X = []
    y = []
    for i in range(k):
        incomeCentroid = np.random.uniform(20000.0, 200000.0)
        ageCentroid = np.random.uniform(20.0, 70.0)
        for j in range(int(pointsPerCluster)):
            X.append([np.random.normal(incomeCentroid, 10000.0), np.random.normal(ageCentroid, 2.0)])
            y.append(i)
    X = np.array(X)
    y = np.array(y)
    return X, y

(X, y) = createClusteredData(100, 5)
```

- Scaling the Data
This section scales the generated data using MinMaxScaler.
```python
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

scaling = MinMaxScaler(feature_range=(-1, 1)).fit(X)
X = scaling.transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y.astype(float))
plt.show()
```

- Training SVM Model
```python
from sklearn import svm

C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
```

- Plotting Predictions
```python
def plotPredictions(clf):
    xx, yy = np.meshgrid(np.arange(-1, 1, .001), np.arange(-1, 1, .001))
    npx = xx.ravel()
    npy = yy.ravel()
    samplePoints = np.c_[npx, npy]
    Z = clf.predict(samplePoints)

    plt.figure(figsize=(8, 6))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.astype(float))
    plt.show()

plotPredictions(svc)
```

- Making Predictions
```python
print(svc.predict(scaling.transform([[200000, 40]])))
print(svc.predict(scaling.transform([[50000, 65]])))
```

# License

This README file provides a comprehensive overview of the project, including installation instructions, usage, project structure, and examples of the different steps involved in the project. The project structure follows best practices for organizing Python projects, making it easy to navigate and maintain.
