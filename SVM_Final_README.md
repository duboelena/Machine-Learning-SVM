Data from Kaggle: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data
Install and load the following:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import learning_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

The first portion of the notebook is data configuration and EDA, followed by train/test splitting and scaling, then finally model and results. 
