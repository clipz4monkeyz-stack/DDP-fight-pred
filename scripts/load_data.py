#!/usr/bin/env python3
"""
Load UFC datasets for the DDP-fight-pred project.
This script contains the imports provided by the user and example code to download the two CSVs from the given URLs.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

url_fight = "https://raw.githubusercontent.com/ThasankaK/UFC-Dataset-and-Model-Predictor/refs/heads/master/ufc_event_fight_stats.csv"
url_fighter = "https://raw.githubusercontent.com/ThasankaK/UFC-Dataset-and-Model-Predictor/refs/heads/master/ufc_fighters_avg.csv"
