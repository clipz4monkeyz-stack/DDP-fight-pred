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

def load_data():
    """Download the two CSVs and return them as pandas DataFrames."""
    df_fight = pd.read_csv(url_fight)
    df_fighter = pd.read_csv(url_fighter)
    return df_fight, df_fighter

def main():
    df_fight, df_fighter = load_data()
    print("fight dataset shape:", df_fight.shape)
    print(df_fight.columns.tolist()[:20])
    print("fighter dataset shape:", df_fighter.shape)
    print(df_fighter.columns.tolist()[:20])

    # Print a small sample of each dataframe
    print("\nFight head:\n", df_fight.head().to_string(index=False))
    print("\nFighter head:\n", df_fighter.head().to_string(index=False))

if __name__ == "__main__":
    main()