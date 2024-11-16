import pandas as pd
import os

def read_data():
    training = pd.read_csv(os.path.join('data', 'twitter_training.csv'))
    test = pd.read_csv(os.path.join('data', 'twitter_validation.csv'))

    if training is None or test is None:
        raise Exception("Unable to read data")

    new_columns = ["id", "origin", "label", "text"]
    training.columns = new_columns

    return training, test