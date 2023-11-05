import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.model_selection import train_test_split
import os
import warnings

warnings.filterwarnings("ignore")

INTERIM_FOLDER = 'data/interim/'
INTERIM_FILENAME = 'processed.csv'
INTERIM_PATH = os.path.join(INTERIM_FOLDER, INTERIM_FILENAME)

TRAINING_DATA_FOLDER = 'data/training/'

NUM_PRETRAIN = 20000
NUM_PRETRAIN_VAL = 5000
NUM_TRAIN = 100000
NUM_TRAIN_VAL = 10000

# load the data
print(f'loading processed dataset from {INTERIM_PATH}')
df = pd.read_csv(INTERIM_PATH)
df = df[['toxic','detoxified']].rename(columns={'toxic':'input','detoxified':'target'})

# split the datasets
temp, pretrain = train_test_split(df, test_size=NUM_PRETRAIN / len(df), random_state=42)
temp, pretrain_val = train_test_split(temp, test_size=NUM_PRETRAIN_VAL / len(temp), random_state=42)
temp, train = train_test_split(temp, test_size=NUM_TRAIN / len(temp), random_state=42)
test, train_val = train_test_split(temp, test_size=NUM_TRAIN_VAL / len(temp), random_state=42)

# convert datasets into huggingface format
pretrain_dataset = Dataset.from_dict(pretrain.to_dict(orient='list'))
pretrain_val_dataset = Dataset.from_dict(pretrain_val.to_dict(orient='list'))
train_dataset = Dataset.from_dict(train.to_dict(orient='list'))
train_val_dataset = Dataset.from_dict(train_val.to_dict(orient='list'))
test_dataset = Dataset.from_dict(test.to_dict(orient='list'))

# save datasets on disk
print(f'saving split datasets into {TRAINING_DATA_FOLDER}')
pretrain_dataset.save_to_disk(os.path.join(TRAINING_DATA_FOLDER, 'pretrain'))
pretrain_val_dataset.save_to_disk(os.path.join(TRAINING_DATA_FOLDER, 'pretrain_val'))
train_dataset.save_to_disk(os.path.join(TRAINING_DATA_FOLDER, 'train'))
train_val_dataset.save_to_disk(os.path.join(TRAINING_DATA_FOLDER, 'train_val'))
test_dataset.save_to_disk(os.path.join(TRAINING_DATA_FOLDER, 'test'))

print('done')