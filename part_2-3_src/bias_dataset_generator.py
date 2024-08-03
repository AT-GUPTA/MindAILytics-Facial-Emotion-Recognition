from data_loader import split_dataset, balanceDataFrame, dataset_details, add_bias
import numpy as np
import torch
import pandas as pd

df = pd.read_csv("./Datasets/datasets.csv")

# Seed for reproducibility
random_state_on = True
seed = 42
if random_state_on:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Split dataset
train_per = 0.7
validation_per = 0.15


train_df, validation_df, test_df = split_dataset(balanceDataFrame(df), train_per, validation_per, seed)
field = 'gender'
print(train_df.info())
print(dataset_details(train_df, field))
field = 'gender'
print(dataset_details(train_df, field))
# bias_train = add_bias(train_df, 15, 2)
# print(dataset_details(bias_train))

# bias_train = add_bias(train_df, field, 0, 2)
# print(dataset_details(bias_train, field))

# bias_train = add_bias(train_df, field, 25, 2)
# print(dataset_details(bias_train, field))
# # bias_train = add_bias(train_df, 50, 2)
# # print(dataset_details(bias_train))
# bias_train = add_bias(train_df, field, 40, 2)
# print(dataset_details(bias_train, field))

# bias_train = add_bias(train_df, field, 100, 2)
# print(dataset_details(bias_train, field))