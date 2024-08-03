import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from data_loader import gather_image_data, FacesDataset, balanceDataFrame, split_dataset
import train_utils as tu
from model import ConvNet
import pandas as pd


classes = ('focused', 'happy', 'neutral', 'surprised')

# Load and prepare data
print("--------------------- Loading Data ---------------------", flush=True)
# df = gather_image_data()
csv_file = './Datasets/datasets.csv'
df = pd.read_csv(csv_file)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4728,), (0.2987,))
])


# Seed for reproducibility
random_state_on = True
seed = 0
if random_state_on:
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

classes = ('focused', 'happy', 'neutral', 'surprised')

# Split dataset
train_per = 0.7
validation_per = 0.15

train_df, validation_df, test_df = split_dataset(balanceDataFrame(df), train_per, validation_per, seed)

train_dataset, validation_dataset, test_dataset = FacesDataset(train_df, transform), FacesDataset(validation_df, transform), FacesDataset(test_df, transform)

# Data loaders
batch_size = 64
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model setup
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

in_channels = 1
out_channels = 32
kernel_size = 3
drop_out = 0.2
network=1

model = ConvNet(network=network, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, drop_out=drop_out).to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.01)
# num_epochs = 30

# Training loop
# print("\n--------------------- Training Loop ---------------------")

# tu.trainingPhase(model, network, train_loader, validation_loader, device,  criterion, optimizer, scheduler, num_epochs)

# Testing Phase
# print("\n--------------------- Testing and Evaluation ---------------------")

# token = 'model_' + str(network) + '_'
# model_path = './Models'
# model_name = tu.getBestModel(model_path, token)
# checkpoint = torch.load(model_name, map_location=torch.device(device))

# # # Restore model and optimizer states
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

# # tu.testingPhase(model, test_loader, device, classes, printPictures = True, showConfusionMatrix = False)
# print(f'\n------------------------------------------------')
# print(f'Model From Part II')
# print(f'------------------------------------------------\n')
# tu.metricspersubFeatures(model, test_dataset, device, batch_size)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'\n------------------------------------------------')
print(f'Model - Trainned using K-Folds')
print(f'------------------------------------------------\n')

model_name = 'Model_K_Fold.pht'

model, best_labels, best_predictions, all_results = tu.StratifiedKFold_init(df, device, transform, model_name, True, print_sample_pics = False)

tu.print_metrics(all_results)

tu.print_model_metrics(best_labels, best_predictions)
tu.testingPhase(model, test_loader, device, classes, printPictures = False, showConfusionMatrix = False)
tu.metricspersubFeatures(model, test_dataset, device, batch_size)

print(f'\n------------------------------------------------')
print(f'Model From Part II - Testing Bias')
print(f'------------------------------------------------\n')

model_name = './Models/Model_From_Part_II.pht'
checkpoint = torch.load(model_name, map_location=torch.device(device))
model = ConvNet(network=1, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, drop_out=drop_out).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


tu.testingPhase(model, test_loader, device, classes, printPictures = False, showConfusionMatrix = True)
tu.metricspersubFeatures(model, test_dataset, device, batch_size)



print(f'\n------------------------------------------------')
print(f'Final Version of the Model - Testing Bias')
print(f'------------------------------------------------\n')

model_name = 'Model_Final_Version.pht'

model, best_labels, best_predictions, all_results = tu.StratifiedKFold_init(df, device, transform, model_name, True, print_sample_pics = False)

tu.print_metrics(all_results)

tu.print_model_metrics(best_labels, best_predictions)
tu.testingPhase(model, test_loader, device, classes, printPictures = False, showConfusionMatrix = False)
tu.metricspersubFeatures(model, test_dataset, device, batch_size)