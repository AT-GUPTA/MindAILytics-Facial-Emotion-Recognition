import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from data_loader import gather_image_data, FacesDataset, balanceDataFrame, split_dataset
import train_utils as tu
from model import ConvNet
from sklearn.metrics import classification_report
import os
from PIL import Image, ImageOps


def getIndividualImg(img_path, transform):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4728,), (0.2987,))
    ])
    img = Image.open(img_path)
    if transform:
        img = transform(img)
    img = ImageOps.grayscale(img)
    return img


def load_model_and_predict_one_img(model_name, transform):
    classes = ('focused', 'happy', 'neutral', 'surprised')
    checkpoint = torch.load(model_name, map_location=device)
    model = ConvNet(network=int(model_name.split('_')[1])).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Select just the first batch of the test loader
    images, labels, _ = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    all_predictions = []
    all_labels = []

    # Evaluation loop
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    index = np.random.randint(low=0, high=(len(test_loader) - 1))
    tu.displayPics(images[index], None, labels[index], all_predictions[index], classes, one_pic=True)


# Function to load model and evaluate
def load_model_and_evaluate(model_name, test_dataset, kernel_size=3, network=1):
    classes = ('focused', 'happy', 'neutral', 'surprised')
    checkpoint = torch.load(model_name, map_location=device)
    model = ConvNet(network=network, kernel_size=kernel_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_predictions = []
    all_labels = []

    # Evaluation loop
    with torch.no_grad():
        for data in test_loader:
            if len(data) == 3:
                images, labels, _ = data
            else:
                images, labels = data

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    print(f"Evaluating Model: {model_name}")
    print(classification_report(all_labels, all_predictions, target_names=['focused', 'happy', 'neutral', 'surprised']))
    tu.getMicroAvg(all_labels, all_predictions)

    # Confusion Matrix
    tu.draw_confusion_matrix(all_labels, all_predictions, classes, model_name)


def compare_best_model_kernel_variations(transform, directory='./Models', test_dataset=None, individualImg=False):
    assert test_dataset is not None, "test_dataset must be provided"

    # Define the specific model files to look for
    model_filenames = ['model_1_2.8174.pht', 'Kernel_5_model_1_2.0494.pht', 'Kernel_7_model_1_1.3953.pht']
    model_paths = [os.path.join(directory, filename) for filename in model_filenames]

    no_models = True

    # Check if specified model files exist
    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Model file does not exist: {model_path}")
        else:
            no_models = False
    if no_models:
        return
    # Evaluate each specified model
    for model_path in model_paths:
        # Extract kernel size from filename
        if os.path.exists(model_path):
            try:
                if 'Kernel_5_model_1_2.0494.pht' in model_path:
                    kernel_size = 5
                    network = 1
                elif 'Kernel_7_model_1_1.3953.pht' in model_path:
                    kernel_size = 7
                    network = 1
                else:
                    kernel_size = 3
                    network = 1
                    print(f"\nEvaluating model file: {model_path}")
                if individualImg:
                    load_model_and_predict_one_img(model_path, transform)
                else:
                    
                    load_model_and_evaluate(model_path, test_dataset,kernel_size=kernel_size, network=network)
            except RuntimeError as e:
                print(f'Problem with {model_path}, {e}')
            


# Function to compare the best models
def compare_best_models(transform, directory='./Models', test_dataset=None, individualImg=False):
    assert test_dataset is not None, "test_dataset must be provided"

    model_files = sorted(os.listdir(directory))
    best_models = {}

    # Collect the best models based on lowest loss
    for file in model_files:
        if file.startswith('model_') and file.endswith('.pht'):
            model_type = file.split('_')[1]
            loss = float(file.split('_')[-1].replace('.pht', ''))
            if model_type not in best_models or best_models[model_type][1] > loss:
                best_models[model_type] = (file, loss)
    print(best_models.items())
    # Evaluate best models
    for model_type, (model_file, loss) in best_models.items():
        model_path = os.path.join(directory, model_file)
        print(f"\nEvaluating {model_type} with model file: {model_file} having loss: {loss}")
        network = int(model_path.split('_')[1])
        if individualImg:
            load_model_and_predict_one_img(model_path, transform)
        else:
            load_model_and_evaluate(model_path, test_dataset, kernel_size=3, network=network)


print("--------------------- Loading Data ---------------------", flush=True)
df = gather_image_data()
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4728,), (0.2987,))
])
dataset = FacesDataset(df, transform)

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

train_dataset, validation_dataset, test_dataset = FacesDataset(train_df, transform), FacesDataset(validation_df,
                                                                                                  transform), FacesDataset(
    test_df, transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("--------------------- Testing ---------------------", flush=True)


# Example on how to call compare_best_models function

# compare_best_models(transform, directory='../Models', test_dataset=test_dataset, individualImg=False)

compare_best_model_kernel_variations(transform, directory='./Models', test_dataset=validation_dataset, individualImg = False)
