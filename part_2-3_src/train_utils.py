import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import os
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels
import torchvision.transforms as transforms
import seaborn as sns
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from model import ConvNet
import data_loader as dl
import torch.nn as nn
from PIL import Image
from data_loader import add_bias, FacesDataset, balanceDataFrame, split_dataset

# Displays images with Actual, Predicted, and name of pictures
def displayPics(images, names, all_labels, all_predictions, classes, one_pic=False):
    if one_pic:
        fig, ax = plt.subplots(figsize=(5, 3))
        image = np.transpose(images.cpu().numpy(),
                             (1, 2, 0))  # Transpose image to (height, width, channels) for matplotlib
        ax.imshow(image, cmap='gray')
        title = f'P: {all_predictions}\nT: {all_labels}'
        label = classes[all_labels]
        prediction = classes[all_predictions]
        title = f'P: {prediction}\nT: {label}'
        if names:
            title += f'\n{names}'
        ax.set_title(title, size=12)
        ax.axis('off')
    else:
        rows = int(len(images) / 5)
        _, axs = plt.subplots(rows, 5, figsize=(15, rows * 3))

        for i in range(rows):
            for j in range(5):
                index = i * 5 + j
                if index < len(all_labels):
                    image = np.transpose(images[index].cpu().numpy(),
                                         (1, 2, 0))  # Transpose image to (height, width, channels) for matplotlib
                    label = classes[all_labels[index]]
                    prediction = ''
                    if all_predictions:
                        prediction = classes[all_predictions[index]]
                    axs[i, j].imshow(image, cmap='gray')
                    title = f'P: {prediction}\nT: {label}'
                    if names:
                        img_name = names[index]
                        title += f'\n{img_name}'
                    axs[i, j].set_title(title, size=12)
                axs[i, j].axis('off')
    plt.tight_layout()
    plt.show()


# Returns true if the loss is continually increased beyond the set tolerance
def early_stopping(tolerance_counter, loss_score) -> bool:
    counter = 0
    stop = False
    if len(loss_score) > 1:
        for i in range(len(loss_score) - 1):
            if (loss_score[i + 1] > loss_score[i]):
                counter += 1
            else:
                counter = 0
            if (counter > tolerance_counter):
                stop = True
                break
    return stop


# Loops through Models folder and return the Model with the minimum loss based on the specified network (Main Model = 1, Variant 1 = 2, Variant 2 = 3)
def getBestModel(modelPath='./drive/MyDrive/Models', token='model_1_') -> str:
    best_score = float('inf')
    best_model_name = None
    for models in os.listdir(modelPath):
        p = Path(os.path.join(modelPath, models))
        model_name = p.stem
        print(f'{model_name=}, {len(model_name.split(token))=}')
        if len(model_name.split(token)) > 1:
            score = float(model_name.split(token)[1])
            if (score < best_score):
                best_score = score
                best_model_name = os.path.join(modelPath, models)
    return best_model_name


# Saves the model into the Models folder
def saveModel(model, optimizer, network, model_path, fold, epoch, best_valid_loss, test):
  model_name = 'Kernel_3_model_' + str(network) + '_' + str(f'{best_valid_loss:.4f}') + '.pht'
  model_path = os.path.join(model_path, model_name)
  if not test:
    torch.save({
              'fold' : fold,
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'BEST_SCORE_VAL': best_valid_loss
          }, model_path)
  print(f'{model_name} with Best Validation Loss [{best_valid_loss:.4f}] has been saved!')


# Draes Confusion matrix based on the labels and predictioned supplied
def draw_confusion_matrix(all_labels, all_predictions, classes, model_name=None):
    cm = confusion_matrix(all_labels, all_predictions)
    title_model_name = ''
    if model_name:
        title_model_name = ' for ' + model_name
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix {title_model_name}')
    plt.show()


# Calculates Micro Avg
def getMicroAvg(all_labels, all_predictions, out_dict = False):

    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions)

    np.seterr(invalid='ignore')

    tp_fp_sum = tp + fp
    micro_precision = np.where(tp_fp_sum > 0, tp / tp_fp_sum, 0)
    tp_fn_sum = tp + fn
    micro_recall = np.where(tp_fn_sum > 0, tp / tp_fn_sum, 0)

    precision_recall_sum = micro_precision + micro_recall
    micro_f1 = np.where(precision_recall_sum > 0, 2 * (micro_precision * micro_recall) / precision_recall_sum, 0)   

    micro_avg_precision = micro_precision.mean() if micro_precision.sum() > 0 else 0
    micro_avg_recall = micro_recall.mean() if micro_recall.sum() > 0 else 0
    micro_avg_f1 = micro_f1.mean() if micro_f1.sum() > 0 else 0

    if not out_dict:
        return f'micro avg\t\t\t{micro_avg_precision:.2f}\t\t\t {micro_avg_recall:.2f}\t\t\t {micro_avg_f1:.2f}'
    else:
        return {'micro_avg_precision': micro_avg_precision, 'micro_avg_recall': micro_avg_recall, 'micro_avg_f1': micro_avg_f1}


# Helper function designed to overfit to test if a model can convege or not
# Characterized by a large number of epochs (1,000 or 10,000) with a batch size of 1
def test_overfitting(dataset, batch_size, device, criterion, optimizer, num_epochs, model, scheduler):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    total = 0
    correct = 0
    training_loss = 0
    for epoch in range(num_epochs):
        images, labels, names = next(iter(loader))

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        training_loss += loss.item()

        print(f'{loss.item()=}')

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()
    accuracy = (correct / total) * 100.0
    print(f'{(training_loss/num_epochs)=}, {accuracy=}')


# Responsible for Model Trainning for one Epoch
def traningOneEpoch(model, train_loader, device, criterion, optimizer, epoch=''):
    n_train_total_steps = len(train_loader)
    training_loss = 0
    total = 0
    correct = 0
    for i, (images, labels, names, _, _) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        training_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'{epoch} step [{i + 1}/{n_train_total_steps}], loss: {loss.item():.4f}')
    accuracy = (correct / total) * 100.0
    return accuracy, training_loss


# Performs validation for the neural network model.
def validationOneEpoch(model, validation_loader, criterion, all_predictions, all_labels, device):
    validation_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for (images, labels, names, _, _) in validation_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        accuracy = (correct / total) * 100.0
    return accuracy, validation_loss


# Main function to train the models
def trainingPhase(model, network, train_loader, validation_loader, device, criterion, optimizer, scheduler, num_epochs):
    BEST_SCORE_VAL = float('inf')
    tolerance = 5
    val_loss_score = []
    all_predictions = []
    all_labels = []

    model_path = './Models'

    for epoch in range(num_epochs):

        model.train()

        epoch_str = f'Epoch[{epoch + 1}/{num_epochs}]'

        accuracy, training_loss = traningOneEpoch(model, train_loader, device, criterion, optimizer, epoch_str)

        scheduler.step()

        model.eval()

        accuracy, validation_loss = validationOneEpoch(model, validation_loader, criterion, all_predictions, all_labels,
                                                       device)

        validation_loss = validation_loss / len(validation_loader)

        val_loss_score.append(validation_loss)

        print(f'Epoch[{epoch + 1}/{num_epochs}], Validation loss: {validation_loss:.4f}, accuracy: {accuracy:.2f}')

        if early_stopping(tolerance, val_loss_score):
            print(f'Stopping Early!')
            break
        if validation_loss < BEST_SCORE_VAL:
            BEST_SCORE_VAL = validation_loss
            saveModel(model, network, optimizer, model_path, epoch, BEST_SCORE_VAL)

    print('Finished Training')
    labels = ['focused', 'happy', 'neutral', 'surprised']
    print(classification_report(all_labels, all_predictions, target_names=labels))
    getMicroAvg(all_labels, all_predictions)
    draw_confusion_matrix(all_labels, all_predictions)


# Main function to testing the Models
def testingPhase(model, test_loader, device, classes, printPictures=True, showConfusionMatrix=False):
    images, labels, predicted = None, None, None

    model.eval()

    all_predictions = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        n_correct = 0
        n_samples = 0

        for (images, labels, names, _, _) in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if printPictures:
                displayPics(images, names, all_labels, all_predictions, classes)

    print('Finished Testing')
    # labels = ['focused', 'happy', 'neutral', 'surprised']
    # print(classification_report(all_labels, all_predictions, target_names=labels))

    print_model_metrics(all_labels, all_predictions)

    if showConfusionMatrix:
        draw_confusion_matrix(all_labels, all_predictions, classes)


# Helper Function that was is used to Pick up the trainning where it stopped
def continueTraining(model, num_epochs, train_loader, validation_loader, device, criterion, optimizer, scheduler,
                     modelPath='./drive/MyDrive/Models', network=1):
    token = 'model_' + str(network) + '_'
    # Load the model
    checkpoint = torch.load(getBestModel(modelPath, token))

    # Restore model and optimizer states
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_valid_loss = checkpoint['BEST_SCORE_VAL']
    epoch = checkpoint['epoch']

    BEST_SCORE_VAL = float('inf')
    tolerance = 5
    val_loss_score = []
    all_predictions = []
    all_labels = []

    starting_epoch = int(epoch)

    for epoch in range(starting_epoch, num_epochs):
        model.train()

        epoch_str = f'Epoch[{epoch + 1}/{num_epochs}]'

        accuracy, training_loss = traningOneEpoch(model, train_loader, device, criterion, optimizer, epoch_str)

        scheduler.step()

        model.eval()

        accuracy, validation_loss = validationOneEpoch(model, validation_loader, criterion, all_predictions, all_labels)

        validation_loss = validation_loss / len(validation_loader)

        val_loss_score.append(validation_loss)

        print(f'Epoch[{epoch + 1}/{num_epochs}], Validation loss: {validation_loss:.4f}, accuracy: {accuracy:.2f}')

        if early_stopping(tolerance, val_loss_score):
            print(f'Stopping Early!')
            break
        if validation_loss < BEST_SCORE_VAL:
            BEST_SCORE_VAL = validation_loss
            saveModel(model, network, modelPath, epoch, BEST_SCORE_VAL)

    print('Finished Training')
    labels = ['focused', 'happy', 'neutral', 'surprised']
    print(classification_report(all_labels, all_predictions, target_names=labels))
    draw_confusion_matrix(all_labels, all_predictions)


def calculateDFMetrics(data, subgroup_column):
    subgroup_data = data.groupby(subgroup_column).apply(lambda x: x.drop(columns=subgroup_column))
    accuracy = accuracy_score(subgroup_data['emotion'], subgroup_data['predicted'])
    precision = precision_score(subgroup_data['emotion'], subgroup_data['predicted'], average='weighted')
    recall = recall_score(subgroup_data['emotion'], subgroup_data['predicted'], average='weighted')
    f1 = f1_score(subgroup_data['emotion'], subgroup_data['predicted'], average='weighted')
    confusion = confusion_matrix(subgroup_data['emotion'], subgroup_data['predicted'])
    return accuracy, precision, recall, f1, confusion
    
# Calculates macro for gender and age groups
def metricspersubFeatures(model, ds, device,batch_size):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
       
    all_ages_predicted = [[],[],[]]
    all_ages_labels = [[],[],[]]
    
    all_genders_predicted = [[],[]]
    all_genders_labels = [[],[]]

    model.eval()

    with torch.no_grad():

        for (images, labels, names, ages, genders) in loader:

            images = images.to(device)
            labels = labels.to(device)
            ages = ages.to(device)
            genders = genders.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            age_groups = [1,2,3]
            #Loop through the age groups and add the labels and predicted values
            for c in age_groups:
                #[ages == c] will return the indices where ages == c is true
                #predicted[ages == c] will return only the predicted values for the specified age_group
                all_ages_predicted[c-1].extend(predicted[ages == c].cpu().numpy())
                #same Technique for the labels
                all_ages_labels[c-1].extend(labels[ages == c].cpu().numpy())
            
            #We do the same for the gender
            gender_groups = [1,2]
            for c in gender_groups:
                all_genders_predicted[c-1].extend(predicted[genders == c].cpu().numpy())
                all_genders_labels[c-1].extend(labels[genders == c].cpu().numpy())

    labels = ['focused', 'happy', 'neutral', 'surprised']
    age_groups = [1,2,3]
    gender_groups = [1,2]

    #Name of the age group and genders categories
    age_groups_labels = ["Young Adult", "Middle Aged Adult", "Old-aged Adult"]
    gender_groups_labels = ["Female", "Male"]

    #Display the metrics per age group
    for age_group in age_groups:
        print(f'\n---------------------------------------------------')
        print(f'{age_groups_labels[age_group - 1]}')
        print(f'---------------------------------------------------')
        #Filters the labels and predicted to include only the emotions that exist in both labels and predicted
        #Otherwise classification_report will not work properly if labels has 4 rows but predicted has only 3 for example.
        target_names_filtered = [labels[i] for i in unique_labels(all_ages_labels[age_group - 1], all_ages_predicted[age_group - 1])]
        print(classification_report(all_ages_labels[age_group - 1], all_ages_predicted[age_group - 1], target_names=target_names_filtered, zero_division=0.0))
    #Display the metrics per gender group
    for gender_group in gender_groups:
        print(f'\n---------------------------------------------------')
        print(f'{gender_groups_labels[gender_group - 1]}')
        print(f'---------------------------------------------------')
        target_names_filtered = [labels[i] for i in unique_labels(all_genders_labels[gender_group - 1], all_genders_predicted[gender_group - 1])]
        print(classification_report(all_genders_labels[gender_group - 1], all_genders_predicted[gender_group - 1], target_names=target_names_filtered, zero_division=0.0))

#Visual Helper to show the distribution of the classes in a fold
def plot_combined_emotion_distribution(train_emotions, test_emotions, fold, save=True):
    unique_emotions = np.unique(np.concatenate([train_emotions, test_emotions]))
    train_counts = np.bincount(train_emotions, minlength=len(unique_emotions))
    test_counts = np.bincount(test_emotions, minlength=len(unique_emotions))
    width = 0.35

    file_name = f'Fold {fold} - Combined Emotion Distribution'

    fig, ax = plt.subplots()
    ax.bar(unique_emotions - width/2, train_counts, width, label='Train')
    ax.bar(unique_emotions + width/2, test_counts, width, label='Test')
    ax.set_title(file_name)
    ax.set_xlabel('Emotion')
    ax.set_ylabel('Count')
    ax.set_xticks(unique_emotions)
    ax.legend()
    if save:
        file_name = './part_3_visualizations/' + file_name 
        plt.savefig(file_name)
        plt.close()
    else:
        plt.show()

#Trainning using Stratified K-Folds to make sure all the emotions are equally represented in the trainning and the data validation process
def StratifiedKFold_init(df, device, transform, model_name, skipTrainning = True, print_sample_pics = False):

    n_splits = 10
    epochs = 30
    batch_size = 64
    leanring_rate = 0.001
    tolerance = 3

    in_channels = 1
    out_channels = 32
    kernel_size = 3
    drop_out = 0.2
    network=1

    BEST_SCORE_VAL = float('inf')

    model = ConvNet(network=network, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, drop_out=drop_out).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=leanring_rate)

    model_path = './Models'

    model_name = os.path.join(model_path, model_name)

    if skipTrainning:
        checkpoint = torch.load(model_name, map_location=torch.device(device))
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()



    best_model = None
    best_model_optimizer = None
    best_model_fold = None
    best_model_epoch = None

    random_state_on = True

    

    seed = 0

    all_results = []


    if random_state_on:
        seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    balancedDF = dl.balanceDataFrame(df)

    emotions =  balancedDF['emotion'].values

    fold = 0
    epoch = 0

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_index, test_index in kf.split(balancedDF, emotions):

        fold += 1
        if not skipTrainning:
            print(f'Starting Trainning for Fold :{fold}')

        train_df = balancedDF.iloc[train_index]
        test_df = balancedDF.iloc[test_index]

        if print_sample_pics:
            plot_sample_images(train_df, 'trainning', fold, seed)
            plot_sample_images(test_df, 'testing', fold, seed)

        train_emotions = train_df['emotion'].values
        test_emotions = test_df['emotion'].values

        #Displays the distributino of the classes @ each fold
        print(f"Fold {fold} - train distribution: {np.bincount(train_emotions)}")
        print(f"Fold {fold} - test distribution: {np.bincount(test_emotions)}")

        plot_combined_emotion_distribution(train_emotions, test_emotions, fold, True)

        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        train_dataset = dl.FacesDataset(train_df, transform)
        test_dataset = dl.FacesDataset(test_df, transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if not skipTrainning:
            model = ConvNet(network=network, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, drop_out=drop_out).to(device)

            criterion = nn.CrossEntropyLoss()

            optimizer = optim.Adam(model.parameters(), lr=leanring_rate)

            # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma = 0.05)

            earlyStopped = False

            val_loss_score = []

            epoch = 0

            while not earlyStopped and epoch < epochs:

                epoch += 1

                model.train()

                epoch_str = f'Epoch[{epoch}/{epochs}]'

                trainning_accuracy, training_loss = traningOneEpoch(model, train_loader, device, criterion, optimizer, epoch_str)
                training_loss = training_loss / len(train_loader)

                val_loss_score.append(training_loss)

                print(f'Fold[{fold}/{n_splits}], Epoch[{epoch}/{epochs}], Trainning loss: {training_loss:.4f}, Trainning accuracy: {trainning_accuracy:.2f}')

                if early_stopping(tolerance, val_loss_score):
                    print(f'Early Stopping @ Epoch: {epoch}')
                    earlyStopped = True

            # scheduler.step()
        model.eval()

        all_predictions = []
        all_labels = []

        validation_accuracy, validation_loss = validationOneEpoch(model, test_loader, criterion, all_predictions, all_labels, device)
        validation_loss = validation_loss / len(test_loader)

        print(f'Fold[{fold}/{n_splits}], Validation loss: {validation_loss:.4f}, accuracy: {validation_accuracy:.2f}')

        all_results.append({'all_labels': all_labels, 'all_predictions': all_predictions})

        test = True

        if validation_loss < BEST_SCORE_VAL:
            BEST_SCORE_VAL = validation_loss
            best_predictions = all_predictions
            best_labels = all_labels
            best_model = model
            best_model_optimizer = optimizer
            best_model_fold = fold
            best_model_epoch = epoch      
            if not skipTrainning:  
                saveModel(best_model, best_model_optimizer, network, model_path, best_model_fold, best_model_epoch, BEST_SCORE_VAL, test)
        if not skipTrainning:
            print(f'Trainnig done for fold {fold}')

    if not skipTrainning:
        saveModel(best_model, best_model_optimizer, network, model_path, best_model_fold, best_model_epoch, BEST_SCORE_VAL, False)
    return best_model, best_labels, best_predictions, all_results

#Helper method that prints macro and micro metrics for the provided labels and predictions
def print_metrics(all_results):
    classification_rep_macro_accumulator = [0.0,0.0,0.0,0.0]
    classification_rep_micro_accumulator = [0.0,0.0,0.0]
    labels = ['focused', 'happy', 'neutral', 'surprised']

    n_splits = len(all_results)

    print_folds = True if n_splits > 1 else False

    for fold, result in enumerate(all_results):
        if print_folds:
            print('\n------------------------------------------------')
            print(f'Metrics for FOLD: {fold + 1}')
            print('------------------------------------------------\n')

        classification_rep = classification_report(result['all_labels'], result['all_predictions'], target_names=labels, output_dict=True, zero_division=0.0)
        classification_rep_micro = getMicroAvg(result['all_labels'], result['all_predictions'], True)

        classification_rep_macro_accumulator[0] += classification_rep['macro avg']['precision']
        classification_rep_macro_accumulator[1] += classification_rep['macro avg']['recall']
        classification_rep_macro_accumulator[2] += classification_rep['macro avg']['f1-score']
        classification_rep_macro_accumulator[3] += classification_rep['accuracy']

        classification_rep_micro_accumulator[0] += classification_rep_micro['micro_avg_precision']
        classification_rep_micro_accumulator[1] += classification_rep_micro['micro_avg_recall']
        classification_rep_micro_accumulator[2] += classification_rep_micro['micro_avg_f1']

        header = ['precision','recall','f1-score','support']

        print(f'\t\t{header[0]}\t  {header[1]}\t{header[2]}\t {header[3]}')

        for key, value in classification_rep.items():
            if key in labels:
                precision = value['precision']
                recall = value['recall']
                f1_score = value['f1-score']
                support = value['support']
                print(f'{key:12}\t{precision:9.2f}\t{recall:8.2f}\t{f1_score:8.2f}\t{support:8}')

        print(f'\n')
        print(f"Accuracy\t\t\t\t\t{classification_rep['accuracy']:8.2f}\t{classification_rep['macro avg']['support']:8}")
        print(f"macro avg\t{classification_rep['macro avg']['precision']:9.2f}\t{classification_rep['macro avg']['recall']:8.2f}\t{classification_rep['macro avg']['f1-score']:8.2f}")
        print(f"micro avg\t{classification_rep_micro['micro_avg_precision']:9.2f}\t{classification_rep_micro['micro_avg_recall']:8.2f}\t{classification_rep_micro['micro_avg_f1']:8.2f}")

    if print_folds:
        print('\n------------------------------------------------')
        print(f'Averages:')
        print('------------------------------------------------\n')
        print(f"Accuracy\t\t\t\t\t{classification_rep_macro_accumulator[3]/n_splits:8.2f}")
        print(f"macro avg\t{classification_rep_macro_accumulator[0]/n_splits:9.2f}\t{classification_rep_macro_accumulator[1]/n_splits:8.2f}\t{classification_rep_macro_accumulator[2]/n_splits:8.2f}")
        print(f"micro avg\t{classification_rep_micro_accumulator[0]/n_splits:9.2f}\t{classification_rep_micro_accumulator[1]/n_splits:8.2f}\t{classification_rep_micro_accumulator[2]/n_splits:8.2f}")

#Helper Method that prints a classification report for the specified labels and prediction by creating the all_result dic that is used in print_metrics
def print_model_metrics(best_labels, best_predictions):
    print('\n------------------------------------------------')
    print(f'Classification Report For the Model:')
    print('------------------------------------------------\n')
    all_results = []
    all_results.append({'all_labels': best_labels, 'all_predictions': best_predictions})
    print_metrics(all_results)

# def kfold_metrics(all_results, best_labels, best_predictions):
#     classification_rep_macro_accumulator = [0.0,0.0,0.0,0.0]
#     classification_rep_micro_accumulator = [0.0,0.0,0.0]
#     labels = ['focused', 'happy', 'neutral', 'surprised']
#     n_splits = len(all_results)

#     for fold, result in enumerate(all_results):
#         print('\n------------------------------------------------')
#         print(f'Metrics for FOLD: {fold + 1}')
#         print('------------------------------------------------\n')

#         classification_rep = classification_report(result['all_labels'], result['all_predictions'], target_names=labels, output_dict=True, zero_division=0.0)
#         classification_rep_micro = getMicroAvg(result['all_labels'], result['all_predictions'], True)

#         classification_rep_macro_accumulator[0] += classification_rep['macro avg']['precision']
#         classification_rep_macro_accumulator[1] += classification_rep['macro avg']['recall']
#         classification_rep_macro_accumulator[2] += classification_rep['macro avg']['f1-score']
#         classification_rep_macro_accumulator[3] += classification_rep['accuracy']

#         classification_rep_micro_accumulator[0] += classification_rep_micro['micro_avg_precision']
#         classification_rep_micro_accumulator[1] += classification_rep_micro['micro_avg_recall']
#         classification_rep_micro_accumulator[2] += classification_rep_micro['micro_avg_f1']

#         header = ['precision','recall','f1-score','support']

#         print(f'\t\t{header[0]}\t  {header[1]}\t{header[2]}\t {header[3]}')

#         for key, value in classification_rep.items():
#             if key in labels:
#                 precision = value['precision']
#                 recall = value['recall']
#                 f1_score = value['f1-score']
#                 support = value['support']
#                 print(f'{key:12}\t{precision:9.2f}\t{recall:8.2f}\t{f1_score:8.2f}\t{support:8}')

#         print(f'\n')
#         print(f"Accuracy\t\t\t\t\t{classification_rep['accuracy']:8.2f}\t{classification_rep['macro avg']['support']:8}")
#         print(f"macro avg\t{classification_rep['macro avg']['precision']:9.2f}\t{classification_rep['macro avg']['recall']:8.2f}\t{classification_rep['macro avg']['f1-score']:8.2f}")
#         print(f"micro avg\t{classification_rep_micro['micro_avg_precision']:9.2f}\t{classification_rep_micro['micro_avg_recall']:8.2f}\t{classification_rep_micro['micro_avg_f1']:8.2f}")

#     print('\n------------------------------------------------')
#     print(f'Averages:')
#     print('------------------------------------------------\n')
#     print(f"Accuracy\t\t\t\t\t{classification_rep_macro_accumulator[3]/n_splits:8.2f}")
#     print(f"macro avg\t{classification_rep_macro_accumulator[0]/n_splits:9.2f}\t{classification_rep_macro_accumulator[1]/n_splits:8.2f}\t{classification_rep_macro_accumulator[2]/n_splits:8.2f}")
#     print(f"micro avg\t{classification_rep_micro_accumulator[0]/n_splits:9.2f}\t{classification_rep_micro_accumulator[1]/n_splits:8.2f}\t{classification_rep_micro_accumulator[2]/n_splits:8.2f}")

#     print('\n------------------------------------------------')
#     print(f'Classification Report For the Model:')
#     print('------------------------------------------------\n')
#     print(classification_report(best_labels, best_predictions, target_names=labels))

#Visual Helper that prints a sample of pictures in a fold
def plot_sample_images(df, df_name, split, seed):
    emotions = df['emotion'].unique()
    fig, axs = plt.subplots(4, 5, figsize=(12, 12))
    fig.suptitle(f'Sample Images for {df_name}, split: {split}', fontsize=16)
    for row_idx, emotion in enumerate(emotions):
        sample_df = df[df['emotion'] == emotion].sample(n=5, random_state=seed)
        for col_idx, row in enumerate(sample_df.itertuples()):
            img = Image.open(row.image_path).resize((224, 224))
            ax = axs[row_idx, col_idx]
            ax.imshow(img, cmap='gray')
            ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    file_name = './part_3_visualizations/' + df_name + '_Split_' + str(split)
    plt.savefig(file_name)
    plt.close()

#Method responsible of training a model based on the field, percentage to drop, and which category to be dropped
def bias_trainning(field, percentage_data_to_drop, cat_to_drop):
  num_epochs = 30
  batch_size = 64
  leanring_rate = 0.001
  best_model_path = './Models/'
  best_model = None
  best_model_optimizer = None
  best_epoch = 0

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.4728,),(0.2987,))])

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


  classes = ['focused', 'happy', 'neutral', 'surprised']

  train_per = 0.7
  validation_per = 0.15

  df = pd.read_csv("./Datasets/datasets.csv")

  bias_train = add_bias(balanceDataFrame(df), field, percentage_data_to_drop, cat_to_drop)

  train_df, validation_df, test_df = split_dataset(balanceDataFrame(bias_train), train_per, validation_per, seed)

  train_dataset, validation_dataset, test_dataset = FacesDataset(train_df, transform), FacesDataset(validation_df, transform), FacesDataset(test_df, transform)

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

  in_channels = 1
  out_channels = 32
  kernel_size = 3
  drop_out = 0.2
  network=1

  model = ConvNet(network=network, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, drop_out=drop_out).to(device)


  criterion = nn.CrossEntropyLoss()

  optimizer = optim.Adam(model.parameters(), lr=leanring_rate)

  BEST_SCORE_VAL = float('inf')
  tolerance = 5
  val_loss_score = []


  best_predictions = []
  best_labels = []



  for epoch in range(num_epochs):

    model.train()

    epoch_str = f'Epoch[{epoch+1}/{num_epochs}]'

    accuracy, training_loss = traningOneEpoch(model, train_loader, device, criterion, optimizer, epoch_str)

    model.eval()

    all_predictions = []
    all_labels = []

    accuracy, validation_loss = validationOneEpoch(model, validation_loader, criterion, all_predictions, all_labels, device)

    validation_loss = validation_loss / len(validation_loader)


    val_loss_score.append(validation_loss)

    print(f'Epoch[{epoch+1}/{num_epochs}], Validation loss: {validation_loss:.4f}, accuracy: {accuracy:.2f}')

    if early_stopping(tolerance, val_loss_score):
      print(f'Stopping Early!')
      break
    if validation_loss < BEST_SCORE_VAL:
      BEST_SCORE_VAL = validation_loss
      best_predictions = all_predictions
      best_epoch = epoch
      best_labels = all_labels
      best_model = model
      best_model_optimizer = optimizer
      model_name = 'Kernel_3_model_' + f'Bias_{str(percentage_data_to_drop)}' + f'_{field}_' + f'{str(cat_to_drop)}' + '_' + str(network) + '_' + str(f'{BEST_SCORE_VAL:.4f}') + '.pht'
      saveModel(model_name, best_model, best_model_optimizer, network, best_model_path, 0, best_epoch, BEST_SCORE_VAL, False)



  print('Finished Training')
  labels = ['focused', 'happy', 'neutral', 'surprised']
  print_model_metrics(best_labels, best_predictions)
  draw_confusion_matrix(best_labels, best_predictions, classes)
