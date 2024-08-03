# COMP472 Project

Github Repository:  (https://github.com/RevoSam/COMP472/tree/main)

This repository contains the work for a deep learning project focused on facial expression recognition using Convolutional Neural Networks (CNNs). The project is divided into two main parts: data preprocessing and CNN model training and evaluation.

## Repository Structure

### Datasets
The final dataset used for this project has been preprocessed and curated from multiple public sources. The raw data has been cleaned, balanced, and split into training, validation, and test sets. Due to storage limitations on GitHub, we have uploaded the final curated dataset, ready for use in model training and evaluation.

#### Sources:
- [Facial Emotion Recognition Dataset](https://www.kaggle.com/datasets/tapakah68/facial-emotion-recognition)
- [OSF Dataset](https://osf.io/f7zbv/)
- [MULTIRACIAL FACE DATABASE](https://jacquelinemchen.wixsite.com/sciplab/face-database)
- [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset)
- [Real and Fake Face Detection](https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection)
- [spenceryee/CS229](https://github.com/spenceryee/CS229)

### Code Files
The `_src` and `Scripts` directories contain Python scripts used for data cleaning, preprocessing, visualization, and dataset processing. The main contributions in part two are the PyTorch code for CNNs, including different variants, and code for evaluation, saving, loading, and testing the models.

#### Main Scripts:
- `model.py`: Contains the definition of the ConvNet class with three different network variants for facial expression recognition.
- `data_loader.py`: A script to load the preprocessed dataset, including functionalities for data balancing and splitting.
- `train_utils.py`: Utilities for training the models, including functions for early stopping, saving models, and drawing confusion matrices.
- `main.py`: The main script to run the training and testing phases of the project.

### Environment Setup
To run the code in this repository, ensure you have Python 3.6+ installed. Then, install the required libraries using pip:
```bash
pip install torch torchvision pandas matplotlib pillow scikit-learn seaborn numpy
```

### Running the Code

#### Data Preprocessing
Data preprocessing has already been performed, and the final dataset is provided. Therefore, no explicit preprocessing steps are required before model training.

#### Data Visualization
Data visualization graphs and related code subtends directly from part 1. Users can run `visualization.py` inside `part_1_src`. This will show class distibution, sample images from each class, and pixel intensity distribution.
```bash
python visualisation.py
```
#### Training the Models
To train the CNN models, navigate to the project's `part_2_src` directory and execute the `main.py` script:
```bash
python main.py
```
This script will automatically load the dataset, perform model training for all variants, and save the best-performing models.

#### Testing the Models
For testing and evaluating the saved models, use the `standalone_test.py` script inside  `part_2_src`. This script loads the saved models, evaluates them on the test dataset, and prints the classification report and confusion matrix:
```bash
python standalone_test.py
```

### Saved Models
Due to file size constraints, the trained models are stored on Google Drive. Download the models from the provided link and place them in a directory named `Models` in the project's root. Use the `standalone_test.py` script to load and evaluate these models. 

[Link to google drive to check saved models](https://drive.google.com/drive/folders/1-w1JAn1FU4UtxrF4phjeL3hhTQ-ARwbN)

### Further Instructions
- All code should be executed in the same directory where the python file is present to ensure paths are correctly resolved.
- The repository includes a brief description and purpose for each script and detailed steps to run the scripts for data cleaning, visualization, and model training and evaluation.
- Ensure to follow the steps outlined in the `standalone_test.py` for evaluating the pre-trained models.
