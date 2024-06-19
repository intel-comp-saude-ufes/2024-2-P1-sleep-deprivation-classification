# Sleep Deprivation Classification

This study was done using Python 3.11.9, please guarantee python version before running anyy scripts in order to guarantee compatibility. This project focuses on the study of hyperparameter tunning for the following models: SVM, KNN and Random Forest. The goal is to better understand data in a signal format and the complexity related to the topic.

## 📥 Downloading data

The dataset used [A Resting-state EEG Dataset for Sleep Deprivation](https://openneuro.org/datasets/ds004902/versions/1.0.4) was curated by Xiang, C., Fan, X., Bai, D. et al. A resting-state EEG dataset for sleep deprivation. Sci Data 11, 427 (2024). https://doi.org/10.1038/s41597-024-03268-2 . 

To download the dataset, aws cli was used, but it can be manually downloaded by browser or using other ways cited by authors. 

```sh
mkdir data
cd data
aws s3 sync --no-sign-request s3://openneuro.org/ds004902 ds004902-download/
```

In order to main dataset organized, the command should be run inside a directory named data as shown aboved using unix commands.

## 🔨 Configuring environment

Before installing dependencies, create and start your virtual environment. 
```sh
python -m venv {virtual_env_name}
```

on linux run the command:
```sh
source {virtual_env_name}/bin/activate
```

on window run the command:
```sh
{virtual_env_name}\Scripts\activate
```

Before installing dependencies via terminal, guarantee that the tag `{virtual_env_name}` appears before directory path. Once confirmed, run the command:

```sh
pip install -r requirements.txt
```

## 🧪 Experiments

Experiments were done using Google Colab. Due to this nature, information regarding model training can be found in the notebook [Model_training.ipynb](Model_training.ipynb). The type of experiment is configured using DATA_TYPE variable, acting as a form of enum of the following values:
- 1: Use only closed eye data
- 2: Use only open eye data
- 3: Use both data

Results were saved in a models.zip folder, containing all models trained and csv file regarding both training and test metrics.

## Results

The use of machine learning models is capable of identifying whether an individual had normal sleep or whether they suffered sleep deprivation. Random Forest stands out in terms of accuracy, reaching 93% when using data both with eyes closed and with eyes open. 

It was not possible to identify specific behaviors for hyperparameters of the different models, with the exception of the SVM Kernel type and the weight of the KNN data points. With the Kernel function set to sigmoid, the model easily suffered overfitting. Regarding KNN, due to the nature of the data presenting a continuity between its "_features_", the use of distance as a weight criterion positively influenced the model's performance.

EEG data is highly complex, requiring multidisciplinary work with specialists for better use within computational intelligence. In this study, we focused on the use of the signal and the application of transformations, but there are several other approaches that could be applied. Methods such as extraction or selection of _features_ require greater knowledge regarding the study of waves and brain activity of certain channels. For future work, with more sophisticated data representation, it would be interesting to repeat the grid search experiment and add different neural networks to the study.

