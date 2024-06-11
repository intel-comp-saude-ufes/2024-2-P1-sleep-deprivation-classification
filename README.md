# Sleep Deprivation Classification

This study was done using Python 3.11.9, please guarantee python version before running anyy scripts in order to guarantee compatibility.

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


