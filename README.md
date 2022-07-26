# Model trainer for the classification service

This repository can be used to train supervised classifiers on new taxonomies.

---

## Setup

Clone this repository and install the required libraries as follows:

```console
git clone git@github.com:IntelCompH2020/taxonomical-classification.git
cd taxonomical-classification
bash setup_environment.sh
```

## How to use

1) Create a working directory for the new taxonomy.

The following command will create a new directory inside `./taxonomies` with the given TAXONOMY_NAME.

```console
TAXONOMY_NAME=<ADD_TAXONOMY_NAME_HERE>
bash new_taxonomy.sh taxonomy_name=$TAXONOMY_NAME
```

The new directory will have the following folder structure:

- `output/`: Directory where the model checkpoints will be stored.
- `logs/`: Directory where the .err and .out slurm log files will be stored.
- `tb/`: Directory where tensorboard files will be stored, for visualization of the training progess.
- `hyperparameters.config.sh`: Configuration file where the user can modify the hyperparameters' default values.
- `generate_run.sh`: Script to generate the run.sh file based on the hyperparameters.config.sh.
- `finetune_classifier.py`: Main code that runs on top of the Trainer class from HuggingFace.
- `data_loader.py`: Script to load any parquet table for model training or inference.

2) Download the model to be finetuned.

Once the working directory has been created, the base model to be finetuned has to be added to the `./models` directory.

This can be done by simply dragging a checkpoint from your local file system, or alternatively it can be downloaded from the internet. We provide a few bash scripts that download models publicly available in the HuggingFace Hub.

As an example, the following command would download a [RoBERTa-large](https://huggingface.co/roberta-large) model inside `./models/roberta-large`:

```console
bash models/download_roberta_large.sh
```

3) Go to your working directory.

```console
cd taxonomies/$TAXONOMY_NAME
```

4) Edit the configuration file.

```console
vim hyperparameters.config.sh
```

5) Generate the run.sh file.

```console
bash generate_run.sh
```

6) Launch the script.

To run locally, follow this example:
```console
TRAIN_DATA=../../data/toy_example/patstat_train/
DEV_DATA=../../data/toy_example/patstat_dev
TEST_DATA=../../data/toy_example/patstat_test/
TEXT_COL="text"
LABEL_COL="ipc0"

bash run.sh train_files=$TRAIN_DATA dev_files=$DEV_DATA test_files=$TEST_DATA text_column=$TEXT_COL label_column=$LABEL_COL
```

To run on HPC, follow this example:
```console
TRAIN_DATA=../../data/toy_example/patstat_train/
DEV_DATA=../../data/toy_example/patstat_dev
TEST_DATA=../../data/toy_example/patstat_test/
TEXT_COL="text"
LABEL_COL="ipc0"

sbatch run.sh train_files=$TRAIN_DATA dev_files=$DEV_DATA test_files=$TEST_DATA text_column=$TEXT_COL label_column=$LABEL_COL
```

Note that the `run.sh` script will be different based on the parameters given in the configuration file (e.g. if it is supposed to run locally or in hpc, the number of nodes, etc).

It goes without saying that in both cases you should adapt the paths and column names to your dataset, this is just an example that uses the toy dataset provided in `/data`.


---

Note that the `./scripts` directory contains the main code used to train classifiers, but there is no need to edit those files since every configurable parameter will be passed as an argument in the bash scripts from the working directory. The files from `./utils` should not be modified either.

## Contact Information

- Aitor Gonzalez Agirre (aitor.gonzalez@bsc.es)
- Joan Llop Palao (joan.lloppalao@bsc.es)
- Marc Pàmies Massip (marc.pamies@bsc.es)
