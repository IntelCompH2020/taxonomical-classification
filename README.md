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

- `logs/`: Directory where the .err and .out slurm log files will be stored.
- `tb/`: Directory where tensorboard files will be stored, for visualization of the training progess.
- `output/`: Directory where the model checkpoints will be stored.
- `config.json`: Configuration file where the user can modify the hyperparameters' default values.
- `train_singlenode.sh`: Script to launch a single-node job to an HPC cluster.
- `train_multinode.sh`: Script to launch a multi-node job to an HPC cluster.
- WiP!!!

2) Download the model to be finetuned.

Once the working directory has been created, the base model to be finetuned has to be added to the `./models` directory.

This can be done by simply dragging a checkpoint from your local file system, or alternatively it can be downloaded from the internet. We provide a few bash scripts that download models publicly available in the Huggingface Hub.

As an example, the following command would download a [RoBERTa-large](https://huggingface.co/roberta-large) model inside `./models/roberta-large`:

```console
	bash models/download_roberta_large.sh
```

3) Go to your working directory.

```console
	cd taxonomies/$TAXONOMY_NAME
```

3) Edit the configuration file.

```console
	vim config.sh
```

5) Launch the finetuning job.

```console
	bash launch_job.sh
```

---

Note that the `./scripts` directory contains the main code used to train classifiers, but there is no need to edit those files since every configurable parameter will be passed as an argument in the bash scripts from the working directory. The files from `./utils` should not be modified either.

## Contact Information

- Aitor Gonzalez Agirre (aitor.gonzalez@bsc.es)
- Joan Llop Palao (joan.lloppalao@bsc.es)
- Marc Pàmies Massip (marc.pamies@bsc.es)
