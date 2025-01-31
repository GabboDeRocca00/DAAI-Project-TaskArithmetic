import torch
import os
from args import parse_arguments
from datasets.common import get_dataloader, maybe_dictionarize
from datasets.registry import get_dataset
from modeling import ImageClassifier, ImageEncoder
from heads import get_classification_head
from task_vectors_modified import NonLinearTaskVector
from utils import train_diag_fim_logtr
import sys
from tqdm import tqdm
from torch import nn
from torch import optim
from torch.backends import cudnn
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
import copy
from glob import glob

def finetune(data_location, model_version, 
             datasets_name = ["DTD", "EuroSAT", "GTSRB", "MNIST", "RESISC45", "SVHN"], 
             datasets_epochs = [76, 12, 11, 5, 15, 4],
             models_location = "models", 
             results_location = "results", 
             stopping_criteria = "last_iter", 
             batch_size = 32,  
             lr = 1e-4, 
             weight_decay = 0, 
             num_workers = 0,
             device = "cuda" if torch.cuda.is_available() else "cpu",  
             model_save = "preserve", 
             samples_nr = 200):
             
    # Convert datasets_name to a list if it's a string
    if isinstance(datasets_name, str):
        datasets_name = [datasets_name]

    # Set-up benchmark mode make training faster
    cudnn.benchmark = True
    # Disable warnings to reduce clutter in the output
    warnings.filterwarnings("ignore")

    # Initialise the dictionaries that will keep track of the parameters used to determine early stopping
    logdet_hF_vec = {}
    val_acc_vec = {}

    # Iterate over the datasets
    for dataset_name in datasets_name:
        # Initialize the key with an empty list if it doesn't exist
        if dataset_name not in logdet_hF_vec:
            logdet_hF_vec[dataset_name] = []
            val_acc_vec[dataset_name] = []

    # Set the command-line arguments
    sys.argv = [
        "script_name.py",  # Placeholder script name
        "--data-location", data_location,
        "--model", "ViT-B-32",
        "--save", models_location
    ]

    # Iterate over the datasets
    for dataset_number, dataset_name in tqdm(enumerate(datasets_name), total=len(datasets_name), desc="\n\n\nIterating datasets"):

        print(f"\nFine-tuning the model version '{model_version}' on '{dataset_name}' dataset \n\n\n")

        # Parse the arguments
        args = parse_arguments()

        # Now you can use the args object with get_classification_head
        head = get_classification_head(args, dataset_name)

        # Instantiate a full model architecture
        encoder = ImageEncoder(args)  # Pre-trained CLIP ViT backbone
        model = ImageClassifier(encoder, head).to(device)  # Build full model

        # Checks if the PreTrained encoder has been saved, if this is not the case, it gets saved in the folder inserted in "models_location"
        ptmodel_location_and_name = Path(models_location + "/encoder_PreTrained.pt")
        if not os.path.exists(ptmodel_location_and_name):
            print("\n\nSaving the encoder of the PreTrained ViT-B-32 model")
            torch.save(obj=model.image_encoder.state_dict(), f=ptmodel_location_and_name)

        # Create a string describing the name of the fine-tuned model and where it will be saved
        model_location_and_name = Path(models_location + "/" + model_version + "_" + "encoder_SingleTask_" + dataset_name + ".pt")

        # Check if an instance of the trained model is already present in the "models_location" folder. In case that case, and if "model_save" is NOT set to "overwrite", the model will not be trained again, but the version already present will be loaded
        if os.path.exists(model_location_and_name) and model_save != "overwrite":
            print(f"\n\n\nTraining not performed:  \nThe model '{model_location_and_name}' is already present. \nIf you want to overwrite it, you have to manually delete it form the folder '{models_location}', or set 'model_save' to 'overwrite'.\n")
            model.image_encoder.load_state_dict(torch.load(f=model_location_and_name))

        else:
            # Obtain the Train split
            train_dataset = get_dataset(
                dataset_name, preprocess=model.train_preprocess,
                location=args.data_location, batch_size=batch_size, num_workers=num_workers)
            train_loader = get_dataloader(train_dataset, is_train=True, args=args)

            # Obtain the Validation split if validation accuracy is used as early stopping criteria
            if stopping_criteria == "max_val_acc":
                val_dataset = get_dataset(
                    dataset_name + "Val", preprocess=model.val_preprocess,
                    location=args.data_location, batch_size=batch_size, num_workers=num_workers)
                val_loader = get_dataloader(val_dataset, is_train=False, args=args)

            # create the folder where to save the state dicts in case of early stopping criteria
            if stopping_criteria == "max_val_acc" or stopping_criteria == "max_logdet_hF":
                # Create a folder where to store the state dictionaries of each epoch
                store_path = f"{results_location}/{stopping_criteria}/{model_version}/{dataset_name}"
                os.makedirs(store_path, exist_ok=True)

            # Define loss and optimiser to be used during training
            loss_fn = nn.CrossEntropyLoss()
            optimiser = optim.SGD(params=model.parameters(), lr=lr, weight_decay=weight_decay)

            # Set the number of epochs for training
            epochs = datasets_epochs[dataset_number]

            # Train the model
            print(f"Training starting on dataset {dataset_name}")
            model.train()
            for epoch in tqdm(range(epochs), "Epoch"):
                train_loss = []
                acc_val = 0
                for batch in train_loader:
                    data = maybe_dictionarize(batch)
                    x, y = data["images"], data["labels"]
                    x, y = x.to(device), y.to(device)
                    y_logits = model(x)
                    loss = loss_fn(y_logits, y)
                    train_loss.append(loss.item())  # Append the loss value to the list
                    optimiser.zero_grad()
                    loss.backward()
                    optimiser.step()

                # if the stopping criteria is the max log trace of the fisher matrix, calculate the log trace of the fisher matrix and append its value to 'logdet_hF_vec' so that we can retrieve the epoch where the fisher matrix had the highest value
                if stopping_criteria == "max_logdet_hF":
                    if epoch == 0:
                        print("\nCalculating Fisher matrix logaritm trace (this will take a looong time)")
                    else:
                        print("\nCalculating Fisher matrix logaritm trace")
                    logdet_hF = train_diag_fim_logtr(args, model, dataset_name, samples_nr)
                    logdet_hF_vec[dataset_name].append(logdet_hF)
                    print(f"Fisher matrix logaritm trace at epoch {epoch + 1} = {logdet_hF}")
                    torch.save(obj=model.image_encoder.state_dict(), f=f"{store_path}/epoch_{epoch + 1}")

                # if the stopping criteria is the max validation accuracy, calculate the the validation accuracy and append its value to 'val_acc_vec' so that we can retrieve the epoch where the accuracy had the highest value
                if stopping_criteria == "max_val_acc":
                    val_acc = 0
                    model.eval()
                    with torch.inference_mode():
                        for batch in val_loader:
                            data = maybe_dictionarize(batch)
                            x, y = data["images"], data["labels"]
                            x, y = x.to(device), y.to(device)
                            y_logits = model(x)
                            y_labels = y_logits.argmax(dim=1)
                            val_acc += (y == y_labels).sum().item() / len(y)
                        val_acc /= len(val_loader)
                        val_acc *= 100
                        val_acc_vec[dataset_name].append(val_acc)
                    print(f"\n\nValidation accuracy at epoch {epoch + 1} = {val_acc} [%]")
                    torch.save(obj=model.image_encoder.state_dict(), f=f"{store_path}/epoch_{epoch + 1}")
                    model.train()

                # Stop the training cycle if the validation accuracy is 100 [%]
                if stopping_criteria == "max_val_acc":
                    if val_acc > 99.99:
                        print("\nTraining interrupted early: the validation accuracy has reached 100 [%]")
                        # Show training loss at each step
                        print(f"\nAverage batch loss at epoch {epoch + 1} = {sum(train_loss) / len(train_loader)}")
                        break
                # Show training loss at each step
                print(f"\nAverage batch loss at epoch {epoch + 1} = {sum(train_loss) / len(train_loader)}")

            # Obtain the epoch number for which the state dictionary will be selected based on the stopping criteria
            if stopping_criteria == "max_val_acc":
                stopping_index = np.argmax(val_acc_vec[dataset_name]) + 1
                print(f"\nstate dictionary chosen at epoch: {stopping_index} out of {epochs}")
                model.image_encoder.load_state_dict(torch.load(f"{store_path}/epoch_{stopping_index}"))
            elif stopping_criteria == "max_logdet_hF":
                stopping_index = np.argmax(logdet_hF_vec[dataset_name]) + 1
                print(f"\nstate dictionary chosen at epoch: {stopping_index} out of {epochs}")
                model.image_encoder.load_state_dict(torch.load(f"{store_path}/epoch_{stopping_index}"))

            print("\nTraining completed\n")

            print(f"\nTraining completed on dataset {dataset_name}\n")

            # Create the folder where to save the fine-tuned model (if not already present)
            if not os.path.exists(models_location):
                os.makedirs(models_location)
