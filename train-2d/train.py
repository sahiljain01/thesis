import gzip
import pickle
import networkx as nx
import pandas as pd
import numpy as np
import random
import os
import argparse 
import wandb
import random
import logging

import torch
from torch_geometric.data import InMemoryDataset
from torch.utils.data import DataLoader, random_split
from torch_geometric.utils import from_networkx, to_networkx
from torch_geometric.loader import DataLoader
from torch.nn import BCELoss

from gin import gin_k_layers
from gin import gin
from basic_gcn import gcn_k_layers

from dataset import LamanDataset

def _wandb_config(architecture, lr, epochs, dim, data_dir, layers):
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="thesis",

        # track hyperparameters and run metadata
        config={
        "architecture": architecture,
        "learning_rate": lr,
        "epochs": epochs,
        "dimensions": dim,
        "data_dir": data_dir,
        "layers": layers,
        }
    )
    
def load_data(data_dir, bs = 64, train_split=0.7):    
    laman_data = LamanDataset("", data_dir)
    proportions = [train_split, 1-train_split]
    lengths = [int(p * len(laman_data)) for p in proportions]
    lengths[-1] = len(laman_data) - sum(lengths[:-1])

    generator1 = torch.Generator().manual_seed(42)
    train_data, test_data = random_split(laman_data, lengths, generator=generator1)

    train_loader = DataLoader(train_data, batch_size = bs, shuffle=True)
    test_loader = DataLoader(test_data, batch_size = bs, shuffle=True)
    return train_loader, test_loader

def check_accuracy(model, loader):
    model.eval()
    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for batch in loader:
            pred, embedding = model(batch.x, batch.edge_index, batch.batch)
            pred = torch.squeeze(pred)
            y = batch.label
            predictions = (pred > 0.5).long() 
            num_correct += (predictions == y).sum() 
            num_samples += predictions.size(0)
                        
    return float(num_correct)/float(num_samples)*100

def train(data_dir, bs, num_epochs, lr, dim, layers, architecture):
    logging.info("Starting training...")
    losses = []

    # model = gin.GIN(num_features=1)
    hard_coded_mapping = {} # key: layer, value: dim_H
    hard_coded_mapping[1] = 40
    hard_coded_mapping[2] = 20
    hard_coded_mapping[3] = 13
    hard_coded_mapping[4] = 10
    hard_coded_mapping[5] = 8
    hard_coded_mapping[6] = 7

    model = None
    if architecture == 'GIN-2D':
        model = gin_k_layers.GIN(num_features=4, layers=layers, dim_h = hard_coded_mapping[layers])
    elif architecture == 'GCN-2D':
        model = gcn_k_layers.GCN(num_features=4, layers=layers, dim_h = hard_coded_mapping[layers])

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                        lr=lr)

    train_loader, test_loader = load_data(data_dir, bs)

    def training_iteration(data):
        model.train()
        loss = 0
        for batch in data:
            optimizer.zero_grad()
            pred, embedding = model(batch.x, batch.edge_index, batch.batch)
            pred = torch.squeeze(pred)
            loss = loss_fn(pred.float(), batch.label.float())
            loss.backward()
            optimizer.step()

        return loss, None

    bestModel = None
    bestAcc = 0 

    wandb.watch(model, log="all")
    
    for epoch in range(num_epochs):
        loss, h = training_iteration(train_loader)
        losses.append(loss)
        logging.info(f"Epoch {epoch} | Train loss {loss}")
        train_acc, test_acc = check_accuracy(model, train_loader), check_accuracy(model, test_loader)
        
        if test_acc > bestAcc:
            bestAcc = test_acc
            bestModel = model 
            
        logging.info(f"Train Accuracy {train_acc} | Test Accuracy {test_acc}")

        if epoch % 10 == 0:
            wandb.log({"epoch": epoch, "train_acc": train_acc, "test_acc": test_acc, "loss": loss})

    torch.save(bestModel.state_dict(), f"architecture{architecture}-best_model-{bs}-{num_epochs}-{lr}-{layers}.pt")
    wandb.finish()
    return loss, None

if __name__ == "__main__":
    # accept command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default="GIN-2D")
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--dim', type=int)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--layers', type=int)

    args = parser.parse_args()
    _wandb_config(args.architecture, args.lr, args.epochs, args.dim, args.data_dir, args.layers)
    
    # add logging
    logging.basicConfig(filename='train.log', filemode='w', format='%harcoded-features-(name)s - %(levelname)s - %(message)s')

    # train
    train(args.data_dir, 64, args.epochs, args.lr, args.dim, args.layers, args.architecture)
