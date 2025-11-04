#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from tkinter.messagebox import NO
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics
import copy
# from models.add_trigger import add_trigger
import math
from transformers import BertTokenizer
from utils.sampling import RedditDataset, load_har_dataset
# from skimage import io
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from transformers import BertTokenizer
from torchvision import datasets, transforms
from load_dataset import collate_fn
import torch.nn.functional as F


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class TextSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        text, label = self.dataset[self.idxs[item]]
        return text, label

    # def __getitem__(self, item):
    #     try:
    #         # Trying to print out the type and value of the index
    #         print("Type of idx:", type(self.idxs[item]), "Value of idx:", self.idxs[item])
    #         # Now attempt to access the dataset with this index
    #         image, label = self.dataset[self.idxs[item]]
    #         # Your existing processing code...
    #     except IndexError as e:
    #         # If an IndexError occurs, print out more information
    #         print("IndexError caught! Index was:", self.idxs[item])
    #         raise e


class LocalUpdate(object):
    def __init__(self, dataset=None, idxs=None, model=None, attack_label=None, local_bs=None, device=None):
        # self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.device = device
        if model != 'lstm':
            if dataset == 'har':
                transform_har = transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                train_csv = './data/HAR/Training_set.csv'
                train_dir = './data/HAR/train/'
                test_csv = './data/HAR/Testing_set.csv'
                test_dir = './data/HAR/test/'
                dataset_train, dataset_test = load_har_dataset(train_csv, train_dir, test_csv, test_dir, transform_har)
                self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=local_bs,
                                            shuffle=True)

            elif dataset == 'imdb':
                self.ldr_train = DataLoader(TextSplit(dataset, idxs), batch_size=local_bs, shuffle=True,
                                            collate_fn=collate_fn)
                self.attack_label = attack_label
                self.model = model
                self.auxiliary_data = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

            else:
                self.ldr_train = DataLoader(DatasetSplit(
                    dataset, idxs), batch_size=local_bs, shuffle=True)
                self.attack_label = attack_label
                self.model = model
                self.auxiliary_data = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
        # else:
        #     self.idxs = idxs
        #     print(idxs)
        #     self.dataset = [dataset[i] for i in idxs]
        #     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #     self.ldr_train = DataLoader(RedditDataset(self.dataset, tokenizer), batch_size=self.args.local_bs, shuffle=True)

        else:
            if dataset == 'imdb':
                self.ldr_train = DataLoader(TextSplit(dataset, idxs), batch_size=local_bs, shuffle=True,
                                            collate_fn=collate_fn)
                self.attack_label = attack_label
                self.model = model
                self.auxiliary_data = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
            else:
                self.dataset = [dataset[i] for i in idxs]  # Create a subset of the dataset based on idxs
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.ldr_train = DataLoader(RedditDataset(self.dataset, tokenizer), batch_size=local_bs,
                                            shuffle=True)

    def get_PLR(self, net):
        # get penultimate layer representations from root dataset
        # return:
        # penultimate layer representations of images in root dataset
        features_list = []
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(
                    self.device), labels.to(self.device)
            net.zero_grad()
            features = net.get_feature(images)
            features_list.append(features)
        features_list = torch.concat(features_list, dim=0)
        return features_list
    # PLR : we can know the PLR and its label.
    # def get_PLR(self, net, target_class=None):
    #     if target_class is None:
    #         target_class = self.args.plr_class

    #     features_list = []
    #     for batch_idx, (images, labels) in enumerate(self.ldr_train):
    #         # Filter images and labels that belong to the target class
    #         target_indices = labels == target_class
    #         target_images = images[target_indices]

    #         if len(target_images) == 0:
    #             continue

    #         target_images = target_images.to(self.args.device)
    #         net.zero_grad()
    #         features = net.get_feature(target_images)
    #         features_list.append(features)

    #     features_list = torch.concat(features_list, dim=0)
    #     return features_list

    # numpy format of tensors
    # def get_PLR(self, net, target_classes=None, num_points_per_class=10):
    #     if target_classes is None:
    #         target_classes = [self.args.plr_class]  # Assuming it's a list of classes if multiple
    #
    #     features_list = []
    #     counts = {cls: 0 for cls in target_classes}  # Initialize counters for each class
    #
    #     for batch_idx, (images, labels) in enumerate(self.auxiliary_data):
    #         for target_class in target_classes:
    #             # Filter images and labels that belong to the target class
    #             target_indices = labels == target_class
    #
    #             if counts[target_class] >= num_points_per_class:
    #                 continue  # Skip if already collected required number of points for this class
    #
    #             target_images = images[target_indices][:num_points_per_class - counts[target_class]]
    #             counts[target_class] += len(target_images)  # Update count
    #
    #             if len(target_images) == 0:
    #                 continue
    #
    #             target_images = target_images.to(self.args.device)
    #             net.zero_grad()
    #             features = net.get_feature(target_images)
    #             features_list.append(features)
    #
    #         if all(count >= num_points_per_class for count in counts.values()):
    #             break  # Exit loop once all classes have enough points
    #
    #     if features_list:
    #         features_list = torch.concat(features_list, dim=0)
    #         return features_list
    #     # else:
    #     #     return torch.tensor([])  # Return an empty tensor if no features were collected
    #
    # # numpy format of PLRs
    # def get_PLR_pair(self, net, target_classes=None, num_points_per_class=10):
    #     if target_classes is None:
    #         target_classes = [self.args.plr_class]  # Assuming it's a list of classes if multiple
    #
    #     features_list = []
    #     targets_list = []
    #
    #     counts = {cls: 0 for cls in target_classes}  # Initialize counters for each class
    #
    #     for batch_idx, (images, labels) in enumerate(self.auxiliary_data):
    #         images, labels = images.to(self.args.device), labels.to(self.args.device)
    #         for target_class in target_classes:
    #             target_indices = labels == target_class
    #
    #             if counts[target_class] >= num_points_per_class:
    #                 continue  # Skip if already collected required number of points for this class
    #
    #             available_points = min(num_points_per_class - counts[target_class], target_indices.sum().item())
    #             if available_points <= 0:
    #                 continue
    #
    #             target_images = images[target_indices][:available_points]
    #             target_labels = labels[target_indices][:available_points]
    #             counts[target_class] += available_points  # Update count
    #
    #             net.zero_grad()
    #             features = net.get_feature(target_images)
    #
    #             # Collect features and labels
    #             features_list.append(features.cpu().detach().numpy())
    #             targets_list.append(target_labels.cpu().detach().numpy())
    #
    #             if all(count >= num_points_per_class for count in counts.values()):
    #                 break  # Exit loop once all classes have enough points
    #
    #     if features_list:
    #         features_array = np.concatenate(features_list, axis=0)
    #         targets_array = np.concatenate(targets_list, axis=0)
    #         return features_array, targets_array
    #     else:
    #         return np.array([]), np.array([])  # Return empty arrays if no features were collected

    def get_multi_layers(self, net, target_classes=None, num_points_per_class=10, plr_class=6):
        if target_classes is None:
            target_classes = [plr_class]  # Assuming it's a list of classes if multiple

        features_list = []
        counts = {cls: 0 for cls in target_classes}  # Initialize counters for each class

        for batch_idx, (images, labels) in enumerate(self.auxiliary_data):
            for target_class in target_classes:
                # Filter images and labels that belong to the target class
                target_indices = labels == target_class

                if counts[target_class] >= num_points_per_class:
                    continue  # Skip if already collected required number of points for this class

                target_images = images[target_indices][:num_points_per_class - counts[target_class]]
                counts[target_class] += len(target_images)  # Update count

                if len(target_images) == 0:
                    continue

                target_images = target_images.to(self.device)
                net.zero_grad()

                # Call the model's get_feature_list function to extract features from multiple layers
                layer_outputs = net.get_feature_list(target_images)  # Returns a list of tensors

                # Concatenate the layer outputs for this batch
                combined_features = torch.cat([f.view(f.size(0), -1) for f in layer_outputs], dim=1)

                # Append the combined features to the features_list
                features_list.append(combined_features)

            if all(count >= num_points_per_class for count in counts.values()):
                break  # Exit loop once all classes have enough points

        if features_list:
            # Concatenate all the batch features into a single tensor
            features_list = torch.cat(features_list, dim=0)
            return features_list

    def get_multi_layers2(self, net, target_classes=None, num_points_per_class=10):
        if target_classes is None:
            target_classes = [self.args.plr_class]  # Assuming it's a list of classes if multiple

        features_list = []
        counts = {cls: 0 for cls in target_classes}  # Initialize counters for each class

        for batch_idx, (images, labels) in enumerate(self.auxiliary_data):
            for target_class in target_classes:
                # Filter images and labels that belong to the target class
                target_indices = labels == target_class

                if counts[target_class] >= num_points_per_class:
                    continue  # Skip if already collected required number of points for this class

                target_images = images[target_indices][:num_points_per_class - counts[target_class]]
                counts[target_class] += len(target_images)  # Update count

                if len(target_images) == 0:
                    continue

                target_images = target_images.to(self.args.device)
                net.zero_grad()

                # Extract features from multiple layers
                features = self.get_multiple_layer_features(net, target_images)
                features_list.append(features)

            if all(count >= num_points_per_class for count in counts.values()):
                break  # Exit loop once all classes have enough points

        if features_list:
            features_list = torch.concat(features_list, dim=0)
            return features_list

    def get_multiple_layer_features(self, net, images):
        layer_outputs = []
        x = images

        # Assuming net is a sequential model or has a list of layers
        for name, layer in net.named_children():
            x = layer(x)

            # Customize based on the structure of your model
            if "layer" in name or "block" in name or "conv" in name:
                layer_outputs.append(x.flatten(1))  # Flatten the output to concatenate later
            elif isinstance(layer, torch.nn.Linear):
                # If we're about to apply a Linear layer, check if we need to flatten first
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)  # Flatten the tensor to (batch_size, -1)
                layer_outputs.append(x)
            else:
                layer_outputs.append(x.flatten(1))  # Or handle other cases specifically

        # Concatenate the features from multiple layers
        features = torch.cat(layer_outputs, dim=1)
        return features

    # def get_PLR(self, net, target_classes=None, num_points_per_class=10):
    #     if target_classes is None:
    #         target_classes = [self.args.plr_class]  # Assuming it's a list of classes if multiple

    #     features_list = []
    #     counts = {cls: 0 for cls in target_classes}  # Initialize counters for each class

    #     for batch_idx, (images, labels) in enumerate(self.auxiliary_data):
    #         images, labels = images.to(self.args.device), labels.to(self.args.device)
    #         for target_class in target_classes:
    #             target_indices = labels == target_class

    #             if counts[target_class] >= num_points_per_class:
    #                 print(f"Already collected required points for class {target_class}")
    #                 continue  # Skip if already collected required number of points for this class

    #             available_points = min(num_points_per_class - counts[target_class], target_indices.sum().item())
    #             if available_points <= 0:
    #                 print(f"No available points for class {target_class}, skipping...")
    #                 continue

    #             target_images = images[target_indices][:available_points]
    #             counts[target_class] += available_points  # Update count

    #             net.zero_grad()
    #             features = net.get_feature(target_images)

    #             # Collect features
    #             features_list.append(features.cpu().detach())  # Removed the numpy conversion

    #             if all(count >= num_points_per_class for count in counts.values()):
    #                 print("Collected enough points for all classes, breaking loop.")
    #                 break  # Exit loop once all classes have enough points

    #     if features_list:
    #         features_tensor = torch.cat(features_list, dim=0)
    #         print(f"Returning features tensor of shape {features_tensor.shape}")
    #         return features_tensor  # Only return the tensor of features
    #     else:
    #         print("No features collected, returning empty tensor.")
    #         return torch.empty(0)  # Return an empty tensor if no features were collected

    # def get_PLR(self, net, target_classes=None, num_points_per_class=10):
    #     if target_classes is None:
    #         # target_classes = [self.args.plr_class]  # Assuming it's a list of classes if multiple
    #             # Randomly select a class from the unique labels in auxiliary_data
    #             unique_classes = torch.unique(torch.tensor([label for _, labels in self.auxiliary_data for label in labels]))
    #             random_class = random.choice(unique_classes).item()
    #             target_classes = [random_class]

    #     features_list = []

    #     counts = {cls: 0 for cls in target_classes}  # Initialize counters for each class

    #     for batch_idx, (images, labels) in enumerate(self.auxiliary_data):
    #         images, labels = images.to(self.args.device), labels.to(self.args.device)
    #         for target_class in target_classes:
    #             target_indices = labels == target_class

    #             if counts[target_class] >= num_points_per_class:
    #                 continue  # Skip if already collected required number of points for this class

    #             available_points = min(num_points_per_class - counts[target_class], target_indices.sum().item())
    #             if available_points <= 0:
    #                 continue

    #             target_images = images[target_indices][:available_points]
    #             counts[target_class] += available_points  # Update count

    #             net.zero_grad()
    #             features = net.get_feature(target_images)

    #             # Collect features
    #             features_list.append(features.cpu().detach())  # Removed the numpy conversion

    #             if all(count >= num_points_per_class for count in counts.values()):
    #                 break  # Exit loop once all classes have enough points

    #     if features_list:
    #         features_tensor = torch.cat(features_list, dim=0)
    #         return features_tensor  # Only return the tensor of features
    #     else:
    #         return torch.empty(0)  # Return an empty tensor if no features were collected

    # get the PLRs of all classes
    # def get_PLR(self, net):
    #     # get penultimate layer representations from root dataset
    #     # return:
    #     # penultimate layer representations of images in root dataset
    #     features_list = []
    #     for batch_idx, (images, labels) in enumerate(self.ldr_train):
    #         images, labels = images.to(
    #                 self.args.device), labels.to(self.args.device)
    #         net.zero_grad()
    #         features = net.get_feature(images)
    #         features_list.append(features)
    #     features_list = torch.concat(features_list, dim=0)
    #     return features_list

    # def get_PLR(self, net, target_class):
    #     # get penultimate layer representations from root dataset for a specific class
    #     # params:
    #     # target_class: the class for which PLRs are required
    #     # return:
    #     # penultimate layer representations of images of the target class in root dataset
    #     features_list = []
    #     for batch_idx, (images, labels) in enumerate(self.ldr_train):
    #         # Filter images and labels that belong to the target class
    #         target_indices = labels == target_class
    #         target_images = images[target_indices]

    #         if len(target_images) == 0:
    #             continue

    #         target_images = target_images.to(self.args.device)
    #         net.zero_grad()
    #         features = net.get_feature(target_images)
    #         features_list.append(features)

    #     features_list = torch.concat(features_list, dim=0)
    #     return features_list
    def train(self, net):
        # if self.args.defence == 'flip':
        #     return self.train_flip(net)
        # # if self.args.model == 'lstm':
        # #     return self.train_lstm(net)
        # if self.args.dataset == 'imdb':
        #     return self.train_text(net)

        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=0.01, momentum=0.9)

        epoch_loss = []
        for iter in range(5):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # print(f"Batch {batch_idx} image shape: {images.shape}")
                images, labels = images.to('cuda:0'), labels.to('cuda:0')
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

        # def train_text(self, net):

    #     net.train()
    #     optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
    #     epoch_loss = []
    #     for iter in range(self.args.local_ep):
    #         batch_loss = []
    #         for batch_idx, (texts, labels) in enumerate(self.ldr_train):
    #             print(f"Batch {batch_idx}: texts shape: {texts.shape}, labels shape: {labels.shape}")
    #             texts, labels = texts.to(self.args.device), labels.to(self.args.device)
    #             net.zero_grad()
    #             log_probs = net(texts)
    #             loss = self.loss_func(log_probs, labels)
    #             loss.backward()
    #             optimizer.step()
    #             batch_loss.append(loss.item())
    #         epoch_loss.append(sum(batch_loss) / len(batch_loss))
    #     return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_text(self, net):
        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (texts, labels) in enumerate(self.ldr_train):
                # print(f"Batch {batch_idx}: texts shape: {texts.shape}, labels shape: {labels.shape}")
                texts, labels = texts.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(texts)  # Output shape should be [batch_size, num_classes]

                # Calculate loss
                loss = F.cross_entropy(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # def train_text(self, net):
    #     net.train()
    #     optimizer = torch.optim.SGD(
    #         net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
    #     epoch_loss = []
    #     for iter in range(self.args.local_ep):
    #         batch_loss = []
    #         for batch_idx, (texts, labels) in enumerate(self.ldr_train):
    #             print(f"Batch {batch_idx}: texts shape: {texts.shape}, labels shape: {labels.shape}")
    #             texts, labels = texts.to(self.args.device), labels.to(self.args.device)
    #             net.zero_grad()
    #             log_probs = net(texts)

    #             # Flatten log_probs and labels for loss computation
    #             log_probs = log_probs.view(-1, log_probs.size(-1))  # [batch_size * seq_length, num_classes]
    #             labels = labels.view(-1)  # [batch_size * seq_length]

    #             loss = F.cross_entropy(log_probs, labels)
    #             loss.backward()
    #             optimizer.step()
    #             batch_loss.append(loss.item())
    #         epoch_loss.append(sum(batch_loss) / len(batch_loss))
    #     return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # def train(self, net):
    #     if self.args.defence == 'flip':
    #         return self.train_flip(net)
    #     if self.args.model == 'lstm':
    #         return self.train_lstm(net)
    #     net.train()
    #     # train and update
    #     optimizer = torch.optim.SGD(
    #         net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

    #     epoch_loss = []
    #     for iter in range(self.args.local_ep):
    #         batch_loss = []
    #         for batch_idx, (images, labels) in enumerate(self.ldr_train):
    #             images, labels = images.to(
    #                 self.args.device), labels.to(self.args.device)
    #             net.zero_grad()
    #             log_probs = net(images)
    #             loss = self.loss_func(log_probs, labels)
    #             loss.backward()
    #             optimizer.step()
    #             batch_loss.append(loss.item())
    #         epoch_loss.append(sum(batch_loss)/len(batch_loss))
    #     return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_lstm(self, net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, batch in enumerate(self.ldr_train):
                input_ids = batch['input_ids'].to(self.args.device)
                attention_mask = batch['attention_mask'].to(self.args.device)
                labels = batch['labels'].to(self.args.device)

                net.zero_grad()

                # Initialize hidden state dynamically based on the actual batch size
                batch_size = input_ids.size(0)
                hidden = net.init_hidden(batch_size)
                hidden = tuple([h.data for h in hidden])  # Detach hidden state

                log_probs, hidden = net(input_ids, hidden)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    # def train_lstm(self, net):
    #     net.train()
    #     benign_optimizer = torch.optim.SGD(
    #         net.parameters(),
    #         lr=self.args.lr,
    #         momentum=self.args.momentum,
    #     )
    #     total_loss = 0.0
    #     hidden = net.init_hidden(self.args.local_bs)  # Initialize hidden state with batch size

    #     for iter in range(self.args.local_ep):
    #         for batch_idx, (data, targets) in enumerate(self.ldr_train):
    #             data, targets = data.to(self.args.device), targets.to(self.args.device)
    #             benign_optimizer.zero_grad()
    #             hidden = tuple([each.data for each in hidden])  # Detach hidden states to prevent backprop through entire training history
    #             output, hidden = net(data, hidden)
    #             class_loss = self.loss_func(output, targets)
    #             torch.nn.utils.clip_grad_norm_(net.parameters(), self.args.clip)
    #             class_loss.backward()
    #             total_loss += class_loss.item()
    #             benign_optimizer.step()

    #     return net.state_dict(), total_loss / len(self.ldr_train)

    # def train_lstm(self, net):
    #     net.train()
    #     optimizer = torch.optim.SGD(
    #         net.parameters(),
    #         lr=self.args.lr,
    #         momentum=self.args.momentum,
    #     )
    #     total_loss = 0.0
    #     criterion = nn.CrossEntropyLoss()

    #     for epoch in range(self.args.local_ep):
    #         epoch_loss = 0.0
    #         for idx in self.idxs:
    #             data, targets = self.dataset[idx]  # Get data and targets by index
    #             data, targets = data.to(self.args.device), targets.to(self.args.device)
    #             hidden = net.init_hidden(data.size(0))
    #             hidden = (hidden[0].to(self.args.device), hidden[1].to(self.args.device))

    #             optimizer.zero_grad()
    #             outputs, hidden = net(data, hidden)
    #             loss = criterion(outputs, targets)

    #             loss.backward()
    #             optimizer.step()

    #             total_loss += loss.item()
    #             epoch_loss += loss.item()

    #         print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(self.idxs)}")

    #     return net.state_dict(), total_loss / self.args.local_ep

    # def train_lstm(self, net):
    #     net.train()
    #     benign_optimizer = torch.optim.SGD(
    #         net.parameters(),
    #         lr=self.args.lr,
    #         momentum=self.args.momentum,
    #     )
    #     total_loss=0.0
    #     hidden = net.init_hidden(20)
    #     ntokens = len(self.args.helper.corpus.dictionary)
    #     for iter in range(self.args.local_ep):
    #         for i, benign_data_idx in enumerate(self.idxs):
    #             benign_data = self.dataset[benign_data_idx]
    #             data_iterator = range(0, benign_data.size(0) - 1, self.args.helper.params['bptt'])
    #             for batch_id, batch in enumerate(data_iterator):
    #                 data, targets = self.args.helper.get_batch(benign_data, batch, False)
    #                 benign_optimizer.zero_grad()
    #                 hidden = self.args.helper.repackage_hidden(hidden)
    #                 output, hidden = net(data, hidden)
    #                 # self.loss_func = nn.CrossEntropyLoss()
    #                 class_loss = self.loss_func(output.view(-1, ntokens), targets)
    #                 torch.nn.utils.clip_grad_norm_(
    #                     net.parameters(), self.args.helper.params['clip']
    #                 )
    #                 class_loss.backward()
    #                 total_loss += class_loss.item()
    #                 benign_optimizer.step()
    #     return net.state_dict(), total_loss / self.args.local_ep

    # def add_trigger(self, image):
    #     return add_trigger(self.args, image)

    def trigger_data(self, images, labels):
        #  To unlearn trigger, label should be clean label
        #  attack_goal == -1 means attack all label to attack_label
        if self.args.attack_goal == -1:
            if math.isclose(self.args.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    bad_data[xx] = self.add_trigger(bad_data[xx])
                images = torch.cat((images, bad_data), dim=0)
                labels = torch.cat((labels, bad_label))
            else:
                for xx in range(len(images)):  # poison_frac% poison data
                    images[xx] = self.add_trigger(images[xx])
                    if xx > len(images) * self.args.poison_frac:
                        break
        else:  # trigger attack_goal to attack_label
            if math.isclose(self.args.poison_frac, 1):  # 100% copy poison data
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    if bad_label[xx] != self.args.attack_goal:  # no in task
                        continue  # jump
                    bad_data[xx] = self.add_trigger(bad_data[xx])
                    images = torch.cat((images, bad_data[xx].unsqueeze(0)), dim=0)
                    labels = torch.cat((labels, bad_label[xx].unsqueeze(0)))
            else:  # poison_frac% poison data
                # count label == goal label
                num_goal_label = len(labels[labels == self.args.attack_goal])
                counter = 0
                for xx in range(len(images)):
                    if labels[xx] != 0:
                        continue
                    images[xx] = self.add_trigger(images[xx])
                    counter += 1
                    if counter > num_goal_label * self.args.poison_frac:
                        break
        return images, labels

    def train_flip(self, net):
        # inverse trigger and unlearn trigger
        # assume defenser know trigger and attack label
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                #  add trigger and keep clean label unchange
                images, labels = self.trigger_data(images, labels)
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_malicious_flipupdate(self, net, test_img=None, dataset_test=None, args=None):
        global_net_dict = copy.deepcopy(net.state_dict())
        # *****save model********
        # benign_dict, _ = self.train(copy.deepcopy(net))
        # torch.save(benign_dict,'./save/benign.pt')
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)

        epoch_loss = []

        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    bad_label[xx] = self.attack_label
                    bad_data[xx][:, 0:5, 0:5] = 1
                images = torch.cat((images, bad_data), dim=0)
                labels = torch.cat((labels, bad_label))
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                net, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))
        attack_list = ['linear.weight', 'conv1.weight', 'layer4.1.conv2.weight', 'layer4.1.conv1.weight',
                       'layer4.0.conv2.weight', 'layer4.0.conv1.weight']
        # *****save model********
        # torch.save(net.state_dict(),'./save/malicious.pt')
        # attack_list=['fc1.weight']
        attack_weight = {}
        for key, var in net.state_dict().items():
            if key in attack_list:
                print("attack")
                attack_weight[key] = 2 * global_net_dict[key] - var
            else:
                attack_weight[key] = var
        return attack_weight, sum(epoch_loss) / len(epoch_loss)
        # return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_malicious_layerAttack(self, net, test_img=None, dataset_test=None, args=None):
        if self.model == 'resnet':
            # attack_list = ['linear.weight', 'conv1.weight', 'layer4.1.conv2.weight',
            #                'layer4.1.conv1.weight', 'layer4.0.conv2.weight', 'layer4.0.conv1.weight']
            attack_list = ['linear.weight',
                           'layer4.1.conv2.weight', 'layer4.1.conv1.weight']
        badnet = copy.deepcopy(net)
        badnet.train()
        # train and update
        optimizer = torch.optim.SGD(
            badnet.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                bad_data, bad_label = copy.deepcopy(
                    images), copy.deepcopy(labels)
                for xx in range(len(bad_data)):
                    bad_label[xx] = self.attack_label
                    bad_data[xx][:, 0:5, 0:5] = 1
                images = torch.cat((images, bad_data), dim=0)
                labels = torch.cat((labels, bad_label))
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                badnet.zero_grad()
                log_probs = badnet(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        bad_net_param = badnet.state_dict()
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                badnet, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))

        net.train()
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        attack_param = {}
        for key, var in net.state_dict().items():
            if key in attack_list:
                attack_param[key] = bad_net_param[key]
            else:
                attack_param[key] = var
        return attack_param, sum(epoch_loss) / len(epoch_loss)

    def train_malicious_labelflip(self, net, test_img=None, dataset_test=None, args=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                for x in range(len(labels)):
                    labels[x] = 9 - labels[x]
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            # attack_param = {}
            # attack_list=['linear.weight','conv1.weight','layer4.1.conv2.weight','layer4.1.conv1.weight','layer4.0.conv2.weight','layer4.0.conv1.weight']
            # for key, var in net.state_dict().items():
            #     if key in attack_list:
            #         attack_param[key] = -var
            #     else:
            #         attack_param[key] = var
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_malicious_badnet(self, net, test_img=None, dataset_test=None, args=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                for xx in range(len(images)):
                    labels[xx] = self.attack_label
                    # print(images[xx][:, 0:5, 0:5])
                    images[xx][:, 0:5, 0:5] = torch.max(images[xx])
                    if xx > len(images) * 0.2:
                        break
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                # if self.args.verbose and batch_idx % 10 == 0:
                #     print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                #         iter, batch_idx * len(images), len(self.ldr_train.dataset),
                #                100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                net, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def train_malicious_biasattack(self, net, test_img=None, dataset_test=None, args=None):
        net.train()
        # train and update
        optimizer = torch.optim.SGD(
            net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(
                    self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        attack_weight = {}
        for key, var in net.state_dict().items():
            attack_weight[key] = var
            if key == 'linear.bias':
                print(attack_weight[key][0])
                attack_weight[key][0] *= 5
                print(attack_weight[key][0])
        if test_img is not None:
            acc_test, _, backdoor_acc = test_img(
                net, dataset_test, args, test_backdoor=True)
            print("local Testing accuracy: {:.2f}".format(acc_test))
            print("local Backdoor accuracy: {:.2f}".format(backdoor_acc))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)
    # def save_pic(image):
    #     io.imsave('x.jpg', images.reshape(28,28).numpy())
