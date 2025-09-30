
# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9
# Define the data partition method

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Subset
from sklearn.preprocessing import LabelEncoder
import torch
import cv2
from glob import glob
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import json
import random
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import json
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import os
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import json
import matplotlib.pyplot as plt



def load_reddit_data(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    data_list = []
    for post_id, post_info in data.items():
        post_info['post_id'] = post_id
        # Randomly assign a label for demonstration purposes (0 or 1)
        post_info['label'] = torch.randint(0, 2, (1,)).item()
        data_list.append(post_info)

    return data_list

def split_reddit_data(data, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_data, test_data

class RedditDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get('title', '')  # Using 'title' instead of 'post_body'
        label = item.get('label', 0)  # Default label to 0 if missing

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }



def load_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data

def reddit_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    all_idxs = list(range(len(dataset)))
    dict_users = {}
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# import numpy as np

# def reddit_iid(dataset, num_users):
#     # 'dataset' is expected to be a list of dictionaries, as loaded by load_reddit_data
#     post_ids = [post['post_id'] for post in dataset]  # Extract the unique post IDs
#     num_items = int(len(post_ids) / num_users)
#     dict_users = {i: [] for i in range(num_users)}  # Initialize the user dictionary with empty lists

#     # Randomly select 'num_items' number of posts for each user
#     for i in range(num_users):
#         dict_users[i] = np.random.choice(post_ids, num_items, replace=False)
#         # Since we're not replacing, we don't need to modify post_ids after each selection

#     return dict_users


# def reddit_iid(json_file, num_users):
#     # Load the dataset from a JSON file
#     dataset = load_json(json_file)

#     # Extract the indices (unique post IDs) and other details
#     post_ids = list(dataset.keys())
#     num_items = int(len(post_ids) / num_users)
#     dict_users = {}

#     # Initialize the user dictionary with empty sets
#     for i in range(num_users):
#         dict_users[i] = set()

#     for i in range(num_users):
#         # Randomly select 'num_items' number of posts for each user
#         dict_users[i] = set(np.random.choice(post_ids, num_items, replace=False))
#         # Remove the selected posts from the list of all post IDs
#         post_ids = [post_id for post_id in post_ids if post_id not in dict_users[i]]

#     return dict_users

# Example usage:
# dict_users = reddit_iid('path_to_your_dataset.json', args.num_users)


# def reddit_iid(dataset, num_users):
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, list(dataset.index)
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users

def reddit_noniid_dirichlet(dataset, num_users, alpha):
    labels = dataset['label'].values
    num_classes = len(set(labels))
    net_dataidx_map = {i: [] for i in range(num_users)}

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = np.cumsum(proportions)[:-1] * len(idx_k)
        proportions = np.round(proportions).astype(int)
        idx_batch = np.split(idx_k, proportions)

        for i in range(num_users):
            net_dataidx_map[i].extend(idx_batch[i].tolist())

    for i in range(num_users):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map



# def reddit_iid(dataset, num_users):
#     num_items = int(len(dataset) / num_users)
#     dict_users = {}
#     all_idxs = list(range(len(dataset)))
#     np.random.shuffle(all_idxs)  # Ensure random distribution
#     for i in range(num_users):
#         dict_users[i] = all_idxs[i * num_items:(i + 1) * num_items]
#     return dict_users



# def reddit_noniid_dirichlet(dataset, num_users, alpha):
#     labels = dataset['label'].values
#     num_classes = len(set(labels))
#     N = len(dataset)

#     net_dataidx_map = {i: [] for i in range(num_users)}

#     for k in range(num_classes):
#         idx_k = np.where(labels == k)[0]
#         np.random.shuffle(idx_k)

#         proportions = np.random.dirichlet(np.repeat(alpha, num_users))
#         proportions = np.cumsum(proportions)[:-1] * len(idx_k)
#         proportions = np.round(proportions).astype(int)
#         idx_batch = np.split(idx_k, proportions)

#         for i in range(num_users):
#             net_dataidx_map[i].extend(idx_batch[i].tolist())

#     for i in range(num_users):
#         np.random.shuffle(net_dataidx_map[i])
#     return net_dataidx_map


def reddit_prob_noniid(dataset, num_users, num_classes, skew=0.5):
    labels = dataset['label'].values
    num_items = int(len(dataset) / num_users)
    dict_users = {i: [] for i in range(num_users)}

    # Assign a probability to each class being chosen
    class_weights = {i: (np.random.beta(a=skew, b=1.0, size=1)[0]) for i in range(num_classes)}

    all_idxs = list(range(len(dataset)))
    while len(all_idxs) > 0:
        for user in range(num_users):
            if len(all_idxs) == 0:
                continue
            class_prob = [class_weights[labels[idx]] for idx in all_idxs]
            class_prob = np.array(class_prob) / np.sum(class_prob)
            chosen_index = np.random.choice(all_idxs, p=class_prob)
            dict_users[user].append(chosen_index)
            all_idxs.remove(chosen_index)

    return dict_users



def load(image_path):
    # Load an image from a file path
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

def load_data(dataset_path):
    files_path = dataset_path
    images, labels = [], []

    for dir_path in glob(files_path +'/*'):
        label = dir_path.split('/')[-1].split('_')[-1]  # Adjust this depending on your folder naming convention
        for image_path in glob(dir_path +'/*'):
            image = load(image_path)
            images.append(image)
            labels.append(label)

    images = np.array(images, dtype="float32") / 255.0  # Normalizing the images
    labels = np.array(labels)
    return images, labels

# Example of loading the data
# images, labels = load_data("./data/Kather")


def train_test_split(images, labels, test_size=1000):
    # This function splits the data into train and test sets.
    # test_size specifies the number of samples in the test set.
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
    return X_train, X_test, Y_train, Y_test


def labels_binarizer(Y_train, Y_test):
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    Y_train = lb.fit_transform(Y_train).argmax(axis=-1)
    Y_test = lb.transform(Y_test).argmax(axis=-1)
    return Y_train, Y_test






# class KatherPytorchDataset(Dataset):
#     def __init__(self, images, labels, transform=None):
#         """
#         Initialize the dataset.
#         :param images: a numpy array of images.
#         :param labels: a list or numpy array of labels.
#         :param transform: torchvision transforms for preprocessing images.
#         """
#         self.images = images
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         """
#         Return the total number of samples in the dataset.
#         """
#         return len(self.images)

#     def __getitem__(self, idx):
#         """
#         Generate one sample of data.
#         :param idx: the index of the sample.
#         """
#         image = self.images[idx]
#         label = self.labels[idx]

#         if self.transform:
#             # Apply the transform to the image if specified
#             image = self.transform(image)

#         # Convert image to a PyTorch tensor and adjust the channels as required by PyTorch (C, H, W)
#         image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)

#         # Convert label to a PyTorch tensor
#         label_tensor = torch.tensor(label, dtype=torch.long)

#         return image_tensor, label_tensor





class KatherPytorchDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert image from float32 to uint8 if necessary
        if image.dtype != np.uint8:
            # Assuming the image is scaled between 0 and 1
            image = (image * 255).astype(np.uint8)

        # Convert numpy array to PIL Image
        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        # Convert label to a PyTorch tensor
        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def __len__(self):
        return len(self.images)




def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset ) /num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards *num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[: ,idxs_labels[1 ,:].argsort()]
    idxs = idxs_labels[0 ,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand *num_imgs:(rand +1 ) *num_imgs]), axis=0)
    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset ) /num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def prob_noniid(dataset_label, num_clients, num_classes, q):
    """
    Sample Non-I.I.D. client data based on the probability
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    proportion = non_iid_distribution_group(dataset_label, num_clients, num_classes, q)
    dict_users = non_iid_distribution_client(proportion, num_clients, num_classes)
    #  output clients' labels information
    # check_data_each_client(dataset_label, dict_users, num_clients, num_classes)
    return dict_users

def prob_noniid_2(dataset_label, num_clients, num_classes, q):
    """
    Sample Non-I.I.D. client data based on the probability
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    proportion = non_iid_distribution_group(dataset_label, num_clients, num_classes, q)
    dict_users = non_iid_distribution_client_2(proportion, num_clients, num_classes)
    #  output clients' labels information
    # check_data_each_client(dataset_label, dict_users, num_clients, num_classes)
    return dict_users

def non_iid_distribution_group(dataset_label, num_clients, num_classes, q):
    dict_users, all_idxs = {}, [i for i in range(len(dataset_label))]
    for i in range(num_classes):
        dict_users[i] = set([])
    for k in range(num_classes):
        idx_k = np.where(np.array(dataset_label) == k)[0]
        num_idx_k = len(idx_k)

        selected_q_data = set(np.random.choice(idx_k, int(num_idx_k *q) , replace=False))
        dict_users[k] = dict_users[k ] |selected_q_data
        idx_k = list(set(idx_k) - selected_q_data)
        all_idxs = list(set(all_idxs) - selected_q_data)
        for other_group in range(num_classes):
            if other_group == k:
                continue
            selected_not_q_data = set \
                (np.random.choice(idx_k, int(num_idx_k *( 1 -q ) /(num_classes -1)) , replace=False))
            dict_users[other_group] = dict_users[other_group ] |selected_not_q_data
            idx_k = list(set(idx_k) - selected_not_q_data)
            all_idxs = list(set(all_idxs) - selected_not_q_data)
    print(len(all_idxs) ,' samples are remained')
    print('random put those samples into groups')
    num_rem_each_group = len(all_idxs) // num_classes
    for i in range(num_classes):
        selected_rem_data = set(np.random.choice(all_idxs, num_rem_each_group, replace=False))
        dict_users[i] = dict_users[i ] |selected_rem_data
        all_idxs = list(set(all_idxs) - selected_rem_data)
    print(len(all_idxs) ,' samples are remained after relocating')
    return dict_users

def non_iid_distribution_client(group_proportion, num_clients, num_classes):
    num_each_group = num_clients // num_classes
    num_data_each_client = len(group_proportion[0]) // num_each_group
    dict_users, all_idxs = {}, [i for i in range(num_data_each_client *num_clients)]
    for i in range(num_classes):
        group_data = list(group_proportion[i])
        for j in range(num_each_group):
            selected_data = set(np.random.choice(group_data, num_data_each_client, replace=False))
            dict_users[ i *10 +j] = selected_data
            group_data = list(set(group_data) - selected_data)
            all_idxs = list(set(all_idxs) - selected_data)
    print(len(all_idxs) ,' samples are remained')
    return dict_users


def non_iid_distribution_client_2(group_proportion, num_clients, num_classes):
    num_each_group = num_clients // num_classes
    num_data_each_client = len(group_proportion[0]) // num_each_group
    dict_users, all_idxs = {}, [i for i in range(num_data_each_client *num_clients)]
    for i in range(num_classes):
        group_data = list(group_proportion[i])
        for j in range(num_each_group):
            selected_data = set(np.random.choice(group_data, num_data_each_client, replace=True))
            dict_users[ i *10 +j] = selected_data
            group_data = list(set(group_data) - selected_data)
            all_idxs = list(set(all_idxs) - selected_data)
    print(len(all_idxs) ,' samples are remained')
    return dict_users

def check_data_each_client(dataset_label, client_data_proportion, num_client, num_classes):
    for client in client_data_proportion.keys():
        client_data = dataset_label[list(client_data_proportion[client])]
        print('client', client, 'distribution information:')
        for i in range(num_classes):
            print('class ', i, ':', len(client_data[client_data==i] ) /len(client_data))




# def dirichlet_distribution_label_imbalance(dataset, num_clients, alpha):
#     """
#     Split dataset based on Dirichlet distribution to simulate label imbalance.

#     :param dataset: PyTorch dataset (like MNIST, CIFAR-10)
#     :param num_clients: Number of clients
#     :param alpha: Concentration parameter for the Dirichlet distribution
#     :return: A dictionary mapping each client to its corresponding subset of the dataset
#     """
#     # Assuming the labels are the second element in the dataset tuples
#     labels = np.array([label for _, label in dataset])

#     # Convert labels to integer encoding if they are not integers
#     if not np.issubdtype(labels.dtype, np.integer):
#         le = LabelEncoder()
#         labels = le.fit_transform(labels)

#     # Number of classes
#     num_classes = np.max(labels) + 1

#     # Initialize client data indices
#     client_indices = {i: np.array([], dtype='int64') for i in range(num_clients)}

#     # Allocate samples to clients based on Dirichlet distribution
#     class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
#     for i in range(num_classes):
#         # Draw samples from a Dirichlet distribution
#         proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

#         # Split the data of class i to clients
#         np.random.shuffle(class_indices[i])
#         class_i_data_split = np.array_split(class_indices[i], np.cumsum(np.round(proportions * len(class_indices[i]), 0).astype('int')[:-1]))

#         for client in range(num_clients):
#             client_indices[client] = np.concatenate((client_indices[client], class_i_data_split[client]), axis=0)

#     # Create a PyTorch Subset for each client
#     client_datasets = {client: Subset(dataset, indices) for client, indices in client_indices.items()}

#     return client_datasets





def dirichlet_distribution_label_imbalance(dataset, num_clients, alpha):
    """
    Split dataset based on Dirichlet distribution to simulate label imbalance.
    Each client will receive a different subset of the dataset, aiming for a non-IID distribution.

    :param dataset: PyTorch dataset with an accessible 'targets' attribute.
    :param num_clients: The number of clients (users) among which the dataset is to be split.
    :param alpha: Concentration parameter for the Dirichlet distribution which controls the degree of non-uniformity.
    :return: A dictionary where each key is a client ID and each value is an array of image indices.
    """
    # Extract labels
    labels = np.array(dataset.targets)

    # Number of classes
    num_classes = len(set(labels))

    # Initialize client data indices
    client_indices = {i: [] for i in range(num_clients)}

    # Allocate samples to clients based on Dirichlet distribution
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    for class_idx in class_indices:
        # Draw class-specific samples from a Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))

        # Ensure the proportions sum to 1
        proportions = proportions / proportions.sum()

        # Calculate the number of samples for each client
        samples_per_client = (proportions * len(class_idx)).astype(int)

        # Assign samples to clients
        np.random.shuffle(class_idx)
        indices = np.split(class_idx, np.cumsum(samples_per_client)[:-1])
        for i, client_idx in enumerate(indices):
            client_indices[i].extend(client_idx.tolist())

    # Convert to integer indices and create a PyTorch Subset for each client
    client_datasets = {client: Subset(dataset, np.array(indices).astype('int64'))
                       for client, indices in client_indices.items() if len(indices) > 0}

    return client_datasets





def mnist_noniid_dirichlet(dataset, num_users, alpha):
    """
    Sample non-I.I.D client data from MNIST dataset using Dirichlet distribution
    :param dataset: MNIST dataset
    :param num_users: Number of users/clients
    :param alpha: Concentration parameter for the Dirichlet distribution
    :return: Dictionary of user data indices
    """
    labels = dataset.train_labels.numpy()
    num_classes = len(np.unique(labels))
    N = len(dataset)

    # Initialize data index map for each client
    net_dataidx_map = {i: [] for i in range(num_users)}

    # Assign data to each client based on Dirichlet distribution
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = np.split(idx_k, proportions)

        for i in range(num_users):
            net_dataidx_map[i].extend(idx_batch[i].tolist())

    # Shuffle data indices for each client
    for i in range(num_users):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map



def fmnist_noniid_dirichlet(dataset, num_users, alpha):
    """
    Sample non-I.I.D client data from Fashion-MNIST dataset using Dirichlet distribution
    :param dataset: Fashion-MNIST dataset
    :param num_users: Number of users/clients
    :param alpha: Concentration parameter for the Dirichlet distribution
    :return: Dictionary of user data indices
    """
    labels = dataset.targets.numpy()  # Accessing labels for Fashion-MNIST
    num_classes = len(np.unique(labels))
    N = len(dataset)

    net_dataidx_map = {i: [] for i in range(num_users)}

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = np.split(idx_k, proportions)

        for i in range(num_users):
            net_dataidx_map[i].extend(idx_batch[i].tolist())

    for i in range(num_users):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map




def cifar10_noniid_dirichlet(dataset, num_users, alpha):
    """
    Sample non-I.I.D client data from CIFAR-10 dataset using Dirichlet distribution
    :param dataset: CIFAR-10 dataset
    :param num_users: Number of users/clients
    :param alpha: Concentration parameter for the Dirichlet distribution
    :return: Dictionary of user data indices
    """
    labels = np.array(dataset.targets)  # Accessing labels for CIFAR-10
    num_classes = len(np.unique(labels))
    N = len(dataset)

    net_dataidx_map = {i: [] for i in range(num_users)}

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = np.split(idx_k, proportions)

        for i in range(num_users):
            net_dataidx_map[i].extend(idx_batch[i].tolist())

    for i in range(num_users):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map



def cifar100_noniid_dirichlet(dataset, num_users, alpha):
    """
    Sample non-I.I.D client data from CIFAR-100 dataset using Dirichlet distribution
    :param dataset: CIFAR-100 dataset
    :param num_users: Number of users/clients
    :param alpha: Concentration parameter for the Dirichlet distribution
    :return: Dictionary of user data indices
    """
    labels = np.array(dataset.targets)  # Assuming dataset.targets holds the labels for CIFAR-100
    num_classes = 100  # Explicitly setting for CIFAR-100
    N = len(dataset)

    net_dataidx_map = {i: [] for i in range(num_users)}

    # Assign samples to users class by class
    for k in range(num_classes):
        # Indices of samples for class k
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        # Divide class k samples into num_users parts with Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        # Ensure proportions sum to 1, then scale by number of samples in class k
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = np.split(idx_k, proportions)

        # Distribute indices among users
        for i in range(num_users):
            net_dataidx_map[i].extend(idx_batch[i].tolist())

    # Optionally shuffle data indices for each client to randomize order
    for i in range(num_users):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map


def quantity_based_label_imbalance(dataset, num_clients, num_labels_per_client):
    """
    Sample non-I.I.D client data from a dataset with quantity-based label imbalance
    :param dataset: Dataset (e.g., MNIST, Fashion-MNIST, CIFAR-10)
    :param num_clients: Number of clients
    :param num_labels_per_client: Number of labels per client
    :return: Dictionary of user data indices
    """
    # Handle different types of label storage (list or NumPy array)
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'train_labels'):
        labels = dataset.train_labels.numpy()
    else:
        raise AttributeError("Dataset does not have 'targets' or 'train_labels' attribute")

    unique_labels = np.unique(labels)

    if num_labels_per_client > len(unique_labels):
        raise ValueError("num_labels_per_client cannot be more than the number of unique labels in the dataset")

    idx_shards = {label: np.where(labels == label)[0] for label in unique_labels}

    for label in unique_labels:
        np.random.shuffle(idx_shards[label])

    dict_users = {i: [] for i in range(num_clients)}

    for client in range(num_clients):
        chosen_labels = np.random.choice(unique_labels, num_labels_per_client, replace=False)
        for label in chosen_labels:
            num_samples = len(idx_shards[label]) // num_clients
            client_samples = idx_shards[label][:num_samples]
            idx_shards[label] = idx_shards[label][num_samples:]
            dict_users[client].extend(client_samples)

    for client in dict_users.keys():
        np.random.shuffle(dict_users[client])

    return dict_users





class AddGaussianNoise(object):
    def __init__(self, mean=0., variance=1., total=1):
        self.mean = mean
        self.std = np.sqrt(variance / total)

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)



def apply_noise(dataset, noise_level, total, mean, std):
    """
    Apply Gaussian noise to the dataset with appropriate normalization.
    :param dataset: Dataset object
    :param noise_level: The level of noise to apply
    :param total: Total number of users (used to adjust noise level)
    :param mean: Mean for normalization
    :param std: Standard deviation for normalization
    :return: Dataset with noise applied
    """
    noise_transform = transforms.Compose([
        transforms.ToTensor(),
        AddGaussianNoise(0., noise_level, total),
        transforms.Normalize(mean, std)
    ])

    # Apply the transformation to the dataset
    dataset.transform = noise_transform
    return dataset


def apply_noise_imdb(dataset, noise_level, total, mean=None, std=None):
    """
    Apply noise to the IMDb dataset by modifying the text or embeddings.
    :param dataset: Dataset object (IMDb)
    :param noise_level: The level of noise to apply (between 0 and 1)
    :param total: Total number of users
    :param mean: Not needed for text, placeholder for compatibility
    :param std: Not needed for text, placeholder for compatibility
    :return: Dataset with noise applied
    """
    noisy_dataset = []

    for text, label in dataset:
        noisy_text = add_text_noise(text, noise_level)
        noisy_dataset.append((noisy_text, label))

    return noisy_dataset

def add_text_noise(text, noise_level):
    """
    Add noise to text by randomly changing or removing words.
    :param text: The input text
    :param noise_level: The fraction of words to alter
    :return: The noisy text
    """
    words = text.split()
    num_words_to_modify = int(noise_level * len(words))

    for _ in range(num_words_to_modify):
        # Randomly pick a word to modify
        idx = random.randint(0, len(words) - 1)
        if random.random() > 0.5:
            # Replace the word with a random word (simulating noise)
            words[idx] = random.choice(words)
        else:
            # Or randomly delete the word (simulating dropout noise)
            words.pop(idx)

    return ' '.join(words)

def kather_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users = {}
    all_idxs = list(range(len(dataset)))
    np.random.shuffle(all_idxs)  # Ensure random distribution

    for i in range(num_users):
        dict_users[i] = all_idxs[i * num_items:(i + 1) * num_items]

    return dict_users


def kather_noniid_dirichlet(dataset, num_users, alpha):
    labels = [label for _, label in dataset]
    num_classes = len(set(labels))
    N = len(dataset)

    net_dataidx_map = {i: [] for i in range(num_users)}

    for k in range(num_classes):
        idx_k = np.where(np.array(labels) == k)[0]
        np.random.shuffle(idx_k)

        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = np.cumsum(proportions)[:-1] * len(idx_k)
        proportions = np.round(proportions).astype(int)
        idx_batch = np.split(idx_k, proportions)

        for i in range(num_users):
            net_dataidx_map[i].extend(idx_batch[i].tolist())

    for i in range(num_users):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map


def kather_prob_noniid(dataset, num_users, num_classes, skew=0.5):
    labels = [label for _, label in dataset]
    num_items = int(len(dataset) / num_users)
    dict_users = {i: [] for i in range(num_users)}

    # Assign a probability to each class being chosen
    class_weights = {i: (np.random.beta(a=skew, b=1.0, size=1)[0]) for i in range(num_classes)}

    while len(labels) > 0:
        for user in range(num_users):
            if len(labels) == 0:
                continue
            class_prob = [class_weights[label] for _, label in dataset]
            # Normalize to form a probability distribution
            class_prob /= np.sum(class_prob)
            chosen_index = np.random.choice(range(len(dataset)), p=class_prob)
            dict_users[user].append(chosen_index)
            # Remove chosen index
            dataset.pop(chosen_index)
            labels.pop(chosen_index)

    return dict_users



def imagenet_iid(dataset, num_users):
    """
    Sample I.I.D. client data from ImageNet dataset
    :param dataset: ImageFolder dataset loaded with ImageNet data
    :param num_users: Number of users/clients
    :return: dict of image indices for each client
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, list(range(len(dataset)))
    np.random.shuffle(all_idxs)
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def imagenet_noniid_dirichlet(dataset, num_users, alpha):
    """
    Sample non-I.I.D client data from ImageNet dataset using Dirichlet distribution
    :param dataset: ImageFolder dataset loaded with ImageNet data
    :param num_users: Number of users/clients
    :param alpha: Concentration parameter for the Dirichlet distribution
    :return: Dictionary of user data indices
    """
    labels = np.array([y for _, y in dataset.samples])
    num_classes = len(np.unique(labels))
    N = len(dataset)

    # Initialize data index map for each client
    net_dataidx_map = {i: [] for i in range(num_users)}

    # Assign data to each client based on Dirichlet distribution
    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = np.cumsum(proportions)[:-1] * len(idx_k)
        proportions = proportions.astype(int)
        idx_batch = np.split(idx_k, proportions)

        for i in range(num_users):
            net_dataidx_map[i].extend(idx_batch[i].tolist())

    # Shuffle data indices for each client
    for i in range(num_users):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map


# class HARImageDataset(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None, is_test=False):
#         self.annotations = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#         self.is_test = is_test
#         if not is_test:
#             self.label_encoder = LabelEncoder()
#             self.annotations['label'] = self.label_encoder.fit_transform(self.annotations['label'])

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
#         image = Image.open(img_name).convert("RGB")
#         if self.transform:
#             image = self.transform(image)

#         if self.is_test:
#             return image, -1  # For test data, we don't have labels

#         label = self.annotations.iloc[idx, 1]
#         return image, torch.tensor(label)



# class HARImageDataset(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None, is_test=False):
#         self.annotations = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#         self.is_test = is_test
#         if not is_test:
#             self.label_encoder = LabelEncoder()
#             self.annotations['label'] = self.label_encoder.fit_transform(self.annotations['label'])

#     def __len__(self):
#         return len(self.annotations)

#     def __getitem__(self, idx):
#         img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
#         image = Image.open(img_name).convert("RGB")  # Ensure images are loaded as RGB
#         if self.transform:
#             image = self.transform(image)

#         if self.is_test:
#             return image, -1  # For test data, we don't have labels

#         label = self.annotations.iloc[idx, 1]
#         # print(f"Loaded image shape: {image.shape}, Label: {label}")  # Debug print
#         return image, torch.tensor(label)


class HARImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, is_test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.is_test = is_test
        if not is_test:
            self.label_encoder = LabelEncoder()
            self.annotations['label'] = self.label_encoder.fit_transform(self.annotations['label'])
            print(f"Encoded labels: {self.annotations['label']}")  # Debug print

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")  # Ensure images are loaded as RGB
        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image, -1  # For test data, we don't have labels

        label = self.annotations.iloc[idx, 1]
        # print(f"Loaded image shape: {image.shape}, Encoded Label: {label}")  # Debug print
        return image, torch.tensor(label)




def load_har_dataset(train_csv, train_dir, test_csv, test_dir, transform):
    train_dataset = HARImageDataset(csv_file=train_csv, root_dir=train_dir, transform=transform)
    test_dataset = HARImageDataset(csv_file=test_csv, root_dir=test_dir, transform=transform, is_test=True)
    return train_dataset, test_dataset




def har_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def har_noniid_dirichlet(dataset, num_users, alpha):
    """
    Sample non-I.I.D client data from HAR dataset using Dirichlet distribution
    :param dataset: HAR dataset
    :param num_users: Number of users/clients
    :param alpha: Concentration parameter for the Dirichlet distribution
    :return: Dictionary of user data indices
    """
    labels = np.array([sample[1] for sample in dataset.imgs])  # Accessing labels for HAR dataset
    num_classes = len(np.unique(labels))
    N = len(dataset)

    net_dataidx_map = {i: [] for i in range(num_users)}

    for k in range(num_classes):
        idx_k = np.where(labels == k)[0]
        np.random.shuffle(idx_k)

        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = np.split(idx_k, proportions)

        for i in range(num_users):
            net_dataidx_map[i].extend(idx_batch[i].tolist())

    for i in range(num_users):
        np.random.shuffle(net_dataidx_map[i])

    return net_dataidx_map


def har_quantity_based_label_imbalance(dataset, num_clients, num_labels_per_client):
    """
    Sample non-I.I.D client data from HAR dataset with quantity-based label imbalance
    :param dataset: HAR dataset
    :param num_clients: Number of clients
    :param num_labels_per_client: Number of labels per client
    :return: Dictionary of user data indices
    """
    labels = np.array([sample[1] for sample in dataset.imgs])  # Accessing labels for HAR dataset
    unique_labels = np.unique(labels)

    if num_labels_per_client > len(unique_labels):
        raise ValueError("num_labels_per_client cannot be more than the number of unique labels in the dataset")

    idx_shards = {label: np.where(labels == label)[0] for label in unique_labels}

    for label in unique_labels:
        np.random.shuffle(idx_shards[label])

    dict_users = {i: [] for i in range(num_clients)}

    for client in range(num_clients):
        chosen_labels = np.random.choice(unique_labels, num_labels_per_client, replace=False)
        for label in chosen_labels:
            num_samples = len(idx_shards[label]) // num_clients
            client_samples = idx_shards[label][:num_samples]
            idx_shards[label] = idx_shards[label][num_samples:]
            dict_users[client].extend(client_samples)

    for client in dict_users.keys():
        np.random.shuffle(dict_users[client])

    return dict_users


# def prob_noniid_2(labels, num_users, num_classes, alpha):
#     # Function to partition data probabilistically
#     # Implementation here depends on the specific requirements
#     pass


# For NLP task


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def plt_line_chart(metric_data, img_path):
    color_par = {
        'clean': '#D62728',
        'backdoor': '#1F77B4',
        'ini': '#1F77B4',
        'mid': '#FF7F0E',
        'end': '#2CA02C',
        'random': '#8A2AA0'
    }

    marker_par = {
        'clean': '.',
        'backdoor': 'o',
        'ini': 'v',
        'mid': 's',
        'end': 'p',
        'random': '*'
    }
    # r1 = list(map(lambda x: x[0] - x[1], zip(metric_data['avg'], metric_data['std'])))  # 上方差
    # r2 = list(map(lambda x: x[0] + x[1], zip(metric_data['avg'], metric_data['std'])))  # 下方差
    # plt.plot(iters, avg, color=color,label=name_of_alg,linewidth=3.5)
    # plt.fill_between(metric_data['t'], r1, r2, color=color_par['std'], alpha=0.2)
    fig = plt.figure()
    if 'xmark' in metric_data:
        ax = fig.add_subplot(1, 1, 1)
        # 添加一个子图，同时使用工厂函数为该子图自动创建一个坐标系区域；axes的位置由row,col,index指定
        ax.set_xticks(metric_data['x'])
        # 设置主刻度位置
        ax.set_xticklabels(metric_data['xmark'], rotation=30, fontsize=7)

    for i, k in enumerate(metric_data.keys()):
        if k in ['clean', 'ini', 'mid', 'end', 'random', 'backdoor']:
            plt.plot(
                metric_data['x'], metric_data[k],
                color=color_par[k], marker=marker_par[k],
                alpha=1, linewidth=1, label=k
            )

    plt.legend()  # 显示图例
    plt.grid(ls='--')  # 生成网格
    plt.xlabel(metric_data['xlabel'])
    plt.ylabel(metric_data['ylabel'])
    plt.title(metric_data['title'])
    # x_major_locator = MultipleLocator(1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    # y_major_locator = MultipleLocator(0.1)
    # 把y轴的刻度间隔设置为0.1，并存在变量里
    # ax = plt.gca()
    # ax为两条坐标轴的实例
    # 设置主刻度的标签， 带入主刻度旋转角度和字体大小参数
    # ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为x_major_locator的倍数
    # ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为y_major_locator的倍数
    # plt.ylim(0.5, 1.05)

    plt.savefig(img_path)
    plt.clf()


def label_acc(pred_label, true_label):
    return metrics.accuracy_score(true_label, pred_label)


def save_csv(cache, csv_path):
    colums = list(cache.keys())
    values = list(cache.values())
    values_T = list(map(list, zip(*values)))
    save = pd.DataFrame(columns=colums, data=values_T)
    f1 = open(csv_path, mode='w', newline='')
    save.to_csv(f1, encoding='gbk', index=False)
    f1.close()


def read_csv(csv_path):
    pd_data = pd.read_csv(csv_path, sep=',', header='infer')
    # pd_data['Status'] = pd_data['Status'].values
    return pd_data


def save_json(cache, json_path):
    # 保存文件
    tf = open(json_path, "w")
    tf.write(json.dumps(cache))
    tf.close()


def read_json(json_path):
    # 读取文件
    tf = open(json_path, "r")
    new_dict = json.load(tf)
    return new_dict



def imdb_iid(dataset, num_users):
    # Implementation of iid partitioning for IMDB dataset
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def imdb_noniid_dirichlet(dataset, num_users, alpha):
    # Implementation of Dirichlet-based non-iid partitioning for IMDB dataset
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    min_size = 0
    while min_size < 10:
        idx_batch = [[] for _ in range(num_users)]
        for k in np.unique(labels):
            idx_k = np.where(labels == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            proportions = np.array \
                ([p * (len(idx_j) < len(labels) / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [np.concatenate((idx_j, idx_k[i:j])) for idx_j, i, j in zip(idx_batch, [0] + proportions.tolist(), proportions.tolist() + [len(idx_k)])]
        min_size = min([len(idx_j) for idx_j in idx_batch])
    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = np.array(idx_batch[j], dtype='int64')
    return dict_users

def quantity_based_label_imbalance_imdb(dataset, num_users, thre_labels):
    # Implementation of quantity-based label imbalance partitioning for IMDB dataset
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    unique_labels = np.unique(labels)
    label_distribution = np.random.choice(unique_labels, size=(num_users, thre_labels), replace=True)
    for i in range(num_users):
        selected_labels = label_distribution[i]
        user_indices = np.concatenate([np.where(labels == label)[0] for label in selected_labels])
        np.random.shuffle(user_indices)
        dict_users[i] = user_indices[:int(len(user_indices) / num_users)]
    return dict_users

def prob_noniid_2_imdb(labels, num_users, num_classes, alpha):
    # Implementation of probability-based non-iid partitioning for IMDB dataset
    label_prob = np.random.dirichlet(np.repeat(alpha, num_classes))
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    for k in range(num_classes):
        label_indices = np.where(labels == k)[0]
        np.random.shuffle(label_indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        proportions = (np.cumsum(proportions) * len(label_indices)).astype(int)[:-1]
        for i in range(num_users):
            dict_users[i] = np.concatenate((dict_users[i], label_indices[:proportions[i]]))
            label_indices = label_indices[proportions[i]:]
    return dict_users


def quantity_skew(dataset, num_users, beta):
    """
    Sample non-I.I.D client data from a dataset using Dirichlet distribution for Quantity Skew
    :param dataset: Dataset object (e.g., MNIST, CIFAR, etc.)
    :param num_users: Number of users/clients
    :param beta: Concentration parameter for the Dirichlet distribution controlling quantity skew
    :return: Dictionary of user data indices
    """
    print("samping the data now")
    labels = np.array([label for _, label in dataset])  # Assuming dataset is a collection of (data, label) tuples
    N = len(dataset)

    # Shuffle the data indices
    idxs = np.random.permutation(N)
    min_size = 0

    # Ensure each user gets a minimum number of samples
    while min_size < 10:
        proportions = np.random.dirichlet(np.repeat(beta, num_users))
        proportions = proportions / proportions.sum()
        min_size = np.min(proportions * len(idxs))

    proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
    idx_batch = np.split(idxs, proportions)

    # Initialize data index map for each client
    net_dataidx_map = {i: idx_batch[i].tolist() for i in range(num_users)}
    print("samping the data end")
    return net_dataidx_map


def real_partition_emnist(dataset_train, n_parties):
    u_train = dataset_train.targets.bincount().numpy()
    num_user = u_train.shape[0]
    user = np.zeros(num_user + 1, dtype=np.int32)
    for i in range(1, num_user + 1):
        user[i] = user[i - 1] + u_train[i - 1]
    no = np.random.permutation(num_user)
    batch_idxs = np.array_split(no, n_parties)
    net_dataidx_map = {i: np.zeros(0, dtype=np.int32) for i in range(n_parties)}
    for i in range(n_parties):
        for j in batch_idxs[i]:
            net_dataidx_map[i] = np.append(net_dataidx_map[i], np.arange(user[j], user[j + 1]))

    return net_dataidx_map



if __name__ == '__main__':
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
