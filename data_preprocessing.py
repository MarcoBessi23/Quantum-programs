import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os


def save_data(train_images_pca, test_images_pca, train_labels, test_labels, file_path):

    np.savez(file_path, 
             train_images_pca=train_images_pca, 
             test_images_pca=test_images_pca,
             train_labels=train_labels,
             test_labels=test_labels)

def load_data(file_path):
    
    data = np.load(file_path)
    return data['train_images_pca'], data['test_images_pca'], data['train_labels'], data['test_labels']

def PCA_data(save_path='./Data/pca_Fashion.npz'):

    if os.path.exists(save_path):
        print("Loading preprocessed data...")
        return load_data(save_path)
    

    print('preprocessing data ...')
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=False, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=False, transform=transform)

    train_dataset_filtered = [(img, label) for img, label in train_dataset if label == 0 or label == 1]
    test_dataset_filtered = [(img, label) for img, label in test_dataset if label == 0 or label == 1]

    train_images = []
    train_labels = []
    test_images = []
    test_labels = []

    count_0, count_1 = 0, 0
    for img, label in train_dataset_filtered:
        if count_0 < 500 and label == 0:
            train_images.append(img.view(-1).numpy())
            train_labels.append(label)
            count_0 += 1
        elif count_1 < 500 and label == 1:
            train_images.append(img.view(-1).numpy())
            train_labels.append(label)
            count_1 += 1

        # Se abbiamo raggiunto 500 esempi per entrambe le classi, interrompiamo il loop
        if count_0 >= 500 and count_1 >= 500:
            break

    count_0, count_1 = 0, 0
    for img, label in test_dataset_filtered:
        if count_0 < 50 and label == 0:
            test_images.append(img.view(-1).numpy())
            test_labels.append(label)
            count_0 += 1
        elif count_1 < 50 and label == 1:
            test_images.append(img.view(-1).numpy())
            test_labels.append(label)
            count_1 += 1

        # Se abbiamo raggiunto 500 esempi per entrambe le classi, interrompiamo il loop
        if count_0 >= 50 and count_1 >= 50:
            break

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    print(train_images.shape, train_labels.shape)  #(1000, 784), (1000,)
    print(test_images.shape, test_labels.shape)    #(100, 784), (100,)



    ##IMPORTANT TO USE PCA
    scaler = StandardScaler()
    train_images_scaled = scaler.fit_transform(train_images)
    test_images_scaled = scaler.transform(test_images)

    pca = PCA(n_components=8)
    train_images_pca = pca.fit_transform(train_images_scaled)
    test_images_pca = pca.transform(test_images_scaled)
    
    train_images_pca = train_images_pca.astype(np.float64) #NECESSARY TO USE QUANTUM ENCODING
    test_images_pca  = test_images_pca.astype(np.float64)  #NECESSARY TO USE QUANTUM ENCODING

    print(train_images_pca.shape)  #(1000,8)
    print(test_images_pca.shape)   #(100,8)
    save_data(train_images_pca, test_images_pca, train_labels, test_labels, save_path)

    return train_images_pca, test_images_pca, train_labels, test_labels

