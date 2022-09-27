# -*- coding: utf-8 -*-

import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def posterior(test_X, w1, w10, w2, w20, w3, w30):
  a1 = np.dot(test_X, w1) + w10
  a2 = np.dot(test_X, w2) + w20
  a3 = np.dot(test_X, w3) + w30
  
  #sigma = np.exp(a1) + np.exp(a2) + np.exp(a3)
  #p1 = np.exp(a1) / sigma
  #p2 = np.exp(a2) / sigma
  #p3 = np.exp(a3) / sigma
     
  #Use posterior to classify
  maxi = max(a1[0], a2[0], a3[0])

  if maxi == a1[0]:
    return 1
  elif maxi == a2[0]:
    return 2
  else:
    return 3


if __name__=="__main__":
    #Read csv file
    wine_csv = pd.read_csv("./Wine.csv", encoding="big5", header=None)
    wine_data = wine_csv.to_numpy()
    
    """
    Data preprocessing
    """
    #Randomly choose 18 instances from each class for test set
    split_index1 = np.random.choice([i for i in range(0,59)], 18, replace=False)
    split_index2 = np.random.choice([i for i in range(59,130)], 18, replace=False)
    split_index3 = np.random.choice([i for i in range(130,178)], 18, replace=False)
    test_index = np.concatenate((split_index1, split_index2, split_index3))
    train_index = np.setdiff1d([i for i in range(0, 178)], test_index)
    
    #Create the training set and testing set
    train_set = np.empty([124, 14])
    test_set = np.empty([54, 14])   
    train_ind = 0
    test_ind = 0
    for i in range(0, 178):
      if i in train_index:
        for j in range(14):
          train_set[train_ind][j] = wine_data[i][j]
        train_ind = train_ind + 1
      else:
        for j in range(14):
          test_set[test_ind][j] = wine_data[i][j]
        test_ind = test_ind + 1
    
    #Randomly shuffle the testing set
    np.random.shuffle(test_set)

    #Separate sets into feature part(x) and label part(y) 
    train_x = train_set[0:, 1:14]
    train_y = train_set[:, 0]
    test_x = test_set[0:, 1:14]
    test_y = test_set[:, 0]
    

    #Normalize the training data and testing data with training data
    norm = np.linalg.norm(train_x)
    train_x = train_x/norm  
    test_x = test_x/norm
    
    
    """
    Starting MAP: Following procedure of chapter4
    """
    #Compute in-class mean
    
    train_1 = np.array(train_x[0:41, :])
    train_2 = np.array(train_x[41:94, :])
    train_3 = np.array(train_x[94:124, :])
    data_num = 124
    
    mean_1 = np.mean(train_1, axis=0)
    mean_2 = np.mean(train_2, axis=0)
    mean_3 = np.mean(train_3, axis=0)
    
    #Compute in-class cov
    cov_1 = np.zeros((13, 13))
    cov_2 = np.zeros((13, 13))
    cov_3 = np.zeros((13, 13))
    
    for x in train_1:
      cov_1 += np.dot(np.transpose([x - mean_1]), [x - mean_1]) / train_1.shape[0]
    for x in train_2:
      cov_2 += np.dot(np.transpose([x - mean_2]), [x - mean_2]) / train_2.shape[0]
    for x in train_3:
      cov_3 += np.dot(np.transpose([x - mean_3]), [x - mean_3]) / train_3.shape[0]
    
    
    #Assume shared covariance
    sh_cov = ((cov_1 * train_1.shape[0])+(cov_2 * train_2.shape[0])+(cov_3 * train_3.shape[0])) / data_num
    
    u, s, v = np.linalg.svd(sh_cov, full_matrices=False)
    sh_cov_inv1 = np.matmul(v.T * 1 / s, u.T)
    
    #Use close form solution to solve parameters
    w1 = np.dot(sh_cov_inv1, mean_1.reshape(-1, 1))
    w2 = np.dot(sh_cov_inv1, mean_2.reshape(-1, 1))
    w3 = np.dot(sh_cov_inv1, mean_3.reshape(-1, 1))
 
    w1_0 = (-0.5) * np.dot(mean_1, np.dot(sh_cov_inv1, mean_1)) + np.log(float(train_1.shape[0]) / data_num) 
    w2_0 = (-0.5) * np.dot(mean_2, np.dot(sh_cov_inv1, mean_2)) + np.log(float(train_2.shape[0]) / data_num) 
    w3_0 = (-0.5) * np.dot(mean_3, np.dot(sh_cov_inv1, mean_3)) + np.log(float(train_3.shape[0]) / data_num)
      
    
    #Prediction
    print("===== Result of prediction =====\n")
    pred = []
    
    for i in range(test_x.shape[0]):
      pred.append(posterior(test_x[i, :], w1, w1_0, w2, w2_0, w3, w3_0))
    
    accuracy = 0
    for j in range(test_x.shape[0]):
      if pred[j] == test_y[j]:
        accuracy += 1
    
    error_rate = float(test_x.shape[0] - accuracy) / test_x.shape[0]
    print("Accuracy: " + str(accuracy) + " correctly predicted out of " + str(test_x.shape[0]) +" testing data")
    print("Error rate: " + str(error_rate) + "%\n")
    
    print("Testing Label: ")
    print([int(i) for i in test_y])
    print("Predicted result: ")
    print(pred)
    print("\n")
    
    
    """
    PCA plot
    """
    
    print("Generating PCA plot.....\n\n")
    targets = [1, 2, 3]
    
    #n = 2
    pca1 = PCA(n_components=2)
    pca1.fit(train_x)
    train_xn = pca1.fit_transform(train_x)
    
    fig = plt.figure(figsize = (8,8))
    ax1 = fig.add_subplot(1,1,1) 
    ax1.set_xlabel('Principal Component 1', fontsize = 15)
    ax1.set_ylabel('Principal Component 2', fontsize = 15)
    ax1.set_title('2 component PCA', fontsize = 20)
    
    ax1.scatter(train_xn[0:41, 0], train_xn[0:41, 1], c = 'r', s = 50)
    ax1.scatter(train_xn[41:94, 0], train_xn[41:94, 1], c = 'b', s = 50)
    ax1.scatter(train_xn[94:124, 0], train_xn[94:124, 1], c = 'g', s = 50)
    ax1.legend(targets)
    ax1.grid()


    #n = 1
    pca2 = PCA(n_components=1)
    pca2.fit(train_x)
    train_xn = pca2.fit_transform(train_x)
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_title('1 component PCA', fontsize = 20)
    
    ax.scatter(train_xn[0:41, 0], 41*[1], c = 'r')
    ax.scatter(train_xn[41:94, 0], 53*[1],c = 'b')
    ax.scatter(train_xn[94:124, 0],30*[1], c = 'g')
    ax.legend(targets)
    ax.grid()

    
    #n = 3
    pca3 = PCA(n_components=3)
    pca3.fit(train_x)
    train_xn = pca3.fit_transform(train_x)
    
    ax2 = plt.axes(projection='3d')
    ax2.scatter(train_xn[0:41, 0], train_xn[0:41, 1], train_xn[0:41, 2], c = 'r', s = 50)
    ax2.scatter(train_xn[41:94, 0], train_xn[41:94, 1], train_xn[41:94, 2],c = 'b', s = 50)
    ax2.scatter(train_xn[94:124, 0], train_xn[94:124, 1], train_xn[94:124, 2],c = 'g', s = 50)
    ax2.legend(targets)
    ax2.grid()
    
