# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 18:06:22 2021

@author: ritup
"""

import random
import numpy as np
import pandas as pd

class Centroid:
    def __init__(self, location=0):
        self.location = location
        self.closest_users = set()

    def get_k_means(self, user_feature_map, num_features_per_user, k):
        # Don't change the following two lines of code.
        random.seed(42)
        # Gets the inital users, to be used as centroids.
        inital_centroid_users = random.sample(sorted(list(user_feature_map.keys())), k)
        # Write your code here.
        centroids = [user_feature_map[a] for a in inital_centroid_users]
        return centroids            
    
    # Following method calculate the distance from each user_id to all centroids and assign the
    # user to the centroid with minimum distance.
    # Distance calculation is done using Manhatten distance.
    # Formula used = |x1-x2|+|y1-y2|+|z1-z2|+|w1-w2|
    # Function returns cluster and label.
    # cluster is a dictionary that contains details of each cluster like centroid and user_id of 
    # users that belong to that cluster.
    # label is a dataframe that contains label for each user_id.
    def calculate_distance_and_clusterize(self, centroids, user_feature_map):
        user_label = pd.DataFrame(columns=['User id', 'Label', 'Features'])
        user_label['User id'] = [u for u in user_feature_map]   
        cluster = {}  
        for i in range(len(centroids)): 
            cluster[i+1] = {"centroid": [], "users": []}
            cluster[i+1]["centroid"] = [p for p in centroids[i]]
        labels = [] 
        features = []
        for x in user_feature_map:
            point = np.array(user_feature_map[x])
            dist = []
            for y in centroids:
                c = np.array(y)
                diff = np.absolute(c - point)
                distance = sum(diff)
                dist.append(distance)
                l = dist.index(min(dist))+1
            labels.append(l)
            features.append(point)
            cluster[l]['users'].append(x)
        user_label['Label'] = labels 
        user_label['Features'] = features
        return cluster, user_label
        
    
    # Below method calculate new centroids.
    # It uses average to find new cluster.
    # It returns old clusters and new clusters.
    def find_new_centroid(self, cluster, user_feature_map):
        old_centroid = []
        old_centroid = [cluster[i]["centroid"] for i in cluster]
        new_centroid = []
        for i in cluster:
            centroid = np.array([0 for a in range(len(user_feature_map['uid_0']))])
            a = cluster[i]["users"]
            for j in a:
                loc = np.array(user_feature_map[j])
                centroid = centroid + loc
            centroid = centroid / len(a)
            new_centroid.append(list(centroid))
        return old_centroid, new_centroid
    

    # This method finds within cluster distance.
    # This is to measure the homogeneity within the clusters.
    # This has been calculated by measuring the distance between each point 
    # and centroid of the cluster and then taking average of all distances.
    # Manhatten distance is used to calculate the distance. 
    # It returns average within-cluster-distance value.      
    def within_cluster_distance(self, cluster, user_feature_map):
        avg_clusters = []
        for i in cluster:
            c = np.array(cluster[i]["centroid"])
            users = cluster[i]["users"]
            sum_clusters = []
            for j in users:
                x = np.array(user_feature_map[j])
                point_dist = np.absolute(c - x)
                distance = sum(point_dist)
                sum_clusters.append(distance)
            avg_clusters.append(sum(sum_clusters) / len(users))
        wss = sum(avg_clusters) / len(cluster)
        return wss
     
    
    # This method calculates between cluster distance, ie, how seperate the clusters are from eac other.
    # This is to measure the variability among the clusters.
    # This has been done by finding the distance between each cluster with remaining clusters and
    # then taking average.
    # It returns average between-cluster-distance value.
    def between_cluster_distance(self, centroids):
        dist = []
        for i in range(len(centroids)):
            c_i = np.array(centroids[i])
            for j in range(i+1, len(centroids)):
                c_j = np.array(centroids[j])
                c_c = np.absolute(c_i - c_j)
                distance = sum(c_c)
                dist.append(distance)
        bss = sum(dist) / len(centroids)
        return bss
               
    
    
    # This is the main method to be called in order to fit the model.
    # It returns centroids, cluster details and label. 
    # The process is made to run atleast 10 times.          
    def fit(self, user_feature_map, num_features_per_user, k):
        centroids = self.get_k_means(user_feature_map, num_features_per_user, k)
        old_centroid = []
        new_centroid = []
        for i in range(30):
            cluster, label = self.calculate_distance_and_clusterize(centroids, user_feature_map)
            old_centroid, new_centroid = self.find_new_centroid(cluster, user_feature_map)
            if np.all(old_centroid == new_centroid) and i > 10:
                print("\nNo. of iterations : ", i)
                break 
            else:
                centroids = new_centroid
        return centroids, cluster, label
    
    

    # This method is to generate a small report with all details.
    def report(self, centroid, cluster, user_feature_map, label):
        wss = self.within_cluster_distance(cluster, user_feature_map)
        bss = self.between_cluster_distance(centroid)
        #self.plot(user_feature_map, label)
        dataframe = pd.DataFrame(columns= ['Cluster no.', 'Centroid', 'Users'])
        dataframe['Cluster no.'] = [i for i in cluster]
        l = []
        m = []
        for i in cluster:
            l.append(cluster[i]['centroid'])
            m.append(cluster[i]['users'])
        dataframe['Centroid'] = [i for i in l]
        dataframe['Users'] = [i for i in m]
        print("\nDetailed Report")
        print("===============\n")
        print("Cluster Details : ")
        print('------------------')
        print(dataframe,"\n")
        print("No. of users in each cluster :")
        print('------------------------------')
        for i in cluster:
            l = len(cluster[i]['users'])
            p = (l / len(user_feature_map)) * 100
            print("Cluster {} : {}\t  ({:.2f}%)". format(i, l, p))
        print("\n")
        print("Within Cluster average distance : {:.3f}\n". format(wss))
        print("Between Cluster average distance : {:.3f}\n". format(bss))
        print("Labels :")
        print('--------')
        print(label)
        print('\n!!!! Thank You !!!!')
        
            
if __name__ == '__main__':
     
    # A text file has been used to read the data from.       
    with open("C:\\Users\\ritup\\OneDrive\\Desktop\\Py_Practice\\Cyberlens\\user_feature_map.txt", 'r') as f1:  
        input_file = f1.read()
     
    # Cleaning the data from the text file to transform into desired form.    
    input_file = input_file.replace('{\n ', "").replace('}', '').replace('\"', '').replace("\"", '').strip()
    file = input_file.split(',\n ') 
    f = []
    for i in file:
        p1 = i.strip()
        p2 = p1.split(':')
        f.append(p2)
    lst = []
    for i in f:
        l = []
        l.append(i[0])
        l1 = []
        s1 = i[1].strip()
        s1 = s1.replace('[', '').replace(']', '').replace(',', '')
        f0 = s1.split(' ')
        for j in f0:
            l1.append(float(j))
        l.append(l1)
        lst.append(l)
        
    # Transforming the data from the text file into a dictionary.    
    user_feature_map = dict(lst)
    
    # Getting input for the value of k from the user and checking for number.
    k = input("Enter value for k = ")  
    assert k.isdigit(), "!!! Please enter number !!!"
    
    # Checking whether number is greater than 0.    
    k = int(k)
    assert k > 0, "!!! Number should be greater than 0 !!!"
    
    a = list(user_feature_map.values())[0]
    num_features_per_user = len(a)
    
    # Centroid class initialization
    kMeans = Centroid()
    
    # Calling the fit method to find the centroids
    cluster_centroid, cluster, label = kMeans.fit(user_feature_map, num_features_per_user, k) 
    
    
    # Printing the resultant centroids
    print("\nCluster centroids are : ") 
    print('------------------------')
    for i, j in enumerate(cluster_centroid):
        j1 = [round(n, 3) for n in j]
        print(f'Cluster {i+1} centroid : {j1}')  
        
    ans = input("Would you like a complete report ? press y/n : ")       
    
    if ans == 'Y' or ans == 'y':
        kMeans.report(cluster_centroid, cluster, user_feature_map, label)
    else:
        print("!!!! Thank you !!!!")
                  
