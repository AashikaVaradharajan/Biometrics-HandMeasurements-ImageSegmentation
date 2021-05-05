# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:22:02 2021

@author: Aashika Varadharajan

@Project: K-MEANS CLUSTERING 
"""
# IMPORTING ALL THE REQUIRED LIBRARIES
import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image

# FILEPATH CONTAINS THE PATH WHERE THE IMAGE IS STORED, AND THE IMAGE IS DISPLAYED 
filepath = "KMeans/InputImage_Original.jpeg"
img = Image.open(filepath)

# RESIZING THE IMAGE AS THE PROCESSOR TAKES VERY LONG FOR LARGE ITERATIONS, 
# SAVING THE RESIZED IMAGE INTHE PATH FOR REFERENCE
resized_img = img.resize((100, 100))
resized_img.save("KMeans/resizedImage.jpg")

# COLOR IMAGE IS LOADED INTO THE VARIABLE X AS (H x W x 3) 
X = cv2.imread("KMeans/resizedImage.jpg")
# SINCE THE IMAGE IS LOADED IN THE FORM OF BGR, CONVERTING THE COLOR TO RGB
X = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)

# RESHAPING X TO MAKE IT A 2D-ARRAY
pixel_values = X.reshape((-1,3))
pixel_values = np.float32(pixel_values)

# GETTING USER INPUT FOR NUMBER OF CLUSTERS
k = int(input("Enter the value for number of clusters: "))
conv_list = []
label_list = []
centers_list = []
label_array = []

# CRITERIA FOR TRAINING THE K-MEANS BLACK-BOX FUNCTION
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)

# ITERATION VALUE UPTO WHICH THE ITERATIVE K-MEANS WILL BE CALCULATED
attempts = 100

# FLAGS TO SET THE RANDOM CENTERS FOR K-MEANS in OPEN CV - BLACK-BOX
flags = cv2.KMEANS_RANDOM_CENTERS


"""
# KMEANS BLACK-BOX FUNCTION USING OPEN-CV
The inputs to the function are:
    data: which is the RGB values of the image
    K: which is the number of clusters 
    flags: random centroids
    Outputs: Convergence value, Cluster Labels and Centers for the clusters.
"""
Convergence, Cluster_labels, Centers = cv2.kmeans(data=pixel_values,K=k, bestLabels=None, criteria=criteria,
                                                  attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)

#Converting the centers into an array of intergers for plotting the segmented Image 
center_fin = np.uint8(Centers)

# Getting the values of pixels and labels together  in an  
Pixels_and_Labels = np.column_stack((pixel_values, Cluster_labels)) 
segmentation = center_fin[Cluster_labels.flatten()]    
segmentedImage = segmentation.reshape(X.shape)
cv2.imwrite('KMeans/segmentedImage_K'+str(k)+ '_Image.jpg', segmentedImage)
    
print("Open the segmented image in the local path and check")
 

