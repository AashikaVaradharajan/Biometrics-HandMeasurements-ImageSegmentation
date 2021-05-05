# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 09:22:02 2021

@author: Aashika Varadharajan

@Project: ITERATIVE K-MEANS CLUSTERING FOR IMAGE SEGMENTATION
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

# GETTING USER INPUT FOR NUMBER OF CLUSTERS AFTER CHOOSING THE OPTIMUM VALUE OF k
k = int(input("Enter the value for number of clusters: "))
conv_list = []
label_list = []
centers_list = []
label_array = []
# CRITERIA FOR TRAINING THE K-MEANS BLACK-BOX FUNCTION
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
# ITERATION VALUE UPTO WHICH THE ITERATIVE K-MEANS WILL BE CALCULATED
attempts = 100

#ITERATION FOR 100 TIMES TO GET 100 DIFFERENT VALUES FOR CLUSTER LABELS WITH DIFFERENT CENTROID EACH TIME IN RANDOM
for i in range(attempts):
    flags = cv2.KMEANS_RANDOM_CENTERS
    Convergence, Cluster_labels, centers = cv2.kmeans(data=pixel_values,K=k, bestLabels=None, criteria=criteria, attempts=1, flags=cv2.KMEANS_RANDOM_CENTERS)
    conv_list.append(Cluster_labels)   
    label_list.append(Cluster_labels) 
    center_fin = np.uint8(centers) 
    centers_list.append(centers)
    #APPENDING LABELS TO THE CORRESPONDING PIXEL VALUES TO GET THE PROBABILITY     
    final_array = np.column_stack((pixel_values, Cluster_labels)) 
    segmentation = center_fin[Cluster_labels.flatten()]    
    segmentedImage = segmentation.reshape(X.shape)
    #SAVING THE SEGMENTED IMAGE AFTER EVERY ITERATION WITH THE UNIQUE VALUE WITH RESPECT TO K and ITERATION
    cv2.imwrite('KMeans/segmentedImage_K'+str(k)+ '_Image_'+str(i)+'.jpg', segmentedImage)
    
print("done")
freq_list = []
final_array = np.int64(final_array)
final_array = np.column_stack((final_array, Cluster_labels)) 
# FINDING THE BEST LABEL BASED ON THE LABEL ID WITH HIGHEST FREQUENCY
for row in final_array:
    counts = np.bincount(row)
    b = np.argmax(counts)
    freq_list.append(b)
    freq_arr = np.array(freq_list)
res_array = np.column_stack((final_array,freq_arr))    



Gray_Scaled = X.reshape((-1,2))
grey_slice = Gray_Scaled[5000:15000]


# Frequency plotting of clusters
from mpl_toolkits import mplot3d
# Creating dataset
z = final_array[:, 3]
x = grey_slice[:, 0]
y = grey_slice[:, 1]
 
# Creating figure
fig = plt.figure(figsize = (16, 9))
ax = plt.axes(projection ="3d")
   
# Add x, y gridlines 
ax.grid(b = True, color ='grey', 
        linestyle ='-.', linewidth = 0.3, 
        alpha = 0.2)

# Define color of graph
# my_cmap = plt.get_cmap('hsv')

 
# Creating plot
sctt = ax.scatter3D(x, y, z,
                    alpha = 0.8,
                    c = (x + y + z), 
                    cmap = 'gray', 
                    marker ='^')
 
plt.title("scatter plot-frequency mapping")
ax.set_xlabel('X-axis', fontweight ='bold') 
ax.set_ylabel('Y-axis', fontweight ='bold') 
ax.set_zlabel('Z-axis', fontweight ='bold')
fig.colorbar(sctt, ax = ax, shrink = 0.5, aspect = 5)
 
# show plot
plt.show()





