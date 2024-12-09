import os
from collections import defaultdict
import cv2
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

def split_dataset(dataset, test_ratio=0.2, seed=42):
    random.seed(seed)
    train_set = []
    test_set = []

    for user, images in dataset.items():
        # Shuffle the images for randomness
        random.shuffle(images)
        
        # Ensure at least one image per word goes to the test set
        num_test = max(1, int(len(images) * test_ratio))
        
        test_images = images[:num_test]
        train_images = images[num_test:]
        
        test_set.extend(test_images)
        train_set.extend(train_images)
    
        return train_set, test_set




# Path to the extracted folder
extracted_path = "isolated_words_per_user"

# Dictionary to store images by user
images_by_user = defaultdict(list)

# Traverse the folder structure
for root, dirs, files in os.walk(extracted_path):
    for file in files:
        if file.endswith(('.png')):  # Filter image files
            # Extract user_id from folder name 
            user_id = os.path.basename(root)  # Get the folder name
            file_path = os.path.join(root, file)
            
            # Add image to the dictionary
            images_by_user[user_id].append(file_path)



for user, image_list in images_by_user.items():
    #print(f"User ID: {user}, Number of Images: {len(image_list)}")
    if user == 'user001':
      #split the data
      train_set, test_set = split_dataset(images_by_user)
      print("Training Set:", train_set)
      print("\n\n\n\n\n")
      print("Testing Set:", test_set)

        
      all_descriptors = []     
#apply sift on the train set
      for image in train_set:
        print(image)
        img = cv2.imread(image)
        gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        #kp = sift.detect(gray,None)
        kp, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            all_descriptors.append(descriptors)
        '''
        kmeans = KMeans(n_clusters=2)
        kmeans.fit(kp)

        plt.scatter( c=kmeans.labels_)
        plt.show()
        '''
        # Print the number of keypoints
        #print(f"Number of keypoints detected: {len(kp)}")
        img=cv2.drawKeypoints(gray,kp,img)
        cv2.imwrite('sift_keypoints.jpg',img)
        
    # Combine all descriptors into a single array
      all_descriptors = np.vstack(all_descriptors)

    # Step 2: Use K-Means to cluster the descriptors
      num_clusters = 50  # Number of clusters
      kmeans = KMeans(n_clusters=num_clusters, random_state=42)
      kmeans.fit(all_descriptors)

    # Step 3: Assign each descriptor to a cluster
      cluster_labels = kmeans.labels_

    # Optional: Organize descriptors by clusters
      clusters = {i: [] for i in range(num_clusters)}
      for idx, label in enumerate(cluster_labels):
        clusters[label].append(all_descriptors[idx])

    # Output cluster information
      for cluster_id, descriptors in clusters.items():
        print(f"Cluster {cluster_id}: {len(descriptors)} descriptors")
        
    
    #measure accuracy on test set
