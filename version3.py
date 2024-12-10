import os
from collections import defaultdict
import cv2
import random
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


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

# Step 2: Cluster Descriptors (K-Means for BoVW)
def cluster_descriptors(descriptors_list, num_clusters):
    # Combine all descriptors into one array
    all_descriptors = np.vstack(descriptors_list)
    print(f"Clustering {all_descriptors.shape[0]} descriptors into {num_clusters} clusters.")
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, verbose=1)
    kmeans.fit(all_descriptors)
    return kmeans

# Step 3: Create Bag of Words Histogram for Each Image
'''
def create_bow_histograms(descriptors_list, kmeans):
    num_clusters = kmeans.n_clusters
    histograms = []
    
    for descriptors in descriptors_list:
        if descriptors is not None:
            labels = kmeans.predict(descriptors)
            histogram, _ = np.histogram(labels, bins=np.arange(num_clusters + 1))
            histograms.append(histogram)
        else:
            histograms.append(np.zeros(num_clusters))
    
    histograms = normalize(histograms, norm='l2')
    return histograms
'''

def create_bow_histograms(descriptors_list, kmeans):
    num_clusters = kmeans.n_clusters
    histograms = []
    
    for descriptors in descriptors_list:
        if descriptors is not None:
            # Assign descriptors to the nearest cluster centers
            labels = kmeans.predict(descriptors)
            # Build a histogram of cluster assignments
            histogram, _ = np.histogram(labels, bins=np.arange(num_clusters + 1))
            histograms.append(histogram)
        else:
            # If no descriptors, create an empty histogram
            histograms.append(np.zeros(num_clusters))
    
    # Normalize histograms for consistency
    histograms = normalize(histograms, norm='l2')
    return np.array(histograms)




# Path to the extracted folder
extracted_path = "isolated_words_per_user"

# Dictionary to store images by user
images_by_user = defaultdict(list)


labels = []
user_id=''
# Traverse the folder structure
for root, dirs, files in os.walk(extracted_path):
    for file in files:
        if file.endswith(('.png')):  # Filter image files
            # Extract user_id from folder name 
            user_id = os.path.basename(root)  # Get the folder name
            file_path = os.path.join(root, file)
            
            # Add image to the dictionary
            images_by_user[user_id].append(file_path)
    labels.append(user_id)

num_clusters = 100  # Number of visual words
sift = cv2.SIFT_create()
descriptors_list  = []
for user, image_list in images_by_user.items():
    
    #print(f"User ID: {user}, Number of Images: {len(image_list)}")
   # if user == 'user001':
      
      for image in image_list:
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)

       # Step 1: Extract SIFT descriptors from all images
       # Step 3: Perform k-means clustering
kmeans = cluster_descriptors(descriptors_list, num_clusters)

print(f"Number of images: {len(descriptors_list)}")
print(f"Number of labels: {len(labels)}")

      
      # Step 4: Create BoVW histograms
bow_histograms = create_bow_histograms(descriptors_list, kmeans)
      


print(f"Number of histograms: {len(bow_histograms)}")
print(f"Number of labels: {len(labels)}")


      # Step 5: Train a Classifier
X_train, X_test, y_train, y_test = train_test_split(bow_histograms, labels, test_size=0.2, random_state=42)
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
      
      # Step 6: Evaluate Classifier
y_pred = svm.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")



'''
      
      # Step 2: Perform k-means clustering to build the visual vocabulary
      kmeans = cluster_descriptors(all_descriptors, num_clusters)
      
      # Step 3: Create Bag of Visual Words histograms
      bow_histograms = create_bow_histograms(all_descriptors, kmeans)
      
      # Step 4: Use `bow_histograms` as features for classification
      print(f"BoVW histograms shape: {len(bow_histograms)} x {len(bow_histograms[0])}")
      # bow_histograms is ready for classifier training (e.g., SVM, Random Forest)


# Example: Labels corresponding to image_paths
labels = [1, 2,3,4,5,6,7,8,9,10,11,12]  # Replace with actual labels

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(bow_histograms, labels, test_size=0.2, random_state=42)

# Train SVM classifier
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
'''