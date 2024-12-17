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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import time
from sklearn.model_selection import GridSearchCV

# Function to perform augmentation
def augment_image(image):
    augmented_images = []

    # Rotate the image
    for angle in [90, 180]:  #, 270
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)  # Scale is 1.0
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(rotated)

    

    return augmented_images    
'''
    # Scale the image
    for scale in [0.5, 1.5]:  # Scaling factors
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        augmented_images.append(scaled)
'''

# Step 2: Cluster Descriptors (K-Means for BoVW)
def cluster_descriptors(descriptors_list, num_clusters):
    # Combine all descriptors into one array
    all_descriptors = np.vstack(descriptors_list)
    print(f"Clustering {all_descriptors.shape[0]} descriptors into {num_clusters} clusters.")
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, verbose=1)
    kmeans.fit(all_descriptors)
    return kmeans

# Step 3: Create Bag of Words Histogram for Each Image
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
    #histograms = normalize(histograms, norm='l2')
    return np.array(histograms)


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

# Apply augmentation to each image
augmented_dataset = []  # List to hold augmented images and their labels

for user, image_list in images_by_user.items():
    for image_path in image_list:
        # Read the image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Error: Unable to read image at {image_path}")
            continue
        
        # Augment the image
        augmented_images = augment_image(image)
        
        # Save each augmented image along with its label (user)
        for augmented_image in augmented_images:
            augmented_dataset.append((augmented_image, user))  # Tuple: (image, label)

num_clusters = 1000  
sift = cv2.SIFT_create()
orb = cv2.ORB_create()
descriptors_list  = []
labels = []
start_time = time.time()

for image , user in augmented_dataset:
        # Extract SIFT / ORB descriptors from all images
        #keypoints, descriptors = sift.detectAndCompute(image, None)
        kp = orb.detect(image,None)
        kp, descriptors = orb.compute(image, kp)
        if descriptors is not None:
            descriptors_list.append(descriptors)
            labels.append(user)
end_time = time.time()
computation = end_time - start_time
#print(f"Time using SIFT algorithm for feature extraction:  {computation:.4f} seconds") 
print(f"Time using ORB algorithm for feature extraction:  {computation:.4f} seconds") 

# Perform k-means clustering 
kmeans = cluster_descriptors(descriptors_list, num_clusters)

# Create BoVW histograms
bow_histograms = create_bow_histograms(descriptors_list, kmeans)


# split the dataset randomly
combined = list(zip(bow_histograms, labels))
random.shuffle(combined)
bow_histograms, labels = zip(*combined)

X_train, X_test, y_train, y_test = train_test_split(bow_histograms, labels, test_size=0.2, random_state=42)



#  Train & Evaluate an SVM Classifier
'''
# parameter grid for SVC
param_grid = [
    {'kernel': ['poly'], 'degree': [2, 3, 4, 5]},  # Polynomial kernel degrees
    {'kernel': ['rbf'], 'gamma': [0.1, 1, 10, 100]}  # RBF kernel gamma values
]

# Initialize GridSearchCV
grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=3, scoring='accuracy', verbose=2)

# Fit to training data
grid_search.fit(X_train, y_train)

# Best hyperparameters and model
print(f"SVM Best Parameters: {grid_search.best_params_}")
print(f"SVM Best Validation Accuracy: {grid_search.best_score_}")
best_model = grid_search.best_estimator_


best_model.fit(X_train, y_train)  # Train on train + validation
test_predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f"SVM Final Test Accuracy: {test_accuracy}")

'''

svm = SVC(kernel='poly', degree=2, random_state=42)
svm.fit(X_train, y_train)
      
y_pred = svm.predict(X_test)
print(f"SVM with polynomial kernel of degree 2 Accuracy: {accuracy_score(y_test, y_pred):.2f}")

#  Train & Evaluate a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}")


#  Train & Evaluate a Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred):.2f}")
