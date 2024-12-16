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
from sklearn.metrics import confusion_matrix
import seaborn as sns


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
    

# Function to perform augmentation
def augment_image(image):
    augmented_images = []

    # Rotate the image
    for angle in [90, 180, 270]:  
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)  # Scale is 1.0
        rotated = cv2.warpAffine(image, M, (w, h))
        augmented_images.append(rotated)

    # Scale the image
    for scale in [0.5, 1.5]:  # Scaling factors
        scaled = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        augmented_images.append(scaled)

    return augmented_images    


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
    histograms = normalize(histograms, norm='l2')
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

num_clusters = 2000  
sift = cv2.SIFT_create()
descriptors_list  = []
labels = []
#for user, image_list in images_by_user.items():
#    for image in image_list:
for image , user in augmented_dataset:
        #image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        # Step 1: Extract SIFT descriptors from all images
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
            labels.append(user)
 
# Step 2: Perform k-means clustering to build the visual vocabulary
kmeans = cluster_descriptors(descriptors_list, num_clusters)

print(f"Number of images: {len(descriptors_list)}")
print(f"Number of labels: {len(labels)}")

# Step 3: Create BoVW histograms
bow_histograms = create_bow_histograms(descriptors_list, kmeans)

print(f"Number of histograms: {len(bow_histograms)}")
print(f"Number of labels: {len(labels)}")

# Step 4: split the dataset randomly
random.shuffle(bow_histograms)
X_train, X_test, y_train, y_test = train_test_split(bow_histograms, labels, test_size=0.2, random_state=42)

# Step 5: Train an SVM Classifier
#svm = SVC(kernel='linear', random_state=42)
#svm = SVC(kernel='poly', degree=3, random_state=42)
svm = SVC(kernel='rbf', gamma=0.1, random_state=42)
svm.fit(X_train, y_train)
      
# Step 5: Evaluate Classifier
y_pred = svm.predict(X_test)
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred):.2f}")

'''
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f"KNN Accuracy: {accuracy_score(y_test, y_pred):.2f}")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")


nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred):.2f}")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}")




'''


'''
      
      # Step 2: Perform k-means clustering to build the visual vocabulary
      kmeans = cluster_descriptors(all_descriptors, num_clusters)
      
      # Step 3: Create Bag of Visual Words histograms
      bow_histograms = create_bow_histograms(all_descriptors, kmeans)
      
      # Step 4: Use `bow_histograms` as features for classification
      print(f"BoVW histograms shape: {len(bow_histograms)} x {len(bow_histograms[0])}")
      # bow_histograms is ready for classifier training (e.g., SVM, Random Forest)

'''