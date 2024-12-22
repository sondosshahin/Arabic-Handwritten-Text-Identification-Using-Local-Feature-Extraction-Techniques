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
import matplotlib.pyplot as plt


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


# Unpack features and labels from augmented_dataset
X, y = zip(*augmented_dataset)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Continue with the rest of your code
descriptors_list = []
labels = []
start_time = time.time()

# Extract features from training images
for image, user in zip(X_train, y_train):
    keypoints, descriptors = sift.detectAndCompute(image, None)
    #kp = orb.detect(image, None)
    #kp, descriptors = orb.compute(image, kp)
    if descriptors is not None:
        descriptors_list.append(descriptors)
        labels.append(user)

end_time = time.time()
computation = end_time - start_time
print(f"Time using ORB algorithm for feature extraction: {computation:.4f} seconds")


# Perform k-means clustering 
kmeans = cluster_descriptors(descriptors_list, num_clusters)

# Create BoVW histograms
bow_histograms = create_bow_histograms(descriptors_list, kmeans)

# Initialize lists for test descriptors and true labels
test_descriptors_list = []
true_labels = []


total_keypoints = 0
num_images = len(X_test)  
# Extract descriptors for the test set
for image, user in zip(X_test, y_test):
    kp, descriptors = sift.detectAndCompute(image, None)
    #kp = orb.detect(image, None)
    num_keypoints = len(kp)
    total_keypoints += num_keypoints
    #kp, descriptors = orb.compute(image, kp)
    if descriptors is not None:
        test_descriptors_list.append(descriptors)
        true_labels.append(user)
average_keypoints = total_keypoints / num_images
print(f"SIFT Average number of keypoints across all images: {average_keypoints:.2f}")

# Quantize test descriptors into visual words using the trained kmeans model
test_bow_histograms = create_bow_histograms(test_descriptors_list, kmeans)

# Predict the user by comparing test histograms to training histograms
predicted_users = []
for test_hist in test_bow_histograms:
    min_distance = float('inf')
    predicted_user = None
    
    # Compare test histogram with each training histogram
    for train_hist, train_user in zip(bow_histograms, labels):
        # Calculate distance (e.g., Euclidean distance) between histograms
        distance = np.linalg.norm(test_hist - train_hist)
        
        # Keep track of the closest match
        if distance < min_distance:
            min_distance = distance
            predicted_user = train_user
    
    predicted_users.append(predicted_user)

# Evaluate performance
accuracy = sum(p == t for p, t in zip(predicted_users, true_labels)) / len(true_labels)
print(f"Accuracy: {accuracy * 100:.2f}%")



# Ensure BoVW histograms are numpy arrays
X_train_bow = np.array(bow_histograms)
y_train = np.array(labels)

X_test_bow = np.array(test_bow_histograms)
y_test = np.array(true_labels)




#  Train & Evaluate a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_bow, y_train)
y_pred = rf.predict(X_test_bow)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred):.2f}")


#  Train & Evaluate a Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train_bow, y_train)
y_pred = nb.predict(X_test_bow)
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred):.2f}")


svm = SVC(kernel='poly', degree=2, random_state=42)  # You can choose other kernels if needed
svm.fit(X_train_bow, y_train)

# Test the SVM
accuracy = svm.score(X_test_bow, y_test)
print(f"SVM Accuracy: {accuracy * 100:.2f}%")




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


'''
# Data
classifiers = ['Naive Bayes', 'SVM', 'Random Forest']
sift_accuracies = [0.28, 0.42, 0.69]  # Example values
orb_accuracies = [0.21, 0.23, 0.22]   # Example values

# X-axis positions
x = np.arange(len(classifiers))  # the label locations
width = 0.35  # the width of the bars

# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
bar1 = ax.bar(x - width/2, sift_accuracies, width, label='SIFT', color='skyblue')
bar2 = ax.bar(x + width/2, orb_accuracies, width, label='ORB', color='orange')

# Add labels, title, and legend
ax.set_xlabel('Classifiers')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Comparison of SIFT and ORB with Classifiers')
ax.set_xticks(x)
ax.set_xticklabels(classifiers)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()
