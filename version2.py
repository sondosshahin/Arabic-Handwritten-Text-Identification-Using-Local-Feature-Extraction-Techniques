import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# Step 1: SIFT Feature Extraction
def extract_sift_features(image_paths):
    sift = cv2.SIFT_create()
    descriptors_list = []
    
    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = sift.detectAndCompute(image, None)
        if descriptors is not None:
            descriptors_list.append(descriptors)
    
    return descriptors_list

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
            # Assign descriptors to the nearest cluster center
            labels = kmeans.predict(descriptors)
            # Build a histogram of cluster assignments
            histogram, _ = np.histogram(labels, bins=np.arange(num_clusters + 1))
            histograms.append(histogram)
        else:
            histograms.append(np.zeros(num_clusters))
    
    # Normalize histograms for consistency
    histograms = normalize(histograms, norm='l2')
    return histograms

# Step 4: Main Function
def main():
    # Example: Replace with your dataset
    image_paths = [
        "path/to/image1.jpg",
        "path/to/image2.jpg",
        "path/to/image3.jpg",
        # Add paths for all images
    ]
    
    num_clusters = 100  # Number of visual words
    
    # Step 1: Extract SIFT descriptors from all images
    descriptors_list = extract_sift_features(image_paths)
    
    # Step 2: Perform k-means clustering to build the visual vocabulary
    kmeans = cluster_descriptors(descriptors_list, num_clusters)
    
    # Step 3: Create Bag of Visual Words histograms
    bow_histograms = create_bow_histograms(descriptors_list, kmeans)
    
    # Step 4: Use `bow_histograms` as features for classification
    print(f"BoVW histograms shape: {len(bow_histograms)} x {len(bow_histograms[0])}")
    # bow_histograms is ready for classifier training (e.g., SVM, Random Forest)
    
if __name__ == "__main__":
    main()
