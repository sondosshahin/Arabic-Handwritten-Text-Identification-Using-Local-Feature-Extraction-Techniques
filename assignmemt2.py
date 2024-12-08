import os
from collections import defaultdict
import cv2


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
    print(f"User ID: {user}, Number of Images: {len(image_list)}")
    if user == 'user001':
      for image in image_list:
          print(image)
          img = cv2.imread(image)
          gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
          sift = cv2.SIFT_create()
          kp = sift.detect(gray,None)

          img=cv2.drawKeypoints(gray,kp,img)

          cv2.imwrite('sift_keypoints.jpg',img)


'''
for user_id, images in images_by_user.items():
    print(f"Processing images for user: {user_id}")
    for image_path in images:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        keypoints, descriptors = cv2.SIFT.detectAndCompute(image, None)
        print(f"Extracted {len(keypoints)} keypoints from {image_path}")
'''