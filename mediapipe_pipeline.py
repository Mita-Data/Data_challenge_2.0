import mediapipe as mp
import torch


# FUNCTION TO EXTRACT MASK FROM MEDIAPIPE IMAGES

def get_mask(image, results):
    # Create a mask for the skin area
    mask = np.zeros_like(image)

    # Define indices for the facial landmarks that typically represent the skin area
    skin_landmark_indices = list(range(0, 468))  # Indices for all landmarks

    if results.multi_face_landmarks: # Check if any face is detected
        for face_landmarks in results.multi_face_landmarks:
            points = []
            for idx in skin_landmark_indices:
                x = int(face_landmarks.landmark[idx].x * image.shape[1])
                y = int(face_landmarks.landmark[idx].y * image.shape[0])
                points.append([x, y])

            points = np.array(points)

            # Create a convex hull around the skin landmarks to approximate the skin area
            hull = cv2.convexHull(points)

            # Draw the convex hull on the mask
            cv2.fillConvexPoly(mask, hull, (255, 255, 255))
    return mask

################################################################################################
# test if there is a GPU available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the model
# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# open df_train.pkl
with open('df_train.pkl', 'rb') as f:
    df_train = pickle.load(f)
image_dir = "crops_100K"
print(df_train.index[:20])
df_train.head(-10)

################################################################################################

# Load an image and convert it to RGB
#--------------------------------------------
image_dir = "crops_100K"
path = f"{image_dir}/{df_train.iloc[0]['filename']}"
image = cv2.imread(path)

# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Apply the MediaPipe Face Mesh model to the image
# ------------------------------------------------

# Process the image to find face landmarks
results = face_mesh.process(image)

# Create a mask for the skin area
mask = get_mask(image, results)
print(mask.shape)

# Apply the mask to the original image and show the result
skin_area = cv2.bitwise_and(image, mask)
print(skin_area.shape)

#####################################################################################

# Display the original image and the skin area
# ---------------------------------------------

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(skin_area)
plt.title('Skin Area')
plt.suptitle('Skin Area Extraction with MediaPipe Face Mesh Model\n'+path)


#####################################################################################

# Process all images in the dataset
# ---------------------------------

# Process all the images
#-----------------------

# Initialize variables
mask_dict = {}
face_pixels = []
image_pixels = []
image_size = []
mask_filename = []
df_train.index

# Process all the images (loop over the images)
for i in df_train.index[:10] :
    if i == 100: print(i)
    if i == 1000: print(i)
    if i % 5000 == 0: print(i)
    
    # load image (.loc will use index, whereas iloc will use row number)
    path = f"{image_dir}/{df_train.loc[i]['filename']}"
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # plt.imshow(image)
    # plt.show()

    # get the keypoints and save to file
    try :
        
        # Process the image to find face landmarks
        results = face_mesh.process(image)
        
        # save the keypoints
        if results.multi_face_landmarks:
            face_landmarks_list = []
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
                face_landmarks_list.append(landmarks)

            with open('test/' + str(i) + '_results.pkl', 'wb') as file:
                pickle.dump(face_landmarks_list, file)
            
            mask_filename.append(str(i) + '_results.pkl')
            

        # Create a mask for the skin area and the face area image
        mask = get_mask(image, results)
        skin_area = cv2.bitwise_and(image, mask) # Create the masked image
        cv2.imwrite('test/' + str(i) + '_masked.jpg', cv2.cvtColor(skin_area, cv2.COLOR_RGB2BGR))

        # Save data to the mask_dict
        mask_dict[df_train.iloc[i]['filename']] = mask
        image_pixels.append(mask.shape[0] * mask.shape[1])
        face_pixels.append(np.sum(mask[:,:,0] == 255))
        image_size.append((mask.shape[0] , mask.shape[1]))

    except UserWarning:
        print(f"Error with file {path}")

# Save the mask_dict to a file
with open('test/mask_dict.pkl', 'wb') as f: pickle.dump(mask_dict, f)

# Add new columns to the df_train and save
df_train['image_pixels'] = image_pixels
df_train['face_pixels'] = face_pixels
df_train['image_size'] = image_size
with open('df_train.pkl', 'wb') as f: pickle.dump(df_train, f)

# show the df_train
df_train.head(-10)