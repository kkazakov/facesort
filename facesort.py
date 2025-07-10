import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from deepface import DeepFace
from retinaface import RetinaFace
import matplotlib.pyplot as plt
import shutil
import sys
import cv2
import argparse
from tqdm import tqdm

# parse command line arguments
parser = argparse.ArgumentParser(description='FaceSort - Sort images by detected faces')
parser.add_argument('-o', '--overwrite', action='store_true', help='Allow overwriting existing files')
parser.add_argument('-i', '--input', type=str, default='./images/', help='Input directory containing images (default: ./images/)')
parser.add_argument('-m', '--model', type=str, default='VGG-Face', 
                    choices=['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace'],
                    help='DeepFace model for face verification (default: VGG-Face)')
parser.add_argument('-d', '--distance', type=str, default='cosine',
                    choices=['cosine', 'euclidean', 'euclidean_l2'],
                    help='Distance metric for face comparison (default: cosine)')
parser.add_argument('-t', '--threshold', type=float, default=None,
                    help='Custom threshold for face matching (default: model-specific threshold)')
parser.add_argument('--enforce-detection', action='store_true', 
                    help='Enforce face detection (skip images where faces cannot be detected)')
parser.add_argument('--debug', action='store_true',
                    help='Show debug information including distance values')
args = parser.parse_args()

# variables and paths declaration
# work paths
images_directory = args.input if args.input.endswith('/') else args.input + '/'
work_path = images_directory + 'facesort/'
faces_directory = images_directory + 'faces/'

# specify which image formats are supported
image_types = ('.jpg', '.jpeg', '.png', '.bmp')

# temporary variable name for face comparison
temp_face = 'temp_face.png'

# individual face image variables
person_face_prefix = 'person_'
person_face_filetype = '.png'

# options for DeepFace verification models and metrics
deepface_models = ['VGG-Face', 'Facenet', 'Facenet512', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace', 'Dlib', 'SFace',]
deepface_metrics = ['cosine', 'euclidean', 'euclidean_l2']

# declaration of variables used for initial image discovery
image_list = []

# declaration of variables for face detection
people_list = []
PersonID = 0

# set overwrite flag for backward compatibility
arg = '-o' if args.overwrite else ''

print("\nFaceSort - https://github.com/kkazakov/facesort/", "\n")

# display current face matching configuration
print(f"Face matching configuration:")
print(f"  Model: {args.model}")
print(f"  Distance metric: {args.distance}")
if args.threshold is not None:
    print(f"  Custom threshold: {args.threshold} (lower = stricter)")
else:
    print(f"  Threshold: model default")
print(f"  Enforce detection: {args.enforce_detection}")
print(f"  Debug mode: {args.debug}")
print()

# validate input directory exists
if not os.path.exists(images_directory):
    print("Error: Input directory does not exist:", images_directory)
    exit(2)

# check if faces_directory already exists and has files
if os.path.exists(faces_directory):
    # check if faces_directory is empty
    # if not, warn the user about overwriting previous data/images and exit
    dir = os.listdir(faces_directory)
    if len(dir) != 0 and arg != '-o':
        print("To avoid losing or overwriting data please make sure the", work_path, "and", faces_directory, "are empty of any data you don't want to lose.")
        print("Use \"python facesort.py -o\" or \"python facesort.py --overwrite\" to allow overwriting files. (Not recommended)")
        print("Use \"python facesort.py -i <folder>\" to specify a different input folder.")
        exit(1)

    # if user still has files but uses -o parameter we continue anyway
    elif len(dir) != 0 and arg == '-o':
        print("Directories were not empty but overwrite command was used.")



# declaration of functions

# search directory provided in argument for images
# make a list (append image_list) of all images found
def scan_images(directory="./"):
    for folder_image in os.listdir(directory):
        if folder_image.endswith(image_types):
            image_list.append(folder_image)

    # if no images are found in the folder, notify user and exit
    if len(image_list) == 0:
        print("No images were found in the provided directory: ",directory)
        exit(2)
    
    # print out the amount of images found 
    print("Found", len(image_list), "images in", directory)


# check faces folder if face already exists
def check_face(image, directory=faces_directory, face_to_check=temp_face):
    
    # clear people_list to start with an empty list
    people_list.clear()

    # use PersonID as a global variable to use it across multiple functions
    global PersonID

    # make a list of all faces found in (faces) directory
    for face in os.listdir(directory):
        if face.endswith(image_types):
            people_list.append(face)

    # call match_face to check if face_to_check is found in directory
    copy_face = match_face(image, directory, face_to_check, people_list)
    
    # if face_to_check was not found in the faces_directory, create a new image and tie it to a PersonID
    if copy_face:
        # add image in the list tied to their PersonID
        list_of_faces[PersonID].append(image)

        old_name = (directory + face_to_check)
        new_name = (directory + person_face_prefix + str(PersonID) + person_face_filetype)

        # person not found in folder yet, add them to person list and copy temp image with their PersonID
        shutil.copyfile(old_name, new_name)
        PersonID = int(PersonID) + 1

    # remove all elements from people_list
    people_list.clear()


# check directory in parameter to see if face_to_check is already in it
def match_face(image, directory, face_to_check, list_of_people=people_list):

    for person in list_of_people:

        # variables for DeepFace verification
        img1_path = directory + face_to_check
        img2_path = directory + person
        
        try:
            # prepare verification parameters (without threshold - we'll handle it manually)
            verify_params = {
                'img1_path': img1_path,
                'img2_path': img2_path,
                'enforce_detection': args.enforce_detection,
                'model_name': args.model,
                'distance_metric': args.distance,
                'align': True
            }
            
            verify = DeepFace.verify(**verify_params)
        except Exception as e:
            if args.debug:
                print(f"Error verifying faces {face_to_check} and {person}: {e}")
            continue

        # get the actual distance value
        distance = verify['distance']
        
        # determine if faces match based on custom or default threshold
        if args.threshold is not None:
            # use custom threshold
            faces_match = distance <= args.threshold
        else:
            # use DeepFace's default verification result
            faces_match = verify['verified']
        
        # debug output
        if args.debug:
            print(f"Comparing {face_to_check} vs {person}: distance={distance:.4f}, threshold={args.threshold if args.threshold else 'default'}, match={faces_match}")

        # check result of face verification
        if faces_match and face_to_check != person:
            # face already exists, add the current image to the list of this persons ID
            # extract PersonID from the face that face_to_check matched with
            temp_id = str(person).removeprefix(person_face_prefix)
            temp_id = temp_id.removesuffix(person_face_filetype)

            # add image to PersonID's list in list_of_faces
            list_of_faces[int(temp_id)].append(image)

            # return False if matching face was found
            return False

    # return True if no matching faces were found    
    return True


# extract faces from images in image_list
def extract_faces():
    # create faces_directory if it doesn't exist
    if not os.path.exists(faces_directory):
        os.mkdir(faces_directory)
    
    # create progress bar for face extraction with time-based ETA
    pbar = tqdm(total=len(image_list), desc="Extracting faces", unit="image", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')
    
    # loop through all images found in the folder
    for image in image_list:
        # use RetinaFace to get all faces in the image
        faces = RetinaFace.extract_faces(images_directory + image, align = True, allow_upscaling = False)

        # update progress bar with current status
        pbar.set_postfix_str(f"Found {len(faces)} faces in {image}")

        # loop through all faces in the image
        for face in faces:
            # validate face dimensions before processing
            if face is None or face.size == 0 or len(face.shape) < 2 or face.shape[0] <= 0 or face.shape[1] <= 0:
                continue
            
            # additional validation for minimum face size
            if face.shape[0] < 10 or face.shape[1] < 10:
                continue

            # save temporary temp_face image to faces_directory
            plt.imsave(fname = faces_directory + temp_face, format='png', arr=face)

            # call check_face function to verify and sort faces
            check_face(image, faces_directory, temp_face)

        # update progress bar
        pbar.update(1)

    # close progress bar
    pbar.close()


# sort images into individual PersonID's folders
def sort_images():
    # create work_path if it doesn't exist
    if not os.path.exists(work_path):
        os.mkdir(work_path)

    # calculate total number of images to copy for progress bar
    total_images = sum(len(list_of_faces[individual]) for individual in range(PersonID))

    # create progress bar for image copying with time-based ETA
    pbar = tqdm(total=total_images, desc="Assigning images", unit="image",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

    # loop through all individual gathered faces listed in PersonID
    for individual in range(PersonID):
        # path variable for each PersonID's individual folder
        individual_path = work_path + person_face_prefix + str(individual) + '/'
        # create PersonID folder if it doesn't exist yet
        if not os.path.exists(individual_path):
            os.mkdir(individual_path)

        # copy PersonID's face to their folder
        base_face = faces_directory + person_face_prefix + str(individual) + person_face_filetype
        folder_face = individual_path + '_' + person_face_prefix + str(individual) + person_face_filetype
        shutil.copyfile(base_face, folder_face)

        # loop through all pictures tied to the PersonID's list
        for pictures in list_of_faces[int(individual)]:
            # variables to copy images for each person
            copy_from_path = images_directory + pictures
            copy_to_path = individual_path + pictures

            # update progress bar with current status
            pbar.set_postfix_str(f"Assigning: Copying image {pictures} to {person_face_prefix}{individual}")

            # copy each image to the PersonID's folder
            shutil.copyfile(copy_from_path, copy_to_path)

            # update progress bar
            pbar.update(1)

    # close progress bar
    pbar.close()


print("\nDepending on your system and amount of images this process may take a while.\n")

# scan folder for all images
scan_images(images_directory)

# every image that matches a PersonID will be added to the list for sorting to individual folders
list_of_faces = [[] for x in range(len(image_list)*len(image_list))]

# extract faces from images
extract_faces()

# sort faces to folders
sort_images()

# delete temporary image face file
if (os.path.isfile(faces_directory+temp_face)):
    os.remove(faces_directory+temp_face)


print("Process completed successfully.")
print(len(image_list),"image(s) were found and sorted into folders by",int(PersonID)-1,"unique faces.")
print("\nThank you for using FaceSort.")
print("https://github.com/kkazakov/facesort/", "\n")
exit(0)
