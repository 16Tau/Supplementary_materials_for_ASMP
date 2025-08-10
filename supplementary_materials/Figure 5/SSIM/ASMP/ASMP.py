"""
The primary function of the current code file is to perform untargeted attacks on a specified convolutional neural network model using ASMP based on the infinite norm.
First, set `actual_images_folder_absolute_path` to the absolute path of the folder containing the original images.
Second, set `output_directory` to the absolute path of the folder where the adversarial examples will be saved.
Third, set `index_file_absolute_path` to the absolute path of the index file.
Fourth, set `weight_file_absolute_path` to the absolute path of the weight file.
Fifth, adjust `degree` to set the maximum tampering range for each pixel.
Sixth, use `"from torchvision.models import googlenet"` to import the official convolutional neural network model structure, and adjust `model_current` to specify the current convolutional neural network model.
Finally, execute the code file to generate the untargeted adversarial examples.
"""



import json
import shutil
import matplotlib.pyplot as plt
from torchvision.models import googlenet
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import math
from skimage.metrics import structural_similarity as ssim
from skimage import io

def calculate_ssim(img1, img2):
    return ssim(img1, img2, win_size=3)

def calculate_ssim_for_folders(folder1, folder2):
    file_list1 = os.listdir(folder1)
    file_list2 = os.listdir(folder2)

    if len(file_list1) != len(file_list2):
        raise ValueError("The quantity in the two folders is not equal")

    ssim_results = []
    total_ssim = 0

    ssim_bins = {
        "0.995-1": 0,
        "0.996-1": 0,
        "0.997-1": 0,
        "0.998-1": 0,
        "0.999-1": 0
    }

    for file_name1, file_name2 in zip(file_list1, file_list2):
        if file_name1.endswith(('.png', '.jpg', '.jpeg', '.bmp')) and file_name2.endswith(
                ('.png', '.jpg', '.jpeg', '.bmp')):
            img1_path = os.path.join(folder1, file_name1)
            img2_path = os.path.join(folder2, file_name2)

            img1 = io.imread(img1_path)
            img2 = io.imread(img2_path)

            ssim_value = calculate_ssim(img1, img2)

            ssim_results.append((file_name1, file_name2, ssim_value))
            total_ssim += ssim_value

            if ssim_value >= 0.990:
                ssim_bins["0.995-1"] += 1
            if ssim_value >= 0.992:
                ssim_bins["0.996-1"] += 1
            if ssim_value >= 0.994:
                ssim_bins["0.997-1"] += 1
            if ssim_value >= 0.996:
                ssim_bins["0.998-1"] += 1
            if ssim_value >= 0.999:
                ssim_bins["0.999-1"] += 1

    print("\nASR:")
    for bin_name, count in ssim_bins.items():
        print(f"{bin_name}: {count*100/len(ssim_results)}%")

    return ssim_results

def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))


# Calculate untargeted attack categories
def untarget_index_image_path(image_path, index_path, weight_path, index, model_cnn):
    # Load image
    img = Image.open(image_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    with open(index_path, "r") as f:
        class_indict = json.load(f)
    # Create model
    model = model_cnn(num_classes=1000).to(device)
    # Load model weights
    model.load_state_dict(torch.load(weight_path))
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    top_probs, top_indices = torch.topk(classification_probability, 1000)
    for i in range(1000):
        if top_indices[i] == index and i != 999:
            attack_index = top_indices[i+1]
            break
        if i == 999:
            attack_index = top_indices[0]
    return(attack_index)


# Input the image into the model for category prediction, and input it as the path of the image file
def predict_image_path(image_path, index_path, weight_path, index, model_cnn):
    # Load image
    img = Image.open(image_path)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    with open(index_path, "r") as f:
        class_indict = json.load(f)
    # Create model
    model = model_cnn(num_classes=1000).to(device)
    # Load model weights
    model.load_state_dict(torch.load(weight_path))
    # Set the model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        classification_probability = torch.softmax(output, dim=0)
    # Get the index of the class with the highest probability
    predicted_class_index = torch.argmax(classification_probability).item()
    return(predicted_class_index,output[index])


# Calculate the pixel weight matrix of a single specified image with category index number "index"
def pixel_weight_matrix_image_path(image_path, weight_path, index, model_cnn):
    img = Image.open(image_path)
    plt.imshow(img)
    img = data_transform(img)
    img = torch.unsqueeze(img, dim=0)
    # Create model
    model = model_cnn(num_classes=1000).to(device)
    model.load_state_dict(torch.load(weight_path))
    # Set the model to evaluation mode
    model.eval()
    output = torch.squeeze(model(img.to(device))).cpu()
    classification_probability = torch.softmax(output, dim=0)
    top_probs, top_indices = torch.topk(classification_probability, 3)
    img = img.to(device)
    model.eval()
    img.requires_grad_()
    output = model(img)
    pred_score = output[0, index]
    pred_score.backward(retain_graph=True)
    gradients = img.grad
    channel_r = gradients[0, 0, :, :].cpu().detach().numpy()
    channel_g = gradients[0, 1, :, :].cpu().detach().numpy()
    channel_b = gradients[0, 2, :, :].cpu().detach().numpy()
    return channel_r, channel_g, channel_b

# Perform initialization operations on the images
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# The absolute path to the folder where the actual images
actual_images_folder_absolute_path = r"..."

# The absolute path to the folder where adversarial samples are saved
output_directory = r"..."

# The absolute path of index file
index_file_absolute_path = r"..."
assert os.path.exists(index_file_absolute_path), "file: '{}' dose not exist.".format(index_file_absolute_path)
with open(index_file_absolute_path, "r") as f:
    class_indict = json.load(f)

# The absolute path of weight file for convolutional neural network model
weight_file_absolute_path = r"..."

# The maximum degree of tampering of a single pixel
degree = 4

# Iterative step size
iteration_step_size = 0.001

# The total number of images in the folder that require tampering attacks
the_total_number_of_tampered_images = 0

image_files = sorted([f for f in os.listdir(actual_images_folder_absolute_path) if f.endswith('.png')], key=sort_func)

# Record the total number of original images
image_num = 0

# Record the total number of successfully attacked images
success_num = 0

# Calculate the maximum number of iterations
max_num_iterative = int(degree / 255 * (2.4285 - (-2.0357)) / iteration_step_size)

# The currently used convolutional neural network model
model_current = googlenet

for image_file in image_files:
    print(f"The image number currently being processed is: {image_file}")
    image_num = image_num + 1

    the_total_number_of_tampered_images = the_total_number_of_tampered_images + 1

    actual_image_absolute_path = os.path.join(actual_images_folder_absolute_path, image_file)

    image = Image.open(actual_image_absolute_path)
    image = image.resize((224, 224))
    actual_image = np.array(image)

    # The R, G, B three channel matrix of the actual image
    actual_image_channel_R = actual_image[:, :, 0]
    actual_image_channel_R = actual_image_channel_R.astype(np.float64)
    actual_image_channel_G = actual_image[:, :, 1]
    actual_image_channel_G = actual_image_channel_G.astype(np.float64)
    actual_image_channel_B = actual_image[:, :, 2]
    actual_image_channel_B = actual_image_channel_B.astype(np.float64)

    # Standardize the actual image
    actual_image_transform_matrix_R = ((actual_image_channel_R / 255) - 0.485) / 0.229
    actual_image_transform_matrix_G = ((actual_image_channel_G / 255) - 0.456) / 0.224
    actual_image_transform_matrix_B = ((actual_image_channel_B / 255) - 0.406) / 0.225
    ####################################################################################################
    # Perform category prediction on the original image
    actual_image_index, x = predict_image_path(actual_image_absolute_path, index_file_absolute_path, weight_file_absolute_path, 0, model_current)
    print(f"Top-1 category index number of the actual image:{actual_image_index}")
    ####################################################################################################
    # Mark whether the untarget attack was successful
    flag = 0

    # The first iteration image is the original image
    iterative_image_path = actual_image_absolute_path

    # Number of iterations
    num_iterative = 0

    # Define a four-dimensional array to record the matrix of attack samples during each forward iteration tampering process
    iterative_image_matrix_forward = np.zeros((140, 3, 224, 224), dtype=np.float64)

    # Define a four-dimensional array to record the positions of tampered pixels during each round of forward iteration tampering process
    iterative_image_pixel_dot_ischange = np.zeros((140, 3, 224, 224), dtype=np.float64)

    # Create a one-dimensional array to record the target category labels selected during each round of forward iteration tampering process
    iterative_image_label = np.zeros(140, dtype=int)

    # Record original image X0
    iterative_image_matrix_forward[num_iterative][0] = actual_image_channel_R.copy()
    iterative_image_matrix_forward[num_iterative][1] = actual_image_channel_G.copy()
    iterative_image_matrix_forward[num_iterative][2] = actual_image_channel_B.copy()

    # Perform forward iteration tampering
    while flag == 0 and num_iterative <= max_num_iterative:

        num_iterative = num_iterative + 1
        print(f"Current iteration count：{num_iterative}")

        # Calculate the category of untargeted attacks selected for this iteration
        attack_index = untarget_index_image_path(iterative_image_path, index_file_absolute_path, weight_file_absolute_path, actual_image_index, model_current)
        print(f"The selected attack category for this round：{attack_index}")

        # Record the attack target category label of the iterative image input in this round
        iterative_image_label[num_iterative] = attack_index

        # Calculate the pixel weight matrix of the current iterative image in the Top1 category
        iterative_image_pixel_weight_matrix_in_actual_label_R, iterative_image_pixel_weight_matrix_in_actual_label_G, iterative_image_pixel_weight_matrix_in_actual_label_B = pixel_weight_matrix_image_path(
            iterative_image_path, weight_file_absolute_path, actual_image_index, model_current
        )
        #################################################################################################
        # Calculate the pixel weight matrix of the current iterative image in the target category
        iterative_image_pixel_weight_matrix_in_attack_label_R, iterative_image_pixel_weight_matrix_in_attack_label_G, iterative_image_pixel_weight_matrix_in_attack_label_B = pixel_weight_matrix_image_path(
            iterative_image_path, weight_file_absolute_path, attack_index, model_current
        )
        ########################################################################################################
        # Calculate the RGB three channel matrix of iterative images
        image = Image.open(iterative_image_path)
        image = image.resize((224, 224))
        iterative_image = np.array(image)

        # The RGB three channel matrix of the image in this iteration
        iterative_image_channel_R = iterative_image[:, :, 0]
        iterative_image_channel_R = iterative_image_channel_R.astype(np.float64)
        iterative_image_channel_G = iterative_image[:, :, 1]
        iterative_image_channel_G = iterative_image_channel_G.astype(np.float64)
        iterative_image_channel_B = iterative_image[:, :, 2]
        iterative_image_channel_B = iterative_image_channel_B.astype(np.float64)

        # Calculate the standardized matrix of iterative image RGB three channels
        iterative_image_transform_matrix_R = (iterative_image_channel_R / 255 - 0.485) / 0.229
        iterative_image_transform_matrix_G = (iterative_image_channel_G / 255 - 0.456) / 0.224
        iterative_image_transform_matrix_B = (iterative_image_channel_B / 255 - 0.406) / 0.225

        # Calculate the contribution matrix of iterative images to the original labels
        iterative_image_pixel_classification_contribution_matrix_in_actual_label_R = iterative_image_transform_matrix_R * iterative_image_pixel_weight_matrix_in_actual_label_R
        iterative_image_pixel_classification_contribution_matrix_in_actual_label_G = iterative_image_transform_matrix_G * iterative_image_pixel_weight_matrix_in_actual_label_G
        iterative_image_pixel_classification_contribution_matrix_in_actual_label_B = iterative_image_transform_matrix_B * iterative_image_pixel_weight_matrix_in_actual_label_B

        # Calculate the contribution matrix of iterative images in the target category
        iterative_image_pixel_classification_contribution_matrix_in_attack_label_R = iterative_image_transform_matrix_R * iterative_image_pixel_weight_matrix_in_attack_label_R
        iterative_image_pixel_classification_contribution_matrix_in_attack_label_G = iterative_image_transform_matrix_G * iterative_image_pixel_weight_matrix_in_attack_label_G
        iterative_image_pixel_classification_contribution_matrix_in_attack_label_B = iterative_image_transform_matrix_B * iterative_image_pixel_weight_matrix_in_attack_label_B
        ############################################################################################################
        # Calculate the tampering matrix of iterative images based on the iteration step size
        iterative_image_transform_matrix_R_increase = iterative_image_transform_matrix_R.copy()
        iterative_image_transform_matrix_G_increase = iterative_image_transform_matrix_G.copy()
        iterative_image_transform_matrix_B_increase = iterative_image_transform_matrix_B.copy()

        iterative_image_transform_matrix_R_decrease = iterative_image_transform_matrix_R.copy()
        iterative_image_transform_matrix_G_decrease = iterative_image_transform_matrix_G.copy()
        iterative_image_transform_matrix_B_decrease = iterative_image_transform_matrix_B.copy()

        for i in range(224):
            for j in range(224):
                iterative_image_transform_matrix_R_increase[i][j] = iterative_image_transform_matrix_R_increase[i][j] + iteration_step_size
                iterative_image_transform_matrix_G_increase[i][j] = iterative_image_transform_matrix_G_increase[i][j] + iteration_step_size
                iterative_image_transform_matrix_B_increase[i][j] = iterative_image_transform_matrix_B_increase[i][j] + iteration_step_size

                iterative_image_transform_matrix_R_decrease[i][j] = iterative_image_transform_matrix_R_decrease[i][j] - iteration_step_size
                iterative_image_transform_matrix_G_decrease[i][j] = iterative_image_transform_matrix_G_decrease[i][j] - iteration_step_size
                iterative_image_transform_matrix_B_decrease[i][j] = iterative_image_transform_matrix_B_decrease[i][j] - iteration_step_size
        ##############################################################################################################
        # Define a matrix to retain tampered information
        change_matrix_R = iterative_image_channel_R.copy()
        change_matrix_R = change_matrix_R.astype(np.float64)

        change_matrix_G = iterative_image_channel_G.copy()
        change_matrix_G = change_matrix_G.astype(np.float64)

        change_matrix_B = iterative_image_channel_B.copy()
        change_matrix_B = change_matrix_B.astype(np.float64)

        # Perform positive absolute monotonic tampering
        for i in range(224):
            for j in range(224):
                ############################################################################################################################################################################################
                if iterative_image_pixel_classification_contribution_matrix_in_attack_label_R[i][j] > 0 and iterative_image_pixel_classification_contribution_matrix_in_actual_label_R[i][j] < 0:

                    if iterative_image_transform_matrix_R[i][j] > 0:
                        change_matrix_R[i][j] = math.ceil(
                            (iterative_image_transform_matrix_R_increase[i][j] * 0.229 + 0.485) * 255)
                        change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                        actual_image_channel_R[i][j] + degree)
                        change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][0][i][j] = 1
                    if iterative_image_transform_matrix_R[i][j] < 0:
                        change_matrix_R[i][j] = int(
                            (iterative_image_transform_matrix_R_decrease[i][j] * 0.229 + 0.485) * 255)
                        change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                        actual_image_channel_R[i][j] + degree)
                        change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][0][i][j] = 1

                if iterative_image_pixel_classification_contribution_matrix_in_attack_label_G[i][j] > 0 and iterative_image_pixel_classification_contribution_matrix_in_actual_label_G[i][j] < 0:
                    if iterative_image_transform_matrix_G[i][j] > 0:
                        change_matrix_G[i][j] = math.ceil(
                            (iterative_image_transform_matrix_G_increase[i][j] * 0.224 + 0.456) * 255)
                        change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                        actual_image_channel_G[i][j] + degree)
                        change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][1][i][j] = 1
                    if iterative_image_transform_matrix_G[i][j] < 0:
                        change_matrix_G[i][j] = int(
                            (iterative_image_transform_matrix_G_decrease[i][j] * 0.224 + 0.456) * 255)
                        change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                        actual_image_channel_G[i][j] + degree)
                        change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][1][i][j] = 1

                if iterative_image_pixel_classification_contribution_matrix_in_attack_label_B[i][j] > 0 and iterative_image_pixel_classification_contribution_matrix_in_actual_label_B[i][j] < 0:
                    if iterative_image_transform_matrix_B[i][j] > 0:
                        change_matrix_B[i][j] = math.ceil(
                            (iterative_image_transform_matrix_B_increase[i][j] * 0.225 + 0.406) * 255)
                        change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                        actual_image_channel_B[i][j] + degree)
                        change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][2][i][j] = 1
                    if iterative_image_transform_matrix_B[i][j] < 0:
                        change_matrix_B[i][j] = int(
                            (iterative_image_transform_matrix_B_decrease[i][j] * 0.225 + 0.406) * 255)
                        change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                        actual_image_channel_B[i][j] + degree)
                        change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][2][i][j] = 1
                ############################################################################################################################################################################################
                if iterative_image_pixel_classification_contribution_matrix_in_attack_label_R[i][j] < 0 and iterative_image_pixel_classification_contribution_matrix_in_actual_label_R[i][j] > 0:
                    if iterative_image_transform_matrix_R[i][j] < 0:
                        change_matrix_R[i][j] = math.ceil(
                            (iterative_image_transform_matrix_R_increase[i][j] * 0.229 + 0.485) * 255)
                        change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                        actual_image_channel_R[i][j] + degree)
                        change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][0][i][j] = 1
                    if iterative_image_transform_matrix_R[i][j] > 0:
                        change_matrix_R[i][j] = int(
                            (iterative_image_transform_matrix_R_decrease[i][j] * 0.229 + 0.485) * 255)
                        change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], actual_image_channel_R[i][j] - degree,
                                                        actual_image_channel_R[i][j] + degree)
                        change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][0][i][j] = 1


                if iterative_image_pixel_classification_contribution_matrix_in_attack_label_G[i][j] < 0 and iterative_image_pixel_classification_contribution_matrix_in_actual_label_G[i][j] > 0:
                    if iterative_image_transform_matrix_G[i][j] < 0:
                        change_matrix_G[i][j] = math.ceil(
                            (iterative_image_transform_matrix_G_increase[i][j] * 0.224 + 0.456) * 255)
                        change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                        actual_image_channel_G[i][j] + degree)
                        change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][1][i][j] = 1
                    if iterative_image_transform_matrix_G[i][j] > 0:
                        change_matrix_G[i][j] = int(
                            (iterative_image_transform_matrix_G_decrease[i][j] * 0.224 + 0.456) * 255)
                        change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], actual_image_channel_G[i][j] - degree,
                                                        actual_image_channel_G[i][j] + degree)
                        change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][1][i][j] = 1


                if iterative_image_pixel_classification_contribution_matrix_in_attack_label_B[i][j] < 0 and iterative_image_pixel_classification_contribution_matrix_in_actual_label_B[i][j] > 0:
                    if iterative_image_transform_matrix_B[i][j] < 0:
                        change_matrix_B[i][j] = math.ceil(
                            (iterative_image_transform_matrix_B_increase[i][j] * 0.225 + 0.406) * 255)
                        change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                        actual_image_channel_B[i][j] + degree)
                        change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][2][i][j] = 1
                    if iterative_image_transform_matrix_B[i][j] > 0:
                        change_matrix_B[i][j] = int(
                            (iterative_image_transform_matrix_B_decrease[i][j] * 0.225 + 0.406) * 255)
                        change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], actual_image_channel_B[i][j] - degree,
                                                        actual_image_channel_B[i][j] + degree)
                        change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][2][i][j] = 1

        # Generate images that have been tampered with in this iteration
        image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B], axis=-1)
        image_rgb = image_rgb.astype(np.uint8)
        image_pil = Image.fromarray(image_rgb)
        image_pil.save("No target attack image.png")
##########################################################################################################
        # Perform forward calibration
        forward_image_optimize_path = "No target attack image.png"
        image = Image.open(forward_image_optimize_path)
        image = image.resize((224, 224))
        forward_image_optimize = np.array(image)

        # The R, G, B three channel matrix of the positive optimized image
        forward_image_optimize_channel_R = forward_image_optimize[:, :, 0]
        forward_image_optimize_channel_R = forward_image_optimize_channel_R.astype(np.float64)
        forward_image_optimize_channel_G = forward_image_optimize[:, :, 1]
        forward_image_optimize_channel_G = forward_image_optimize_channel_G.astype(np.float64)
        forward_image_optimize_channel_B = forward_image_optimize[:, :, 2]
        forward_image_optimize_channel_B = forward_image_optimize_channel_B.astype(np.float64)

        # Calculate the standardized matrix of Xt+1
        forward_image_optimize_transform_matrix_R = ((forward_image_optimize_channel_R / 255) - 0.485) / 0.229
        forward_image_optimize_transform_matrix_G = ((forward_image_optimize_channel_G / 255) - 0.456) / 0.224
        forward_image_optimize_transform_matrix_B = ((forward_image_optimize_channel_B / 255) - 0.406) / 0.225

        # Calculate the pixel weight matrix of Xt+1 in the original image category
        forward_image_optimize_pixel_weight_matrix_in_actual_label_R, forward_image_optimize_pixel_weight_matrix_in_actual_label_G, forward_image_optimize_pixel_weight_matrix_in_actual_label_B = pixel_weight_matrix_image_path(forward_image_optimize_path, weight_file_absolute_path, actual_image_index, model_current)
        # Calculate the contribution matrix of Xt+1 in the original image category
        forward_image_optimize_contribution_matrix_in_actual_label_R = forward_image_optimize_transform_matrix_R * forward_image_optimize_pixel_weight_matrix_in_actual_label_R
        forward_image_optimize_contribution_matrix_in_actual_label_G = forward_image_optimize_transform_matrix_G * forward_image_optimize_pixel_weight_matrix_in_actual_label_G
        forward_image_optimize_contribution_matrix_in_actual_label_B = forward_image_optimize_transform_matrix_B * forward_image_optimize_pixel_weight_matrix_in_actual_label_B

        # Calculate the pixel weight matrix of Xt+1 in the target category
        forward_image_optimize_pixel_weight_matrix_in_object_label_R, forward_image_optimize_pixel_weight_matrix_in_object_label_G, forward_image_optimize_pixel_weight_matrix_in_object_label_B = pixel_weight_matrix_image_path(forward_image_optimize_path, weight_file_absolute_path, attack_index, model_current)
        # Calculate the contribution matrix of Xt+1 in the target category
        forward_image_optimize_contribution_matrix_in_object_label_R = forward_image_optimize_transform_matrix_R * forward_image_optimize_pixel_weight_matrix_in_object_label_R
        forward_image_optimize_contribution_matrix_in_object_label_G = forward_image_optimize_transform_matrix_G * forward_image_optimize_pixel_weight_matrix_in_object_label_G
        forward_image_optimize_contribution_matrix_in_object_label_B = forward_image_optimize_transform_matrix_B * forward_image_optimize_pixel_weight_matrix_in_object_label_B

        # Calculate the contribution value matrix of the top 1 categories in the iterative image when X is the original image
        actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_actual_label_R = actual_image_transform_matrix_R * forward_image_optimize_pixel_weight_matrix_in_actual_label_R
        actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_actual_label_G = actual_image_transform_matrix_G * forward_image_optimize_pixel_weight_matrix_in_actual_label_G
        actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_actual_label_B = actual_image_transform_matrix_B * forward_image_optimize_pixel_weight_matrix_in_actual_label_B

        # Calculate the contribution matrix on the target category when X is the original image
        actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_R = actual_image_transform_matrix_R * forward_image_optimize_pixel_weight_matrix_in_object_label_R
        actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_G = actual_image_transform_matrix_G * forward_image_optimize_pixel_weight_matrix_in_object_label_G
        actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_B = actual_image_transform_matrix_B * forward_image_optimize_pixel_weight_matrix_in_object_label_B

        # Calculate the growth rate of the contribution value in the Top1 category of the iterative image after hypothesis correction
        if_optimize_growth_contribution_in_actual_label_R = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_actual_label_R - forward_image_optimize_contribution_matrix_in_actual_label_R
        if_optimize_growth_contribution_in_actual_label_G = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_actual_label_G - forward_image_optimize_contribution_matrix_in_actual_label_G
        if_optimize_growth_contribution_in_actual_label_B = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_actual_label_B - forward_image_optimize_contribution_matrix_in_actual_label_B

        # Calculate the growth rate of contribution value in the target category after hypothesis correction
        if_optimize_growth_contribution_in_object_label_R = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_R - forward_image_optimize_contribution_matrix_in_object_label_R
        if_optimize_growth_contribution_in_object_label_G = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_G - forward_image_optimize_contribution_matrix_in_object_label_G
        if_optimize_growth_contribution_in_object_label_B = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_B - forward_image_optimize_contribution_matrix_in_object_label_B

        for i in range(224):
            for j in range(224):
                if if_optimize_growth_contribution_in_object_label_R[i][j] > 0 and if_optimize_growth_contribution_in_object_label_R[i][j] > if_optimize_growth_contribution_in_actual_label_R[i][j]:
                    forward_image_optimize_channel_R[i][j] = actual_image_channel_R[i][j]
                    iterative_image_pixel_dot_ischange[num_iterative][0][i][j] = 0
                if if_optimize_growth_contribution_in_object_label_G[i][j] > 0 and if_optimize_growth_contribution_in_object_label_G[i][j] > if_optimize_growth_contribution_in_actual_label_G[i][j]:
                    forward_image_optimize_channel_G[i][j] = actual_image_channel_G[i][j]
                    iterative_image_pixel_dot_ischange[num_iterative][1][i][j] = 0
                if if_optimize_growth_contribution_in_object_label_B[i][j] > 0 and if_optimize_growth_contribution_in_object_label_B[i][j] > if_optimize_growth_contribution_in_actual_label_B[i][j]:
                    forward_image_optimize_channel_B[i][j] = actual_image_channel_B[i][j]
                    iterative_image_pixel_dot_ischange[num_iterative][2][i][j] = 0

        # Positive calibration is completed, record the calibrated image, and use it for rollback optimization
        iterative_image_matrix_forward[num_iterative][0] = forward_image_optimize_channel_R.copy()
        iterative_image_matrix_forward[num_iterative][1] = forward_image_optimize_channel_G.copy()
        iterative_image_matrix_forward[num_iterative][2] = forward_image_optimize_channel_B.copy()


        # Generate images that have been tampered with in this iteration
        image_rgb = np.stack([forward_image_optimize_channel_R, forward_image_optimize_channel_G, forward_image_optimize_channel_B], axis=-1)
        image_rgb = image_rgb.astype(np.uint8)
        image_pil = Image.fromarray(image_rgb)
        image_pil.save("No target attack image.png")
        iterative_image_path = "No target attack image.png"

        # Input the positively monotonically tampered and positively optimized images of this round into the model for category judgment
        iterative_image_index, x = predict_image_path(iterative_image_path, index_file_absolute_path, weight_file_absolute_path, 0, model_current)

        # Determine whether the untargeted attack was successful
        if iterative_image_index != actual_image_index:
            success_num = success_num + 1
            print("No target attack successful!")
            current_image_path = "No target attack image.png"
            new_image_path = os.path.join(output_directory, image_file)
            shutil.copy(current_image_path, new_image_path)
            flag = 1
            print("Iteration tampering frequency：")
            print(num_iterative)
        if iterative_image_index == actual_image_index:
            print("This round of no target attack failed！")
            iterative_image_path = "No target attack image.png"
            flag = 0
############################################################################################################################
    # Generate adversarial samples for rollback optimization
    flag = 0
    while flag == 0 and num_iterative > 0:
        print(f"Rollback optimization rounds：{num_iterative}")
        adversarial_sample_pth = "No target attack image.png"
        last_image_path = f"last_image.png"
        shutil.copy(adversarial_sample_pth, last_image_path)

        # First, synthesize a reference image based on the recorded forward iterative images
        image_rgb = np.stack([iterative_image_matrix_forward[num_iterative - 1][0], iterative_image_matrix_forward[num_iterative - 1][1], iterative_image_matrix_forward[num_iterative - 1][2]], axis=-1)
        image_rgb = image_rgb.astype(np.uint8)
        image_pil = Image.fromarray(image_rgb)
        image_pil.save("forward_image.png")
        forward_image_path = "forward_image.png"

        image = Image.open(forward_image_path)
        # Resize the image to the specified size
        image = image.resize((224, 224))
        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Based on the RGB display mode, extract the data of each channel
        forward_image_R_channel = image_array[:, :, 0]
        forward_image_R_channel = forward_image_R_channel.astype(np.float64)
        forward_image_G_channel = image_array[:, :, 1]
        forward_image_G_channel = forward_image_G_channel.astype(np.float64)
        forward_image_B_channel = image_array[:, :, 2]
        forward_image_B_channel = forward_image_B_channel.astype(np.float64)

        # Calculate the pixel weight matrix of the original image on the top 1 label of the forward iteration
        actual_image_forward_pixel_weight_matrix_in_actual_label_R, actual_image_forward_pixel_weight_matrix_in_actual_label_G, actual_image_forward_pixel_weight_matrix_in_actual_label_B = pixel_weight_matrix_image_path(
            actual_image_absolute_path, weight_file_absolute_path, actual_image_index, model_current)

        # Calculate the pixel weight matrix of the original image on the target label
        actual_image_forward_pixel_weight_matrix_in_object_label_R, actual_image_forward_pixel_weight_matrix_in_object_label_G, actual_image_forward_pixel_weight_matrix_in_object_label_B = pixel_weight_matrix_image_path(
            actual_image_absolute_path, weight_file_absolute_path, iterative_image_label[num_iterative], model_current
        )

        # Calculate the pixel weight matrix of adversarial samples on the Top1 label in forward iteration
        adversarial_sample_pixel_weight_matrix_in_actual_label_R, adversarial_sample_pixel_weight_matrix_in_actual_label_G, adversarial_sample_pixel_weight_matrix_in_actual_label_B = pixel_weight_matrix_image_path(
            adversarial_sample_pth, weight_file_absolute_path, actual_image_index, model_current)

        # Calculate the pixel weight matrix of adversarial samples on the target label
        adversarial_sample_pixel_weight_matrix_in_object_label_R, adversarial_sample_pixel_weight_matrix_in_object_label_G, adversarial_sample_pixel_weight_matrix_in_object_label_B = pixel_weight_matrix_image_path(
            adversarial_sample_pth, weight_file_absolute_path, iterative_image_label[num_iterative], model_current
        )


        image = Image.open(adversarial_sample_pth)
        # Resize the image to the specified size
        image = image.resize((224, 224))
        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Based on the RGB display mode, extract the data of each channel
        adversarial_sample_R_channel = image_array[:, :, 0]
        adversarial_sample_R_channel = adversarial_sample_R_channel.astype(np.float64)
        adversarial_sample_G_channel = image_array[:, :, 1]
        adversarial_sample_G_channel = adversarial_sample_G_channel.astype(np.float64)
        adversarial_sample_B_channel = image_array[:, :, 2]
        adversarial_sample_B_channel = adversarial_sample_B_channel.astype(np.float64)
        adversarial_sample_transform_matrix_R = ((adversarial_sample_R_channel / 255) - 0.485) / 0.229
        adversarial_sample_transform_matrix_G = ((adversarial_sample_G_channel / 255) - 0.456) / 0.224
        adversarial_sample_transform_matrix_B = ((adversarial_sample_B_channel / 255) - 0.406) / 0.225


        image = Image.open(actual_image_absolute_path)
        # Resize the image to the specified size
        image = image.resize((224, 224))
        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Based on the RGB display mode, extract the data of each channel
        actual_image_R_channel = image_array[:, :, 0]
        actual_image_R_channel = actual_image_R_channel.astype(np.float64)
        actual_image_G_channel = image_array[:, :, 1]
        actual_image_G_channel = actual_image_G_channel.astype(np.float64)
        actual_image_B_channel = image_array[:, :, 2]
        actual_image_B_channel = actual_image_B_channel.astype(np.float64)

        actual_image_transform_matrix_R = ((actual_image_R_channel / 255) - 0.485) / 0.229
        actual_image_transform_matrix_G = ((actual_image_G_channel / 255) - 0.456) / 0.224
        actual_image_transform_matrix_B = ((actual_image_B_channel / 255) - 0.406) / 0.225

        # Calculate the contribution matrix of the original image in the actual category of forward iteration
        actual_image_forward_pixel_classification_contribution_matrix_in_actual_label_R = actual_image_transform_matrix_R * actual_image_forward_pixel_weight_matrix_in_actual_label_R
        actual_image_forward_pixel_classification_contribution_matrix_in_actual_label_G = actual_image_transform_matrix_G * actual_image_forward_pixel_weight_matrix_in_actual_label_G
        actual_image_forward_pixel_classification_contribution_matrix_in_actual_label_B = actual_image_transform_matrix_B * actual_image_forward_pixel_weight_matrix_in_actual_label_B
        # Calculate the contribution matrix of the original image in the target category
        actual_image_forward_pixel_classification_contribution_matrix_in_object_label_R = actual_image_transform_matrix_R * actual_image_forward_pixel_weight_matrix_in_object_label_R
        actual_image_forward_pixel_classification_contribution_matrix_in_object_label_G = actual_image_transform_matrix_G * actual_image_forward_pixel_weight_matrix_in_object_label_G
        actual_image_forward_pixel_classification_contribution_matrix_in_object_label_B = actual_image_transform_matrix_B * actual_image_forward_pixel_weight_matrix_in_object_label_B

        # Calculate the contribution matrix of adversarial samples in the actual categories of forward iteration
        adversarial_sample_pixel_classification_contribution_matrix_in_actual_label_R = adversarial_sample_transform_matrix_R * adversarial_sample_pixel_weight_matrix_in_actual_label_R
        adversarial_sample_pixel_classification_contribution_matrix_in_actual_label_G = adversarial_sample_transform_matrix_G * adversarial_sample_pixel_weight_matrix_in_actual_label_G
        adversarial_sample_pixel_classification_contribution_matrix_in_actual_label_B = adversarial_sample_transform_matrix_B * adversarial_sample_pixel_weight_matrix_in_actual_label_B
        # Calculate the contribution matrix of adversarial samples in the target category
        adversarial_sample_pixel_classification_contribution_matrix_in_object_label_R = adversarial_sample_transform_matrix_R * adversarial_sample_pixel_weight_matrix_in_object_label_R
        adversarial_sample_pixel_classification_contribution_matrix_in_object_label_G = adversarial_sample_transform_matrix_G * adversarial_sample_pixel_weight_matrix_in_object_label_G
        adversarial_sample_pixel_classification_contribution_matrix_in_object_label_B = adversarial_sample_transform_matrix_B * adversarial_sample_pixel_weight_matrix_in_object_label_B

        # Calculate the difference in contribution values between the original image and adversarial samples on the original category labels
        contribution_value_growth_in_actual_label_R = adversarial_sample_pixel_classification_contribution_matrix_in_actual_label_R - actual_image_forward_pixel_classification_contribution_matrix_in_actual_label_R
        contribution_value_growth_in_actual_label_G = adversarial_sample_pixel_classification_contribution_matrix_in_actual_label_G - actual_image_forward_pixel_classification_contribution_matrix_in_actual_label_G
        contribution_value_growth_in_actual_label_B = adversarial_sample_pixel_classification_contribution_matrix_in_actual_label_B - actual_image_forward_pixel_classification_contribution_matrix_in_actual_label_B
        # Calculate the difference in contribution values between the original image and adversarial samples on the target category label
        contribution_value_growth_in_object_label_R = adversarial_sample_pixel_classification_contribution_matrix_in_object_label_R - actual_image_forward_pixel_classification_contribution_matrix_in_object_label_R
        contribution_value_growth_in_object_label_G = adversarial_sample_pixel_classification_contribution_matrix_in_object_label_G - actual_image_forward_pixel_classification_contribution_matrix_in_object_label_G
        contribution_value_growth_in_object_label_B = adversarial_sample_pixel_classification_contribution_matrix_in_object_label_B - actual_image_forward_pixel_classification_contribution_matrix_in_object_label_B


        for i in range(224):
            for j in range(224):
                if not(contribution_value_growth_in_object_label_R[i][j] > contribution_value_growth_in_actual_label_R[i][j]) and iterative_image_pixel_dot_ischange[num_iterative][0][i][j] == 1:
                    adversarial_sample_R_channel[i][j] = forward_image_R_channel[i][j]
                if not(contribution_value_growth_in_object_label_G[i][j] > contribution_value_growth_in_actual_label_G[i][j]) and iterative_image_pixel_dot_ischange[num_iterative][1][i][j] == 1:
                    adversarial_sample_G_channel[i][j] = forward_image_G_channel[i][j]
                if not(contribution_value_growth_in_object_label_B[i][j] > contribution_value_growth_in_actual_label_B[i][j]) and iterative_image_pixel_dot_ischange[num_iterative][2][i][j] == 1:
                    adversarial_sample_B_channel[i][j] = forward_image_B_channel[i][j]


        gap_R = adversarial_sample_R_channel - actual_image_channel_R
        gap_R = np.clip(gap_R, -degree, degree)
        attack_image_matrix_R = actual_image_channel_R + gap_R
        attack_image_matrix_R = np.clip(attack_image_matrix_R, 0, 255)

        gap_G = adversarial_sample_G_channel - actual_image_channel_G
        gap_G = np.clip(gap_G, -degree, degree)
        attack_image_matrix_G = actual_image_channel_G + gap_G
        attack_image_matrix_G = np.clip(attack_image_matrix_G, 0, 255)

        gap_B = adversarial_sample_B_channel - actual_image_channel_B
        gap_B = np.clip(gap_B, -degree, degree)
        attack_image_matrix_B = actual_image_channel_B + gap_B
        attack_image_matrix_B = np.clip(attack_image_matrix_B, 0, 255)

        # Combine three channels into an RGB image
        image_rgb = np.stack([attack_image_matrix_R, attack_image_matrix_G, attack_image_matrix_B], axis=-1)
        # Convert data type to 8-bit unsigned integer
        image_rgb = image_rgb.astype(np.uint8)
        # Create PIL image object
        image_pil = Image.fromarray(image_rgb)
        image_pil.save("No target attack image.png")

        # Calculate and query whether the current attack image has been successfully attacked
        attack_image_path = "No target attack image.png"
        attack_image_index, x = predict_image_path(attack_image_path, index_file_absolute_path, weight_file_absolute_path, actual_image_index, model_current)
        if attack_image_index != actual_image_index:
            print("This round of rollback optimization has been successful!")
            iterative_image_path = f"{output_directory}\\{image_num}.png"
            shutil.copy(attack_image_path, iterative_image_path)
            num_iterative = num_iterative - 1
        else:
            flag = 1
            print("This round of rollback optimization failed!")
            iterative_image_path = f"{output_directory}\\{image_num}.png"
            shutil.copy(last_image_path, iterative_image_path)
    print(f"The success rate of the attack is：{success_num / image_num}")

ssim_results = calculate_ssim_for_folders(actual_images_folder_absolute_path, output_directory)
