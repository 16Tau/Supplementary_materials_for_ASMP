"""
The primary function of the current code file is to perform targeted attacks on a specified convolutional neural network model using ASMP based on the L0 norm.
First, set `actual_images_folder_absolute_path` to the absolute path of the folder containing the original images.
Second, set `output_directory` to the absolute path of the folder where the adversarial examples will be saved.
Third, set `index_file_absolute_path` to the absolute path of the index file.
Fourth, set `weight_file_absolute_path` to the absolute path of the weight file.
Fifth, adjust the value of `tampering_ratio` to control the allowed ratio of tampered pixels.
Sixth, use `"from torchvision.models import googlenet"` to import the official convolutional neural network model structure, and adjust `model_current` to specify the current convolutional neural network model.
Finally, execute the code file to generate the targeted adversarial examples.
"""



import json
import shutil
import matplotlib.pyplot as plt
from torchvision.models import vit_b_16
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import math


def read_matrix(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        matrix = [list(map(float, line.split())) for line in lines]
    return np.array(matrix)


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))


# Calculate targeted attack categories
def target_index_image_path(image_path, index_path, weight_path, model_cnn):
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
    attack_index = top_indices[499]
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
index_file_absolute_path = r".."
assert os.path.exists(index_file_absolute_path), "file: '{}' dose not exist.".format(index_file_absolute_path)
with open(index_file_absolute_path, "r") as f:
    class_indict = json.load(f)

# The absolute path of weight file for convolutional neural network model
weight_file_absolute_path = r"..."

# Iterative step size
iteration_step_size = 0.001

# The total number of images in the folder that require tampering attacks
the_total_number_of_tampered_images = 0

# The number of images that have been successfully tampered with and attacked
num_images_success = 0

image_files = sorted([f for f in os.listdir(actual_images_folder_absolute_path) if f.endswith('.png')], key=sort_func)

# Record the total number of original images
image_num = 0

# Record the total number of successfully attacked images
success_num = 0

# The allowed ratio of tampered pixels
tampering_ratio = 0.2

# Maximum number of tampered elements
maximum_number_of_tampered_pixel = 3 * 224 * 224 * tampering_ratio

# The currently used convolutional neural network model
model_current = vit_b_16

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

    # Calculate the attack target categories for targeted attacks
    attack_index = target_index_image_path(actual_image_absolute_path, index_file_absolute_path, weight_file_absolute_path, model_current)
    print(f"The target attack category is：{attack_index}")
    ####################################################################################################
    # Mark whether the target attack was successful
    flag = 0

    # The first iteration image is the original image
    iterative_image_path = actual_image_absolute_path

    # Number of iterations
    num_iterative = 0

    # Define a four-dimensional array to record the matrix of attack samples during each forward iteration tampering process
    iterative_image_matrix_forward = np.zeros((141, 3, 224, 224), dtype=np.float64)

    # Define a four-dimensional array to record the positions of tampered pixels during each round of forward iteration tampering process
    iterative_image_pixel_dot_ischange = np.zeros((141, 3, 224, 224), dtype=np.float64)

    # Create a one-dimensional array to record the target category labels selected during each round of forward iteration tampering process
    iterative_image_top1_label = np.zeros(141, dtype=int)

    # Record original image X0
    iterative_image_matrix_forward[num_iterative][0] = actual_image_channel_R.copy()
    iterative_image_matrix_forward[num_iterative][1] = actual_image_channel_G.copy()
    iterative_image_matrix_forward[num_iterative][2] = actual_image_channel_B.copy()

    # Perform forward iteration tampering
    while flag == 0 and num_iterative < 140:

        num_iterative = num_iterative + 1
        print(f"Current iteration count：{num_iterative}")

        # Calculate the Top-1 category of the iterative image in this round
        iterative_image_top1_index, x = predict_image_path(iterative_image_path, index_file_absolute_path, weight_file_absolute_path, 0, model_current)
        print(f"The Top-1 category of the current iterative image：{iterative_image_top1_index}")

        # Record the Top-1 category labels of the iterative images input in this round
        iterative_image_top1_label[num_iterative] = iterative_image_top1_index

        # Calculate the pixel weight matrix of the current iterative image in the Top1 category
        iterative_image_pixel_weight_matrix_in_iterative_top1_label_R, iterative_image_pixel_weight_matrix_in_iterative_top1_label_G, iterative_image_pixel_weight_matrix_in_iterative_top1_label_B = pixel_weight_matrix_image_path(
            iterative_image_path, weight_file_absolute_path, iterative_image_top1_index, model_current
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

        # Calculate the contribution matrix of iterative images to the top 1 labels in the iteration
        iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_R = iterative_image_transform_matrix_R * iterative_image_pixel_weight_matrix_in_iterative_top1_label_R
        iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_G = iterative_image_transform_matrix_G * iterative_image_pixel_weight_matrix_in_iterative_top1_label_G
        iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_B = iterative_image_transform_matrix_B * iterative_image_pixel_weight_matrix_in_iterative_top1_label_B

        # Calculate the contribution matrix of iterative images in the target category
        iterative_image_pixel_classification_contribution_matrix_in_attack_label_R = iterative_image_transform_matrix_R * iterative_image_pixel_weight_matrix_in_attack_label_R
        iterative_image_pixel_classification_contribution_matrix_in_attack_label_G = iterative_image_transform_matrix_G * iterative_image_pixel_weight_matrix_in_attack_label_G
        iterative_image_pixel_classification_contribution_matrix_in_attack_label_B = iterative_image_transform_matrix_B * iterative_image_pixel_weight_matrix_in_attack_label_B

        # Define a marking matrix to mark the positions of elements that are allowed to be tampered with
        mark_matrix = np.zeros((3, 224, 224), dtype=np.float64)

        # In the first iteration, sort the feature monotonic pixels based on their absolute contribution values
        if num_iterative == 1:
            feature_monotonicity_rate = np.zeros((3, 224, 224), dtype=np.float64)

            for i in range(224):
                for j in range(224):
                    if iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_R[i][j] * iterative_image_pixel_classification_contribution_matrix_in_attack_label_R[i][j] < 0:
                        feature_monotonicity_rate[0][i][j] = np.abs(iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_R[i][j]) + np.abs(iterative_image_pixel_classification_contribution_matrix_in_attack_label_R[i][j])
                    if iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_G[i][j] * iterative_image_pixel_classification_contribution_matrix_in_attack_label_G[i][j] < 0:
                        feature_monotonicity_rate[1][i][j] = np.abs(iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_G[i][j]) + np.abs(iterative_image_pixel_classification_contribution_matrix_in_attack_label_G[i][j])
                    if iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_B[i][j] * iterative_image_pixel_classification_contribution_matrix_in_attack_label_B[i][j] < 0:
                        feature_monotonicity_rate[2][i][j] = np.abs(iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_B[i][j]) + np.abs(iterative_image_pixel_classification_contribution_matrix_in_attack_label_B[i][j])

            shape = feature_monotonicity_rate.shape
            flat_array = feature_monotonicity_rate.flatten()
            sorted_indices = np.argsort(flat_array)
            sequence_feature_monotonicity_rate = np.zeros_like(flat_array)
            sequence_feature_monotonicity_rate[sorted_indices[::-1]] = np.arange(1, len(flat_array) + 1)
            sequence_feature_monotonicity_rate = sequence_feature_monotonicity_rate.reshape(shape)

            for i in range(224):
                for j in range(224):
                    if sequence_feature_monotonicity_rate[0][i][j] <= maximum_number_of_tampered_pixel:
                        mark_matrix[0][i][j] = 1
                    if sequence_feature_monotonicity_rate[1][i][j] <= maximum_number_of_tampered_pixel:
                        mark_matrix[1][i][j] = 1
                    if sequence_feature_monotonicity_rate[2][i][j] <= maximum_number_of_tampered_pixel:
                        mark_matrix[2][i][j] = 1

        if num_iterative != 1:
            feature_monotonicity_rate = np.zeros((3, 224, 224), dtype=np.float64)
            for i in range(224):
                for j in range(224):
                    if iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_R[i][j] * iterative_image_pixel_classification_contribution_matrix_in_attack_label_R[i][j] < 0:
                        feature_monotonicity_rate[0][i][j] = np.abs(iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_R[i][j]) + np.abs(iterative_image_pixel_classification_contribution_matrix_in_attack_label_R[i][j])
                    if iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_G[i][j] * iterative_image_pixel_classification_contribution_matrix_in_attack_label_G[i][j] < 0:
                        feature_monotonicity_rate[1][i][j] = np.abs(iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_G[i][j]) + np.abs(iterative_image_pixel_classification_contribution_matrix_in_attack_label_G[i][j])
                    if iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_B[i][j] * iterative_image_pixel_classification_contribution_matrix_in_attack_label_B[i][j] < 0:
                        feature_monotonicity_rate[2][i][j] = np.abs(iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_B[i][j]) + np.abs(iterative_image_pixel_classification_contribution_matrix_in_attack_label_B[i][j])

            shape = feature_monotonicity_rate.shape
            flat_array = feature_monotonicity_rate.flatten()
            sorted_indices = np.argsort(flat_array)
            sequence_feature_monotonicity_rate = np.zeros_like(flat_array)
            sequence_feature_monotonicity_rate[sorted_indices[::-1]] = np.arange(1, len(flat_array) + 1)
            sequence_feature_monotonicity_rate = sequence_feature_monotonicity_rate.reshape(shape)

            num = 0
            for i in range(224):
                for j in range(224):
                    if iterative_image_channel_R[i][j] != actual_image_channel_R[i][j]:
                        mark_matrix[0][i][j] = 1
                        num = num + 1
                    if iterative_image_channel_G[i][j] != actual_image_channel_G[i][j]:
                        mark_matrix[1][i][j] = 1
                        num = num + 1
                    if iterative_image_channel_B[i][j] != actual_image_channel_B[i][j]:
                        mark_matrix[2][i][j] = 1
                        num = num + 1
            sequence = 1
            while num < maximum_number_of_tampered_pixel:
                coords = np.where(sequence_feature_monotonicity_rate == sequence)
                sequence = sequence + 1
                if mark_matrix[coords[0][0]][coords[1][0]][coords[2][0]] != 1:
                    mark_matrix[coords[0][0]][coords[1][0]][coords[2][0]] = 1
                    num = num + 1
        print(f"The number of modifiable elements marked in the mark_matrix matrix：{np.sum(mark_matrix)}")
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
                if iterative_image_pixel_classification_contribution_matrix_in_attack_label_R[i][j] > 0 and iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_R[i][j] < 0 and mark_matrix[0][i][j] == 1:

                    if iterative_image_transform_matrix_R[i][j] > 0:
                        change_matrix_R[i][j] = math.ceil(
                            (iterative_image_transform_matrix_R_increase[i][j] * 0.229 + 0.485) * 255)
                        change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][0][i][j] = 1
                    if iterative_image_transform_matrix_R[i][j] < 0:
                        change_matrix_R[i][j] = int(
                            (iterative_image_transform_matrix_R_decrease[i][j] * 0.229 + 0.485) * 255)
                        change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][0][i][j] = 1

                if iterative_image_pixel_classification_contribution_matrix_in_attack_label_G[i][j] > 0 and iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_G[i][j] < 0 and mark_matrix[1][i][j] == 1:
                    if iterative_image_transform_matrix_G[i][j] > 0:
                        change_matrix_G[i][j] = math.ceil(
                            (iterative_image_transform_matrix_G_increase[i][j] * 0.224 + 0.456) * 255)
                        change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][1][i][j] = 1
                    if iterative_image_transform_matrix_G[i][j] < 0:
                        change_matrix_G[i][j] = int(
                            (iterative_image_transform_matrix_G_decrease[i][j] * 0.224 + 0.456) * 255)
                        change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][1][i][j] = 1

                if iterative_image_pixel_classification_contribution_matrix_in_attack_label_B[i][j] > 0 and iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_B[i][j] < 0 and mark_matrix[2][i][j] == 1:
                    if iterative_image_transform_matrix_B[i][j] > 0:
                        change_matrix_B[i][j] = math.ceil(
                            (iterative_image_transform_matrix_B_increase[i][j] * 0.225 + 0.406) * 255)
                        change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][2][i][j] = 1
                    if iterative_image_transform_matrix_B[i][j] < 0:
                        change_matrix_B[i][j] = int(
                            (iterative_image_transform_matrix_B_decrease[i][j] * 0.225 + 0.406) * 255)
                        change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][2][i][j] = 1
                ############################################################################################################################################################################################
                if iterative_image_pixel_classification_contribution_matrix_in_attack_label_R[i][j] < 0 and iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_R[i][j] > 0 and mark_matrix[0][i][j] == 1:
                    if iterative_image_transform_matrix_R[i][j] < 0:
                        change_matrix_R[i][j] = math.ceil(
                            (iterative_image_transform_matrix_R_increase[i][j] * 0.229 + 0.485) * 255)
                        change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][0][i][j] = 1
                    if iterative_image_transform_matrix_R[i][j] > 0:
                        change_matrix_R[i][j] = int(
                            (iterative_image_transform_matrix_R_decrease[i][j] * 0.229 + 0.485) * 255)
                        change_matrix_R[i][j] = np.clip(change_matrix_R[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][0][i][j] = 1


                if iterative_image_pixel_classification_contribution_matrix_in_attack_label_G[i][j] < 0 and iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_G[i][j] > 0 and mark_matrix[1][i][j] == 1:
                    if iterative_image_transform_matrix_G[i][j] < 0:
                        change_matrix_G[i][j] = math.ceil(
                            (iterative_image_transform_matrix_G_increase[i][j] * 0.224 + 0.456) * 255)
                        change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][1][i][j] = 1
                    if iterative_image_transform_matrix_G[i][j] > 0:
                        change_matrix_G[i][j] = int(
                            (iterative_image_transform_matrix_G_decrease[i][j] * 0.224 + 0.456) * 255)
                        change_matrix_G[i][j] = np.clip(change_matrix_G[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][1][i][j] = 1


                if iterative_image_pixel_classification_contribution_matrix_in_attack_label_B[i][j] < 0 and iterative_image_pixel_classification_contribution_matrix_in_iterative_top1_label_B[i][j] > 0 and mark_matrix[2][i][j] == 1:
                    if iterative_image_transform_matrix_B[i][j] < 0:
                        change_matrix_B[i][j] = math.ceil(
                            (iterative_image_transform_matrix_B_increase[i][j] * 0.225 + 0.406) * 255)
                        change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][2][i][j] = 1
                    if iterative_image_transform_matrix_B[i][j] > 0:
                        change_matrix_B[i][j] = int(
                            (iterative_image_transform_matrix_B_decrease[i][j] * 0.225 + 0.406) * 255)
                        change_matrix_B[i][j] = np.clip(change_matrix_B[i][j], 0, 255)
                        iterative_image_pixel_dot_ischange[num_iterative][2][i][j] = 1

        # Generate images that have been tampered with in this iteration
        image_rgb = np.stack([change_matrix_R, change_matrix_G, change_matrix_B], axis=-1)
        image_rgb = image_rgb.astype(np.uint8)
        image_pil = Image.fromarray(image_rgb)
        image_pil.save("target attack image.png")
##########################################################################################################
        # Perform forward calibration
        forward_image_optimize_path = "target attack image.png"
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

        # Calculate the pixel weight matrix of Xt+1 in the top 1 category of iterative images
        forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_R, forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_G, forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_B = pixel_weight_matrix_image_path(forward_image_optimize_path, weight_file_absolute_path, iterative_image_top1_index, model_current)
        # Calculate the contribution matrix of Xt+1 in the top 1 category of iterative images
        forward_image_optimize_contribution_matrix_in_iterative_top1_label_R = forward_image_optimize_transform_matrix_R * forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_R
        forward_image_optimize_contribution_matrix_in_iterative_top1_label_G = forward_image_optimize_transform_matrix_G * forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_G
        forward_image_optimize_contribution_matrix_in_iterative_top1_label_B = forward_image_optimize_transform_matrix_B * forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_B

        # Calculate the pixel weight matrix of Xt+1 in the target category
        forward_image_optimize_pixel_weight_matrix_in_object_label_R, forward_image_optimize_pixel_weight_matrix_in_object_label_G, forward_image_optimize_pixel_weight_matrix_in_object_label_B = pixel_weight_matrix_image_path(forward_image_optimize_path, weight_file_absolute_path, attack_index, model_current)
        # Calculate the contribution matrix of Xt+1 in the target category
        forward_image_optimize_contribution_matrix_in_object_label_R = forward_image_optimize_transform_matrix_R * forward_image_optimize_pixel_weight_matrix_in_object_label_R
        forward_image_optimize_contribution_matrix_in_object_label_G = forward_image_optimize_transform_matrix_G * forward_image_optimize_pixel_weight_matrix_in_object_label_G
        forward_image_optimize_contribution_matrix_in_object_label_B = forward_image_optimize_transform_matrix_B * forward_image_optimize_pixel_weight_matrix_in_object_label_B

        # Calculate the contribution value matrix of the top 1 categories in the iterative image when X is the original image
        actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_R = actual_image_transform_matrix_R * forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_R
        actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_G = actual_image_transform_matrix_G * forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_G
        actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_B = actual_image_transform_matrix_B * forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_B

        # Calculate the contribution matrix on the target category when X is the original image
        actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_R = actual_image_transform_matrix_R * forward_image_optimize_pixel_weight_matrix_in_object_label_R
        actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_G = actual_image_transform_matrix_G * forward_image_optimize_pixel_weight_matrix_in_object_label_G
        actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_B = actual_image_transform_matrix_B * forward_image_optimize_pixel_weight_matrix_in_object_label_B

        # Calculate the growth rate of the contribution value in the Top1 category of the iterative image after hypothesis correction
        if_optimize_growth_contribution_in_iterative_top1_label_R = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_R - forward_image_optimize_contribution_matrix_in_iterative_top1_label_R
        if_optimize_growth_contribution_in_iterative_top1_label_G = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_G - forward_image_optimize_contribution_matrix_in_iterative_top1_label_G
        if_optimize_growth_contribution_in_iterative_top1_label_B = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_iterative_top1_label_B - forward_image_optimize_contribution_matrix_in_iterative_top1_label_B

        # Calculate the growth rate of contribution value in the target category after hypothesis correction
        if_optimize_growth_contribution_in_object_label_R = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_R - forward_image_optimize_contribution_matrix_in_object_label_R
        if_optimize_growth_contribution_in_object_label_G = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_G - forward_image_optimize_contribution_matrix_in_object_label_G
        if_optimize_growth_contribution_in_object_label_B = actual_image_contribution_matrix_with_forward_image_optimize_pixel_weight_matrix_in_object_label_B - forward_image_optimize_contribution_matrix_in_object_label_B

        for i in range(224):
            for j in range(224):
                if if_optimize_growth_contribution_in_object_label_R[i][j] > 0 and if_optimize_growth_contribution_in_object_label_R[i][j] > if_optimize_growth_contribution_in_iterative_top1_label_R[i][j]:
                    forward_image_optimize_channel_R[i][j] = actual_image_channel_R[i][j]
                    iterative_image_pixel_dot_ischange[num_iterative][0][i][j] = 0
                if if_optimize_growth_contribution_in_object_label_G[i][j] > 0 and if_optimize_growth_contribution_in_object_label_G[i][j] > if_optimize_growth_contribution_in_iterative_top1_label_G[i][j]:
                    forward_image_optimize_channel_G[i][j] = actual_image_channel_G[i][j]
                    iterative_image_pixel_dot_ischange[num_iterative][1][i][j] = 0
                if if_optimize_growth_contribution_in_object_label_B[i][j] > 0 and if_optimize_growth_contribution_in_object_label_B[i][j] > if_optimize_growth_contribution_in_iterative_top1_label_B[i][j]:
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
        image_pil.save("target attack image.png")
        iterative_image_path = "target attack image.png"

        # Input the positively monotonically tampered and positively optimized images of this round into the model for category judgment
        iterative_image_index, x = predict_image_path(iterative_image_path, index_file_absolute_path, weight_file_absolute_path, attack_index, model_current)

        # Determine whether the targeted attack was successful
        if iterative_image_index == attack_index:
            success_num = success_num + 1
            print("target attack successful!")
            current_image_path = "target attack image.png"
            new_image_path = os.path.join(output_directory, image_file)
            shutil.copy(current_image_path, new_image_path)
            flag = 1
            print("Iteration tampering frequency：")
            print(num_iterative)
        if iterative_image_index != attack_index:
            print("This round of target attack failed！")
            iterative_image_path = "target attack image.png"
            flag = 0
############################################################################################################################
    if iterative_image_index != attack_index:
        current_image_path = "target attack image.png"
        new_image_path = os.path.join(output_directory, image_file)
        shutil.copy(current_image_path, new_image_path)

    # When adversarial samples are successfully generated in a positive direction, Generate adversarial samples for rollback optimization
    if iterative_image_index == attack_index:
        flag = 0
        while flag == 0 and num_iterative > 0:
            print(f"Rollback optimization rounds：{num_iterative}")
            adversarial_sample_pth = "target attack image.png"
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
            actual_image_forward_pixel_weight_matrix_in_forward_top1_label_R, actual_image_forward_pixel_weight_matrix_in_forward_top1_label_G, actual_image_forward_pixel_weight_matrix_in_forward_top1_label_B = pixel_weight_matrix_image_path(
                actual_image_absolute_path, weight_file_absolute_path, iterative_image_top1_label[num_iterative], model_current)

            # Calculate the pixel weight matrix of the original image on the target label
            actual_image_forward_pixel_weight_matrix_in_object_label_R, actual_image_forward_pixel_weight_matrix_in_object_label_G, actual_image_forward_pixel_weight_matrix_in_object_label_B = pixel_weight_matrix_image_path(
                actual_image_absolute_path, weight_file_absolute_path, attack_index, model_current
            )

            # Calculate the pixel weight matrix of adversarial samples on the Top1 label in forward iteration
            adversarial_sample_pixel_weight_matrix_in_forward_top1_R, adversarial_sample_pixel_weight_matrix_in_forward_top1_G, adversarial_sample_pixel_weight_matrix_in_forward_top1_B = pixel_weight_matrix_image_path(
                adversarial_sample_pth, weight_file_absolute_path, iterative_image_top1_label[num_iterative], model_current)

            # Calculate the pixel weight matrix of adversarial samples on the target label
            adversarial_sample_pixel_weight_matrix_in_object_label_R, adversarial_sample_pixel_weight_matrix_in_object_label_G, adversarial_sample_pixel_weight_matrix_in_object_label_B = pixel_weight_matrix_image_path(
                adversarial_sample_pth, weight_file_absolute_path, attack_index, model_current
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

            # Calculate the contribution matrix of the original image in the top 1 category of forward iteration
            actual_image_forward_pixel_classification_contribution_matrix_in_forward_top1_label_R = actual_image_transform_matrix_R * actual_image_forward_pixel_weight_matrix_in_forward_top1_label_R
            actual_image_forward_pixel_classification_contribution_matrix_in_forward_top1_label_G = actual_image_transform_matrix_G * actual_image_forward_pixel_weight_matrix_in_forward_top1_label_G
            actual_image_forward_pixel_classification_contribution_matrix_in_forward_top1_label_B = actual_image_transform_matrix_B * actual_image_forward_pixel_weight_matrix_in_forward_top1_label_B
            # Calculate the contribution matrix of the original image in the target category
            actual_image_forward_pixel_classification_contribution_matrix_in_object_label_R = actual_image_transform_matrix_R * actual_image_forward_pixel_weight_matrix_in_object_label_R
            actual_image_forward_pixel_classification_contribution_matrix_in_object_label_G = actual_image_transform_matrix_G * actual_image_forward_pixel_weight_matrix_in_object_label_G
            actual_image_forward_pixel_classification_contribution_matrix_in_object_label_B = actual_image_transform_matrix_B * actual_image_forward_pixel_weight_matrix_in_object_label_B

            # Calculate the contribution matrix of adversarial samples in the top 1 categories of forward iteration
            adversarial_sample_pixel_classification_contribution_matrix_in_forward_top1_label_R = adversarial_sample_transform_matrix_R * adversarial_sample_pixel_weight_matrix_in_forward_top1_R
            adversarial_sample_pixel_classification_contribution_matrix_in_forward_top1_label_G = adversarial_sample_transform_matrix_G * adversarial_sample_pixel_weight_matrix_in_forward_top1_G
            adversarial_sample_pixel_classification_contribution_matrix_in_forward_top1_label_B = adversarial_sample_transform_matrix_B * adversarial_sample_pixel_weight_matrix_in_forward_top1_B
            # Calculate the contribution matrix of adversarial samples in the target category
            adversarial_sample_pixel_classification_contribution_matrix_in_object_label_R = adversarial_sample_transform_matrix_R * adversarial_sample_pixel_weight_matrix_in_object_label_R
            adversarial_sample_pixel_classification_contribution_matrix_in_object_label_G = adversarial_sample_transform_matrix_G * adversarial_sample_pixel_weight_matrix_in_object_label_G
            adversarial_sample_pixel_classification_contribution_matrix_in_object_label_B = adversarial_sample_transform_matrix_B * adversarial_sample_pixel_weight_matrix_in_object_label_B

            # Calculate the difference in contribution values between the original image and adversarial samples on the original category labels
            contribution_value_growth_in_actual_label_R = adversarial_sample_pixel_classification_contribution_matrix_in_forward_top1_label_R - actual_image_forward_pixel_classification_contribution_matrix_in_forward_top1_label_R
            contribution_value_growth_in_actual_label_G = adversarial_sample_pixel_classification_contribution_matrix_in_forward_top1_label_G - actual_image_forward_pixel_classification_contribution_matrix_in_forward_top1_label_G
            contribution_value_growth_in_actual_label_B = adversarial_sample_pixel_classification_contribution_matrix_in_forward_top1_label_B - actual_image_forward_pixel_classification_contribution_matrix_in_forward_top1_label_B
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
            attack_image_matrix_R = actual_image_channel_R + gap_R
            attack_image_matrix_R = np.clip(attack_image_matrix_R, 0, 255)

            gap_G = adversarial_sample_G_channel - actual_image_channel_G
            attack_image_matrix_G = actual_image_channel_G + gap_G
            attack_image_matrix_G = np.clip(attack_image_matrix_G, 0, 255)

            gap_B = adversarial_sample_B_channel - actual_image_channel_B
            attack_image_matrix_B = actual_image_channel_B + gap_B
            attack_image_matrix_B = np.clip(attack_image_matrix_B, 0, 255)

            # Combine three channels into an RGB image
            image_rgb = np.stack([attack_image_matrix_R, attack_image_matrix_G, attack_image_matrix_B], axis=-1)
            # Convert data type to 8-bit unsigned integer
            image_rgb = image_rgb.astype(np.uint8)
            # Create PIL image object
            image_pil = Image.fromarray(image_rgb)
            image_pil.save("target attack image.png")

            # Calculate and query whether the current attack image has been successfully attacked
            attack_image_path = "target attack image.png"
            attack_image_index, x = predict_image_path(attack_image_path, index_file_absolute_path, weight_file_absolute_path, actual_image_index, model_current)
            if attack_image_index == attack_index:
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