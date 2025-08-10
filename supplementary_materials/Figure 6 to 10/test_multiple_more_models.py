import torch
from torch.utils.data import DataLoader
import os
import argparse
from torchvision import transforms, datasets, models
from AlexNet import AlexNet
import torch.nn as nn


def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on a test dataset and return accuracy.

    Args:
        model: The neural network model
        test_loader: DataLoader for the test dataset
        device: Device to run the model (CPU or GPU)

    Returns:
        float: Accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main(network_name, weights_path):
    """
    Test a single neural network on multiple test datasets.

    Args:
        network_name (str): Name of the network ('AlexNet', 'ResNet18', 'DenseNet121', 'VGG16', 'GoogLeNet')
        weights_path (str): Path to the weights file (.pth)
    """
    # Parameters for test datasets
    step_size = 0.0001
    ini_value = 0.99999
    step_value = 1 - step_size
    num_tests = 6

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Network configurations
    networks = {
        'AlexNet': {
            'model_class': AlexNet,
            'model_args': {'num_classes': 4},
        },
        'ResNet18': {
            'model_class': models.resnet18,
            'model_args': {'weights': None},
            'modify_fn': lambda m: setattr(m, 'fc', nn.Linear(m.fc.in_features, 4)),
        },
        'DenseNet121': {
            'model_class': models.densenet121,
            'model_args': {'weights': None},
            'modify_fn': lambda m: setattr(m, 'classifier', nn.Linear(m.classifier.in_features, 4)),
        },
        'VGG16': {
            'model_class': models.vgg16,
            'model_args': {'weights': None},
            'modify_fn': lambda m: m.classifier.__setitem__(6, nn.Linear(4096, 4)),
        },
        'GoogLeNet': {
            'model_class': models.googlenet,
            'model_args': {'weights': None, 'aux_logits': True},
            'modify_fn': lambda m: (
                setattr(m, 'fc', nn.Linear(1024, 4)),
                setattr(m.aux1, 'fc2', nn.Linear(m.aux1.fc2.in_features, 4)),
                setattr(m.aux2, 'fc2', nn.Linear(m.aux2.fc2.in_features, 4))
            ),
        }
    }

    # Validate network name
    if network_name not in networks:
        raise ValueError(f"Invalid network '{network_name}'. Choose from {list(networks.keys())}")

    print(f"\nTesting {network_name}...")

    # Initialize model
    network = networks[network_name]
    model = network['model_class'](**network['model_args'])
    if 'modify_fn' in network:
        network['modify_fn'](model)
    model = model.to(device)

    # Load weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file {weights_path} not found for {network_name}")
    model.load_state_dict(torch.load(weights_path))

    # Initialize Excel workbook 
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"{network_name} Test Accuracy"

    # Set headers
    headers = ['Original'] + [
        f'SSIM ({(step_value - (i - 1) * step_size):.5f}, {(ini_value - (i - 1) * step_size):.5f})' for i in
        range(1, num_tests)]
    for col, header in enumerate(headers, start=1):
        ws[f'{get_column_letter(col)}1'] = header

    # Test on each dataset
    accuracies = []
    ranges=[0.99975,0.99925,0.99875,0.99825,0.99775,0.99725]

    for l in range(num_tests):
        # Determine test dataset path
        if l == 0:
            print("Testing on original dataset")
            image_path = "../datasets/copper_foil_1/dataset/test"
        else:
            ssim_val = step_value - (l - 1) * step_size
            ini_val = ini_value - (l - 1) * step_size
            print(f"Testing on SSIM ({ranges[l]:.5f}, {ranges[l-1]:.5f})")
            image_path = f"../datasets/copper_foil_1/noise/SSIMed/({ranges[l-1]:.5f}, {ranges[l]:.5f})"


        # Verify dataset path
        if not os.path.exists(image_path):
            print(f"Warning: Dataset path {image_path} does not exist, skipping")
            accuracies.append('N/A')
            continue

        # Load test dataset
        test_dataset = datasets.ImageFolder(root=image_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # Evaluate model
        accuracy = evaluate_model(model, test_loader, device)
        print(f"Accuracy on {image_path}: {accuracy:.2f}%")
        accuracies.append(round(accuracy, 2))
    # Clean up
    del model
    torch.cuda.empty_cache()  # Clear GPU memory


if __name__ == '__main__':
    for i in range(41,43):
        main('GoogLeNet', f'GoogLeNet_models/{i}/GoogLeNet.pth')