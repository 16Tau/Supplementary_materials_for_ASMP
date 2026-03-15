import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import vit_b_32
from PIL import Image
import json
import numpy as np
import math
import os
import time


def sort_func(file_name):
    return int(''.join(filter(str.isdigit, file_name)))

class Attack:
    def __init__(self, attack, model_name, epsilon, targeted, random_start, norm, loss, device):
        self.model = model_name.to(device)
        self.device = device
        self.epsilon = (epsilon - 0.406) / 0.225
        self.targeted = targeted
        self.random_start = random_start
        self.norm = norm
        self.loss = loss

    def get_logits(self, x):
        return self.model(x)

    def get_grad(self, loss, x):
        grad = torch.autograd.grad(loss, x, retain_graph=False, create_graph=False)[0]
        return grad


class ANDA(Attack):
    def __init__(
        self,
        model_name,
        epsilon=(16 / 255 - 0.406) / 0.225,
        alpha=1.6 / 255,
        epoch=10,
        n_ens=25,
        aug_max=0.3,
        sample=False,
        targeted=False,
        random_start=False,
        norm='linfty',
        loss='crossentropy',
        device=None,
        attack='ANDA',
        **kwargs
    ):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = 0
        self.n_ens = n_ens
        self.aug_max = aug_max
        self.sample = sample

        def is_sqr(n):
            a = int(math.sqrt(n))
            return a * a == n

        assert is_sqr(self.n_ens), "n_ens must be square number."

        self.thetas = self.get_thetas(int(math.sqrt(self.n_ens)), -self.aug_max, self.aug_max)

    def get_theta(self, i, j):
        theta = torch.tensor([[[1, 0, i], [0, 1, j]]], dtype=torch.float)
        return theta

    def get_thetas(self, n, min_r=-0.5, max_r=0.5):
        range_r = torch.linspace(min_r, max_r, n)
        thetas = []
        for i in range_r:
            for j in range_r:
                thetas.append(self.get_theta(i, j))
        thetas = torch.cat(thetas, dim=0)
        return thetas

    def transform(self, thetas, data):
        grids = F.affine_grid(thetas, data.size(), align_corners=False).to(data.device)
        output = F.grid_sample(data, grids, align_corners=False)
        return output

    def get_loss(self, logits, labels):
        return F.cross_entropy(logits, labels, reduction="sum")

    def forward(self, data, label, **kwargs):
        assert data.shape[0] == 1, "ANDA only supports batchsize=1"

        data = data.clone().detach().to(self.device)
        xt = data.clone().detach()
        label = label.clone().detach().to(self.device)

        stat = ANDA_STATISTICS(data_shape=data.shape, device=self.device)

        for t in range(self.epoch):

            xt_batch = xt.repeat(self.n_ens, 1, 1, 1)
            xt_batch.requires_grad = True

            aug_xt_batch = self.transform(self.thetas, xt_batch)

            labels = label.repeat(xt_batch.shape[0])

            logits = self.get_logits(aug_xt_batch)

            loss = self.get_loss(logits, labels)

            grad = self.get_grad(loss, xt_batch)

            stat.collect_stat(grad)

            # mean update
            noise = stat.noise_mean
            xt = xt + self.alpha * noise.sign()

            delta = xt - data
            delta[0][0] = torch.clamp(delta[0][0], min=-0.13699, max=0.13699)
            delta[0][1] = torch.clamp(delta[0][1], min=-0.14005, max=0.14005)
            delta[0][2] = torch.clamp(delta[0][2], min=-0.13941, max=0.13941)

            xt = delta + data

            xt[0][0] = torch.clamp(xt[0][0], -2.1179, 2.2489).detach()
            xt[0][1] = torch.clamp(xt[0][1], -2.0357, 2.4285).detach()
            xt[0][2] = torch.clamp(xt[0][2], -1.8044, 2.64).detach()

        delta = xt - data
        return delta.detach()


class ANDA_STATISTICS:
    def __init__(self, device, data_shape=(1, 3, 224, 224)):
        self.data_shape = data_shape
        self.device = device
        self.n_models = 0
        self.noise_mean = torch.zeros(data_shape, dtype=torch.float).to(device)
        self.noise_cov_mat_sqrt = torch.empty((0, np.prod(data_shape)), dtype=torch.float).to(device)

    def collect_stat(self, noise):
        mean = self.noise_mean
        cov_mat_sqrt = self.noise_cov_mat_sqrt
        bs = noise.shape[0]

        mean = mean * self.n_models / (self.n_models + bs) + noise.data.sum(dim=0, keepdim=True) / (self.n_models + bs)

        dev = (noise.data - mean).view(bs, -1)
        cov_mat_sqrt = torch.cat((cov_mat_sqrt, dev), dim=0)

        self.noise_mean = mean
        self.noise_cov_mat_sqrt = cov_mat_sqrt
        self.n_models += bs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = vit_b_32(weights=None)

weight_path = r"..."
state_dict = torch.load(weight_path, map_location=device)
model.load_state_dict(state_dict)
model.eval().to(device)

with open(r"...", 'r') as f:
    idx2label = json.load(f)

index_file_absolute_path = r"..."

weight_file_absolute_path = r"..."

actual_images_folder_absolute_path = "..."
output_directory = "..."

# The total number of images in the folder that require tampering attacks
the_total_number_of_tampered_images = 0

image_files = sorted([f for f in os.listdir(actual_images_folder_absolute_path) if f.endswith('.png')],
                     key=sort_func)

# Record the total number of original images
image_num = 0

# Record the total number of successfully attacked images
success_num = 0

data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
     ])

image_start_time = time.time()
for image_file in image_files:
    print(f"The image number currently being processed is: {image_file}")
    image_num = image_num + 1

    the_total_number_of_tampered_images = the_total_number_of_tampered_images + 1

    actual_image_absolute_path = os.path.join(actual_images_folder_absolute_path, image_file)
    img = Image.open(actual_image_absolute_path).convert("RGB")

    image = Image.open(actual_image_absolute_path)
    image = image.resize((224, 224))
    actual_image = np.array(image)

    x = data_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1).item()


    attack = ANDA(
        model_name=model,
        epsilon=4/255,
        alpha=1/255,
        epoch=100,
        n_ens=25,
        aug_max=0.3,
        targeted=False,
        device=device
    )

    delta = attack.forward(x, torch.tensor([pred]))

    adv = x + delta
    adv[0][0] = torch.clamp(((adv[0][0] * 0.229 + 0.485) * 255), 0, 255)
    adv[0][1] = torch.clamp(((adv[0][1] * 0.224 + 0.456) * 255), 0, 255)
    adv[0][2] = torch.clamp(((adv[0][2] * 0.225 + 0.406) * 255), 0, 255)

    adv_img = adv.squeeze().permute(1, 2, 0).cpu().numpy()
    adv_img = (adv_img * 1).astype(np.uint8)
    adv_img = Image.fromarray(adv_img)

    adv_image_path = f"{output_directory}\\{image_num}.png"

    adv_img.save(adv_image_path)

image_end_time = time.time()
image_elapsed_time = image_end_time - image_start_time