import os
import torch
from torchvision import models, transforms
from PIL import Image
from torchattacks import PGD
import time


image_start_time = time.time()

input_dir = r"..."
output_dir = r"..."
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.vit_b_32(pretrained=True)
model.eval()
model.to(device)

mean = torch.tensor([0, 0, 0], device=device).view(3,1,1)
std = torch.tensor([1, 1, 1], device=device).view(3,1,1)

def normalize(x):
    return (x - mean) / std

def denormalize(x):
    return x * std + mean

to_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

eps = 8 / 255
alpha = 1 / 255
steps = 100

attack = PGD(
    model,
    eps=eps,
    alpha=alpha,
    steps=steps,
    random_start=True
)
attack.set_mode_targeted_by_label()


for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        continue


    img_path = os.path.join(input_dir, img_name)
    img = Image.open(img_path).convert("RGB")

    x_pixel = to_tensor(img).unsqueeze(0).to(device)
    x_norm = normalize(x_pixel)

    with torch.no_grad():
        logits = model(x_norm)
        probs = torch.softmax(logits, dim=1)
        _, topk_indices = torch.topk(probs, k=500, dim=1)

    orig_label = topk_indices[0, 0].item()
    target_label = topk_indices[0, 499].item()
    target = torch.tensor([target_label], device=device)

    adv_x_norm = attack(x_norm, target)

    adv_x_pixel = denormalize(adv_x_norm)
    adv_x_pixel = torch.clamp(adv_x_pixel, 0.0, 1.0)

    adv_img = transforms.ToPILImage()(adv_x_pixel.squeeze(0).cpu())
    adv_img.save(os.path.join(output_dir, img_name))

    print(f"[OK] {img_name} | orig={orig_label} → target(top500)={target_label}")

image_end_time = time.time()
image_elapsed_time = image_end_time - image_start_time
print(f"{image_elapsed_time / 5}")