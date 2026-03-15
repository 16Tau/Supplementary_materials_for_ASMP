import os
import time
import torch
from torchvision import models, transforms
from PIL import Image
from torchattacks import CW


# =====================================================
input_dir = r"..."
output_dir = r"..."
os.makedirs(output_dir, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.alexnet(pretrained=True).to(device)
model.eval()

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

attack = CW(
    model,
    c=1e-4,
    lr=0.01,
    steps=100,
    kappa=0
)
attack.set_mode_targeted_by_label()

total_attack_time = 0.0
num_images = 0

for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        continue

    img = Image.open(os.path.join(input_dir, img_name)).convert("RGB")
    x_pixel = to_tensor(img).unsqueeze(0).to(device)
    x_norm = normalize(x_pixel)

    with torch.no_grad():
        logits = model(x_norm)
        _, topk_indices = torch.topk(logits, k=500, dim=1)

    target_label = topk_indices[0, 499].item()
    target = torch.tensor([target_label], device=device)

    start = time.perf_counter()
    adv_x_norm = attack(x_norm, target)
    end = time.perf_counter()
    # --------------------------

    total_attack_time += (end - start)
    num_images += 1

    adv_x_pixel = torch.clamp(denormalize(adv_x_norm), 0, 1)
    adv_img = transforms.ToPILImage()(adv_x_pixel.squeeze(0).cpu())
    adv_img.save(os.path.join(output_dir, img_name))

    print(f"[OK] {img_name} | target(top500)={target_label}")


avg_time = total_attack_time / num_images
print(f": {avg_time:.6f} ")
