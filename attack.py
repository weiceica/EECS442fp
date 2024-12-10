import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from tqdm import tqdm


def square_attack(model, image, side_length=4, max_iters=100):
    """
    Perform the square attack on a single image.
    Parameters:
        model: The PyTorch model to attack.
        image: Input image (torch.Tensor) of shape [1, C, H, W].
        side_length: Length of the side of the square to perturb.
        max_iters: Maximum number of iterations to run the attack.
    Returns:
        adv_image: The perturbed adversarial image.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.clone().detach().to(device)

    adv_image = image.clone().detach()
    adv_image.requires_grad = False
    _, _, H, W = adv_image.shape

    for _ in range(max_iters):
        x_start = np.random.randint(0, H - side_length)
        y_start = np.random.randint(0, W - side_length)
        perturbation = torch.rand_like(adv_image[:, :, x_start:x_start+side_length, y_start:y_start+side_length])
        adv_image[:, :, x_start:x_start+side_length, y_start:y_start+side_length] = perturbation

        _ = model(adv_image)

    return adv_image


def process_office31_dataset(input_dir, output_dir, model, side_length=4, max_iters=100):
    """
    Process the Office31 dataset and apply the Square Attack.
    Parameters:
        input_dir: Directory of the Office31 dataset.
        output_dir: Directory to save perturbed images.
        model: The PyTorch model to attack.
        side_length: Length of the side of the square to perturb.
        max_iters: Maximum number of iterations to run the attack.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    inv_transform = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
        transforms.ToPILImage()
    ])

    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        output_subfolder = os.path.join(output_dir, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        for image_file in tqdm(files, desc=f"Processing {relative_path}"):
            image_path = os.path.join(root, image_file)
            try:
                image = Image.open(image_path).convert('RGB')
                input_tensor = transform(image).unsqueeze(0).to(device)

                adv_image = square_attack(model, input_tensor, side_length, max_iters)
                adv_image = inv_transform(adv_image.squeeze(0).cpu().detach())
                adv_image.save(os.path.join(output_subfolder, image_file))

            except Exception as e:
                print(f"Error processing {image_path}: {e}")


if __name__ == "__main__":
    # Directories
    input_dir = "./office31"
    output_dir = "./office31perturbed"

    # Load model
    model = resnet18(weights='ResNet18_Weights.DEFAULT')
    model.eval()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Process the dataset
    process_office31_dataset(input_dir, output_dir, model, side_length=10, max_iters=100)
