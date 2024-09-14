import os
import torch
from torchvision import io, transforms
from PIL import Image

def load_images_from_folder(folder_path):
    images = {}
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(folder_path, filename)
            img = io.read_image(img_path)
            images[filename] = img.div(255.0)  # Normalize image
    return images

def save_greyscale_images(images, output_folder):
    # Define a transform to convert images to grayscale
    to_grayscale = transforms.Grayscale(num_output_channels=1)

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    for filename, img_tensor in images.items():
        # Convert the image tensor to grayscale
        grayscale_img_tensor = to_grayscale(img_tensor)

        # Convert the tensor back to a PIL image
        grayscale_img = transforms.ToPILImage()(grayscale_img_tensor.squeeze(0))

        # Save the grayscale image with the same filename in the output folder
        output_path = os.path.join(output_folder, filename)
        grayscale_img.save(output_path)

if __name__ == "__main__":
    # Specify the input and output folders
    input_folder = "real"
    output_folder = "greyscale"

    # Load images from the input folder
    images = load_images_from_folder(input_folder)

    # Save the images as grayscale in the output folder
    save_greyscale_images(images, output_folder)
