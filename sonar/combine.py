from PIL import Image

def combine(name):
    path_A = '/blue/azare/samgallic/Research/new_cycle_gan/checkpoints/less_noise/histograms/Gamma2Rayleigh/'
    path_B = '/blue/azare/samgallic/Research/new_cycle_gan/checkpoints/less_noise/histograms/Rayleigh2Gamma/'

    # Paths to your 5 images
    image_paths = [path_A + "Epoch_1_Rayleigh.png", path_A + "Epoch_50_Rayleigh.png", 
                path_A + "Epoch_100_Rayleigh.png", path_A + "Epoch_150_Rayleigh.png", 
                path_A + "Epoch_200_Rayleigh.png"]

    # Load the images
    images = [Image.open(img) for img in image_paths]

    # Resize all images to the same size (optional, adjust width and height as needed)
    width, height = 846, 545  # Adjust dimensions as needed
    images = [img.resize((width, height)) for img in images]

    # Calculate the size of the final image
    final_width = width * 3  # Three images in the top row
    final_height = height * 2  # Two rows

    # Create a new blank image with a white background
    final_image = Image.new("RGB", (final_width, final_height), (255, 255, 255))

    # Paste the images into the final image
    # Top row (3 images)
    final_image.paste(images[0], (0, 0))
    final_image.paste(images[1], (width, 0))
    final_image.paste(images[2], (width * 2, 0))

    # Bottom row (2 images)
    final_image.paste(images[3], (width // 2, height))
    final_image.paste(images[4], (width * 3 // 2, height))

    # Save or show the final image
    final_image.save('combine.png')
