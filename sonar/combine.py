from PIL import Image

def combine(name):
    path_A = '/blue/azare/samgallic/Research/new_cycle_gan/checkpoints/' + name + '/histograms/Gamma2Rayleigh/'
    path_B = '/blue/azare/samgallic/Research/new_cycle_gan/checkpoints/' + name + '/histograms/Rayleigh2Gamma/'

    # Paths to your 5 images
    image_paths_A = [path_A + "Epoch_1_Rayleigh.png", path_A + "Epoch_50_Rayleigh.png", 
                path_A + "Epoch_100_Rayleigh.png", path_A + "Epoch_200_Rayleigh.png", 
                path_A + "Epoch_250_Rayleigh.png"]
    image_paths_B = [path_B + "Epoch_1_Gamma.png", path_B + "Epoch_50_Gamma.png", 
                path_B + "Epoch_100_Gamma.png", path_B + "Epoch_200_Gamma.png", 
                path_B + "Epoch_250_Gamma.png"]

    # Load the images
    images_A = [Image.open(img) for img in image_paths_A]
    images_B = [Image.open(img) for img in image_paths_B]

    # Resize all images to the same size (optional, adjust width and height as needed)
    width, height = 846, 545  # Adjust dimensions as needed
    images_A = [img.resize((width, height)) for img in images_A]
    images_B = [img.resize((width, height)) for img in images_B]

    # Calculate the size of the final image
    final_width = width * 3  # Three images in the top row
    final_height = height * 2  # Two rows

    # Create a new blank image with a white background
    final_image_A = Image.new("RGB", (final_width, final_height), (255, 255, 255))
    final_image_B = Image.new("RGB", (final_width, final_height), (255, 255, 255))

    # Paste the images into the final image
    # Top row (3 images)
    final_image_A.paste(images_A[0], (0, 0))
    final_image_A.paste(images_A[1], (width, 0))
    final_image_A.paste(images_A[2], (width * 2, 0))

    # Bottom row (2 images)
    final_image_A.paste(images_A[3], (width // 2, height))
    final_image_A.paste(images_A[4], (width * 3 // 2, height))

    # Save or show the final image
    final_image_A.save(path_A + 'final_gamma2rayleigh.png')

    # Paste the images into the final image
    # Top row (3 images)
    final_image_B.paste(images_B[0], (0, 0))
    final_image_B.paste(images_B[1], (width, 0))
    final_image_B.paste(images_B[2], (width * 2, 0))

    # Bottom row (2 images)
    final_image_B.paste(images_B[3], (width // 2, height))
    final_image_B.paste(images_B[4], (width * 3 // 2, height))

    # Save or show the final image
    final_image_B.save(path_B + 'final_rayleigh2gamma.png')
    
combine('black_debugged_no_disc')