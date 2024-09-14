import os
import shutil

# Define the directory containing the images
def seperate(name):
    
    source_directory = '/blue/azare/samgallic/Research/new_cycle_gan/results/' + name + '/test_latest/images'

    # Define target directories for real_A and fake_A images
    real_B_directory = 'results/organized/' + name + '/real_B'
    fake_B_directory = 'results/organized/' + name + '/fake_B'
    real_A_directory = 'results/organized/' + name + '/real_A'
    fake_A_directory = 'results/organized/' + name + '/fake_A'
    rec_A_directory = 'results/organized/' + name + '/rec_A'
    rec_B_directory = 'results/organized/' + name + '/rec_B'

    # Create target directories if they don't already exist
    os.makedirs(real_B_directory, exist_ok=True)
    os.makedirs(fake_B_directory, exist_ok=True)
    os.makedirs(real_A_directory, exist_ok=True)
    os.makedirs(fake_A_directory, exist_ok=True)
    os.makedirs(rec_B_directory, exist_ok=True)
    os.makedirs(rec_A_directory, exist_ok=True)

    # Iterate through all files in the source directory
    for filename in os.listdir(source_directory):
        if filename.endswith('.png'):  # Filter to only process .png files
            # Check if the file is a real_B image
            if 'real_B' in filename:
                # Copy the real_B image to the real_B directory
                shutil.copy(os.path.join(source_directory, filename), 
                            os.path.join(real_B_directory, filename))
            # Check if the file is a fake_B image
            elif 'fake_B' in filename:
                # Copy the fake_B image to the fake_B directory
                shutil.copy(os.path.join(source_directory, filename), 
                            os.path.join(fake_B_directory, filename))
            elif 'fake_A' in filename:
                # Copy the fake_A image to the fake_A directory
                shutil.copy(os.path.join(source_directory, filename), 
                            os.path.join(fake_A_directory, filename))
            elif 'real_A' in filename:
                # Copy the real_A image to the real_A directory
                shutil.copy(os.path.join(source_directory, filename), 
                            os.path.join(real_A_directory, filename))
            elif 'rec_B' in filename:
                # Copy the fake_B image to the fake_B directory
                shutil.copy(os.path.join(source_directory, filename), 
                            os.path.join(rec_B_directory, filename))
            elif 'rec_A' in filename:
                # Copy the fake_A image to the fake_A directory
                shutil.copy(os.path.join(source_directory, filename), 
                            os.path.join(rec_A_directory, filename))
def main():
    seperate('normal2noisy_emb')

if __name__ == '__main__':
    main()