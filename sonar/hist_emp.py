import torch
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# Define the image transformations
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

# Directories containing the images
A_dir = '/blue/azare/samgallic/Research/new_cycle_gan/datasets/gray_forest/trainB'
B_dir = '/blue/azare/samgallic/Research/new_cycle_gan/datasets/normal2noisy_forest/trainB'

# Load filenames from both directories
A_filenames = sorted([file for file in os.listdir(A_dir) if file.endswith(('png', 'jpg', 'jpeg'))])
B_filenames = sorted([file for file in os.listdir(B_dir) if file.endswith(('png', 'jpg', 'jpeg'))])

# Ensure both directories have the same filenames
assert A_filenames == B_filenames, "Filenames do not match between directories"

results_div = []
results_sub = []

# Process images with matching filenames and perform element-wise division
for filename in A_filenames:
    a_path = os.path.join(A_dir, filename)
    b_path = os.path.join(B_dir, filename)
    
    # Open the images
    image1 = Image.open(a_path)
    image2 = Image.open(b_path)
    
    # Apply the transformations
    tensora = transform(image1)
    tensorb = transform(image2)
    
    # Element-wise division (with epsilon to avoid division by zero)
    valid_indices = (tensora != 0) & (tensorb != 0)
    result_div = tensorb[valid_indices] / tensora[valid_indices]
    result_sub = tensorb - tensora
    
    # Append the result to the list
    results_div.append(result_div)
    results_sub.append(result_sub)

# Concatenate all results into a single tensor
all_results_div = torch.cat([r.flatten() for r in results_div])
all_results_sub = torch.cat([r.flatten() for r in results_sub])
print(len(all_results_div))
print(len(all_results_sub))

# Plot histogram of the result
plt.hist(all_results_div, bins=500, alpha=0.75, color='blue')
plt.title('Histogram of Element-wise Division of Transformed Images')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.xlim(xmin=-50, xmax = 50)
plt.grid(True)
plt.savefig('division.png')
plt.close()

plt.hist(all_results_sub, bins=500, alpha=0.75, color='blue')
plt.title('Histogram of Element-wise Subtraction of Transformed Images')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.xlim(xmin=-10, xmax = 10)
plt.grid(True)
plt.savefig('subtraction.png')
