# conditional.py
import torch
import kornia

def ConditionalHists(noisy_tensor_dict: dict, clean_tensor_dict: dict, bins: torch.Tensor, calc):
    # Pixel value ranges
    ranges = [
        (0, 49),
        (50, 99),
        (100, 149),
        (150, 199),
        (200, 255)
    ]

    # Initialize a list of empty tensors to store noise values for each range
    noise_values = [torch.tensor([]).to(calc.device) for _ in ranges]

    # Get sorted lists of filenames
    noisy_filenames = sorted(noisy_tensor_dict.keys())
    clean_filenames = sorted(clean_tensor_dict.keys())

    # Ensure that the number of files matches
    if len(noisy_filenames) != len(clean_filenames):
        print("Mismatch in the number of files between the noisy and clean datasets.")
        exit()

    # Iterate over the filenames
    for noisy_filename, clean_filename in zip(noisy_filenames, clean_filenames):
        if noisy_filename != clean_filename:
            print(f"Filename mismatch: {noisy_filename} vs {clean_filename}")
            continue  # Skip mismatched files

        noisy = noisy_tensor_dict[noisy_filename]
        clean = clean_tensor_dict[clean_filename]

        # Calculate the noise (difference)
        noise = noisy - clean

        # Collect noise values based on the original pixel value ranges
        for idx, (lower, upper) in enumerate(ranges):
            mask = (clean >= lower) & (clean <= upper)
            noise_in_range = noise[mask].flatten()

            # Concatenate new noise values to the existing tensor for this range
            noise_values[idx] = torch.cat((noise_values[idx], noise_in_range))

    histograms = dict()

    # Compute histograms for each range
    for idx, range_tuple in enumerate(ranges):
        data = noise_values[idx]  # Shape: (N,)
        if data.numel() == 0:
            # Handle empty data case
            histograms[range_tuple] = torch.zeros_like(bins)
            continue

        # Compute histogram using calc.compute_histogram
        hist = calc.compute_histogram(data, bins)  # hist shape: (1, num_bins)
        hist = hist.squeeze()  # Shape: (num_bins,)
        histograms[range_tuple] = hist / hist.sum()

    return histograms

def loss(noisy_sample: torch.Tensor, normal_sample: torch.Tensor, histograms: dict, bins: torch.Tensor, calc):
    # Pixel value ranges
    ranges = [
        (0, 49),
        (50, 99),
        (100, 149),
        (150, 199),
        (200, 255)
    ]

    noise = noisy_sample - normal_sample

    total_weighted_loss = 0.0
    total_weight = 0.0

    # Compute difference for each range
    for idx, range_tuple in enumerate(ranges):
        lower, upper = range_tuple

        # Create a mask for pixels in the normal sample within the current range
        mask = (normal_sample >= lower) & (normal_sample <= upper)
        noise_in_range = noise[mask].flatten()
        normal_values_in_range = normal_sample[mask].flatten()

        if noise_in_range.numel() == 0:
            continue  # No data in this range, skip

        # Compute histogram of the noise_in_range
        hist = calc.compute_histogram(noise_in_range, bins).squeeze()

        # Normalize the histogram
        if hist.sum() > 0:
            hist = hist / hist.sum()
        else:
            continue  # Avoid division by zero

        # Retrieve the target histogram for the current range
        target_hist = histograms[range_tuple]

        # Compute the difference between the histograms (e.g., L1 loss)
        diff = torch.abs(hist - target_hist)

        # Compute the mean difference as the loss for this range
        loss_value = diff.mean()

        # Compute the weight for this range based on the normal pixel values
        range_weight = normal_values_in_range.size(0)

        # Accumulate the weighted loss and total weight
        total_weighted_loss += loss_value * range_weight
        total_weight += range_weight

    # Return the weighted average loss across all ranges
    if total_weight == 0:
        return 0.0  # Avoid division by zero if no ranges were processed
    else:
        return total_weighted_loss / total_weight
