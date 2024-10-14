import unittest
import torch

def loss(sample, empirical):
    # Ensure tensors are on the same device (i.e., CUDA if available)
    assert sample.device == empirical.device, "Tensors should be on the same device."

    # Ensure that the tensors do not contain NaN or Inf values
    if torch.isnan(sample).any() or torch.isnan(empirical).any():
        raise ValueError("Input tensors contain NaN values")
    if torch.isinf(sample).any() or torch.isinf(empirical).any():
        raise ValueError("Input tensors contain Inf values")

    # Determine the min and max from both tensors
    data_min = min(sample.min(), empirical.min())
    data_max = max(sample.max(), empirical.max())

    # Define the bins based on the min and max values
    bins = torch.linspace(data_min.item(), data_max.item(), steps=500, device=sample.device)

    # Compute histograms (make sure they are on the same device)
    hist1 = torch.histc(sample, bins=len(bins), min=bins.min().item(), max=bins.max().item())
    hist2 = torch.histc(empirical, bins=len(bins), min=bins.min().item(), max=bins.max().item())

    # Normalize histograms
    hist1 /= hist1.sum()
    hist2 /= hist2.sum()

    # Debugging: Check histograms
    print(f"Histogram 1: {hist1}")
    print(f"Histogram 2: {hist2}")

    # Subtract the histograms element-wise
    hist_diff = torch.abs(hist1 - hist2).mean()

    # Return the difference (convert tensor to float)
    return hist_diff.item()  # Ensure the return type is a float

class TestDistanceCalc(unittest.TestCase):
    def test_loss_identical_tensors(self):
        """Test that loss returns 0 for identical tensors."""
        sample = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        empirical = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        calculated_loss = loss(sample, empirical)
        self.assertAlmostEqual(calculated_loss, 0.0, places=5)

    def test_loss_all_zeros(self):
        """Test that loss returns 0 when both tensors are all zeros."""
        sample = torch.zeros(100, device='cuda')
        empirical = torch.zeros(100, device='cuda')
        calculated_loss = loss(sample, empirical)
        self.assertAlmostEqual(calculated_loss, 0.0, places=5)

    def test_loss_different_shapes(self):
        """Test that loss returns 0 for tensors with different order but same shape."""
        sample = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        empirical = torch.tensor([3.0, 2.0, 1.0], device='cuda')
        calculated_loss = loss(sample, empirical)
        self.assertAlmostEqual(calculated_loss, 0.0, places=5)

    def test_loss_different_distributions(self):
        """Test that loss returns a non-zero value for tensors with different distributions."""
        sample = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        empirical = torch.tensor([100.0, 200.0, 300.0], device='cuda')
        calculated_loss = loss(sample, empirical)
        self.assertGreater(calculated_loss, 0.0)

    def test_loss_inf_values(self):
        """Test that loss raises an error with Inf values."""
        sample = torch.tensor([1.0, float('inf'), 3.0], device='cuda')
        empirical = torch.tensor([3.0, 2.0, 1.0], device='cuda')
        with self.assertRaises(ValueError):
            loss(sample, empirical)

    def test_loss_nan_values(self):
        """Test that loss raises an error with NaN values."""
        sample = torch.tensor([1.0, float('nan'), 3.0], device='cuda')
        empirical = torch.tensor([3.0, 2.0, 1.0], device='cuda')
        with self.assertRaises(ValueError):
            loss(sample, empirical)

    def test_loss_large_tensors(self):
        """Test that the loss works for large tensors."""
        sample = torch.randn(10000, device='cuda')
        empirical = torch.randn(10000, device='cuda')
        calculated_loss = loss(sample, empirical)
        self.assertIsInstance(calculated_loss, float)  # Ensure the loss is returned as a number

# Run the tests
if __name__ == '__main__':
    unittest.main()
