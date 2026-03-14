import math
import torch
from torchvision import io
from torch.nn import functional as F


def multi_scale_image(image: torch.Tensor) -> tuple[torch.Tensor]:
    """
    Resize the image into five specific scales.

    Args:
        image (torch.Tensor) - The image tensor which has to be scaled.

    Returns:
        The list of tensors of the image in five different scales.
    """
    scales = (1/(2 * math.sqrt(2)), 1/2, 1/math.sqrt(2), 1, math.sqrt(2))
    scaled_images = [
        F.interpolate(
            input = image,
            scale_factor = scale,
            mode = "bilinear",
            align_corners = False
        ) for scale in scales
    ]
    return scaled_images


if __name__ == "__main__":
    sample_image_path = "/home/ubuntu/data/datasets/roxford5k/jpg/oxford_002881.jpg"
    sample_image = io.read_image(sample_image_path).unsqueeze(0).float()
    scaled_images = multi_scale_image(sample_image)
    print(len(scaled_images), scaled_images[0].shape)

