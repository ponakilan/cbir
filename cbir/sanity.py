import os
from torchvision import io

root_dir = "/home/ubuntu/data/datasets/rparis6k/jpg"

print("Scanning for fake JPEGs...")
for file in os.listdir(root_dir):
    if file.endswith(("jpg", "jpeg")):
        image_path = os.path.join(root_dir, file)
        try:
            io.read_image(image_path)
        except RuntimeError as e:
            print(f"Bad file found: {file} - {e}")
            os.remove(os.path.join(root_dir, file))