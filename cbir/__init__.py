import os
import torch
from torchvision import io, transforms
import matplotlib.pyplot as plt
from PIL import Image

from cbir.search import batched_coattention_search
from cbir.models import QueryFeatureExtractor, GeM

FEATURES_PATH = "/home/ubuntu/data/feature_cache/roxford5k_features.pkl"
DB_BASE_DIR = "/home/ubuntu/data/datasets/roxford5k/jpg"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512)),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

feature_cache = torch.load(FEATURES_PATH)

db_image_ids = list(feature_cache.keys())
list_of_cached_tensors = list(feature_cache.values())

db_tensor = torch.stack(list_of_cached_tensors, dim=0)
db_tensor = db_tensor.to(device)

feature_extractor = QueryFeatureExtractor()
feature_extractor.eval()

gem = GeM(p=3)

def db_image_search(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        V_q = feature_extractor(image)

    V_q = gem(V_q)

    scores = batched_coattention_search(V_q, db_tensor)
    top_indices = torch.argsort(scores, descending=True)[:6]

    res_image_paths = []

    for index in top_indices:
        image_id = db_image_ids[index]
        image_path = os.path.join(DB_BASE_DIR, image_id)
        res_image_paths.append(image_path + ".jpg")
    
    return res_image_paths
