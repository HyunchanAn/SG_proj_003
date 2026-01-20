import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from vsams.models.classifier import SurfaceClassifier
import os
import glob
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define Class Mappings (Synchronized with labeler.py)
MATERIALS = ["Metal", "Plastic", "Glass", "Painted", "Wood", "Other"]
FINISHES = ["Mirror", "Rough", "Hairline", "Matte", "Glossy", "Pattern", "Other"]

MAT_MAP = {name: i for i, name in enumerate(MATERIALS)}
FIN_MAP = {name: i for i, name in enumerate(FINISHES)}

class SurfaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        if not os.path.exists(root_dir):
            print(f"Warning: Root directory {root_dir} does not exist.")
            return

        # Folder structure: dataset/train/Material_Finish/*.jpg
        for class_folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, class_folder)
            if not os.path.isdir(folder_path):
                continue
            
            # Parse labels from folder name
            try:
                mat_name, fin_name = class_folder.split('_')
                mat_idx = MAT_MAP.get(mat_name, MAT_MAP["Other"])
                fin_idx = FIN_MAP.get(fin_name, FIN_MAP["Other"])
            except ValueError:
                print(f"Skipping folder with invalid format: {class_folder}")
                continue

            for img_path in glob.glob(os.path.join(folder_path, "*.[jJ][pP][gG]")) + \
                             glob.glob(os.path.join(folder_path, "*.[pP][nN][gG]")):
                self.samples.append((img_path, mat_idx, fin_idx))
        
        print(f"Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mat_label, fin_label = self.samples[idx]
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return image, mat_label, fin_label

def train_model(num_epochs=10, batch_size=8, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Augmentations (Albumentations)
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Rotate(limit=15, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    
    # 2. Dataset & Dataloader
    train_dir = os.path.join("dataset", "train")
    dataset = SurfaceDataset(train_dir, transform=train_transform)
    
    if len(dataset) == 0:
        print("No data found. Please collect data using labeler.py first.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 3. Model
    model = SurfaceClassifier(
        num_materials=len(MATERIALS), 
        num_finishes=len(FINISHES)
    ).to(device)
    
    # 4. Training Components
    criterion_material = nn.CrossEntropyLoss()
    criterion_finish = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for images, mat_labels, fin_labels in dataloader:
            images = images.to(device)
            mat_labels = mat_labels.to(device)
            fin_labels = fin_labels.to(device)
            
            optimizer.zero_grad()
            mat_out, fin_out = model(images)
            
            loss_mat = criterion_material(mat_out, mat_labels)
            loss_fin = criterion_finish(fin_out, fin_labels)
            total_loss = loss_mat + loss_fin
            
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    # 5. Save Model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/v_sams_model.pth')
    print("Training Complete. Model saved to checkpoints/v_sams_model.pth")

if __name__ == "__main__":
    train_model(num_epochs=5)
