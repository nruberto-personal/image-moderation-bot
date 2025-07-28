# train.py
import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel, Trainer, TrainingArguments
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
from tqdm import tqdm

# === Settings ===
DATA_DIR = "dataset"
BATCH_SIZE = 4  # Reduced batch size to avoid memory issues
EPOCHS = 5
MODEL_NAME = "openai/clip-vit-base-patch32"

# === Custom CLIP Classifier ===
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=2):
        super().__init__()
        self.clip = clip_model
        # Freeze CLIP parameters
        for param in self.clip.parameters():
            param.requires_grad = False
        # Add classification head
        self.classifier = nn.Linear(self.clip.config.projection_dim, num_classes)
        
    def forward(self, pixel_values, labels=None):
        # Get CLIP image features
        image_outputs = self.clip.get_image_features(pixel_values=pixel_values)
        # Classify
        logits = self.classifier(image_outputs)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# === Load Model + Processor ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
model = CLIPClassifier(clip_model).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# === Prepare Data ===
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, labels, processor):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # Open and resize image to avoid decompression bomb
            image = Image.open(self.image_paths[idx]).convert("RGB")
            # Resize to reasonable size
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Process image with CLIP processor
            encoding = self.processor(images=image, return_tensors="pt", padding=True)
            
            # Add label
            encoding["labels"] = torch.tensor(self.labels[idx])
            
            # Remove batch dimension
            return {k: v.squeeze(0) for k, v in encoding.items()}
        except Exception as e:
            print(f"Error processing {self.image_paths[idx]}: {e}")
            # Return a dummy item if image fails to load
            dummy_image = Image.new('RGB', (224, 224), color='gray')
            encoding = self.processor(images=dummy_image, return_tensors="pt", padding=True)
            encoding["labels"] = torch.tensor(self.labels[idx])
            return {k: v.squeeze(0) for k, v in encoding.items()}

# Collect image paths and labels
image_paths, labels = [], []
label_map = {"appropriate": 0, "inappropriate": 1}
for label_name in os.listdir(DATA_DIR):
    class_dir = os.path.join(DATA_DIR, label_name)
    if os.path.isdir(class_dir):
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(class_dir, img_file))
                labels.append(label_map[label_name])

# Split into train/val
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, stratify=labels)

# Create datasets
train_dataset = ImageDataset(train_paths, train_labels, processor)
val_dataset = ImageDataset(val_paths, val_labels, processor)

# === Training ===
training_args = TrainingArguments(
    output_dir="./clip-checkpoints",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    load_best_model_at_end=True,
    dataloader_pin_memory=False,  # Disable pin_memory for CPU
    remove_unused_columns=False,  # Keep all columns
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# Save the trained model
torch.save(model.state_dict(), "clip-timelines-model/model.pt")
processor.save_pretrained("clip-timelines-model")

print("✅ Training completed and model saved!")
