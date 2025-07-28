# load_model.py
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Load the same model architecture
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes=2):
        super().__init__()
        self.clip = clip_model
        for param in self.clip.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.clip.config.projection_dim, num_classes)
        
    def forward(self, pixel_values, labels=None):
        image_outputs = self.clip.get_image_features(pixel_values=pixel_values)
        logits = self.classifier(image_outputs)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
model = CLIPClassifier(clip_model).to(device)

# Load the best checkpoint
checkpoint_path = "./clip-checkpoints/checkpoint-660"  # This should be your best checkpoint
model.load_state_dict(torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location=device))
model.eval()

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("✅ Model loaded successfully!")

# Test function
def predict_image(image_path):
    """Predict if an image is appropriate (0) or inappropriate (1)"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    
    inputs = processor(images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        probabilities = torch.softmax(logits, dim=1)
        prediction = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][prediction].item()
    
    return prediction, confidence

# Example usage
if __name__ == "__main__":
    # Test with a sample image (replace with actual image path)
    # result, conf = predict_image("path/to/test/image.jpg")
    # print(f"Prediction: {'Appropriate' if result == 0 else 'Inappropriate'} (confidence: {conf:.2f})")
    print("Model is ready for predictions!") 