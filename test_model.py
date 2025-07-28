# test_model.py
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import json
from datetime import datetime
import glob

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

class ImageModerator:
    def __init__(self, checkpoint_path="./clip-checkpoints/checkpoint-660"):
        """Initialize the image moderator with trained model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP model and processor
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load trained classifier
        self.model = CLIPClassifier(self.clip_model).to(self.device)
        self.model.load_state_dict(torch.load(f"{checkpoint_path}/pytorch_model.bin", map_location=self.device))
        self.model.eval()
        
        # Prediction log
        self.predictions_log = []
        
        print(f"✅ Model loaded successfully on {self.device}")
    
    def predict_image(self, image_path):
        """Predict if an image is appropriate (0) or inappropriate (1)"""
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            inputs = self.processor(images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs["logits"]
                probabilities = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][prediction].item()
            
            return prediction, confidence
        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
            return None, 0.0
    
    def get_moderation_decision(self, prediction, confidence):
        """Get moderation decision based on prediction and confidence"""
        if prediction == 0:  # Appropriate
            if confidence > 0.8:
                return "APPROVED", "High confidence appropriate"
            else:
                return "REVIEW", "Low confidence appropriate"
        else:  # Inappropriate
            if confidence > 0.85:
                return "AUTO-FLAG", "High confidence inappropriate"
            elif confidence > 0.65:
                return "FLAG", "Medium confidence inappropriate"
            else:
                return "REVIEW", "Low confidence inappropriate"
    
    def test_single_image(self, image_path):
        """Test a single image and print results"""
        print(f"\n🔍 Testing: {image_path}")
        print("-" * 50)
        
        prediction, confidence = self.predict_image(image_path)
        
        if prediction is None:
            return
        
        decision, reason = self.get_moderation_decision(prediction, confidence)
        
        # Print results
        result_text = "Appropriate" if prediction == 0 else "Inappropriate"
        print(f"Prediction: {result_text}")
        print(f"Confidence: {confidence:.3f}")
        print(f"Decision: {decision}")
        print(f"Reason: {reason}")
        
        # Log prediction
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "prediction": prediction,
            "confidence": confidence,
            "decision": decision,
            "reason": reason
        }
        self.predictions_log.append(log_entry)
        
        return prediction, confidence, decision
    
    def test_folder(self, folder_path, file_extensions=("*.jpg", "*.jpeg", "*.png")):
        """Test all images in a folder"""
        print(f"\n📁 Testing folder: {folder_path}")
        print("=" * 60)
        
        # Find all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        if not image_files:
            print(f"❌ No image files found in {folder_path}")
            return
        
        print(f"Found {len(image_files)} images to test")
        
        # Test each image
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}]", end=" ")
            self.test_single_image(image_path)
    
    def save_log(self, filename="prediction_log.json"):
        """Save prediction log to JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.predictions_log, f, indent=2)
        print(f"\n📝 Prediction log saved to {filename}")
    
    def print_summary(self):
        """Print summary of all predictions"""
        if not self.predictions_log:
            print("No predictions to summarize")
            return
        
        total = len(self.predictions_log)
        decisions = {}
        for entry in self.predictions_log:
            decision = entry["decision"]
            decisions[decision] = decisions.get(decision, 0) + 1
        
        print(f"\n📊 SUMMARY ({total} images tested)")
        print("=" * 40)
        for decision, count in decisions.items():
            percentage = (count / total) * 100
            print(f"{decision}: {count} ({percentage:.1f}%)")

# Example usage
if __name__ == "__main__":
    # Initialize the moderator
    moderator = ImageModerator()
    
    # Test a single image
    # moderator.test_single_image("test_samples/photo.jpg")
    
    # Test all images in a folder
    # moderator.test_folder("test_samples")
    
    # Test multiple individual images
    test_images = [
        # "test_samples/image1.jpg",
        # "test_samples/image2.png",
        # Add your image paths here
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            moderator.test_single_image(image_path)
        else:
            print(f"❌ Image not found: {image_path}")
    
    # Print summary and save log
    moderator.print_summary()
    moderator.save_log()
    
    print("\n🎯 Ready to test! Uncomment the lines above to test your images.")
    print("💡 Usage examples:")
    print("  - moderator.test_single_image('path/to/image.jpg')")
    print("  - moderator.test_folder('path/to/folder')") 