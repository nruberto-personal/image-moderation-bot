# quick_test.py - Simple demo to test your model
from test_model import ImageModerator
import os

def main():
    # Initialize the moderator
    print("🚀 Loading your trained model...")
    moderator = ImageModerator()
    
    # Test a single image (replace with your image path)
    test_image = "test_samples/your_image.jpg"  # Change this to your image path
    
    if os.path.exists(test_image):
        print(f"\n🎯 Testing: {test_image}")
        result, confidence, decision = moderator.test_single_image(test_image)
        
        print(f"\n📋 Results:")
        print(f"  - Prediction: {'Appropriate' if result == 0 else 'Inappropriate'}")
        print(f"  - Confidence: {confidence:.3f}")
        print(f"  - Decision: {decision}")
        
    else:
        print(f"\n❌ Image not found: {test_image}")
        print("💡 To test your model:")
        print("   1. Put some test images in the 'test_samples' folder")
        print("   2. Update the 'test_image' path above")
        print("   3. Run this script again")
        
        # Show what's in the test_samples folder
        if os.path.exists("test_samples"):
            files = os.listdir("test_samples")
            if files:
                print(f"\n📁 Files in test_samples folder:")
                for file in files:
                    print(f"   - {file}")
            else:
                print("\n📁 test_samples folder is empty")

if __name__ == "__main__":
    main() 