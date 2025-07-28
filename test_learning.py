#!/usr/bin/env python3
"""
Test script to verify the model learning functionality
"""

import json
import os
from datetime import datetime

def test_training_data_structure():
    """Test if training data is being saved correctly"""
    if os.path.exists("training_data.json"):
        with open("training_data.json", 'r') as f:
            data = json.load(f)
        
        print(f"✅ Training data file exists with {len(data)} samples")
        
        if len(data) > 0:
            sample = data[0]
            required_fields = ['image_url', 'image_title', 'subreddit', 'model_prediction', 'model_confidence', 'user_feedback', 'timestamp']
            
            missing_fields = [field for field in required_fields if field not in sample]
            if missing_fields:
                print(f"❌ Missing fields in training data: {missing_fields}")
            else:
                print("✅ Training data structure is correct")
                print(f"Sample entry: {sample}")
        
        return len(data)
    else:
        print("❌ No training data file found")
        return 0

def create_sample_training_data():
    """Create sample training data for testing"""
    sample_data = [
        {
            "image_url": "https://example.com/image1.jpg",
            "title": "Sample image 1",
            "subreddit": "test",
            "model_prediction": 0,
            "model_confidence": 0.75,
            "user_feedback": "appropriate",
            "timestamp": datetime.now().isoformat()
        },
        {
            "image_url": "https://example.com/image2.jpg", 
            "title": "Sample image 2",
            "subreddit": "test",
            "model_prediction": 1,
            "model_confidence": 0.80,
            "user_feedback": "inappropriate",
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    with open("training_data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("✅ Created sample training data")

if __name__ == "__main__":
    print("🧪 Testing Model Learning Functionality")
    print("=" * 50)
    
    # Test existing training data
    sample_count = test_training_data_structure()
    
    if sample_count == 0:
        print("\n📝 Creating sample training data for testing...")
        create_sample_training_data()
        test_training_data_structure()
    
    print("\n🎯 Learning Requirements:")
    print("- Minimum samples for retraining: 10")
    print("- Auto-retrain frequency: Every 10 samples")
    print("- Expected improvement: 50-100 samples")
    print("- Reliable performance: 500-1000 samples")
    
    print("\n🚀 Ready to test learning! Run the Streamlit app and:")
    print("1. Go to 'Train Model' tab")
    print("2. Provide feedback on 10+ images")
    print("3. Click 'Retrain Model' or wait for auto-retrain")
    print("4. Test the improved model!") 