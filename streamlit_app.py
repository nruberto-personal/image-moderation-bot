# streamlit_app.py
import streamlit as st
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np
import json
import pandas as pd
from datetime import datetime
import requests
from io import BytesIO
import time
import os
from dotenv import load_dotenv
from safetensors.torch import load_file

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Image Moderation AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .approved {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .flagged {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .review {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
    }
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        height: 20px;
        border-radius: 10px;
        transition: width 0.3s ease;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

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

@st.cache_resource
def load_model():
    """Load the trained model (cached for performance)"""
    with st.spinner("Loading AI model..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load CLIP model and processor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load trained classifier
        model = CLIPClassifier(clip_model).to(device)
        if device.type == "cuda":
            model.load_state_dict(load_file("./clip-checkpoints/checkpoint-660/model.safetensors", device=device))
        else:
            model.load_state_dict(load_file("./clip-checkpoints/checkpoint-660/model.safetensors"))
        model.eval()
        
        return model, processor, device

@st.cache_resource
def load_reddit():
    """Load Reddit client with environment variables"""
    try:
        import praw
        reddit = praw.Reddit(
            client_id=os.getenv('REDDIT_CLIENT_ID'),
            client_secret=os.getenv('REDDIT_CLIENT_SECRET'),
            user_agent=os.getenv('REDDIT_USER_AGENT', 'CLIP_Image_Moderation_Bot/1.0')
        )
        return reddit
    except Exception as e:
        st.error(f"Failed to load Reddit client: {e}")
        st.info("Please check your .env file and Reddit API credentials")
        return None

def get_popular_subreddits():
    """Get a list of popular subreddits with images"""
    return [
        "aww", "pics", "funny", "memes", "wholesomememes", "MadeMeSmile", "Eyebleach",
        "EarthPorn", "FoodPorn", "Art", "photography", "cats", "dogs", "puppies",
        "kittens", "nature", "travel", "architecture", "cars", "motorcycles",
        "sports", "basketball", "soccer", "baseball", "tennis", "golf",
        "fashion", "streetwear", "makeup", "skincare", "fitness", "bodybuilding",
        "cooking", "baking", "gardening", "plants", "succulents", "bonsai",
        "anime", "cosplay", "gaming", "pcgaming", "xbox", "playstation",
        "movies", "television", "books", "music", "hiphopheads", "popheads",
        "science", "technology", "programming", "webdev", "dataisbeautiful",
        "space", "astronomy", "history", "geography", "maps", "wallpapers"
    ]

def search_subreddits(query, limit=10):
    """Search for subreddits matching the query"""
    try:
        reddit = load_reddit()
        subreddits = []
        
        # Search for subreddits
        for subreddit in reddit.subreddits.search(query, limit=limit):
            try:
                # Get subreddit info
                sub_info = reddit.subreddit(subreddit.display_name)
                subreddits.append({
                    'name': sub_info.display_name,
                    'title': sub_info.title,
                    'subscribers': sub_info.subscribers,
                    'description': sub_info.public_description[:100] + "..." if len(sub_info.public_description) > 100 else sub_info.public_description
                })
            except:
                continue
        
        return subreddits
    except Exception as e:
        st.error(f"Error searching subreddits: {e}")
        return []

def get_random_subreddit():
    """Get a random subreddit from Reddit's popular/new subreddits"""
    try:
        reddit = load_reddit()
        
        # Get random subreddit from popular/new
        import random
        random_post = random.choice(list(reddit.subreddit("all").hot(limit=100)))
        
        # Get subreddit name from the post
        subreddit_name = random_post.subreddit.display_name
        
        # Get subreddit info
        sub_info = reddit.subreddit(subreddit_name)
        
        return {
            'name': sub_info.display_name,
            'title': sub_info.title,
            'subscribers': sub_info.subscribers,
            'description': sub_info.public_description[:100] + "..." if len(sub_info.public_description) > 100 else sub_info.public_description
        }
    except Exception as e:
        st.error(f"Error getting random subreddit: {e}")
        return None

def fetch_random_image():
    """Fetch a random image from a random subreddit with automatic retry"""
    reddit = load_reddit()
    max_attempts = 10  # Try up to 10 different subreddits
    
    for attempt in range(max_attempts):
        try:
            # Get random subreddit
            random_sub = get_random_subreddit()
            if not random_sub:
                continue
                
            subreddit_name = random_sub['name']
            
            # Show which subreddit we're trying
            st.info(f"🔍 Trying r/{subreddit_name} (attempt {attempt + 1}/{max_attempts})")
            
            subreddit = reddit.subreddit(subreddit_name)
            
            # Get posts from hot/new
            try:
                posts = list(subreddit.hot(limit=50))
                if not posts:
                    st.warning(f"⚠️ No posts found in r/{subreddit_name}, trying next subreddit...")
                    continue
            except Exception as e:
                st.warning(f"⚠️ Cannot access r/{subreddit_name} (may be private), trying next subreddit...")
                continue
            
            # Try to find an image post
            import random
            random.shuffle(posts)  # Randomize order
            
            for post in posts:
                # Check if it's an image
                if hasattr(post, 'url') and post.url.lower().endswith(('jpg', 'jpeg', 'png')):
                    try:
                        # Download image
                        response = requests.get(post.url, timeout=10)
                        if response.status_code == 200:
                            image = Image.open(BytesIO(response.content)).convert("RGB")
                            st.success(f"✅ Successfully fetched image from r/{subreddit_name}")
                            return {
                                'image': image,
                                'title': post.title,
                                'url': post.url,
                                'score': post.score,
                                'author': str(post.author),
                                'subreddit': random_sub['name'],
                                'subreddit_title': random_sub['title'],
                                'subreddit_subscribers': random_sub['subscribers']
                            }
                    except Exception as e:
                        continue  # Try next post
            
            # If we get here, no images found in this subreddit
            st.warning(f"⚠️ No images found in r/{subreddit_name}, trying next subreddit...")
            
        except Exception as e:
            st.warning(f"⚠️ Error with r/{subreddit_name}: {str(e)}, trying next subreddit...")
            continue
    
    # If we get here, we've tried all attempts
    st.error("❌ Could not find any images after trying multiple subreddits. Please try again.")
    return None

def fetch_reddit_images(subreddit_name, limit=20):
    """Fetch images from a subreddit"""
    try:
        reddit = load_reddit()
        subreddit = reddit.subreddit(subreddit_name)
        
        images = []
        for post in subreddit.hot(limit=limit):
            if hasattr(post, 'url') and post.url.lower().endswith(('jpg', 'jpeg', 'png')):
                try:
                    # Download image
                    response = requests.get(post.url, timeout=10)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content)).convert("RGB")
                        images.append({
                            'image': image,
                            'title': post.title,
                            'url': post.url,
                            'score': post.score,
                            'author': str(post.author),
                            'subreddit': subreddit_name
                        })
                except Exception as e:
                    continue  # Skip problematic images
        
        return images
    except Exception as e:
        st.error(f"Error fetching from r/{subreddit_name}: {e}")
        return []

def predict_image(image, model, processor, device):
    """Predict if an image is appropriate (0) or inappropriate (1)"""
    try:
        # Resize image
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        
        # Process image
        inputs = processor(images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return prediction, confidence
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, 0.0

def get_moderation_decision(prediction, confidence):
    """Get moderation decision based on prediction and confidence"""
    if prediction == 0:  # Appropriate
        if confidence > 0.8:
            return "APPROVED", "High confidence appropriate", "approved"
        else:
            return "REVIEW", "Low confidence appropriate", "review"
    else:  # Inappropriate
        if confidence > 0.85:
            return "AUTO-FLAG", "High confidence inappropriate", "flagged"
        elif confidence > 0.65:
            return "FLAG", "Medium confidence inappropriate", "flagged"
        else:
            return "REVIEW", "Low confidence inappropriate", "review"

def analyze_image(img_data, img_idx, model, processor, device):
    """Helper function to analyze an image and display results"""
    with st.spinner("Analyzing..."):
        prediction, confidence = predict_image(img_data['image'], model, processor, device)
        
        if prediction is not None:
            decision, reason, css_class = get_moderation_decision(prediction, confidence)
            
            # Update session state
            st.session_state.prediction_count += 1
            
            # Log prediction
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "filename": f"reddit_{img_data['subreddit']}_{img_idx}",
                "prediction": prediction,
                "confidence": confidence,
                "decision": decision,
                "reason": reason,
                "reddit_title": img_data['title'],
                "reddit_score": img_data['score'],
                "reddit_author": img_data['author']
            }
            
            if 'prediction_log' not in st.session_state:
                st.session_state.prediction_log = []
            st.session_state.prediction_log.append(log_entry)
            
            # Show result
            st.markdown(f"""
            <div class="result-box {css_class}">
                <h4>Analysis Result</h4>
                <p><strong>Decision:</strong> {decision}</p>
                <p><strong>Confidence:</strong> {confidence:.3f}</p>
                <p><strong>Prediction:</strong> {'✅ Appropriate' if prediction == 0 else '❌ Inappropriate'}</p>
            </div>
            """, unsafe_allow_html=True)

def save_training_data(img_data, model_prediction, model_confidence, user_feedback):
    """Save training data for future model improvement"""
    training_entry = {
        "timestamp": datetime.now().isoformat(),
        "image_url": img_data['url'],
        "image_title": img_data['title'],
        "subreddit": img_data['subreddit'],
        "subreddit_title": img_data['subreddit_title'],
        "subreddit_subscribers": img_data['subreddit_subscribers'],
        "reddit_score": img_data['score'],
        "reddit_author": img_data['author'],
        "model_prediction": model_prediction,
        "model_confidence": model_confidence,
        "user_feedback": user_feedback,  # "appropriate" or "inappropriate"
        "feedback_timestamp": datetime.now().isoformat()
    }
    
    # Save to training data file
    training_file = "training_data.json"
    try:
        # Load existing data
        try:
            with open(training_file, 'r') as f:
                training_data = json.load(f)
        except FileNotFoundError:
            training_data = []
        
        # Add new entry
        training_data.append(training_entry)
        
        # Save back to file
        with open(training_file, 'w') as f:
            json.dump(training_data, f, indent=2)
            
        return True
    except Exception as e:
        st.error(f"Error saving training data: {e}")
        return False

def display_result(prediction, confidence, decision, reason, css_class):
    """Display the moderation result with styling"""
    st.markdown(f"""
    <div class="result-box {css_class}">
        <h3>🔍 Moderation Result</h3>
        <p><strong>Decision:</strong> {decision}</p>
        <p><strong>Reason:</strong> {reason}</p>
        <p><strong>Prediction:</strong> {'✅ Appropriate' if prediction == 0 else '❌ Inappropriate'}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence bar
    st.markdown("### Confidence Level")
    confidence_percent = confidence * 100
    color = "#28a745" if prediction == 0 else "#dc3545"
    
    st.markdown(f"""
    <div class="confidence-bar">
        <div class="confidence-fill" style="width: {confidence_percent}%; background-color: {color};"></div>
    </div>
    <p style="text-align: center; font-weight: bold;">{confidence_percent:.1f}%</p>
    """, unsafe_allow_html=True)

def retrain_model_with_feedback(model, processor, device, training_data_file="training_data.json"):
    """Retrain the model with collected feedback data"""
    try:
        if not os.path.exists(training_data_file):
            return False, "No training data found"
        
        with open(training_data_file, 'r') as f:
            training_data = json.load(f)
        
        if len(training_data) < 10:  # Need at least 10 samples
            return False, f"Need at least 10 samples, currently have {len(training_data)}"
        
        # Prepare training data
        images = []
        labels = []
        
        for entry in training_data:
            try:
                # Load image from URL
                response = requests.get(entry['image_url'], timeout=10)
                if response.status_code == 200:
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    image = image.resize((224, 224), Image.Resampling.LANCZOS)
                    images.append(image)
                    
                    # Convert user feedback to label
                    # 0 = appropriate, 1 = inappropriate
                    label = 0 if entry['user_feedback'] == 'appropriate' else 1
                    labels.append(label)
            except Exception as e:
                continue  # Skip problematic images
        
        if len(images) < 10:
            return False, f"Could not load enough images, only {len(images)} valid samples"
        
        # Convert to tensors
        inputs = processor(images=images, return_tensors="pt", padding=True)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labels_tensor = labels_tensor.to(device)
        
        # Set up training
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop (few epochs for online learning)
        num_epochs = 3
        batch_size = 4
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            # Process in batches
            for i in range(0, len(images), batch_size):
                batch_inputs = {k: v[i:i+batch_size] for k, v in inputs.items()}
                batch_labels = labels_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(**batch_inputs)
                loss = criterion(outputs['logits'], batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        model.eval()
        
        # Save updated model
        torch.save(model.state_dict(), "clip-checkpoints/checkpoint-660/model_updated.pt")
        
        return True, f"Successfully retrained with {len(images)} samples"
        
    except Exception as e:
        return False, f"Training failed: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🛡️ Image Moderation AI</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Settings")
        
        # Confidence thresholds
        st.subheader("Confidence Thresholds")
        auto_flag_threshold = st.slider("Auto-Flag Threshold", 0.7, 0.95, 0.85, 0.05)
        flag_threshold = st.slider("Flag Threshold", 0.5, 0.8, 0.65, 0.05)
        approved_threshold = st.slider("Approved Threshold", 0.6, 0.9, 0.8, 0.05)
        
        st.markdown("---")
        
        # Model info
        st.subheader("🤖 Model Info")
        st.info("CLIP-based image classifier trained on Reddit data")
        
        # Session stats
        if 'prediction_count' not in st.session_state:
            st.session_state.prediction_count = 0
        st.metric("Images Processed", st.session_state.prediction_count)
    
    # Load model
    model, processor, device = load_model()
    
    # Create tabs
    tab1, tab2 = st.tabs(["📤 Upload Image", "🌐 Browse Reddit"])
    
    with tab1:
        # Main content for file upload
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📤 Upload Image")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Drag and drop an image here, or click to browse"
            )
            
            if uploaded_file is not None:
                                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                # Process button
                if st.button("🔍 Analyze Image", type="primary"):
                    with st.spinner("Analyzing image..."):
                        prediction, confidence = predict_image(image, model, processor, device)
                        
                        if prediction is not None:
                            # Get decision
                            decision, reason, css_class = get_moderation_decision(prediction, confidence)
                            
                            # Update session state
                            st.session_state.prediction_count += 1
                            
                            # Display result
                            with col2:
                                st.header("📋 Analysis Results")
                                display_result(prediction, confidence, decision, reason, css_class)
                                
                                # Log prediction
                                log_entry = {
                                    "timestamp": datetime.now().isoformat(),
                                    "filename": uploaded_file.name,
                                    "prediction": prediction,
                                    "confidence": confidence,
                                    "decision": decision,
                                    "reason": reason
                                }
                                
                                if 'prediction_log' not in st.session_state:
                                    st.session_state.prediction_log = []
                                st.session_state.prediction_log.append(log_entry)
                                
                                # Show action buttons
                                st.markdown("### 🎯 Actions")
                                if decision == "AUTO-FLAG":
                                    st.error("🚨 This image has been automatically flagged for removal!")
                                    st.button("Remove Image", type="secondary")
                                elif decision == "FLAG":
                                    st.warning("⚠️ This image has been flagged for review.")
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.button("Remove", type="secondary")
                                    with col_b:
                                        st.button("Approve", type="primary")
                                elif decision == "REVIEW":
                                    st.info("👀 This image needs human review.")
                                    col_a, col_b = st.columns(2)
                                    with col_a:
                                        st.button("Remove", type="secondary")
                                    with col_b:
                                        st.button("Approve", type="primary")
                                else:  # APPROVED
                                    st.success("✅ This image has been approved!")
    
    with tab2:
        st.header("🌐 Browse Reddit Images")
        
        # Create main layout: Search panel on left, Gallery on right
        search_col, gallery_col = st.columns([1, 2])
        
        with search_col:
            st.subheader("🔍 Search Settings")
            
            # Popular subreddits dropdown
            popular_subreddits = get_popular_subreddits()
            
            # Create tabs for different selection methods
            tab_method1, tab_method2, tab_method3 = st.tabs(["📋 Popular", "🔍 Search", "🎯 Train Model"])
            
            with tab_method1:
                st.write("Choose from popular subreddits:")
                
                # Group popular subreddits by category
                categories = {
                    "🐱 Animals & Pets": ["aww", "cats", "dogs", "puppies", "kittens"],
                    "🌍 Nature & Travel": ["EarthPorn", "nature", "travel", "photography", "space"],
                    "😄 Humor & Memes": ["funny", "memes", "wholesomememes", "MadeMeSmile"],
                    "🎨 Art & Design": ["Art", "architecture", "wallpapers", "dataisbeautiful"],
                    "🏃 Sports & Fitness": ["sports", "fitness", "bodybuilding", "basketball"],
                    "🎮 Gaming & Tech": ["gaming", "technology", "programming", "anime"],
                    "🍔 Food & Lifestyle": ["FoodPorn", "cooking", "fashion", "makeup"]
                }
                
                selected_subreddit = None
                for category, subs in categories.items():
                    st.write(f"**{category}**")
                    cols = st.columns(3)
                    for i, sub in enumerate(subs):
                        if cols[i % 3].button(sub, key=f"pop_{sub}"):
                            selected_subreddit = sub
                            st.session_state.selected_subreddit = sub
                            # Auto-fetch images when selecting from popular
                            with st.spinner(f"Fetching images from r/{sub}..."):
                                reddit_images = fetch_reddit_images(sub, 12)
                                if reddit_images:
                                    st.session_state.reddit_images = reddit_images
                                    st.success(f"✅ Fetched {len(reddit_images)} images from r/{sub}")
                                else:
                                    st.error(f"❌ No images found in r/{sub}")
                
                if selected_subreddit:
                    st.success(f"Selected: r/{selected_subreddit}")
            
            with tab_method2:
                st.write("Search for specific subreddits:")
                
                # Search input
                search_query = st.text_input(
                    "Search subreddits",
                    placeholder="e.g., 'cats', 'food', 'art'",
                    help="Type to search for subreddits"
                )
                
                if search_query and len(search_query) >= 2:
                    with st.spinner("Searching subreddits..."):
                        search_results = search_subreddits(search_query, 8)
                        
                        if search_results:
                            st.write("**Search Results:**")
                            for result in search_results:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"**r/{result['name']}** - {result['title']}")
                                    st.caption(f"{result['description']}")
                                with col2:
                                    if st.button(f"Select", key=f"search_{result['name']}"):
                                        selected_subreddit = result['name']
                                        st.session_state.selected_subreddit = result['name']
                                        # Auto-fetch images when selecting from search
                                        with st.spinner(f"Fetching images from r/{result['name']}..."):
                                            reddit_images = fetch_reddit_images(result['name'], 12)
                                            if reddit_images:
                                                st.session_state.reddit_images = reddit_images
                                                st.success(f"✅ Fetched {len(reddit_images)} images from r/{result['name']}")
                                            else:
                                                st.error(f"❌ No images found in r/{result['name']}")
                                        st.success(f"Selected: r/{result['name']}")
                        else:
                            st.info("No subreddits found. Try a different search term.")
            
            # Train Model tab
            with tab_method3:
                st.subheader("🎯 Rapid-Fire Model Training")
                st.write("Get random images from random subreddits and provide feedback to improve the model.")
                
                # Initialize session state
                if 'training_count' not in st.session_state:
                    st.session_state.training_count = 0
                if 'training_agreements' not in st.session_state:
                    st.session_state.training_agreements = 0
                
                # Training stats
                col_stats1, col_stats2 = st.columns(2)
                with col_stats1:
                    st.metric("Images Trained", st.session_state.training_count)
                with col_stats2:
                    if st.session_state.training_count > 0:
                        agreement_rate = (st.session_state.training_agreements / st.session_state.training_count) * 100
                        st.metric("Agreement Rate", f"{agreement_rate:.1f}%")
                    else:
                        st.metric("Agreement Rate", "0%")
                
                # Get random image button
                if st.button("🎲 Get Random Image", type="primary", use_container_width=True):
                    with st.spinner("Fetching random image..."):
                        random_img = fetch_random_image()
                        if random_img:
                            st.session_state.current_training_image = random_img
                            st.success(f"✅ Got image from r/{random_img['subreddit']}")
                        else:
                            st.error("❌ Could not fetch random image. Try again.")
                
                # Retrain model button
                if st.button("🧠 Retrain Model", type="secondary", use_container_width=True):
                    with st.spinner("Retraining model with your feedback..."):
                        success, message = retrain_model_with_feedback(model, processor, device)
                        if success:
                            st.success(f"✅ {message}")
                            # Reload the updated model
                            model.load_state_dict(torch.load("clip-checkpoints/checkpoint-660/model_updated.pt", map_location=device))
                            st.info("🔄 Model updated! New predictions will use your feedback.")
                        else:
                            st.warning(f"⚠️ {message}")
                
                # Show training data info
                if os.path.exists("training_data.json"):
                    with open("training_data.json", 'r') as f:
                        training_data = json.load(f)
                    st.info(f"📊 Training data: {len(training_data)} samples collected")
                    if len(training_data) >= 10:
                        st.success("✅ Ready for retraining! Click 'Retrain Model' to apply your feedback.")
                    else:
                        st.warning(f"⚠️ Need {10 - len(training_data)} more samples for retraining")

        with gallery_col:
            # Check if we're in Train Model tab and have a training image
            if 'current_training_image' in st.session_state:
                st.subheader("📸 Current Image")
                
                img_data = st.session_state.current_training_image
                
                # Display image
                st.image(img_data['image'], caption=img_data['title'], use_container_width=True)
                
                # Image info
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.write(f"**Subreddit:** r/{img_data['subreddit']}")
                    st.write(f"**Title:** {img_data['title'][:50]}...")
                with col_info2:
                    st.write(f"**Score:** {img_data['score']}")
                    st.write(f"**Subscribers:** {img_data['subreddit_subscribers']:,}")
                
                # Model analysis
                st.subheader("🤖 Model Analysis")
                if st.button("🔍 Analyze Image", type="secondary"):
                    with st.spinner("Analyzing..."):
                        prediction, confidence = predict_image(img_data['image'], model, processor, device)
                        
                        if prediction is not None:
                            decision, reason, css_class = get_moderation_decision(prediction, confidence)
                            
                            # Store analysis results
                            st.session_state.current_analysis = {
                                'prediction': prediction,
                                'confidence': confidence,
                                'decision': decision,
                                'reason': reason
                            }
                            
                            # Display result
                            st.markdown(f"""
                            <div class="result-box {css_class}">
                                <h4>Model Prediction</h4>
                                <p><strong>Decision:</strong> {decision}</p>
                                <p><strong>Confidence:</strong> {confidence:.3f}</p>
                                <p><strong>Prediction:</strong> {'✅ Appropriate' if prediction == 0 else '❌ Inappropriate'}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # User feedback (only show if model has analyzed)
                if 'current_analysis' in st.session_state:
                    st.subheader("👤 Your Feedback")
                    st.write("Do you agree with the model's assessment?")
                    
                    col_feedback1, col_feedback2 = st.columns(2)
                    
                    with col_feedback1:
                        if st.button("✅ Appropriate", type="primary", use_container_width=True):
                            user_feedback = "appropriate"
                            model_pred = st.session_state.current_analysis['prediction']
                            model_conf = st.session_state.current_analysis['confidence']
                            
                            # Save training data
                            if save_training_data(img_data, model_pred, model_conf, user_feedback):
                                st.session_state.training_count += 1
                                if model_pred == 0:  # Model said appropriate, user agrees
                                    st.session_state.training_agreements += 1
                                st.success("✅ Feedback saved! Getting next image...")
                                
                                # Check if we should auto-retrain (every 10 samples)
                                if os.path.exists("training_data.json"):
                                    with open("training_data.json", 'r') as f:
                                        training_data = json.load(f)
                                    if len(training_data) % 10 == 0 and len(training_data) >= 10:
                                        st.info("🔄 Auto-retraining model with your feedback...")
                                        success, message = retrain_model_with_feedback(model, processor, device)
                                        if success:
                                            model.load_state_dict(torch.load("clip-checkpoints/checkpoint-660/model_updated.pt", map_location=device))
                                            st.success("✅ Model automatically updated!")
                                    
                                    # Clear current image and get new one
                                    del st.session_state.current_training_image
                                    del st.session_state.current_analysis
                                    st.rerun()
                    
                    with col_feedback2:
                        if st.button("❌ Inappropriate", type="secondary", use_container_width=True):
                            user_feedback = "inappropriate"
                            model_pred = st.session_state.current_analysis['prediction']
                            model_conf = st.session_state.current_analysis['confidence']
                            
                            # Save training data
                            if save_training_data(img_data, model_pred, model_conf, user_feedback):
                                st.session_state.training_count += 1
                                if model_pred == 1:  # Model said inappropriate, user agrees
                                    st.session_state.training_agreements += 1
                                st.success("✅ Feedback saved! Getting next image...")
                                
                                # Check if we should auto-retrain (every 10 samples)
                                if os.path.exists("training_data.json"):
                                    with open("training_data.json", 'r') as f:
                                        training_data = json.load(f)
                                    if len(training_data) % 10 == 0 and len(training_data) >= 10:
                                        st.info("🔄 Auto-retraining model with your feedback...")
                                        success, message = retrain_model_with_feedback(model, processor, device)
                                        if success:
                                            model.load_state_dict(torch.load("clip-checkpoints/checkpoint-660/model_updated.pt", map_location=device))
                                            st.success("✅ Model automatically updated!")
                                    
                                    # Clear current image and get new one
                                    del st.session_state.current_training_image
                                    del st.session_state.current_analysis
                                    st.rerun()
            
            # Display Reddit images for other tabs
            elif 'reddit_images' in st.session_state and st.session_state.reddit_images:
                st.subheader("📸 Image Gallery")
                
                images = st.session_state.reddit_images
                
                # Debug info
                st.write(f"Found {len(images)} images to display")
                
                # Create a simple 2-column layout for the gallery
                for i in range(0, len(images), 2):
                    # Create 2 columns for each row
                    img_col1, img_col2 = st.columns(2)
                    
                    # First image in the row
                    if i < len(images):
                        img_data = images[i]
                        with img_col1:
                            with st.container():
                                # Ensure we have a valid PIL Image
                                if hasattr(img_data['image'], 'size'):
                                    st.image(
                                        img_data['image'], 
                                        caption=f"Score: {img_data['score']}", 
                                        use_container_width=True
                                    )
                                    
                                    # Title
                                    title = img_data['title'][:40] + "..." if len(img_data['title']) > 40 else img_data['title']
                                    st.caption(title)
                                    
                                    # Analyze button
                                    if st.button(f"🔍 Analyze", key=f"analyze_{i}", use_container_width=True):
                                        analyze_image(img_data, i, model, processor, device)
                                else:
                                    st.error(f"Invalid image data at index {i}")
                    
                    # Second image in the row
                    if i + 1 < len(images):
                        img_data = images[i + 1]
                        with img_col2:
                            with st.container():
                                # Ensure we have a valid PIL Image
                                if hasattr(img_data['image'], 'size'):
                                    st.image(
                                        img_data['image'], 
                                        caption=f"Score: {img_data['score']}", 
                                        use_container_width=True
                                    )
                                    
                                    # Title
                                    title = img_data['title'][:40] + "..." if len(img_data['title']) > 40 else img_data['title']
                                    st.caption(title)
                                    
                                    # Analyze button
                                    if st.button(f"🔍 Analyze", key=f"analyze_{i+1}", use_container_width=True):
                                        analyze_image(img_data, i + 1, model, processor, device)
                                else:
                                    st.error(f"Invalid image data at index {i + 1}")
            else:
                st.subheader("📸 Image Gallery")
                st.info("👆 Select a subreddit from the left panel to start browsing")
    
    # History section
    if 'prediction_log' in st.session_state and st.session_state.prediction_log:
        st.markdown("---")
        st.header("📜 Analysis History")
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(st.session_state.prediction_log)
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%H:%M:%S')
        df['prediction'] = df['prediction'].map({0: 'Appropriate', 1: 'Inappropriate'})
        df['confidence'] = df['confidence'].apply(lambda x: f"{x:.3f}")
        
        # Display table
        st.dataframe(
            df[['timestamp', 'filename', 'prediction', 'confidence', 'decision']],
            use_container_width=True
        )
        
        # Download log
        if st.button("📥 Download Analysis Log"):
            json_str = json.dumps(st.session_state.prediction_log, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"moderation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main() 