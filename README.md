# 🖼️ CLIP Image Moderation System

An intelligent image moderation system built with CLIP (Contrastive Language-Image Pre-training) and Streamlit. This system can classify images as appropriate or inappropriate, and learns from user feedback to improve its accuracy over time.

## ✨ Features

- **🤖 AI-Powered Moderation**: Uses CLIP model for intelligent image classification
- **🧠 Learning Capability**: Model learns from user feedback to improve accuracy
- **📱 Interactive Web Interface**: Beautiful Streamlit app with drag-and-drop functionality
- **🔍 Reddit Integration**: Browse and analyze images from Reddit subreddits
- **📊 Real-Time Training**: Automatic model retraining every 10 feedback samples
- **📈 Performance Tracking**: Monitor model accuracy and training progress
- **🎯 Customizable Thresholds**: Set confidence levels for different moderation actions

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
- Reddit API credentials (for Reddit integration)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd clip-timelines-moderation
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your Reddit API credentials:
   ```
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=your_user_agent
   ```

3. **Create virtual environment**
   ```bash
   python -m venv clip-env
   ```

4. **Activate virtual environment**
   - **Windows:**
     ```bash
     clip-env\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source clip-env/bin/activate
     ```

5. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install transformers datasets scikit-learn matplotlib tqdm
   pip install huggingface_hub praw accelerate streamlit safetensors
   ```

6. **Train the initial model**
   ```bash
   python train.py
   ```

7. **Launch the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📸 Screenshots

### Main Interface
<img width="1917" height="826" alt="image" src="https://github.com/user-attachments/assets/f47318e8-defb-4e40-8a0f-5801175ad70b" />

### Upload Image Tab
<img width="1919" height="816" alt="image" src="https://github.com/user-attachments/assets/c5f4d096-6f18-4757-8eef-959e6cb71a3c" />

### Reddit Browsing Tab
<img width="1919" height="829" alt="image" src="https://github.com/user-attachments/assets/bf8fe075-6dc1-4db7-8381-4e41305a4b7c" />
<img width="1919" height="817" alt="image" src="https://github.com/user-attachments/assets/04fe5ef8-158a-4e61-aab3-e103904b916c" />

### Train Model Tab
<img width="1919" height="818" alt="image" src="https://github.com/user-attachments/assets/3c2adf89-fcdf-4ebd-8b9a-de21b7511c09" />

### Model Analysis Results
<img width="1919" height="819" alt="image" src="https://github.com/user-attachments/assets/79f89e98-8bb2-4524-83cf-7480648c2aaf" />
<img width="1096" height="675" alt="image" src="https://github.com/user-attachments/assets/c7a031e7-94ab-4fd1-9c77-426a2379e74c" />

## 🎯 How to Use

### 1. Upload and Analyze Images
- Go to the **"Upload Image"** tab
- Drag and drop an image or click to browse
- View the model's prediction and confidence score
- See the moderation decision based on your thresholds

### 2. Browse Reddit Images
- Navigate to the **"Browse Reddit"** tab
- Choose from popular subreddits or search for specific ones
- Browse images in a gallery format
- Click any image to analyze it

### 3. Train the Model
- Go to the **"Train Model"** tab
- Click **"Get Random Image"** to fetch a random image from Reddit
- Review the model's prediction
- Provide feedback: **"Appropriate"** or **"Inappropriate"**
- The model automatically retrains every 10 samples

### 4. Monitor Performance
- View training statistics in the sidebar
- Track agreement rate between model and user decisions
- Download prediction logs for analysis

## ⚙️ Configuration

### Moderation Thresholds
Adjust these values in the sidebar:
- **Auto-Flag**: Images with high confidence of being inappropriate
- **Flag for Review**: Images requiring human review
- **Approved**: Images with high confidence of being appropriate

### Training Parameters
- **Minimum samples**: 10 (required for retraining)
- **Auto-retrain frequency**: Every 10 samples
- **Learning rate**: 1e-5 (optimized for online learning)

## 📁 Project Structure

```
clip-timelines-moderation/
├── streamlit_app.py          # Main Streamlit application
├── train.py                  # Initial model training script
├── scrape_reddit.py          # Reddit data collection
├── test_model.py             # Command-line model testing
├── load_model.py             # Model loading utilities
├── test_learning.py          # Learning functionality testing
├── .env                      # Environment variables (not in git)
├── .env.example              # Environment variables template
├── .gitignore               # Git ignore rules
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── dataset/                 # Training dataset
│   ├── appropriate/         # Appropriate images
│   └── inappropriate/       # Inappropriate images
├── clip-checkpoints/        # Model checkpoints
├── test_samples/            # Test images
└── screenshots/             # App screenshots (add your own)
```

## 🔧 Environment Variables

Create a `.env` file with the following variables:

```env
# Reddit API Configuration
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_user_agent_string

# Optional: Model Configuration
MODEL_DEVICE=cpu  # or cuda for GPU
BATCH_SIZE=4
LEARNING_RATE=1e-5
```

### Getting Reddit API Credentials

1. Go to [Reddit Apps](https://www.reddit.com/prefs/apps)
2. Click **"Create App"** or **"Create Another App"**
3. Fill in the details:
   - **Name**: Your app name
   - **Type**: Script
   - **Description**: Image moderation system
   - **About URL**: Your GitHub repo URL
   - **Redirect URI**: http://localhost:8080
4. Copy the **Client ID** (under your app name) and **Client Secret**

## 🧠 Model Training

### Initial Training
The model starts with a pre-trained CLIP model and fine-tunes it on your dataset:

```bash
python train.py
```

### Online Learning
The model continuously learns from user feedback:
- Every 10 feedback samples triggers automatic retraining
- Manual retraining available via "Retrain Model" button
- Training data saved to `training_data.json`

### Expected Learning Timeline
- **10-50 samples**: Initial adaptation
- **50-100 samples**: Noticeable improvement
- **100-500 samples**: Significant improvement
- **500+ samples**: Highly reliable performance

## 🐛 Troubleshooting

### Common Issues

1. **"Python was not found"**
   - Ensure Python 3.13+ is installed and in PATH
   - Use full path: `C:\Users\username\AppData\Local\Programs\Python\Python313\python.exe`

2. **"Module not found"**
   - Activate virtual environment: `clip-env\Scripts\activate`
   - Install missing packages: `pip install package_name`

3. **Reddit API errors**
   - Verify API credentials in `.env` file
   - Check Reddit API rate limits
   - Ensure subreddit is public and accessible

4. **Model loading errors**
   - Run `python train.py` to generate initial model
   - Check if `clip-checkpoints/` directory exists

5. **Memory issues**
   - Reduce `BATCH_SIZE` in training parameters
   - Use CPU instead of GPU: `MODEL_DEVICE=cpu`

### Performance Tips

- **GPU Usage**: Set `MODEL_DEVICE=cuda` for faster training
- **Batch Size**: Increase `BATCH_SIZE` if you have sufficient memory
- **Regular Training**: Provide feedback consistently for best results
- **Diverse Data**: Include both appropriate and inappropriate examples

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI CLIP**: Base model for image understanding
- **Hugging Face**: Transformers library and model hosting
- **Streamlit**: Web application framework
- **Reddit**: API for image data collection

## 📞 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section above
2. Review the [Issues](https://github.com/yourusername/clip-timelines-moderation/issues) page
3. Create a new issue with detailed information

---

**Happy moderating! 🎉** 
