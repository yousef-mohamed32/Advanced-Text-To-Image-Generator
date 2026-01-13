# 🎨 AI Text-to-Image Generator

A production-ready text-to-image generation system powered by Stable Diffusion with enhanced attention mechanisms, deployed on Lightning.ai.

## 🌟 Features

- **Advanced AI Models**: Built on Stable Diffusion v1.5 with custom attention mechanisms
- **High-Quality Generation**: Multi-head attention for precise text-image alignment
- **Quality Metrics**: Inception Score and FID evaluation
- **Web Interface**: Beautiful Flask-based UI for easy interaction
- **Scalable Deployment**: Optimized for Lightning.ai cloud platform

## 📁 Project Structure

```
text-to-image-generator/
│
├── src/
│   ├── models/
│   │   ├── attention.py          # Cross-attention & self-attention layers
│   │   ├── unet.py               # Enhanced UNet architecture
│   │   └── evaluator.py          # Image quality metrics
│   │
│   ├── pipeline/
│   │   ├── generator.py          # Core generation system
│   │   └── pipeline.py           # End-to-end pipeline
│   │
│   └── utils/
│       ├── config.py             # Configuration settings
│       └── image_utils.py        # Image processing utilities
│
├── app.py                         # Flask web application
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd text-to-image-generator
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:7860`

### Lightning.ai Deployment

1. **Create a new Lightning Studio**
   - Go to [Lightning.ai](https://lightning.ai)
   - Click "New Studio"
   - Select GPU instance (T4 or better recommended)

2. **Upload your code**
   ```bash
   # In Lightning Studio terminal
   git clone <your-repo-url>
   cd text-to-image-generator
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   python app.py
   ```

5. **Expose the port**
   - Lightning.ai will automatically expose port 7860
   - Click on the "Open App" button in the Studio

## 🎯 Usage

### Web Interface

1. Enter your text description in the prompt box
2. Select quality level (High/Medium/Fast)
3. Choose image size (512x512, 768x768, or 1024x1024)
4. Click "Generate Image"
5. Download your generated image

### API Endpoints

#### Generate Single Image
```bash
curl -X POST http://localhost:7860/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape at sunset",
    "quality": "high",
    "width": 768,
    "height": 768
  }'
```

#### Health Check
```bash
curl http://localhost:7860/health
```

#### Batch Generation
```bash
curl -X POST http://localhost:7860/batch-generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      "A cyberpunk city at night",
      "A magical forest with glowing mushrooms"
    ]
  }'
```

## ⚙️ Configuration

Edit `src/utils/config.py` to customize:

- Model settings (model name, device)
- Generation parameters (steps, guidance scale)
- Image dimensions
- Output directories
- Flask server settings

## 🧠 Model Architecture

### Attention Mechanisms

1. **Cross-Attention**: Aligns text features with image features
   - Multi-head attention (8 heads)
   - Layer normalization
   - Feed-forward network

2. **Self-Attention**: Captures long-range dependencies
   - Multi-head attention (8 heads)
   - Residual connections

3. **Enhanced UNet**: Custom architecture with attention blocks
   - Down-sampling with cross-attention
   - Up-sampling with cross-attention
   - Skip connections

## 📊 Quality Metrics

The system includes built-in evaluation metrics:

- **Inception Score (IS)**: Measures quality and diversity
- **Fréchet Inception Distance (FID)**: Measures similarity to real images

## 🔧 Troubleshooting

### CUDA Out of Memory
- Reduce image size (use 512x512)
- Lower quality setting
- Reduce number of inference steps

### Slow Generation
- Use GPU instead of CPU
- Enable attention slicing (already enabled for CUDA)
- Use lower quality settings for faster results

### Model Download Issues
- Ensure stable internet connection
- Models are cached in `~/.cache/huggingface/`
- First run will take longer to download models

## 📝 Development

### Adding Custom Models

1. Create new model in `src/models/`
2. Import in `src/pipeline/generator.py`
3. Add initialization in `TextToImageGenerator.__init__()`

### Adding New Features

1. Update configuration in `src/utils/config.py`
2. Add utility functions in `src/utils/`
3. Update Flask routes in `app.py`

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - Base model
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - Diffusion library
- [Lightning.ai](https://lightning.ai) - Cloud deployment platform

## 🚀 Roadmap

- [ ] Add more pre-trained models
- [ ] Implement image-to-image generation
- [ ] Add style transfer capabilities
- [ ] Support for custom fine-tuned models
- [ ] Batch processing optimization
- [ ] Advanced prompt engineering features

---
