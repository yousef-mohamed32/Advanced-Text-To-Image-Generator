"""
Flask Web Application for Text-to-Image Generation
Optimized for Lightning.ai deployment
"""

from flask import Flask, request, jsonify, render_template_string
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline.pipeline import GenerativeAIPipeline
from src.utils.config import Config
from src.utils.image_utils import pil_to_base64, enhance_prompt

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request

# Global pipeline instance (lazy loading)
pipeline = None


def get_pipeline():
    """Lazy load the pipeline to avoid startup delays"""
    global pipeline
    if pipeline is None:
        print("🔄 Initializing AI Pipeline...")
        Config.ensure_directories()
        pipeline = GenerativeAIPipeline(model_name=Config.MODEL_NAME)
        print("✅ Pipeline initialized!")
    return pipeline


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Text-to-Image Generator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .input-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 16px;
            font-family: inherit;
            resize: vertical;
            transition: border-color 0.3s;
        }
        
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .settings {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .setting-item {
            display: flex;
            flex-direction: column;
        }
        
        select, input[type="number"] {
            padding: 10px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        select:focus, input[type="number"]:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .generate-btn {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 18px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .generate-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        .generate-btn:active {
            transform: translateY(0);
        }
        
        .generate-btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        
        #result {
            margin-top: 30px;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .result-image {
            width: 100%;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-top: 20px;
        }
        
        .metrics {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }
        
        .metrics h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        
        .examples {
            margin-top: 20px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .examples h3 {
            color: #333;
            margin-bottom: 15px;
        }
        
        .example-chip {
            display: inline-block;
            background: white;
            padding: 8px 15px;
            margin: 5px;
            border-radius: 20px;
            cursor: pointer;
            border: 2px solid #e0e0e0;
            transition: all 0.3s;
            font-size: 14px;
        }
        
        .example-chip:hover {
            border-color: #667eea;
            background: #667eea;
            color: white;
        }
        
        .download-btn {
            display: inline-block;
            margin-top: 15px;
            padding: 12px 30px;
            background: #28a745;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            transition: background 0.3s;
        }
        
        .download-btn:hover {
            background: #218838;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 AI Image Generator</h1>
            <p>Transform your words into stunning visuals with AI</p>
        </div>
        
        <div class="content">
            <div class="input-group">
                <label for="prompt">✍️ Describe your image:</label>
                <textarea 
                    id="prompt" 
                    rows="4" 
                    placeholder="Example: A serene mountain landscape at sunset with a crystal-clear lake reflecting the colorful sky"
                ></textarea>
            </div>
            
            <div class="settings">
                <div class="setting-item">
                    <label for="quality">🎯 Quality:</label>
                    <select id="quality">
                        <option value="high">High Quality (50 steps)</option>
                        <option value="medium" selected>Medium Quality (30 steps)</option>
                        <option value="low">Fast (20 steps)</option>
                    </select>
                </div>
                
                <div class="setting-item">
                    <label for="size">📐 Size:</label>
                    <select id="size">
                        <option value="512">512x512 (Fast)</option>
                        <option value="768" selected>768x768 (Recommended)</option>
                        <option value="1024">1024x1024 (Slow)</option>
                    </select>
                </div>
            </div>
            
            <button class="generate-btn" onclick="generateImage()">
                🚀 Generate Image
            </button>
            
            <div class="examples">
                <h3>💡 Try these examples:</h3>
                <span class="example-chip" onclick="setPrompt('A futuristic cyberpunk city at night with neon lights')">Cyberpunk City</span>
                <span class="example-chip" onclick="setPrompt('A magical forest with glowing mushrooms and fireflies')">Magical Forest</span>
                <span class="example-chip" onclick="setPrompt('A cute robot reading a book in a cozy library')">Robot Reading</span>
                <span class="example-chip" onclick="setPrompt('An astronaut riding a horse on Mars')">Space Adventure</span>
                <span class="example-chip" onclick="setPrompt('A steampunk airship floating above clouds')">Steampunk Ship</span>
            </div>
            
            <div id="result"></div>
        </div>
    </div>

    <script>
        let currentImageData = null;
        
        function setPrompt(text) {
            document.getElementById('prompt').value = text;
        }
        
        async function generateImage() {
            const prompt = document.getElementById('prompt').value.trim();
            const quality = document.getElementById('quality').value;
            const size = parseInt(document.getElementById('size').value);
            const resultDiv = document.getElementById('result');
            const generateBtn = document.querySelector('.generate-btn');

            if (!prompt) {
                resultDiv.innerHTML = '<div class="error">⚠️ Please enter a description for your image!</div>';
                return;
            }

            // Disable button and show loading
            generateBtn.disabled = true;
            generateBtn.textContent = '⏳ Generating...';
            
            resultDiv.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>🎨 Creating your masterpiece...</p>
                    <p style="color: #666; font-size: 14px; margin-top: 10px;">This may take 1-2 minutes depending on quality settings</p>
                </div>
            `;

            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        prompt: prompt,
                        quality: quality,
                        width: size,
                        height: size
                    })
                });

                const data = await response.json();

                if (data.error) {
                    resultDiv.innerHTML = `<div class="error">❌ Error: ${data.error}</div>`;
                } else {
                    currentImageData = data.image;
                    resultDiv.innerHTML = `
                        <h3 style="color: #667eea; margin-bottom: 15px;">✨ Generated Successfully!</h3>
                        <img src="data:image/png;base64,${data.image}" alt="Generated Image" class="result-image">
                        <a href="data:image/png;base64,${data.image}" download="ai_generated_${Date.now()}.png" class="download-btn">
                            📥 Download Image
                        </a>
                        <div class="metrics">
                            <h3>📊 Generation Info</h3>
                            <p><strong>Prompt:</strong> ${data.prompt}</p>
                            <p><strong>Size:</strong> ${data.width}x${data.height}</p>
                            <p><strong>Steps:</strong> ${data.steps}</p>
                            <p style="margin-top: 10px; color: #666;">
                                ✓ Using attention mechanisms for text-image alignment
                            </p>
                        </div>
                    `;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">❌ Error: ${error.message}</div>`;
            } finally {
                generateBtn.disabled = false;
                generateBtn.textContent = '🚀 Generate Image';
            }
        }
        
        // Allow Enter key to generate (with Shift+Enter for new line)
        document.getElementById('prompt').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                generateImage();
            }
        });
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    """Render the main page"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/health')
def health():
    """Health check endpoint for Lightning.ai"""
    return jsonify({'status': 'healthy', 'service': 'text-to-image-generator'})


@app.route('/generate', methods=['POST'])
def generate_image():
    """Generate image from text prompt"""
    try:
        data = request.json
        prompt = data.get('prompt', '').strip()
        quality = data.get('quality', 'medium')
        width = data.get('width', Config.DEFAULT_WIDTH)
        height = data.get('height', Config.DEFAULT_HEIGHT)

        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        # Get quality settings
        steps_map = {
            'high': Config.HIGH_QUALITY_STEPS,
            'medium': Config.MEDIUM_QUALITY_STEPS,
            'low': Config.LOW_QUALITY_STEPS
        }
        num_steps = steps_map.get(quality, Config.MEDIUM_QUALITY_STEPS)

        # Enhance prompt for better quality
        enhanced_prompt = enhance_prompt(prompt, quality_level=quality)

        # Get pipeline and generate
        pipe = get_pipeline()
        image = pipe.system.generate(
            enhanced_prompt,
            num_inference_steps=num_steps,
            guidance_scale=Config.DEFAULT_GUIDANCE_SCALE,
            width=width,
            height=height
        )

        # Convert to base64
        img_base64 = pil_to_base64(image)

        return jsonify({
            'image': img_base64,
            'prompt': prompt,
            'enhanced_prompt': enhanced_prompt,
            'width': width,
            'height': height,
            'steps': num_steps,
            'quality': quality
        })

    except Exception as e:
        print(f"Error generating image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch-generate', methods=['POST'])
def batch_generate():
    """Generate multiple images from multiple prompts"""
    try:
        data = request.json
        prompts = data.get('prompts', [])
        
        if not prompts or not isinstance(prompts, list):
            return jsonify({'error': 'Invalid prompts list'}), 400
        
        if len(prompts) > 5:
            return jsonify({'error': 'Maximum 5 prompts allowed per batch'}), 400

        pipe = get_pipeline()
        results = []
        
        for prompt in prompts:
            enhanced = enhance_prompt(prompt)
            image = pipe.system.generate(enhanced, num_inference_steps=20)
            img_base64 = pil_to_base64(image)
            results.append({
                'prompt': prompt,
                'image': img_base64
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Ensure directories exist
    Config.ensure_directories()
    
    # Run Flask app
    port = int(os.environ.get('PORT', Config.FLASK_PORT))
    print(f"\n{'='*70}")
    print(f"🚀 Starting AI Text-to-Image Generator")
    print(f"{'='*70}")
    print(f"🌐 Server running on http://0.0.0.0:{port}")
    print(f"📱 Lightning.ai will expose this automatically")
    print(f"📍 Look for 'Open' button or port {port} in Lightning.ai UI")
    print(f"{'='*70}\n")
    
    app.run(
        host='0.0.0.0',  # CRITICAL: Must bind to 0.0.0.0 for Lightning.ai
        port=port,
        debug=False,  # Disable debug mode for Lightning.ai
        threaded=True  # Enable threading for better performance
    )
