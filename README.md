# Gemma3-Multimodal-Interface

A simple Gradio interface for interacting with Google's Gemma-3 multimodal model through Hugging Face.

## Overview

This application provides a user-friendly web interface to interact with Google's Gemma-3 multimodal model (gemma-3-4b-it). It supports two modes of operation:
- **Caption Mode**: Upload an image and provide a prompt to get detailed image captions or visual analysis
- **Chat Mode**: Have a text-only conversation with the model

![Gemma3 Multimodal Interface](images/image.png)
*Screenshot of the Gemma3 Multimodal Interface*

## Features

- Interactive web interface built with Gradio
- Support for image captioning and analysis
- Text-only chat capabilities
- Optimized for efficient inference on CUDA-enabled devices

## Prerequisites

- Python 3.x
- PyTorch with CUDA support (recommended)
- Hugging Face account with access to the Gemma-3 model
- Hugging Face API token

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Gemma3-Multimodal-Interface.git
cd Gemma3-Multimodal-Interface
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your Hugging Face token as an environment variable:
```bash
export HF_TOKEN="your_hugging_face_token"
```

## Usage

Run the application:
```bash
python app.py
```

This will launch a Gradio interface accessible at `http://localhost:7860` and also generate a temporary public URL.

### Using the Interface

1. **Caption Mode**:
   - Upload an image
   - Enter a prompt (e.g., "Describe this image in detail")
   - Select "caption" mode
   - Click "Submit"

2. **Chat Mode**:
   - Enter your text prompt
   - Select "chat" mode
   - Click "Submit"

## Technical Details

The application:
- Uses the `google/gemma-3-4b-it` model from Hugging Face
- Implements memory-efficient inference with PyTorch
- Disables certain CUDA optimizations for better compatibility
- Limits generation to 100 tokens per response

## Requirements

```
gradio
transformers
torch
pillow
```

## License

MIT License

## Acknowledgments

- Google for the Gemma-3 model
- Hugging Face for model hosting and transformers library
- Gradio for the web interface framework
