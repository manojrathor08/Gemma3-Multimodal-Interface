import gradio as gr
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import torch
import os
import torch
torch.cuda.empty_cache()

# Retrieve your Hugging Face token from environment variables
hf_token = os.environ.get("HF_TOKEN")

# Disable certain CUDA SDP optimizations to avoid potential issues.
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# Load the multimodal model and processor.
model_id = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto", use_auth_token=hf_token
).eval()
processor = AutoProcessor.from_pretrained(model_id)

def multimodal_query(image, prompt_text, mode="caption"):
    """
    Generates a response from the model based on the mode.
    
    For "caption" mode, it expects both an image and a text prompt.
    For "chat" mode, it expects only a text prompt.
    
    Args:
        image (PIL.Image or None): The input image (ignored in chat mode).
        prompt_text (str): The text prompt.
        mode (str): "caption" for image caption/summary, "chat" for a text-only conversation.
        
    Returns:
        str: The generated response.
    """
    if mode == "caption":
        system_prompt = "You are an assistant that provides detailed image captions."
        # For captioning, include both image and text in the user message.
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text}
        ]
    else:  # chat mode
        system_prompt = "You are a helpful assistant."
        # For chat, only include the text prompt.
        user_content = [{"type": "text", "text": prompt_text}]
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": user_content
        }
    ]
    
    # Process the messages: format, tokenize, and move inputs to the model's device.
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]
    
    # Generate output tokens.
    with torch.inference_mode():
        generated_tokens = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        # Extract only the generated tokens (remove prompt tokens).
        generated_tokens = generated_tokens[0][input_len:]
    
    # Decode tokens into human-readable text.
    response = processor.decode(generated_tokens, skip_special_tokens=True)
    return response

def multimodal_interface(image, prompt, mode):
    """
    Gradio wrapper to ensure the image is a PIL image when provided.
    
    In chat mode, the image parameter is ignored.
    """
    # For caption mode, ensure we have a valid image.
    if mode == "caption":
        if isinstance(image, str):
            image = Image.open(image)
        elif image is None:
            return "Please provide an image for caption mode."
    else:
        image = None  # Ignore image in chat mode
    
    return multimodal_query(image, prompt, mode)

# Create the Gradio interface.
interface = gr.Interface(
    fn=multimodal_interface,
    inputs=[
        gr.Image(type="pil", label="Input Image (Only for Caption Mode)"),
        gr.Textbox(lines=2, placeholder="Enter your prompt...", label="Prompt"),
        gr.Radio(choices=["caption", "chat"], label="Mode", value="caption")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Gemma-3 Multimodal Interface",
    description="Select 'caption' to get an image caption/summary (requires an image) or 'chat' for a text-only conversation.",
)

interface.launch(share=True)
