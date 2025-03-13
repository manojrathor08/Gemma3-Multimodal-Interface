import gradio as gr
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import torch

# Disable certain CUDA SDP optimizations (to avoid potential issues)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)

# Load the multimodal model and processor
model_id = "google/gemma-3-4b-it"
model = Gemma3ForConditionalGeneration.from_pretrained(
    model_id, device_map="auto"
).eval()
processor = AutoProcessor.from_pretrained(model_id)

def multimodal_query(image, prompt_text, mode="caption"):
    """
    Generates a response from the model based on the image and text prompt.
    
    Args:
        image (PIL.Image): The input image.
        prompt_text (str): The prompt to guide the output.
        mode (str): "caption" to get an image caption/summary or "chat" for a chatbot response.
        
    Returns:
        str: The model's generated text.
    """
    # Set system prompt based on mode
    if mode == "caption":
        system_prompt = "You are an assistant that provides detailed image captions."
    else:
        system_prompt = "You are a helpful assistant."
    
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    # Process the messages: format, tokenize, and move inputs to the model's device
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]
    
    # Generate output tokens
    with torch.inference_mode():
        generated_tokens = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        # Slice to get only the generated part (exclude the input prompt tokens)
        generated_tokens = generated_tokens[0][input_len:]
    
    # Decode tokens into text
    response = processor.decode(generated_tokens, skip_special_tokens=True)
    return response

def multimodal_interface(image, prompt, mode):
    """
    Gradio wrapper function to ensure the image is a PIL image and then call multimodal_query.
    """
    # If Gradio returns a file path (string), open it as an image.
    if isinstance(image, str):
        image = Image.open(image)
    return multimodal_query(image, prompt, mode)

# Create the Gradio interface
interface = gr.Interface(
    fn=multimodal_interface,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Textbox(lines=2, placeholder="Enter your prompt...", label="Prompt"),
        gr.Radio(choices=["caption", "chat"], label="Mode", value="caption")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Gemma-3 Multimodal Interface",
    description="Provide an image and a prompt to either get a caption/summary or interact as a chatbot.",
)

interface.launch()
