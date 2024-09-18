import gradio as gr
import torch
from .src.finetune import train_and_validate
import json
from datasets import load_dataset

def finetune_model(model_name, output_dir, dataset_name, image_column, text_column, user_text, num_accumulation_steps, eval_steps, max_steps, train_batch_size, val_batch_size, train_select_start, train_select_end, val_select_start, val_select_end, train_field, val_field, device, min_pixel, max_pixel, image_factor):
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    
    # Call the train_and_validate function with the provided parameters
    train_and_validate(
        model_name=model_name,
        output_dir=output_dir,
        dataset_name=dataset_name,
        image_column=image_column,
        text_column=text_column,
        device=device,
        user_text=user_text,
        num_accumulation_steps=num_accumulation_steps,
        eval_steps=eval_steps,
        max_steps=max_steps,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        train_select_start=train_select_start,
        train_select_end=train_select_end,
        val_select_start=val_select_start,
        val_select_end=val_select_end,
        train_field=train_field,
        val_field=val_field,
        min_pixel=min_pixel,
        max_pixel=max_pixel,
        image_factor=image_factor
    )
    
    return f"Training completed. Model saved in {output_dir}"

# Create the Gradio interface
def load_dataset_sample(dataset_name):
    dataset = load_dataset(dataset_name, streaming=True)
    sample = list(dataset['train'].take(5))
    return sample, list(sample[0].keys())

def update_fields(dataset_name):
    sample, fields = load_dataset_sample(dataset_name)
    return gr.Dropdown(choices=fields, label="Image Column"), gr.Dropdown(choices=fields, label="Text Column"), gr.DataFrame(value=[list(s.values()) for s in sample], headers=list(sample[0].keys()))

def preview_message_structure(dataset_name, image_column, text_column, user_text):
    sample, _ = load_dataset_sample(dataset_name)
    image = sample[0][image_column]
    assistant_text = sample[0][text_column]
    message_structure = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "Image data (not shown)"},
                    {"type": "text", "text": user_text}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": assistant_text}
                ]
            }
        ]
    }
    return json.dumps(message_structure, indent=2)

with gr.Blocks() as iface:
    gr.Markdown("# Qwen2-VL Model Finetuning")
    gr.Markdown("Finetune the Qwen2-VL model on a specified dataset.")
    
    with gr.Row():
        dataset_name = gr.Textbox(label="Dataset Name")
        load_button = gr.Button("Load Dataset")
    
    with gr.Row():
        image_column = gr.Dropdown(label="Image Column")
        text_column = gr.Dropdown(label="Text Column")
        train_field = gr.Dropdown(label="Train Field", choices=["train", "validation", "test"])
        val_field = gr.Dropdown(label="Validation Field", choices=["train", "validation", "test"])
    
    sample_data = gr.DataFrame(label="Sample Data")
    
    load_button.click(update_fields, inputs=[dataset_name], outputs=[image_column, text_column, sample_data])

    model_name = gr.Dropdown(
        label="Model Name",
        choices=["Qwen/Qwen2-VL-2B-Instruct", "Qwen/Qwen2-VL-7B-Instruct"],
        value="Qwen/Qwen2-VL-2B-Instruct"
    )
    
    user_text = gr.Textbox(label="User Instructions", value="Convert this image to text")
    preview_button = gr.Button("Preview Message Structure")
    message_preview = gr.JSON(label="Message Structure Preview")
    
    preview_button.click(preview_message_structure, inputs=[dataset_name, image_column, text_column, user_text], outputs=[message_preview])
    
    output_dir = gr.Textbox(label="Output Directory")
    with gr.Row():
        with gr.Column():
            num_accumulation_steps = gr.Number(label="Number of Accumulation Steps", value=2)
            eval_steps = gr.Number(label="Evaluation Steps", value=10000)
            max_steps = gr.Number(label="Max Steps", value=100000)
        with gr.Column():
            train_batch_size = gr.Number(label="Training Batch Size", value=1)
            val_batch_size = gr.Number(label="Validation Batch Size", value=1)
        with gr.Column():
            train_select_start = gr.Number(label="Training Select Start", value=0)
            train_select_end = gr.Number(label="Training Select End", value=100000)
        with gr.Column():
            val_select_start = gr.Number(label="Validation Select Start", value=0)
            val_select_end = gr.Number(label="Validation Select End", value=10000)
    with gr.Row():
        with gr.Column():
            device = gr.Dropdown(label="Device", choices=["cuda", "cpu", "mps"], value="cuda")
            min_pixel = gr.Number(label="Minimum Pixel Size", value=256)
        with gr.Column():
            max_pixel = gr.Number(label="Maximum Pixel Size", value=384)
            image_factor = gr.Number(label="Image Factor", value=28)
    finetune_button = gr.Button("Start Finetuning")
    result = gr.Textbox(label="Result")
    
    finetune_button.click(
        finetune_model,
            inputs=[model_name,output_dir, dataset_name, image_column, text_column, user_text, num_accumulation_steps, eval_steps, max_steps, train_batch_size, val_batch_size, train_select_start, train_select_end, val_select_start, val_select_end, train_field, val_field],
            outputs=[result]
    )

# Launch the app
if __name__ == "__main__":
    iface.launch(server_port=8083, server_name="compute-50-01")
