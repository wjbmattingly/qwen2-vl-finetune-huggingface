# Fine-tuning Qwen2-VL: A Powerful Vision-Language Model for Handwritten Text Recognition

## Introduction

In the rapidly evolving field of AI and computer vision, the Qwen2-VL model stands out as a game-changer. Developed by the Qwen team, this state-of-the-art vision-language model offers unprecedented capabilities in understanding and processing visual information, including handwritten text. Today, we'll explore how to fine-tune Qwen2-VL for specific tasks, with a focus on its potential for Handwritten Text Recognition (HTR).

## Why Qwen2-VL Matters

Qwen2-VL represents a significant leap forward in multimodal AI. It can understand images of various resolutions and aspect ratios, process videos over 20 minutes long, and even operate mobile devices and robots based on visual input and text instructions. For HTR specifically, Qwen2-VL's ability to handle complex visual information makes it one of the best models available, rivaling even closed-source alternatives.

Key features that make Qwen2-VL exceptional for HTR:

1. State-of-the-art performance on visual understanding benchmarks
2. Ability to handle arbitrary image resolutions
3. Multilingual support, including understanding text in images across various languages
4. Advanced architecture with Naive Dynamic Resolution and Multimodal Rotary Position Embedding (M-ROPE)

## Step-by-Step Guide to Fine-tuning Qwen2-VL

Let's walk through the process of fine-tuning Qwen2-VL for your specific HTR task:

### Step 1: Setup

First, ensure you have the necessary dependencies:

```bash
pip install git+https://github.com/huggingface/transformers
pip install qwen-vl-utils
```

Next, clone this repository and move into the new folder.

```bash
git clone https://github.com/wjbmattingly/qwen2-vl-finetuning-huggingface.git
cd qwen2-vl-finetuning-huggingface
```

Finally, install the required packages for fine-tuning:

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Your Dataset

For HTR, you'll need a dataset of handwritten text images paired with their transcriptions. Organize your data into a format compatible with HuggingFace datasets. For an example, please see [Catmus Medieval](https://huggingface.co/datasets/CATMuS/medieval).

<iframe
	src="https://huggingface.co/datasets/CATMuS/medieval/embed/viewer"
	frameborder="0"
	width="100%"
	height="500px"
></iframe>

Here, we can see that we have the images in the `im` field and our transcription is in the `text` field. For this tutorial, we will be fine-tuning Qwen 2 VL 2B on this line-level HTR dataset.

### Step 3: Load the Model and Processor

Now that we have a dataset prepared, let's go ahead and start the fine-tuning process. Over the next few sections, we will cover all the steps performed by the function in src.finetune.py, `train_and_validate()`. If you would rather just do all these steps as a single command in Python, please skip ahead to the final section of this blog.

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
```

This step initializes the Qwen2-VL model and its associated processor. The `torch_dtype="auto"` and `device_map="auto"` parameters ensure optimal performance based on your hardware.

### Step 4: Prepare the Training Data

We'll use a custom dataset class that formats our data for Qwen2-VL:

```python
from torch.utils.data import Dataset

class HuggingFaceDataset(Dataset):
    def __init__(self, dataset, image_column, text_column, user_text="Convert this image to text"):
        self.dataset = dataset
        self.image_column = image_column
        self.text_column = text_column
        self.user_text = user_text

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_column]
        assistant_text = item[self.text_column]

        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": self.user_text}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": str(assistant_text)}
                    ]
                }
            ]
        }
```

This custom `HuggingFaceDataset` class is crucial for preparing our data in a format suitable for fine-tuning the Qwen2-VL model. Let's break down its components:

1. **Initialization**: The `__init__` method sets up the dataset with the necessary columns for images and text, as well as a default user prompt.

2. **Length**: The `__len__` method returns the total number of items in the dataset, allowing us to iterate over it.

3. **Item Retrieval**: The `__getitem__` method is the core of this class. For each index:
   - It retrieves an item from the dataset.
   - Extracts the image and text from the specified columns.
   - Formats the data into a conversation-like structure that Qwen2-VL expects.

4. **Conversation Format**: The returned dictionary mimics a conversation with:
   - A "user" message containing both the image and a text prompt.
   - An "assistant" message containing the transcription (ground truth).

This structure is essential because it allows the model to learn the connection between the input (image + prompt) and the expected output (transcription). During training, the model will learn to generate the assistant's response based on the user's input, effectively learning to transcribe handwritten text from images.

This class wraps our HuggingFace dataset, formatting each item as a conversation with a user query (including an image) and an assistant response.

### Step 5: Image Processing

To ensure consistent image processing, we use the `ensure_pil_image` function:

```python
from PIL import Image
import base64
from io import BytesIO

def ensure_pil_image(image, min_size=256):
    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, str):
        if image.startswith('data:image'):
            image = image.split(',')[1]
        image_data = base64.b64decode(image)
        pil_image = Image.open(BytesIO(image_data))
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    if pil_image.width < min_size or pil_image.height < min_size:
        scale = max(min_size / pil_image.width, min_size / pil_image.height)
        new_width = int(pil_image.width * scale)
        new_height = int(pil_image.height * scale)
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    return pil_image
```

This function ensures that the input image is a PIL Image object and meets a minimum size requirement. It's particularly useful in our pipeline for several reasons:

1. **Flexibility in Input Types**: It can handle different types of image inputs:
   - PIL Image objects are used directly.
   - Base64-encoded strings (common in web applications) are decoded into images.
   - It raises an error for unsupported types, helping to catch potential issues early.

2. **Minimum Size Enforcement**: The function ensures that images meet a minimum size (default 256x256 pixels). This is crucial because:
   - Many vision models have minimum input size requirements.
   - Consistent image sizes can improve training stability and model performance.
   - It preserves the aspect ratio while resizing, maintaining image integrity.

3. **Quality Preservation**: When resizing is necessary, it uses the LANCZOS algorithm, which is known for producing high-quality resized images.

4. **Error Handling**: The function includes error checking and raises informative exceptions, making debugging easier.

5. **Integration with Data Pipeline**: This function can be easily integrated into our data loading and preprocessing pipeline, ensuring all images are properly formatted before being fed into the model.

By using `ensure_pil_image`, we standardize our image inputs, which is crucial for consistent model training and inference. This is especially important when dealing with datasets that might contain images of varying formats and sizes.


### Step 6: Collate Function

The `collate_fn` is crucial for processing batches of data:

```python
def collate_fn(batch, processor, device):
    messages = [item['messages'] for item in batch]
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    images = [ensure_pil_image(msg[0]['content'][0]['image']) for msg in messages]
    
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    ).to(device)

    input_ids_lists = inputs['input_ids'].tolist()
    labels_list = []
    for ids_list in input_ids_lists:
        label_ids = [-100] * len(ids_list)
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            label_ids[begin_end_indexs[0]+2:begin_end_indexs[1]+1] = ids_list[begin_end_indexs[0]+2:begin_end_indexs[1]+1]
        labels_list.append(label_ids)

    labels_ids = torch.tensor(labels_list, dtype=torch.int64)

    return inputs, labels_ids
```

This `collate_fn` function plays a crucial role in preparing batches of data for training the Qwen2-VL model. Here's a detailed breakdown of its functionality:

1. **Batch Processing**: 
   - It takes a batch of items, where each item contains 'messages' with user and assistant interactions.
   - The function processes these messages to create input suitable for the model.

2. **Text Processing**:
   - It applies a chat template to each message, converting the structured conversation into a format the model can understand.
   - The `tokenize=False` parameter ensures we get the formatted text, not tokenized IDs at this stage.

3. **Image Processing**:
   - For each message, it extracts the image and ensures it's in the correct format using the `ensure_pil_image` function.
   - This step standardizes all images, regardless of their original format or size.

4. **Model Input Creation**:
   - The processor (likely a Qwen2-VL specific processor) is used to create model inputs.
   - It combines the processed texts and images, applies padding, and converts everything to PyTorch tensors.
   - The resulting inputs are moved to the specified device (CPU or GPU).

5. **Label Creation**:
   - It creates labels for training, which is crucial for supervised learning.
   - The `find_assistant_content_sublist_indexes` function is used to identify the portions of the input that correspond to the assistant's responses.
   - Labels are set to -100 for non-assistant parts (which will be ignored during loss calculation) and to the actual token IDs for the assistant's responses.

6. **Output**:
   - The function returns two elements:
     1. The processed inputs ready for the model.
     2. The labels tensor, aligned with the inputs, for calculating the loss during training.

This function is essential for transforming raw data into a format that the Qwen2-VL model can use for training, ensuring that both the visual and textual components are properly integrated and aligned.

### Step 7: Validation Function

We include a validation step to monitor model performance:

```python
def validate(model, val_loader):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            inputs, labels = batch
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    model.train()
    return avg_val_loss
```

This function calculates the average validation loss across all batches.

### Step 8: Training and Validation Loop

The main training function, `train_and_validate`, brings everything together:

```python
def train_and_validate(
    model_name,
    output_dir,
    dataset_name,
    image_column,
    text_column,
    device="cuda",
    user_text="Convert this image to text",
    min_pixel=256,
    max_pixel=384,
    image_factor=28,
    num_accumulation_steps=2,
    eval_steps=10000,
    max_steps=100000,
    train_select_start=0,
    train_select_end=1000,
    val_select_start=0,
    val_select_end=1000,
    train_batch_size=1,
    val_batch_size=1,
    train_field="train",
    val_field="validation"
):
    # Load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixel*image_factor*image_factor, max_pixels=max_pixel*image_factor*image_factor, padding_side="right")

    # Load and prepare dataset
    dataset = load_dataset(dataset_name)
    train_dataset = dataset[train_field].shuffle(seed=42).select(range(train_select_start, train_select_end))
    val_dataset = dataset[val_field].shuffle(seed=42).select(range(val_select_start, val_select_end))

    train_dataset = HuggingFaceDataset(train_dataset, image_column, text_column, user_text)
    val_dataset = HuggingFaceDataset(val_dataset, image_column, text_column, user_text)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device),
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        collate_fn=partial(collate_fn, processor=processor, device=device)
    )

    # Set up optimizer
    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Training loop
    global_step = 0
    progress_bar = tqdm(total=max_steps, desc="Training")

    while global_step < max_steps:
        for batch in train_loader:
            global_step += 1
            inputs, labels = batch
            outputs = model(**inputs, labels=labels)
            
            loss = outputs.loss / num_accumulation_steps
            loss.backward()
            
            if global_step % num_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item() * num_accumulation_steps})

            # Evaluation and model saving
            if global_step % eval_steps == 0 or global_step == max_steps:
                avg_val_loss = validate(model, val_loader)
                save_dir = os.path.join(output_dir, f"model_step_{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                processor.save_pretrained(save_dir)
                model.train()

            if global_step >= max_steps:
                save_dir = os.path.join(output_dir, "final")
                model.save_pretrained(save_dir)
                processor.save_pretrained(save_dir)
                break

    progress_bar.close()
```

This function handles the entire training process, including data loading, model training, validation, and saving checkpoints.

### Step 9: Running the Fine-tuning Process

To start the fine-tuning process, you can now call the `train_and_validate` function with your specific parameters:

```python
from src.finetune import train_and_validate

train_and_validate(
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    output_dir="/output",
    dataset_name="CATMuS/medieval",
    image_column="im",
    text_column="text",
    user_text="Transcribe this handwritten text.",
    train_field="train",
    val_field="validation",
    num_accumulation_steps=2,
    eval_steps=1000,
    max_steps=10000,
    train_batch_size=1,
    val_batch_size=1,
    device="cuda"
)
```

This will start the fine-tuning process on the specified dataset, saving model checkpoints at regular intervals.

## Running Inference with your Fine-Tuned Model

To use your fine-tuned model, you can load it like this:

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

model = Qwen2VLForConditionalGeneration.from_pretrained("/output/final")
processor = AutoProcessor.from_pretrained("/output/final")

# Now you can use the model for inference
```

To use the model, you can follow the supplied documentation from Qwen2 VL:

```python
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
# inputs = inputs.to("cuda") Map the inputs to device, if necessary

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128) # Increase max_new_tokens if you want to generate longer outputs (good for longer texts)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```

## Conclusion

By following these steps, you can fine-tune the Qwen2-VL model for your specific HTR task. The process involves preparing your dataset, setting up the model and processor, and running the training loop with validation steps.

Remember to monitor the training progress and adjust hyperparameters as needed. Once fine-tuning is complete, you can use the saved model for inference on new handwritten text images.

Happy transcribing!