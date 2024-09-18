"""
The original version of this fine-tuning script came from this source: https://github.com/zhangfaen/finetune-Qwen2-VL. I modified this to align it to work specifically with HuggingFace datasets. I also designed it to specifically with with the Gradio app in the main directory, app.py. I also added a validation step to the training loop. I am deeply indebted and grateful for their work. Without this code, this project would have been substantially more difficult.
"""
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

from datasets import load_dataset

from PIL import Image
import base64
from io import BytesIO

from functools import partial
from tqdm import tqdm


def find_assistant_content_sublist_indexes(l):
    """
    Find the start and end indexes of assistant content sublists within a given list.

    This function searches for specific token sequences that indicate the beginning and end
    of assistant content in a tokenized list. It identifies pairs of start and end indexes
    for each occurrence of assistant content.

    Args:
        l (list): A list of tokens to search through.

    Returns:
        list of tuples: A list of (start_index, end_index) pairs indicating the positions
        of assistant content sublists within the input list.

    Note:
        - The start of assistant content is identified by the sequence [151644, 77091].
        - The end of assistant content is marked by the token 151645.
        - This function assumes that each start sequence has a corresponding end token.
    """
    start_indexes = []
    end_indexes = []

    # Iterate through the list to find starting points
    for i in range(len(l) - 1):
        # Check if the current and next element form the start sequence
        if l[i] == 151644 and l[i + 1] == 77091:
            start_indexes.append(i)
            # Now look for the first 151645 after the start
            for j in range(i + 2, len(l)):
                if l[j] == 151645:
                    end_indexes.append(j)
                    break  # Move to the next start after finding the end

    return list(zip(start_indexes, end_indexes))

class HuggingFaceDataset(Dataset):
    """
    A custom Dataset class for handling HuggingFace datasets with image and text pairs.

    This class is designed to work with datasets that contain image-text pairs,
    specifically for use in vision-language models. It processes the data to create
    a format suitable for models like Qwen2-VL, structuring each item as a conversation
    with a user query (including an image) and an assistant response.

    Attributes:
        dataset: The HuggingFace dataset to be wrapped.
        image_column (str): The name of the column containing image data.
        text_column (str): The name of the column containing text data.
        user_text (str): The default user query text to pair with each image.

    """
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

def ensure_pil_image(image, min_size=256):
    """
    Ensures that the input image is a PIL Image object and meets a minimum size requirement.

    This function handles different input types:
    - If the input is already a PIL Image, it's used directly.
    - If the input is a string, it's assumed to be a base64-encoded image and is decoded.
    - For other input types, a ValueError is raised.

    The function also resizes the image if it's smaller than the specified minimum size,
    maintaining the aspect ratio.

    Args:
        image (Union[PIL.Image.Image, str]): The input image, either as a PIL Image object
                                             or a base64-encoded string.
        min_size (int, optional): The minimum size (in pixels) for both width and height. 
                                  Defaults to 256.

    Returns:
        PIL.Image.Image: A PIL Image object meeting the size requirements.

    Raises:
        ValueError: If the input image type is not supported.
    """
    if isinstance(image, Image.Image):
        pil_image = image
    elif isinstance(image, str):
        # Assuming it's a base64 string
        if image.startswith('data:image'):
            image = image.split(',')[1]
        image_data = base64.b64decode(image)
        pil_image = Image.open(BytesIO(image_data))
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Check if the image is smaller than the minimum size
    if pil_image.width < min_size or pil_image.height < min_size:
        # Calculate the scaling factor
        scale = max(min_size / pil_image.width, min_size / pil_image.height)
        new_width = int(pil_image.width * scale)
        new_height = int(pil_image.height * scale)
        
        # Resize the image
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
    
    return pil_image

def collate_fn(batch, processor, device):
    """
    Collate function for processing batches of data for the Qwen2-VL model.

    This function prepares the input data for training or inference by processing
    the messages, applying chat templates, ensuring images are in the correct format,
    and creating input tensors for the model.

    Args:
        batch (List[Dict]): A list of dictionaries, each containing 'messages' with text and image data.
        processor (AutoProcessor): The processor for the Qwen2-VL model, used for tokenization and image processing.
        device (torch.device): The device (CPU or GPU) to which the tensors should be moved.

    Returns:
        Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing:
            - inputs: A dictionary of input tensors for the model (e.g., input_ids, attention_mask).
            - labels_ids: A tensor of label IDs for training, with -100 for non-assistant tokens.

    Note:
        This function assumes that each message in the batch contains both text and image data,
        and that the first content item in each message is an image.
    """
    messages = [item['messages'] for item in batch]
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in messages]
    
    # Ensure all images are PIL Image objects
    images = [ensure_pil_image(msg[0]['content'][0]['image']) for msg in messages]
    
    # Process the text and images using the processor
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    # Move the inputs to the specified device (CPU or GPU)
    inputs = inputs.to(device)

    # Convert input IDs to a list of lists for easier processing
    input_ids_lists = inputs['input_ids'].tolist()
    labels_list = []
    for ids_list in input_ids_lists:
        # Initialize label IDs with -100 (ignored in loss calculation)
        label_ids = [-100] * len(ids_list)
        # Find the indexes of assistant content in the input IDs
        for begin_end_indexs in find_assistant_content_sublist_indexes(ids_list):
            # Set the label IDs for assistant content, skipping the first two tokens
            label_ids[begin_end_indexs[0]+2:begin_end_indexs[1]+1] = ids_list[begin_end_indexs[0]+2:begin_end_indexs[1]+1]
        labels_list.append(label_ids)

    # Convert the labels list to a tensor
    labels_ids = torch.tensor(labels_list, dtype=torch.int64)

    # Return the processed inputs and label IDs
    return inputs, labels_ids

def validate(model, val_loader):
    """
    Validate the model on the validation dataset.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for the validation dataset.

    Returns:
        float: The average validation loss.

    This function sets the model to evaluation mode, performs a forward pass
    on the validation data without gradient computation, calculates the loss,
    and returns the average validation loss across all batches.
    """
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
    """
    Train and validate a Qwen2VL model on a specified dataset.

    Args:
        model_name (str): Name of the pre-trained model to use.
        output_dir (str): Directory to save the trained model.
        dataset_name (str): Name of the dataset to use for training and validation.
        image_column (str): Name of the column containing image data in the dataset.
        text_column (str): Name of the column containing text data in the dataset.
        device (str): Device to use for training ('cuda' or 'cpu').
        user_text (str): Default text prompt for the user input.
        min_pixel (int): Minimum pixel size for image processing.
        max_pixel (int): Maximum pixel size for image processing.
        image_factor (int): Factor for image size calculation.
        num_accumulation_steps (int): Number of steps for gradient accumulation.
        eval_steps (int): Number of steps between evaluations.
        max_steps (int): Maximum number of training steps.
        train_select_start (int): Starting index for selecting training data.
        train_select_end (int): Ending index for selecting training data.
        val_select_start (int): Starting index for selecting validation data.
        val_select_end (int): Ending index for selecting validation data.
        train_batch_size (int): Batch size for training.
        val_batch_size (int): Batch size for validation.
        train_field (str): Field name for training data in the dataset.
        val_field (str): Field name for validation data in the dataset.

    Returns:
        None
    """
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        device_map=device
    )

    processor = AutoProcessor.from_pretrained(model_name, min_pixels=min_pixel*image_factor*image_factor, max_pixels=max_pixel*image_factor*image_factor, padding_side="right")

    # Load and split the dataset
    dataset = load_dataset(dataset_name)
    train_dataset = dataset[train_field].shuffle(seed=42).select(range(train_select_start, train_select_end))
    val_dataset = dataset[val_field].shuffle(seed=42).select(range(val_select_start, val_select_end))

    train_dataset = HuggingFaceDataset(train_dataset, image_column, text_column, user_text)
    val_dataset = HuggingFaceDataset(val_dataset, image_column, text_column, user_text)

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

    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-5)

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

            # Perform evaluation and save model every EVAL_STEPS
            if global_step % eval_steps == 0 or global_step == max_steps:
                avg_val_loss = validate(model, val_loader)

                # Save the model and processor
                save_dir = os.path.join(output_dir, f"model_step_{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                processor.save_pretrained(save_dir)

                model.train()  # Set the model back to training mode

            if global_step >= max_steps:
                save_dir = os.path.join(output_dir, f"final")
                model.save_pretrained(save_dir)
                processor.save_pretrained(save_dir)
                break

        if global_step >= max_steps:
            save_dir = os.path.join(output_dir, f"final")
            model.save_pretrained(save_dir)
            processor.save_pretrained(save_dir)
            break

    progress_bar.close()