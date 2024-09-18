# Qwen2-VL Model Finetuning

This repository contains code for finetuning the Qwen2-VL vision-language model on custom datasets using HuggingFace datasets. It includes a Gradio web interface for easy interaction and a Python script for command-line execution.

## Credits

This project is based on the work from [https://github.com/zhangfaen/finetune-Qwen2-VL](https://github.com/zhangfaen/finetune-Qwen2-VL). I've significantly modified and extended their original fine-tuning script to work specifically with HuggingFace datasets and added a Gradio interface for ease of use.

## Features

- Finetune Qwen2-VL model on custom HuggingFace datasets
- Gradio web interface for interactive model training
- Command-line script for batch processing
- Customizable training parameters
- Validation during training

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/wjbmattingly/qwen2-vl-finetuning-huggingface.git
   cd qwen2-vl-finetuning-huggingface
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Gradio Web Interface

To use the Gradio web interface:

1. Run the following command:
   ```
   python app.py
   ```

2. Open your web browser and navigate to the URL displayed in the console (usually `http://localhost:8083`).

3. Use the interface to select your dataset, set training parameters, and start the finetuning process.

### Command-line Finetuning

To finetune the model using the command line:

1. Open `src/finetune.py` and modify the parameters in the `train_and_validate` function call at the bottom of the file.

2. Run the script:
   ```python
   from src.finetune import train_and_validate

   train_and_validate(
       model_name="Qwen/Qwen2-VL-2B-Instruct",
       output_dir="/output",
       dataset_name="catmus/medieval",
       image_column="im",
       text_column="text",
       user_text="Convert this image to text",
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

This command will start the finetuning process with the specified parameters:

- Using the Qwen2-VL-2B-Instruct model
- Saving the output to "/output"
- Using the "catmus/medieval" dataset
- Using "im" as the image column and "text" as the text column
- Setting a custom user prompt
- Using "train" and "validation" splits for training and validation
- Setting various training parameters like accumulation steps, evaluation frequency, and batch sizes
   ```

## Finetuning Process

The finetuning process involves the following steps:

1. Loading the pre-trained Qwen2-VL model and processor
2. Preparing the dataset using the custom `HuggingFaceDataset` class
3. Setting up data loaders for training and validation
4. Training the model with gradient accumulation and periodic evaluation
5. Saving the finetuned model

Key functions in `src/finetune.py`:

- `train_and_validate`: Main function that orchestrates the finetuning process
- `collate_fn`: Prepares batches of data for the model
- `validate`: Performs validation on the model during training

## Customizing the Training

To finetune on your own HuggingFace dataset:

1. Modify the `dataset_name`, `image_column`, and `text_column` parameters in `train_and_validate`.
2. Adjust other parameters such as `max_steps`, `eval_steps`, and batch sizes as needed.

## Roadmap


Future improvements and features:

- [ ] Implement distributed GPU fine-tuning for faster training on multiple GPUs
- [ ] Add support for training on video datasets to leverage Qwen2-VL's video processing capabilities
- [ ] Develop more complex and custom message structures to handle diverse tasks
- [ ] Expand functionality to support tasks beyond Handwritten Text Recognition (HTR)


