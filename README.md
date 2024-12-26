# Fine-tuning LLava using Odds Ratio Preference Optimization for Alignment

This repository contains the implementation for fine-tuning the LLava model using **Odds Ratio Preference Optimization** for **alignment**. 
## Setup Instructions

Below are the steps to set up and run the project in a virtual environment:
```bash
python -m venv venv
```
Activate the virtual environment to use it for the project:
  ```bash
  source venv/bin/activate
  ```
Install the required Python dependencies
```bash
pip install -r requirements.txt
```
To use **Weights and Biases** for tracking experiments, set **WANDB API key**:

```bash
export WANDB_API_KEY="wandb_api_key_here"
```
To authenticate with Hugging Face, run the following command and log in with your credentials:
```bash
huggingface-cli login
```
Train!
```bash
python train.py
```

