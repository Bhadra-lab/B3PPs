# BrainShuttle-ESM

This repository provides code and data used for predicting Blood–Brain Barrier–Penetrating Peptides (B3PPs).

---

## Running the Prediction

Clone the repository and navigate to the prediction directory. Then set up the Python environment:

```bash
python -m venv env
source env/bin/activate      # Linux / macOS
# env\Scripts\activate       # Windows

pip install -r requirements.txt   # Install required packages
python app.py                     # Start the prediction code
```
## Online Prediction Server

A hosted version of the prediction model is available and can be used easily through the following link:

https://huggingface.co/spaces/Bhadralab/B3PPs_Predict

You can go to this link and **run predictions online without setting up the code locally**.

## Data

The datasets used in this study are provided in the `data/` directory.

## Fine-tuning and Hyperparameter Search

Sample code for fine-tuning and hyperparameter search is provided in the `code/` directory.

