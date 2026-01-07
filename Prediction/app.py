import re
import torch
import pandas as pd
import gradio as gr
from io import StringIO
from transformers import EsmForSequenceClassification, EsmTokenizer

# --- Load tokenizer & model ---
tokenizer = EsmTokenizer.from_pretrained(
    "facebook/esm2_t6_8M_UR50D",
    do_lower_case=False
)

model = EsmForSequenceClassification.from_pretrained("model/best_model5")
model.eval()

# --- FASTA Reader ---
def read_fasta(fasta_string):
    sequences = []
    headers = []
    seq_buffer = []

    for line in StringIO(fasta_string):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if seq_buffer:
                sequences.append("".join(seq_buffer))
                seq_buffer.clear()
            headers.append(line)
        else:
            if not re.match(r'^[ACDEFGHIKLMNPQRSTVWY]+$', line):
                raise ValueError(
                    "Invalid FASTA format: Only natural amino acids (ACDEFGHIKLMNPQRSTVWY) allowed."
                )
            if len(line) > 30:
                raise ValueError(
                    f"Sequence too long: '{line}' ({len(line)} > 30 characters)."
                )
            seq_buffer.append(line)

    if seq_buffer:
        sequences.append("".join(seq_buffer))

    if len(headers) != len(sequences):
        raise ValueError(
            f"FASTA parsing error: Found {len(headers)} headers but {len(sequences)} sequences. "
            "Each header must be followed by a sequence."
        )

    return headers, sequences


def predict_peptide_class(sequences):
    sequences = [str(s) for s in sequences]
    inputs = tokenizer(
        sequences,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=30
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.sigmoid(logits)[:, 1].cpu().numpy()
    classes = ["B3PP" if p > 0.5 else "Non-B3PP" for p in probs]
    return probs, classes


def predict_from_fasta(fasta_input):
    """Processes FASTA input and returns predictions in table + CSV download."""
    try:
        headers, sequences = read_fasta(fasta_input)
        if not sequences:
            df = pd.DataFrame({"Error": ["No valid sequences found."]})
            return df, None

        probs, classes = predict_peptide_class(sequences)
        probs_rounded = [f"{p:.2f}" for p in probs]
        df = pd.DataFrame({
            "Header": headers,
            "Sequence": sequences,
            "Probability": probs_rounded,
            "Predicted Class": classes
        })
        # Save as CSV file
        csv_path = "predictions.csv"
        df.to_csv(csv_path, index=False)
        return df, csv_path

    except ValueError as e:
        df = pd.DataFrame({"Error": [str(e)]})
        return df, None
    except Exception as e:
        df = pd.DataFrame({"Error": [f"Unexpected error: {str(e)}"]})
        return df, None


iface = gr.Interface(
    fn=predict_from_fasta,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Paste your peptide sequences in FASTA format here"
    ),
    outputs=[
        gr.Dataframe(label="Predictions"),
        gr.File(label="Download CSV")
    ],
    title="B3PP Predictor",
    description=(
        "Submit peptide sequences in FASTA format to determine their potential as "
        "blood-brain barrier penetration peptides. Sequences must consist exclusively "
        "of natural amino acids in uppercase letters, with a maximum length of 30 characters."
    )
)

if __name__ == "__main__":
    iface.launch()
