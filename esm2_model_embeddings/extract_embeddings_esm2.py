import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import os
import numpy as np

# Function to extract embeddings for a single sequence
def extract_embedding(sequence, model, tokenizer, device, layer_indices=None, per_token=False):
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract embeddings from multiple specified layers
    if layer_indices is not None:
        embeddings = [outputs.hidden_states[i].squeeze(0) for i in layer_indices]  # List of layer embeddings
        embeddings = torch.cat(embeddings, dim=0)  # Concatenate embeddings across layers
    else:
        embeddings = outputs.last_hidden_state.squeeze(0)  # Use last hidden state if no layer indices are specified

    if per_token:
        # Return per-token embeddings without aggregation
        return embeddings.cpu().numpy()
    else:
        # Return the mean of the embeddings across the sequence length as the representation
        return embeddings.mean(dim=0).cpu().numpy()

# Main function
def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from ESM models.")
    parser.add_argument("model_name", type=str, default="facebook/esm2_t33_650M_UR50D", help="Name of the ESM model.")
    parser.add_argument("input_fasta", type=str, help="Path to the input FASTA file.")
    parser.add_argument("output_dir", type=str, help="Directory to save embeddings.")
    parser.add_argument("--repr_layers", nargs='+', type=int, default=[-1], help="List of layers to extract embeddings from.")
    parser.add_argument("--include", type=str, choices=["mean", "per-token"], default="per-token", help="Type of embedding to extract.")

    args = parser.parse_args()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    # Set the device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Enable output of all hidden states
    model.config.output_hidden_states = True

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Read sequences from the FASTA file
    sequences = []
    for record in SeqIO.parse(args.input_fasta, "fasta"):
        sequences.append((record.id, str(record.seq)))

    # Extract embeddings for each sequence and save them
    print(f"Extracting embeddings for proteins in {args.input_fasta}...")
    for seq_id, sequence in sequences:
        print(f"Processing {seq_id}...")
        per_token = args.include == "per-token"
        embedding = extract_embedding(sequence, model, tokenizer, device, layer_indices=args.repr_layers, per_token=per_token)
        # Save the embedding for this protein in an individual file
        output_file = os.path.join(args.output_dir, f"{seq_id}_embedding.npy")
        np.save(output_file, embedding)
        print(f"Embedding for {seq_id} saved to {output_file}")

    print("Processing completed.")

if __name__ == "__main__":
    main()


