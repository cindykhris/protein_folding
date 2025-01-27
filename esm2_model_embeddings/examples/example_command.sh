#!/bin/bash
# Example usage of extract_embeddings_esm2.py
python extract_embeddings_esm2.py facebook/esm2_t33_650M_UR50D \
  examples/example_input.fasta \
  examples/output_embeddings \
  --repr_layers -1 -2 -3 \
  --include per-token

