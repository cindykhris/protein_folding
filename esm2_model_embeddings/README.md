# Extract Embeddings Using ESM2 Model

This script, `extract_embeddings_esm2.py`, allows you to extract protein embeddings from the ESM2 model provided by Facebook AI. The embeddings can be saved as per-token embeddings or mean embeddings for specified layers of the model.

## Requirements

- Python 3.7+
- Required Python libraries:
  - `torch`
  - `transformers`
  - `biopython`
  - `numpy`

Install the required libraries using pip:
```bash
pip install torch transformers biopython numpy
```

## Usage

The script can be executed via the command line with the following format:

```bash
python extract_embeddings_esm2.py <model_name> <input_fasta> <output_dir> --repr_layers <layers> --include <type>
```

### Arguments

1. `model_name` (str): The name of the ESM model to use. Default is `facebook/esm2_t33_650M_UR50D`.
2. `input_fasta` (str): The path to the input FASTA file containing protein sequences.
3. `output_dir` (str): The directory to save the extracted embeddings.

### Optional Arguments

- `--repr_layers` (list of int): Specify the layers of the model to extract embeddings from. For example, `--repr_layers -1 -2 -3` will extract embeddings from the last three layers.
- `--include` (str): Specify the type of embedding to extract. Options are:
  - `per-token`: Save embeddings for each token (residue).
  - `mean`: Save the mean embedding across all tokens.

### Example Command

```bash
python extract_embeddings_esm2.py facebook/esm2_t33_650M_UR50D \
  /pollard/data/projects/cpino/protein_folding/mc_test.fa \
  /pollard/data/projects/cpino/protein_folding/output_embeddings \
  --repr_layers -1 -2 -3 --include per-token
```

In this example:
- The model `facebook/esm2_t33_650M_UR50D` is used.
- Input FASTA file is located at `/pollard/data/projects/cpino/protein_folding/mc_test.fa`.
- Output embeddings are saved in `/pollard/data/projects/cpino/protein_folding/output_embeddings`.
- Embeddings are extracted from the last three layers (`-1`, `-2`, `-3`).
- Per-token embeddings are generated.

## Output

Each protein in the FASTA file will have its embedding saved as a `.npy` file in the specified output directory. The file name will be `<protein_id>_embedding.npy`, where `<protein_id>` corresponds to the ID in the FASTA file.

## Notes

- Ensure that the input FASTA file is properly formatted, with each sequence having a unique identifier.
- For large FASTA files or complex models, consider running the script on a machine with GPU support for faster processing.

## Troubleshooting

- If the script fails due to memory issues, try using fewer layers or switching to mean embeddings instead of per-token embeddings.
- Use `--repr_layers` to focus on specific layers to reduce the size of the output embeddings.

