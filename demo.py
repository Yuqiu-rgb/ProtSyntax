import argparse

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from Bio.PDB import PDBParser
import warnings

# Ignore common discontinuous residue warnings in Biopython
warnings.filterwarnings("ignore")


def extract_ca_coordinates(pdb_path, chain_id='A'):
    """
    Extracts C-alpha atom coordinates of a specified chain from a PDB file.
    This preprocessing step is essential as ProtSyntax relies on residue-level
    spatial geometric information for its Geometric Gated Attention module.
    """
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_path)

    ca_coords = []
    # Iterate through the specified chain to extract C-alpha coordinates
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    # Process only standard amino acids, skipping heteroatoms like water
                    if residue.id[0] == ' ' and 'CA' in residue:
                        ca_coords.append(residue['CA'].get_coord())

    return np.array(ca_coords)


def parse_args():
    parser = argparse.ArgumentParser(description="Run a minimal ProtSyntax PTM-site prediction demo.")
    parser.add_argument(
        "--model",
        default="Model_weight/ProtSyntax",
        help="Local path or model identifier for the released ProtSyntax checkpoint.",
    )
    parser.add_argument(
        "--sequence",
        default="MKTIIALSYIFCLVFADYKDDDDAMAKTIIALSYIFCLVFA",
        help="Protein sequence to evaluate.",
    )
    parser.add_argument(
        "--pdb",
        default="example_protein.pdb",
        help="Path to the matching protein structure file in PDB format.",
    )
    parser.add_argument("--chain", default="A", help="PDB chain identifier.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for reporting PTM sites.")
    parser.add_argument("--max-length", type=int, default=1024, help="Maximum tokenized sequence length.")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Configure model path and device
    model_name_or_path = args.model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}...")

    # 2. Load Tokenizer and Model
    # trust_remote_code=True is required to load custom architectures like Bi-Gated DeltaNet
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)
    model.eval()

    # 3. Prepare input data
    protein_sequence = args.sequence
    pdb_file_path = args.pdb

    print("Parsing PDB file to extract 3D geometric features...")
    try:
        coords = extract_ca_coordinates(pdb_file_path, chain_id=args.chain)
        # Convert coordinates to a Tensor and add a batch dimension
        coords_tensor = torch.tensor(coords, dtype=torch.float32).unsqueeze(0).to(device)
    except Exception as e:
        print(f"PDB parsing failed: {e}")
        return

    # 4. Sequence Tokenization
    inputs = tokenizer(
        protein_sequence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length
    )

    # Move sequence inputs to the corresponding compute device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # 5. Model Inference
    print("Initiating prediction for PTM sites and functional consequences...")
    with torch.no_grad():
        # ProtSyntax is a multi-modal model; this assumes the forward pass
        # accepts sequence tokens and structural coordinates
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            coordinates=coords_tensor
        )

    # 6. Post-processing results (Adjust based on actual model output structure)
    # Assuming outputs.logits returns the PTM probability for each amino acid site
    probabilities = torch.sigmoid(outputs.logits).squeeze().cpu().numpy()

    print("\n=== Prediction Results ===")
    for i, (amino_acid, prob) in enumerate(zip(protein_sequence, probabilities)):
        if prob > args.threshold:
            print(f"Position {i + 1} ({amino_acid}): PTM Probability = {prob:.4f}")


if __name__ == "__main__":
    main()
