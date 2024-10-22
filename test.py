import argparse
import torch

from utils.helper import compute_mad_score, load_tetra_model, load_mad_model, get_tetraloss_embedding, get_cos_sim, read_embedding_from_file

def parse_args():
    parser = argparse.ArgumentParser(description='Takes as input embeddings from an existing face recognition system and extract the new TetraLoss embeddings.')
    parser.add_argument('--checkpoint_tetra', required=True, help='Path to the pre-trained TetraLoss model checkpoint.')
    parser.add_argument('--checkpoint_mad', required=True, help='Path to the pre-trained MAD model checkpoint.')
    parser.add_argument('--path_original_reference_emb', required=True, help='Path to saved reference embedding extracted from a reference face image using a face recognition system.')
    parser.add_argument('--path_original_probe_emb', required=True, help='Path to saved probe embedding extracted from a probe face image using a face recognition system.')
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tetraloss_model = load_tetra_model(args.checkpoint_tetra, device)
    mad_clf, mad_scaler = load_mad_model(args.checkpoint_mad)

    original_reference_emb = read_embedding_from_file(args.path_original_reference_emb)
    original_probe_emb = read_embedding_from_file(args.path_original_probe_emb)

    tetraloss_reference_emb = get_tetraloss_embedding(tetraloss_model, original_reference_emb, device)
    tetraloss_probe_emb = get_tetraloss_embedding(tetraloss_model, original_probe_emb, device)

    tetraloss_sim_score = get_cos_sim(tetraloss_reference_emb, tetraloss_probe_emb)
    mad_score = compute_mad_score(mad_clf, mad_scaler, original_reference_emb, original_probe_emb)

    print(f"TetraLoss score: {tetraloss_sim_score}")
    print(f"Differential MAD score: {mad_score}") # 1 means bona fide, 0 means morph
    print(f"Combined score: {(tetraloss_sim_score + mad_score) / 2}")
    
if __name__ == "__main__":
    main()