import numpy as np
import torch
import pickle
import os

from model.model import MLP

def get_cos_sim(array1, array2):
    return (np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2)))

def load_tetra_model(checkpoint_path, device):
    model = MLP(embedding_size=512).to(device)
    model.load_state_dict(torch.load(checkpoint_path,  map_location=device))
    model.eval() 
    return model

def load_mad_model(checkpoint_path):
    return pickle.load(open(checkpoint_path, "rb"))

def read_embedding_from_file(input_file):
    return np.genfromtxt(input_file, dtype='float')

def get_tetraloss_embedding(model, original_embedding, device):
    original_emb = torch.FloatTensor(original_embedding.flatten()).to(device)
    original_emb = original_emb.unsqueeze(0) 

    with torch.no_grad():
        new_emb = model(original_emb)
    
    new_emb = new_emb.squeeze(dim=0).detach().cpu().numpy()
    return new_emb

def compute_mad_score(svm, scaler, ref_emb, probe_emb):
    def combine_embeddings(reference_emb, probe_emb):
        embedding_diff = np.subtract(reference_emb, probe_emb)
        return embedding_diff
    pred_score = svm.predict_proba(scaler.transform([combine_embeddings(ref_emb, probe_emb)]))[0]
    return 1 - pred_score[0]

def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)