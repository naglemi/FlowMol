import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rdkit import Chem
import dgl
import math
import subprocess
import os
from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

def dpo_finetune_step(model, receptor, device='cuda:0', use_classifier=False, clf=None, beta=0.1, n_molecules=32,
                      smiles_to_pdbqt=None, dock_and_score=None):
    with torch.no_grad():
        sampled_mols = model.sample_random_sizes(n_molecules=n_molecules, device=device)
    rdkit_mols = []
    scores = []
    for i, smol in enumerate(sampled_mols):
        mol = smol.rdkit_mol
        if mol is None:
            # Invalid molecule, assign low score
            scores.append(-999.0)
            rdkit_mols.append(None)
            continue

        pdbqt_path = f"temp_lig_{i}.pdbqt"
        sdf_path = f"temp_lig_{i}.sdf"

        w = Chem.SDWriter(sdf_path)
        # DO NOT disable kekulization
        try:
            w.write(mol)  # Attempts to kekulize by default
        except (Chem.KekulizeException, Chem.AtomKekulizeException):
            print(f"Warning: Could not kekulize molecule {i}. Skipping this molecule.")
            w.close()
            scores.append(-999.0)
            rdkit_mols.append(None)
            if os.path.exists(sdf_path):
                os.remove(sdf_path)
            continue
        w.close()

        cmd = ["obabel", sdf_path, "-O", pdbqt_path, "--addhs", "--partialcharge", "gasteiger"]
        subprocess.run(cmd, check=True)

        docking_score = dock_and_score(pdbqt_path, rec_pdbqt=receptor)
        scores.append(docking_score)
        rdkit_mols.append(mol)

    # Convert scores to a tensor
    scores = torch.tensor(scores, device=device)
    sorted_indices = torch.argsort(scores, descending=True)
    half = len(rdkit_mols)//2
    winners_idx = sorted_indices[:half]
    losers_idx = sorted_indices[half:]

    # Build a graph batch
    batched_graph = dgl.batch([smol.g for smol in sampled_mols]).to(device)
    node_batch_idx, edge_batch_idx = get_batch_idxs(batched_graph)
    upper_edge_mask = batched_graph.edata['ue_mask']

    # Time fixed at 0.5
    t_fixed = torch.tensor([0.5]*len(sampled_mols), device=device)
    # Get predictions from model.vector_field
    vf_output = model.vector_field(batched_graph, t_fixed, node_batch_idx=node_batch_idx, upper_edge_mask=upper_edge_mask)

    # Compute atom entropy
    atom_probs = vf_output['a'].clamp(1e-9, 1-1e-9)
    atom_entropy = -(atom_probs * atom_probs.log()).sum(dim=-1)

    mol_entropies = []
    for i in range(len(sampled_mols)):
        node_mask = (node_batch_idx == i)
        mol_ent = atom_entropy[node_mask].mean()
        mol_entropies.append(mol_ent)
    mol_entropies = torch.stack(mol_entropies)

    # Ranking-based approach
    if not use_classifier:
        winner_loss = mol_entropies[winners_idx].mean()
        loser_loss = mol_entropies[losers_idx].mean()
        aux_loss = -(winner_loss - loser_loss)
        alpha = 0.1
        # Return aux_loss and scores as a Python list
        return aux_loss*alpha, scores.cpu().tolist()
    else:
        # Classifier-based approach
        if clf is None:
            # Need a classifier, still return scores
            return torch.zeros(1, device=device, requires_grad=True), scores.cpu().tolist()
        log_ratios = []
        n_pairs = min(len(winners_idx), len(losers_idx))
        from rdkit.Chem import AllChem
        for i in range(n_pairs):
            w_idx = winners_idx[i].item()
            l_idx = losers_idx[i].item()
            if rdkit_mols[w_idx] is None or rdkit_mols[l_idx] is None:
                continue
            fp_size = 1024
            fp_w = AllChem.GetMorganFingerprintAsBitVect(rdkit_mols[w_idx], 2, nBits=fp_size)
            fp_l = AllChem.GetMorganFingerprintAsBitVect(rdkit_mols[l_idx], 2, nBits=fp_size)
            w_vec = torch.tensor(fp_w, dtype=torch.float32, device=device)
            l_vec = torch.tensor(fp_l, dtype=torch.float32, device=device)
            x = torch.cat([w_vec, l_vec], dim=0).unsqueeze(0)
            out = clf(x)
            p = torch.softmax(out, dim=-1)[0,1]
            p = torch.clamp(p, 1e-5, 1-1e-5)
            r_w = scores[w_idx]
            r_l = scores[l_idx]
            score = ((r_w - r_l)/beta) + torch.log(p/(1-p))
            loss_pair = -torch.log(torch.sigmoid(score))
            log_ratios.append(loss_pair)
        if len(log_ratios)==0:
            # No valid pairs, still return scores
            return torch.zeros(1, device=device, requires_grad=True), scores.cpu().tolist()
        total_loss = torch.stack(log_ratios).mean()
        return total_loss, scores.cpu().tolist()