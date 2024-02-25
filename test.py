import argparse
import torch
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from models.mol_fm import MolFM
from analysis.molecule_builder import SampledMolecule
from analysis.metrics import SampleAnalyzer
from typing import List
from rdkit import Chem
import pickle

def parse_args():
    p = argparse.ArgumentParser(description='Testing Script')
    p.add_argument('--model_dir', type=Path, help='Path to model directory', default=None)
    p.add_argument('--checkpoint', type=Path, help='Path to checkpoint file', default=None)
    p.add_argument('--output_file', type=Path, help='Path to output file', default=None)

    p.add_argument('--n_mols', type=int, default=100)
    p.add_argument('--n_atoms_per_mol', type=int, default=None)
    p.add_argument('--n_timesteps', type=int, default=20)
    p.add_argument('--visualize', action='store_true', help='Visualize the sampled trajectories')
    p.add_argument('--metrics', action='store_true', help='Compute metrics on the sampled molecules')

    p.add_argument('--seed', type=int, default=None)

    args = p.parse_args()

    if args.model_dir is not None and args.checkpoint is not None:
        raise ValueError('only specify model_dir or checkpoint but not both')
    
    if args.model_dir is None and args.checkpoint is None:
        raise ValueError('must specify model_dir or checkpoint')

    return args


if __name__ == "__main__":

    # parse arguments
    args = parse_args()

    # set seed
    if args.seed is not None:
        seed_everything(args.seed)

    # get model directory and checkpoint file
    if args.model_dir is not None:
        model_dir = args.model_dir
        checkpoint_file = args.model_dir / 'checkpoints' / 'last.ckpt'
    elif args.checkpoint is not None:
        model_dir = args.checkpoint.parent.parent
        checkpoint_file = args.checkpoint

    # load model
    model = MolFM.load_from_checkpoint(checkpoint_file)

    # set device to cuda:0 if available otherwise cpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # move model to device
    model = model.to(device)

    # set model to eval mode
    model.eval()

    if args.n_atoms_per_mol is None:
        molecules: List[SampledMolecule]  = model.sample_random_sizes(args.n_mols, device=device, n_timesteps=args.n_timesteps, visualize=args.visualize)
    else:
        n_atoms = torch.full((args.n_mols,), args.n_atoms_per_mol, dtype=torch.long, device=device)
        molecules: List[SampledMolecule] = model.sample(n_atoms, device=device, n_timesteps=args.n_timesteps, visualize=args.visualize)

    # get output file
    if args.output_file is not None:
        output_file = args.output_file
    else:
        output_file = model_dir / 'samples' / 'sampled_mols.sdf'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # compute metrics if necessary
    if args.metrics:
        metrics = SampleAnalyzer().analyze(molecules)
        metrics_txt_file = output_file.parent / f'{output_file.stem}_metrics.txt'
        metrics_pkl_file = output_file.parent / f'{output_file.stem}_metrics.pkl'
        print(f'Writing metrics to {metrics_txt_file} and {metrics_pkl_file}')
        with open(metrics_txt_file, 'w') as f:
            for k, v in metrics.items():
                f.write(f'{k}: {v}\n')
        with open(metrics_pkl_file, 'wb') as f:
            pickle.dump(metrics, f)
        


    # check that output file is an sdf file
    if output_file.suffix != '.sdf':
        raise ValueError('output file must be an sdf file')
    
    if not args.visualize:
        # print the output_file
        print(f'Writing molecules to {output_file}')
        
        # write molecules to sdf file
        sdf_writer = Chem.SDWriter(str(output_file))
        sdf_writer.SetKekulize(False)
        for mol in molecules:
            rdkit_mol = mol.rdkit_mol
            if rdkit_mol is not None:
                sdf_writer.write(rdkit_mol)
        sdf_writer.close()
    else:
        print('visualize flag set to True, writing a seprate output file for each molecule trajectory')
        for mol_idx, mol in enumerate(molecules):
            mol_output_file = output_file.parent / f'{output_file.stem}_{mol_idx}{output_file.suffix}'
            print(f'Writing molecule {mol_idx} to {mol_output_file}')

            sdf_writer = Chem.SDWriter(str(mol_output_file))
            sdf_writer.SetKekulize(False)
            for traj_mol in mol.traj_mols:
                sdf_writer.write(traj_mol)
            sdf_writer.close()

    

    