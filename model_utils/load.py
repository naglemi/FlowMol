from models.mol_fm import MolFM
from pathlib import Path
import yaml
from data_processing.data_module import MoleculeDataModule

def read_config_file(config_file: Path) -> dict:
    # process config file into dictionary
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def model_from_config(config: dict) -> MolFM:

    atom_type_map = config['dataset']['atom_map']

    # get the sample interval (how many epochs between drawing/evaluating)
    sample_interval = config['training']['evaluation']['sample_interval']
    mols_to_sample = config['training']['evaluation']['mols_to_sample']


    # get the filepath of the n_atoms histogram
    n_atoms_hist_filepath = Path(config['dataset']['processed_data_dir']) / 'train_data_n_atoms_histogram.pt'

    model = MolFM(atom_type_map=atom_type_map, 
                    n_atoms_hist_file=n_atoms_hist_filepath,
                    sample_interval=sample_interval,
                    n_mols_to_sample=mols_to_sample,
                    vector_field_config=config['vector_field'],
                    interpolant_scheduler_config=config['interpolant_scheduler'], 
                    lr_scheduler_config=config['lr_scheduler'],
                    **config['mol_fm'])
    
    return model

def data_module_from_config(config: dict) -> MoleculeDataModule:
    # get the x_subspace of the model
    x_subspace = config['mol_fm']['x_subspace']
    batch_size = config['training']['batch_size']
    num_workers = config['training']['num_workers']

    # determine if we are doing distributed training
    if config['training']['trainer_args']['devices'] > 1:
        distributed = True
    else:
        distributed = False

    data_module = MoleculeDataModule(dataset_config=config['dataset'],
                                     prior_config=config['mol_fm']['prior_config'],
                                     x_subspace=x_subspace,
                                     max_num_edges=int(config['training']['max_num_edges']),
                                     batch_size=batch_size, 
                                     num_workers=num_workers, 
                                     distributed=distributed)
    
    return data_module