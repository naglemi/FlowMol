import argparse
from pathlib import Path
import yaml
import dgl
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
import wandb
import dgl
import sys

# from models.ligand_edm import LigandEquivariantDiffusion
from models.mol_fm import MolFM
from data_processing.dataset import MoleculeDataset

def parse_args():
    p = argparse.ArgumentParser(description='Training Script')
    p.add_argument('--config', type=Path, default=None)
    p.add_argument('--resume', type=Path, default=None, help='Path to run directory or checkpoint file to resume from')

    # TODO: make these arguments do something
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup_length', type=float, default=1)

    # create a boolean argument for whether or not this is a debug run
    p.add_argument('--debug', action='store_true')

    p.add_argument('--seed', type=int, default=None)


    args = p.parse_args()

    if args.config is not None and args.resume is not None:
        raise ValueError('only specify a config file or a resume file but not both')

    return args


if __name__ == "__main__":
    args = parse_args()
    
    # TODO: implement resuming
    if args.resume is not None:
        # determine if we are resuming from a run directory or a checkpoint file
        if args.resume.is_dir():
            # we are resuming from a run directory
            # get the config file from the run directory
            run_dir = args.resume
            ckpt_file = str(run_dir / 'checkpoints' / 'last.ckpt')
        elif args.resume.is_file():
            run_dir = args.resume.parent.parent
            ckpt_file = str(args.resume)
        else:
            raise ValueError('resume argument must be a run directory or a checkpoint file that must already exist')
        
        config_file = run_dir / 'config.yaml'
    else:
        config_file = args.config
        ckpt_file = None


    # set seed
    if args.seed is not None:
        pl.seed_everything(args.seed)
    
    # process config file into dictionary
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # create dataset
    train_dataset = MoleculeDataset(split='train', dataset_config=config['dataset'])
    val_dataset = MoleculeDataset(split='val', dataset_config=config['dataset'])

    # create dataloaders
    train_dataloader = DataLoader(train_dataset, 
                            batch_size=config['training']['batch_size'], 
                            shuffle=True, 
                            collate_fn=dgl.batch, 
                            num_workers=config['training']['num_workers'])
    
    val_dataloader = DataLoader(val_dataset, 
                            batch_size=config['training']['batch_size']*2, 
                            shuffle=True, 
                            collate_fn=dgl.batch, 
                            # num_workers=config['training']['num_workers'],
                            num_workers=0)

    # get the filepath of the n_atoms histogram
    n_atoms_hist_filepath = Path(config['dataset']['processed_data_dir']) / 'train_data_n_atoms_histogram.pt'

    # get the sample interval (how many epochs between drawing/evaluating)
    sample_interval = config['training']['evaluation']['sample_interval']
    mols_to_sample = config['training']['evaluation']['mols_to_sample']

    # create model
    atom_type_map = config['dataset']['atom_map']
    try:
        num_devices = config['training']['trainer_args']['devices']
    except KeyError:
        num_devices = 1
    batches_per_epoch = len(train_dataloader) // num_devices
    model = MolFM(atom_type_map=atom_type_map,
                batches_per_epoch=batches_per_epoch, 
                n_atoms_hist_file=n_atoms_hist_filepath,
                sample_interval=sample_interval,
                n_mols_to_sample=mols_to_sample,
                vector_field_config=config['vector_field'],
                interpolant_scheduler_config=config['interpolant_scheduler'], 
                lr_scheduler_config=config['lr_scheduler'],
                **config['mol_fm'])
    
    # get wandb logger config
    wandb_config = config['wandb']

    # if this is a debug run, set the wandb logger mode to disabled
    if args.debug:
        wandb_config['mode'] = 'offline'
        wandb_config['name'] = 'debug_run'

    # if we are not resuming a run, generate a run_id
    if args.resume is None:
        run_id = wandb.util.generate_id()
        wandb_config['id'] = run_id
    else:
        # we are resuming a run, so get the run_id from the resume file
        run_id = config['resume']['run_id']
        wandb_config['id'] = run_id
        wandb_config['resume'] = 'must'

    # create the logging directory if it doesn't exist
    output_dir = Path(config['training']['output_dir'])
    wandb_config['save_dir'] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # create wandb logger
    wandb_logger = WandbLogger(config=config, **wandb_config)
    # wandb_logger.experiment # not sure why this line is here...

    # get run directory
    run_dir = output_dir / run_id
    if rank_zero_only.rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)

        # print the run directory
        print('Results are being written to: ', run_dir)

    # create ModelCheckpoint callback
    checkpoints_dir = run_dir / 'checkpoints'
    checkpoint_config = config['checkpointing']
    checkpoint_config['dirpath'] = str(checkpoints_dir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(**checkpoint_config)

    # save the config file to the run directory
    # include the run_id so we can resume this run later
    if args.resume is None and rank_zero_only.rank == 0:
        wandb_logger.experiment
        config['resume'] = {}
        config['resume']['run_id'] = run_id
        config['wandb']['name'] = wandb.run.name
        with open(run_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

    # get pl trainer config
    trainer_config = config['training']['trainer_args']

    # compute the validation interval and add arguments for the pl.Trainer object accordingly
    batches_per_epoch = len(train_dataloader)
    trainer_config['val_check_interval'] = int(config['training']['evaluation']['val_loss_interval'] * batches_per_epoch)
    trainer_config['check_val_every_n_epoch'] = None

    # if this is a debug run, set limit_train_batches to 10
    if args.debug:
        trainer_config['limit_train_batches'] = 100

    # create trainer
    trainer = pl.Trainer(logger=wandb_logger, **trainer_config, callbacks=[checkpoint_callback])
    
    # train
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, ckpt_path=ckpt_file)
