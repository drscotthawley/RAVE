import torch
from torch.utils.data import DataLoader, random_split

from rave.model import RAVE
from rave.core import random_phase_mangle, EMAModelCheckPoint
from rave.core import search_for_run



#from udls import SimpleDataset, simple_audio_preprocess
#from effortless_config import Config
from prefigure import get_all_args, push_wandb_config
from aeiou.hpc import HostPrinter
from aeiou.datasets import AudioDataset

import pytorch_lightning as pl
from pytorch_lightning.utilities.distributed import rank_zero_only
from os import environ, path
import numpy as np

import GPUtil as gpu

#from udls.transforms import Compose, RandomApply, Dequantize, RandomCrop
import wandb

@rank_zero_only
def setup_wandb(args):
    config = vars(args) # dict(args)
    wandb.init(project=args.name, config=config, save_code=True)
    return 


if __name__ == "__main__":

    '''
    class args(Config):
        DATA_SIZE = 16
        CAPACITY = 32
        LATENT_SIZE = 128
        RATIOS = [4, 4, 2, 2, 2]
        TAYLOR_DEGREES = 0
        BIAS = True
        NO_LATENCY = False

        MIN_KL = 1e-4
        MAX_KL = 1e-1
        CROPPED_LATENT_SIZE = 0
        FEATURE_MATCH = True

        LOUD_STRIDE = 1

        USE_NOISE = True
        NOISE_RATIOS = [4, 4, 4]
        NOISE_BANDS = 5

        D_CAPACITY = 16
        D_MULTIPLIER = 4
        D_N_LAYERS = 4

        WARMUP = 1000000
        MODE = "hinge"
        CKPT = None

        PREPROCESSED = None
        WAV = None
        SR = 48000
        N_SIGNAL = 65536
        MAX_STEPS = 2000000

        N_GPUS = 1
        BATCH = 8

        NAME = None

    args.parse_args()
    '''
    args = get_all_args()

    torch.manual_seed(args.seed)

    # special parsing for arg lists (TODO: could add this functionality to prefigure later):
    args.ratios = eval(''.join(args.ratios))
    print(f"args = {args}")
    assert args.name is not None

    """model = RAVE(data_size=args.DATA_SIZE,
                 capacity=args.CAPACITY,
                 latent_size=args.LATENT_SIZE,
                 ratios=args.RATIOS,
                 bias=args.BIAS,
                 loud_stride=args.LOUD_STRIDE,
                 use_noise=args.USE_NOISE,
                 noise_ratios=args.NOISE_RATIOS,
                 noise_bands=args.NOISE_BANDS,
                 d_capacity=args.D_CAPACITY,
                 d_multiplier=args.D_MULTIPLIER,
                 d_n_layers=args.D_N_LAYERS,
                 warmup=args.WARMUP,
                 mode=args.MODE,
                 no_latency=args.NO_LATENCY,
                 sr=args.SR,
                 min_kl=args.MIN_KL,
                 max_kl=args.MAX_KL,
                 cropped_latent_size=args.CROPPED_LATENT_SIZE,
                 feature_match=args.FEATURE_MATCH,
                 taylor_degrees=args.TAYLOR_DEGREES)"""

    model = RAVE(data_size=args.data_size,
        capacity=args.capacity,
        latent_size=args.latent_size,
        ratios=args.ratios,
        bias=args.bias,
        loud_stride=args.loud_stride,
        use_noise=args.use_noise,
        noise_ratios=args.noise_ratios,
        noise_bands=args.noise_bands,
        d_capacity=args.d_capacity,
        d_multiplier=args.d_multiplier,
        d_n_layers=args.d_n_layers,
        warmup=args.warmup,
        mode=args.mode,
        no_latency=args.no_latency,
        sr=args.sr,
        min_kl=args.min_kl,
        max_kl=args.max_kl,
        cropped_latent_size=args.cropped_latent_size,
        feature_match=args.feature_match,
        taylor_degrees=0)

    x = torch.zeros(args.batch, 2**14)
    model.validation_step(x, 0)

    if True: # new aeiou dataset class
        dataset = AudioDataset(args.wav, sample_size=args.n_signal, sample_rate=args.sr, augs=args.augs, load_frac=args.load_frac)
    else:
        #args.transforms = eval(args.transforms)
        dataset = SimpleDataset(
            args.PREPROCESSED,
            args.WAV,
            extension="*.wav,*.aif,*.flac",
            preprocess_function=simple_audio_preprocess(args.SR,
                                                        2 * args.N_SIGNAL),
            split_set="full",
            transforms=Compose([
                RandomCrop(args.N_SIGNAL),
                # RandomApply(
                #     lambda x: random_phase_mangle(x, 20, 2000, .99, args.SR),
                #     p=.8,
                # ),
                Dequantize(16),
                lambda x: x.astype(np.float32),
            ]),
        )


    print(f"len(dataset) = {len(dataset)}")
    val = (2 * len(dataset)) // 100
    train = len(dataset) - val
    train, val = random_split(
        dataset,
        [train, val],
        generator=torch.Generator().manual_seed(42),
    )

    train = DataLoader(train, args.batch, True, drop_last=True, num_workers=8)
    val = DataLoader(val, args.batch, False, num_workers=8)

    # CHECKPOINT CALLBACKS
    # validation_checkpoint = pl.callbacks.ModelCheckpoint(
    #     monitor="validation",
    #     filename="best",
    # )
    last_checkpoint = pl.callbacks.ModelCheckpoint(every_n_train_steps=100000)

    val_check = {}
    #if len(train) >= 10000:
    #    val_check["val_check_interval"] = 10000
    #else:
    #    nepoch = 10000 // len(train)
    #    val_check["check_val_every_n_epoch"] = nepoch
    val_check["val_check_interval"] = 100

    setup_wandb(args)

    wandb_logger = pl.loggers.WandbLogger(project=args.name)
    wandb_logger.watch(model)
    push_wandb_config(wandb_logger, args)

    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=args.num_gpus,
        num_nodes=args.num_nodes,
        strategy='ddp',
        #precision=16, # issues with AMP _scale preclude precision=16
        precision=32, 
        callbacks=[last_checkpoint],
        resume_from_checkpoint=search_for_run(args.ckpt),
        log_every_n_steps=1,
        max_epochs=10000000,
        max_steps=args.max_steps,
        limit_val_batches=0,
        **val_check,
    )
    trainer.fit(model, train, val, ckpt_path=args.ckpt)
