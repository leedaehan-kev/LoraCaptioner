from abc import abstractmethod, ABC
from typing import Optional

import wandb
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from src.model import ImageCaptioner
from src.dataset import ConCapDataset
from src.data_structures import AttrDict


class BaseRun(ABC):
    def __init__(self, config, project, run_name, rank: Optional[int] = 0):
        # rank
        self.rank = rank
        assert self.rank == dist.get_rank()

        if rank == 0:  # logging and checkpointing is only done on rank 0
            wandb.init(project=project, name=run_name, config=config)
            self.config = wandb.config
            self.save_dir = f'./model_checkpoints/{wandb.run.id}'
        else:
            self.config = AttrDict(config)
            self.save_dir = None

    def __call__(self):
        for epoch in range(self.config.num_epochs):
            self.train(epoch)
            self.validate(epoch)

    @abstractmethod
    def train(self, epoch):
        pass

    @abstractmethod
    def validate(self, epoch):
        pass


class ImageCaptionerRun(BaseRun):
    def __init__(self, config, project, run_name, rank: Optional[int] = 0):
        super().__init__(config, project, run_name, rank)

        # model
        self.model = ImageCaptioner(self.config.img_encoder_name, self.config.text_encdec_name,
                                    lora_config=self.config.lora_config, device=self.rank)
        self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        # dataset
        train_dset = ConCapDataset('train', self.config.img_encoder_name, self.config.text_encdec_name,
                                   subset_size=self.config.data_subset_size)
        val_dset = ConCapDataset('dev', self.config.img_encoder_name, self.config.text_encdec_name)

        # dataloader for multiprocessing
        self.train_dataloader = DataLoader(train_dset,
                                           batch_size=self.config.batch_size_per_gpu,
                                           shuffle=False,
                                           num_workers=0,
                                           pin_memory=True,
                                           sampler=DistributedSampler(train_dset),
                                           collate_fn=train_dset.collate_fn)
        self.val_dataloader = DataLoader(val_dset,
                                         batch_size=self.config.batch_size_per_gpu,
                                         shuffle=False,
                                         num_workers=0,
                                         pin_memory=True,
                                         sampler=DistributedSampler(val_dset),
                                         collate_fn=val_dset.collate_fn)

    def __call__(self):
        # count parameters
        if self.rank == 0:
            self._count_params()

        # setup variables that span across epochs
        self.batch_idx = 0
        self.running_loss_batch = 0.0
        self.best_val_loss = float('inf')

        # train/eval loop
        super().__call__()

    def _count_params(self):
        """Count number of trainable and total parameters and log to wandb"""
        wandb.run.summary['num_trainable_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        wandb.run.summary['num_total_params'] = sum(p.numel() for p in self.model.parameters())

    def train(self, epoch):
        print(f"GPU {self.rank}: Training epoch {epoch + 1}")

        running_loss_epoch = 0.0
        self.model.train()
        self.train_dataloader.sampler.set_epoch(epoch)

        for caption, image in tqdm(self.train_dataloader):
            self.batch_idx += 1

            # forward
            loss = self.model(caption.to(self.rank), image.to(self.rank))

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss_epoch += loss.item()
            self.running_loss_batch += loss.item()

            # log running_loss_batch
            if self.batch_idx % self.config.log_batch_interval == 0:
                # aggregate over all processes
                self.running_loss_batch = torch.tensor(self.running_loss_batch, device=self.rank)
                dist.all_reduce(self.running_loss_batch)  # sum

                # log (only on rank 0)
                if self.rank == 0:
                    self.running_loss_batch = self.running_loss_batch.item() / dist.get_world_size()
                    wandb.log({'batch': self.batch_idx * dist.get_world_size(),
                               'train_loss_batch': self.running_loss_batch / self.config.log_batch_interval})

                # reset
                self.running_loss_batch = 0.0

        # log running_loss_epoch
        running_loss_epoch = torch.tensor(running_loss_epoch, device=self.rank)
        dist.all_reduce(running_loss_epoch)  # sum

        if self.rank == 0:
            running_loss_epoch = running_loss_epoch.item() / dist.get_world_size()
            wandb.log({'epoch': epoch + 1, 'train_loss': running_loss_epoch / len(self.train_dataloader)})

    def validate(self, epoch):
        running_loss_epoch = 0.0
        self.model.eval()
        self.val_dataloader.sampler.set_epoch(epoch)

        with torch.no_grad():
            for caption, image in tqdm(self.val_dataloader):
                # forward
                loss = self.model(caption.to(self.rank), image.to(self.rank))
                running_loss_epoch += loss.item()

        # log running_loss_epoch
        running_loss_epoch = torch.tensor(running_loss_epoch, device=self.rank)
        dist.all_reduce(running_loss_epoch)  # sum

        if self.rank == 0:
            running_loss_epoch = running_loss_epoch.item() / dist.get_world_size()
            wandb.log({'epoch': epoch + 1, 'val_loss': running_loss_epoch / len(self.val_dataloader)})

            # compare with best loss
            if running_loss_epoch < self.best_val_loss:
                self.best_val_loss = running_loss_epoch
                wandb.run.summary['best_val_loss'] = self.best_val_loss / len(self.val_dataloader)

                # save model
                self.model.save(self.save_dir)
                print(f'Best model saved at {self.save_dir}')
