import wandb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import ImageCaptioner
from src.data import ConCapDataset


class BaseRun:
    def __init__(self, config, project, run_name):
        wandb.init(project=project, name=run_name, config=config)
        self.config = wandb.config

    def __call__(self, debug=False):
        for epoch in range(self.config.num_epochs):
            self.train(epoch)
            self.validate(epoch)

            if debug:
                break

    def train(self, epoch):
        raise NotImplementedError

    def validate(self, epoch):
        raise NotImplementedError


class ImageCaptionerRun(BaseRun):
    def __init__(self, config, project, run_name):
        super().__init__(config, project, run_name)

        # model
        self.model = ImageCaptioner(self.config.img_encoder_name, self.config.text_encdec_name, self.config.lora_config, self.config.device)

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        # dataset
        train_dset = ConCapDataset('train', self.config.img_encoder_name, self.config.text_encdec_name)
        val_dset = ConCapDataset('validation', self.config.img_encoder_name, self.config.text_encdec_name)

        # dataloader
        self.train_dataloader = DataLoader(train_dset,
                                           batch_size=self.config.batch_size,
                                           shuffle=True,
                                           num_workers=self.config.num_workers,
                                           collate_fn=train_dset.collate_fn)
        self.val_dataloader = DataLoader(val_dset,
                                         batch_size=self.config.batch_size,
                                         shuffle=False,
                                         num_workers=self.config.num_workers,
                                         collate_fn=val_dset.collate_fn)

        # device
        self.device = self.config.device

    def __call__(self, *args, **kwargs):
        # count parameters
        wandb.run.summary['num_trainable_params'] = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        wandb.run.summary['num_total_params'] = sum(p.numel() for p in self.model.parameters())

        # setup
        self.batch_idx = 0
        self.running_loss_batch = 0.0
        self.best_val_loss = float('inf')

        # train/eval loop
        super().__call__(*args, **kwargs)

    def train(self, epoch):
        running_loss_epoch = 0.0
        self.model.train()

        for caption, image in tqdm(self.train_dataloader):
            self.batch_idx += 1

            # forward
            outputs = self.model(caption.to(self.device), image.to(self.device))
            loss = outputs.loss

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss_epoch += loss.item()
            self.running_loss_batch += loss.item()

            # log running_loss_batch
            if self.batch_idx % self.config.log_interval_batch == 0:
                wandb.log({'batch': self.batch_idx,
                           'train_loss_batch': self.running_loss_batch / self.config.log_interval_batch})
                self.running_loss_batch = 0.0

        # log running_loss_epoch
        wandb.log({'epoch': epoch + 1, 'train_loss': running_loss_epoch / len(self.train_dataloader)})

    def validate(self, epoch):
        running_loss_epoch = 0.0
        self.model.eval()

        with torch.no_grad():
            for caption, image in tqdm(self.val_dataloader):
                # forward
                outputs = self.model(caption.to(self.device), image.to(self.device))
                loss = outputs.loss
                running_loss_epoch += loss.item()

        # log running_loss_epoch
        wandb.log({'epoch': epoch + 1, 'val_loss': running_loss_epoch / len(self.val_dataloader)})

        # compare with best loss
        if running_loss_epoch < self.best_val_loss:
            self.best_val_loss = running_loss_epoch
            wandb.run.summary['best_val_loss'] = self.best_val_loss
            self.model.text_encdec.save_pretrained(f"{wandb.run.dir}/best_model")
