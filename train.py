# %% library
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.distributions.normal import Normal
from rich.console import Console
from tqdm import tqdm
import argparse
from loader import DefoggingDataset
import gc
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.optim as optim
import random
import torch.nn as nn
from model import Model
from pytorchtools import EarlyStopping


# %% Train
def train(args):

    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    console = Console()

    opt1 = optim.AdamW(params= args.model.parameters() ,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    if args.use_count:
        pass
    else:
        # criterion1 = nn.BCEWithLogitsLoss(reduction='mean')  # input: logit, target \in {0, 1}.
        criterion1 = nn.MSELoss(reduction='mean')


    writer = SummaryWriter(f'./runs/{args.experiment}')
    early_stopping = EarlyStopping(patience=10, verbose=False, path=f'./parameter/{args.experiment}.pth')

    steps_per_epoch = len(args.train_loader)
    for epoch in range(1, args.epochs + 1):

        total_loss = 0
        args.model.train()

        with tqdm(total=steps_per_epoch, leave=False, dynamic_ncols=True) as pbar:
            for i, batch in enumerate(args.train_loader):

                x = batch['input'].to(args.device)
                y = batch['target'].to(args.device)

                opt1.zero_grad()

                pred = args.model(x)

                recon_loss = criterion1(pred , y )
                recon_loss.backward()
                opt1.step()

                # opt2.zero_grad()
                # pred, z_fake = args.model(x)
                # z_real = Normal(loc = torch.zeros_like(z_fake), scale=1).sample()
                # c_fake_loss = criterion2(args.model.C(z_fake), y_fake)
                # c_real_loss = criterion2(args.model.C(z_real), y_real)
                # c_loss = 0.5 * (c_fake_loss + c_real_loss)
                # c_loss.backward()
                # nn.utils.clip_grad_norm_(args.model.C.parameters(), 1.)
                # opt2.step()

                pbar.update(1)

                # train_G_adv_loss += g_adv_loss.item()
                # train_G_recon_loss += recon_loss.item()
                # train_C_fake_loss += c_fake_loss.item()
                # train_C_real_loss += c_real_loss.item()

                total_loss += recon_loss

            # avg_adv_loss = train_G_adv_loss / steps_per_epoch
            # avg_recon_loss = train_G_recon_loss / steps_per_epoch
            avg_recon_loss = total_loss / steps_per_epoch
            # avg_fake_loss = train_C_fake_loss / steps_per_epoch
            # avg_real_loss = train_C_real_loss / steps_per_epoch

            # early_stopping(avg_recon_loss, args.model)
            early_stopping(avg_recon_loss, args.model)


            if early_stopping.early_stop:
                print('Early stopping')
                break

            console.print(f"Train [{epoch:>04}]/[{args.epochs:>04}]: ", end='', style="Bold Cyan")
            # console.print(f"adv_loss:{avg_adv_loss:.4f}", sep=' | ', style='Bold Blue')
            console.print(f"recon_loss:{avg_recon_loss:.4f}", sep=' | ', style='Bold Blue')
            # console.print(f"fake_loss:{avg_fake_loss:.4f}", sep=' | ', style='Bold Blue')
            # console.print(f"real_loss:{avg_real_loss:.4f}", sep=' | ', style='Bold Blue')

            # writer.add_scalar(tag='adv_loss', scalar_value=avg_adv_loss, global_step=epoch)
            writer.add_scalar(tag='recon_loss', scalar_value=avg_recon_loss, global_step=epoch)
            # writer.add_scalar(tag='fake_loss', scalar_value=avg_fake_loss, global_step=epoch)
            # writer.add_scalar(tag='real_loss', scalar_value=avg_real_loss, global_step=epoch)

            if epoch % 10 == 0:
                torch.save(args.model.state_dict() ,
                           os.path.join("D:\프로젝트\메타플레이\jinsoo\parameter", f"{args.experiment}_epoch_{epoch:04d}.pt")
                            )



# %% main
def main():
    parser = argparse.ArgumentParser(description="Defog")

    # parser
    parser.add_argument('--model', type=str, default='wgan-gp', help='')
    parser.add_argument('--data_dir', type=str, default='X:/pysc2-replay-parser/data/defogging', help='')
    parser.add_argument('--matchup', type=str, default='TvP', help='')
    parser.add_argument('--size', type=int, default=32, help='')
    parser.add_argument('--epochs', type=int, default=1000, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--num_workers', type=int, default=5, help='')
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--latent_dim', type=int, default=100, help='')
    parser.add_argument('--optimizer', type = str ,default='adamw', help='')
    parser.add_argument('--learning_rate', type = float ,default=1e-4, help='')
    parser.add_argument('--weight_decay', type = float ,default=5e-5, help='')
    parser.add_argument('--experiment', type = str ,default='temp4(not_use_count)_MSE', help='')
    parser.add_argument('--use_count', action='store_true')
    parser.add_argument('--reparameterize', action='store_true')



    args = parser.parse_args()

    dataset = DefoggingDataset(args)
    split_idx = int(len(dataset) * 0.99) # train 99퍼센트 , 1퍼센트
    train_indices = list(range(len(dataset)))[:split_idx]
    args.train_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=SubsetRandomSampler(train_indices),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False)

    print('loader created')

    args.model = Model(args).to(args.device)
    args.model.initialize_weights()

    gc.collect()
    train(args)


# %% run
if __name__ == "__main__":
    main()