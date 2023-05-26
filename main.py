import argparse

from utils import *
from tqdm import tqdm
from model import *


def main(args):
    train_data, test_data, scaler, dim = load_dataset(args.dataset_dir, args.batch_size, args.train_rate,
                                                      args.miss_rate,
                                                      args.hint_rate)

    generator = Generator(dim)  # generaotr model, a simple full connection model. The end of the model is sigmod
    discriminator = Discriminator(
        dim)  # discriminator model, a simple full connection model. The end of the model is sigmod
    train(generator, discriminator, train_data, args.alpha, args.epoch, args.device)
    print('ok')


def train(generator, discriminator, train_data, alpha, epoch, device):
    if device == 'cuda:0':
        generator.to(device)
        discriminator.to(device)
    optimizer_D = torch.optim.Adam(params=discriminator.parameters())
    optimizer_G = torch.optim.Adam(params=generator.parameters())

    for i in tqdm(range(epoch)):
        x, m, h, z = train_data.get()
        new_x = m * x + (1 - m) * z
        if device == 'cuda:0':
            x = x.to(device)
            m = m.to(device)
            h = h.to(device)
            new_x = new_x.to(device)
        # train D
        optimizer_D.zero_grad()
        D_loss_curr = discriminator_loss(generator, discriminator, m, new_x, h)
        D_loss_curr.backward()
        optimizer_D.step()
        # train G
        optimizer_G.zero_grad()
        G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = generator_loss(generator, discriminator, x, m, new_x, h,
                                                                              alpha)
        G_loss_curr.backward()
        optimizer_G.step()
        if i % 100 == 0:
            print('Iter: {}'.format(i))
            print('Train_loss: {:.4}'.format(np.sqrt(MSE_train_loss_curr.item())))
            print('Test_loss: {:.4}'.format(np.sqrt(MSE_test_loss_curr.item())))
            print()


# GAIN Losses
def discriminator_loss(generator, discriminator, m, new_x, h):
    # Generator
    G_sample = generator(new_x, m)
    # Combine with original data
    Hat_New_X = new_x * m + G_sample * (1 - m)

    # Discriminator
    D_prob = discriminator(Hat_New_X, h)

    # Loss
    D_loss = -torch.mean(m * torch.log(D_prob + 1e-8) + (1 - m) * torch.log(1. - D_prob + 1e-8))
    return D_loss


def generator_loss(generator, discriminator, x, m, new_x, h, alpha):
    # Generator
    G_sample = generator(new_x, m)
    # Combine with original data
    Hat_New_X = new_x * m + G_sample * (1 - m)
    # Discriminator
    D_prob = discriminator(Hat_New_X, h)

    # Loss
    G_loss1 = -torch.mean((1 - m) * torch.log(D_prob + 1e-8))
    MSE_train_loss = torch.mean((m * new_x - m * G_sample) ** 2) / torch.mean(m)
    G_loss = G_loss1 + alpha * MSE_train_loss
    # MSE Performance metric
    MSE_test_loss = torch.mean(((1 - m) * x - (1 - m) * G_sample) ** 2) / torch.mean(1 - m)
    return G_loss, MSE_train_loss, MSE_test_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch', type=int, default=5000)
    parser.add_argument('--miss_rate', type=float, default=0.2)
    parser.add_argument('--hint_rate', type=float, default=0.9)
    parser.add_argument('--alpha', type=int, default=10, help='Loss Hyperparameters')
    parser.add_argument('--train_rate', type=float, default=0.8)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('----dataset_dir', default='Spam.csv')
    args = parser.parse_args()
    main(args)
    print("end")
