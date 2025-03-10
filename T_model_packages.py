import torch
import torch.nn as nn


class ConvMINE(nn.Module):
    def __init__(self, input_channels, output_channels, img_size):
        super(ConvMINE, self).__init__()
        self.img_size = img_size
        self.output_channels = output_channels
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels + output_channels, 64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(4),
            nn.Sigmoid()
        )


    def forward(self, x, y):
        y = torch.nn.functional.pad(y, (0, self.img_size-y.size(2), 0, self.img_size-y.size(3)))
        x_y = torch.cat([x, y], dim=1)  # (batch_size, channels_x + channels_y, height, width)
        x_y = self.conv_layers(x_y)
        t = torch.mean(x_y, dim=[1,2,3])

        return t


# z_y mutual information estimator
class LrMINE(nn.Module):
    def __init__(self, input_channels, output_channels, img_size, batch, nz):
        super(LrMINE, self).__init__()
        # Translated comment
        self.img_size = img_size
        self.batch = batch
        self.input_channels= input_channels
        self.output_channels = output_channels
        self.learner_layers = nn.Sequential(
            nn.Linear(output_channels * input_channels + nz, 4096),
            nn.ReLU(),
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.Sigmoid()
        )


    def forward(self, x, y):
        x = x.view(self.batch, -1)
        x_y = torch.cat([x, y], dim=1)  # (batch_size, channels_x + channels_y, height, width)
        x_y = self.learner_layers(x_y)
        t = torch.mean(x_y, dim=[1])

        return t


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim]
    '''

    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size // 2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size // 2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size // 2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()

        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)