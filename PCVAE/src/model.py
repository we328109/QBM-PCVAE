import random
from typing import ForwardRef
import torch
from torch import nn, softmax
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax
from utils import physic_informed
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import RestrictedBoltzmannMachine
from kaiwu.torch_plugin.qvae_dist_util import MixtureGeneric


class MLP_REG(nn.Module):
    def __init__(
        self,
        ft_dim,
        p=0.2
    ):
        super(MLP_REG, self).__init__()
        self.l1 = nn.Linear(ft_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, 64)
        self.l6 = nn.Linear(64, 1)
        self.p = p 
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.dropout(F.relu(self.l3(x)), p=self.p)
        x = F.dropout(F.relu(self.l4(x)), p=self.p)
        x = F.dropout(F.relu(self.l5(x)), p=self.p)
        return self.l6(x)


class MLP_CLA(nn.Module):
    def __init__(
        self,
        ft_dim,
        p=0.2
    ):
        super(MLP_CLA, self).__init__()
        self.l1 = nn.Linear(ft_dim, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 256)
        self.l4 = nn.Linear(256, 128)
        self.l5 = nn.Linear(128, 64)
        self.l6 = nn.Linear(64, 14)
        self.p = p

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.dropout(F.relu(self.l3(x)), p=self.p)
        x = F.dropout(F.relu(self.l4(x)), p=self.p)
        x = F.dropout(F.relu(self.l5(x)), p=self.p)
        return F.log_softmax(self.l6(x), dim=1)
        #return self.softmax(self.net(x))


class PILP(nn.Module):
    def __init__(
        self,
        ft_dim,
        training=True,
        latent_size=64,
        p=0.1,
        qbm_visible=None,
        dist_beta=10.0,
        qbm_sampler=None,
        qbm_sampler_alpha=0.95,
    ):
        super(PILP, self).__init__()
        # encoder
        self.fc1 = nn.Linear(ft_dim + 7, 512)
        self.fc2 = nn.Linear(512, latent_size)
        # decode
        self.a_pre = MLP_REG(ft_dim+latent_size, p)
        self.b_pre = MLP_REG(ft_dim+latent_size, p)
        self.c_pre = MLP_REG(ft_dim+latent_size, p)
        self.alpha_pre = MLP_REG(ft_dim+latent_size, p)
        self.beta_pre = MLP_REG(ft_dim+latent_size, p)
        self.gamma_pre = MLP_REG(ft_dim+latent_size, p)
        self.crystal_cla = MLP_CLA(ft_dim+latent_size, p)
        # functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.training = training
        self.latent_size = latent_size
        self.qbm_num_visible = qbm_visible or latent_size // 2
        self.qbm_num_hidden = latent_size - self.qbm_num_visible
        self.dist_beta = dist_beta
        self.qbm = RestrictedBoltzmannMachine(
            num_visible=self.qbm_num_visible,
            num_hidden=self.qbm_num_hidden,
        )
        self.qbm_sampler = qbm_sampler or SimulatedAnnealingOptimizer(
            alpha=qbm_sampler_alpha
        )

    def encode(self, gt, x): #(q(z|ft,gt))
        '''
        gt = ground truth, x = feature
        '''
        inputs = torch.cat([gt, x], 1)
        h1 = self.sigmoid(self.fc1(inputs))
        q_logits = self.fc2(h1)
        return q_logits

    def posterior(self, q_logits):
        posterior_dist = MixtureGeneric(q_logits, self.dist_beta)
        zeta = posterior_dist.reparameterize(self.training)
        return posterior_dist, zeta

    def _bm_negative_sample(self):
        return self.qbm.sample(self.qbm_sampler)

    def kl_divergence(self, posterior, zeta):
        entropy = torch.sum(posterior.entropy(), dim=1)
        logit_q = posterior.logit_mu
        log_ratio = posterior.log_ratio(zeta)

        if self.qbm.num_nodes != logit_q.shape[1]:
            raise ValueError(
                "QBM latent size does not match encoder output size."
            )

        logit_q1 = logit_q[:, : self.qbm_num_visible]
        logit_q2 = logit_q[:, self.qbm_num_visible :]
        log_ratio1 = log_ratio[:, : self.qbm_num_visible]

        q1 = torch.sigmoid(logit_q1)
        q2 = torch.sigmoid(logit_q2)
        q1_pert = torch.sigmoid(logit_q1 + log_ratio1)

        cross_entropy = -torch.matmul(
            torch.cat([q1, q2], dim=-1), self.qbm.linear_bias
        ) - torch.sum(
            torch.matmul(q1_pert, self.qbm.quadratic_coef) * q2, dim=1, keepdim=True
        )
        cross_entropy = cross_entropy.squeeze(dim=1)
        s_neg = self._bm_negative_sample()
        cross_entropy = cross_entropy - self.qbm(s_neg).mean()

        kl = cross_entropy - entropy
        return torch.mean(kl)

    def decode(self, z, x, crystal_gt=None):
        '''
        x = feature, z = latent
        '''
        inputs = torch.cat([z, x], 1)
        a,b,c,alpha,beta,gamma\
         =self.a_pre(inputs),self.b_pre(inputs),self.c_pre(inputs),\
             self.alpha_pre(inputs),self.beta_pre(inputs),self.gamma_pre(inputs)
        crystal = self.crystal_cla(inputs)
        #crytal_pre = torch.multinomial(crystal, 1).view(-1)
        #crytal_pre = torch.argmax(crystal, 1).view(-1)
        if crystal_gt is None:
            #print("None")
            crystal_gt = torch.multinomial(crystal, 1).view(-1).view(-1,1)
        #return crystal, physic_informed([a, b, c, alpha, beta, gamma], crystal_gt)
        #print(crystal.shape, crytal_pre.shape)
        #loss_extra = self.extra(a,b,c,alpha,beta,gamma,crytal_pre)
        return crystal, [a, b, c, alpha, beta, gamma]

    def forward(self, gt, x, crystal_gt=None):
        q_logits = self.encode(gt, x)
        posterior, zeta = self.posterior(q_logits)
        cry, pi = self.decode(zeta, x, crystal_gt)
        kl = self.kl_divergence(posterior, zeta)
        return cry, pi, q_logits, zeta, kl
