import torch
import random

from ..utils import *
from ..gradient.mifgsm import MIFGSM
from functools import partial
accumulated_features = {} 

class VITB_BSR(MIFGSM):
    """
    BSR Attack
    'Boosting Adversarial Transferability by Block Shuffle and Rotation'(https://https://arxiv.org/abs/2308.10299)
    
    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        num_scale (int): the number of shuffled copies in each iteration.
        num_block (int): the number of block in the image.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_scale=10, num_block=3
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_scale=5, num_block=3, targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='SIM', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_scale = num_scale
        self.num_block = num_block

        self.model_name = model_name
        self.model = self.model[1]              


        self._register_model()                  
        self.model = wrap_model(self.model.eval().cuda()) # wrap the model again


    def get_length(self, length):
        rand = np.random.uniform(size=self.num_block)
        rand_norm = np.round(rand/rand.sum()*length).astype(np.int32)
        rand_norm[rand_norm.argmax()] += length - rand_norm.sum()
        return tuple(rand_norm)

    def shuffle_single_dim(self, x, dim):
        lengths = self.get_length(x.size(dim))
        x_strips = list(x.split(lengths, dim=dim))
        random.shuffle(x_strips)
        return x_strips


    def shuffle(self, x):
        dims = [2,3]
        random.shuffle(dims)
        x_strips = self.shuffle_single_dim(x, dims[0])
        return torch.cat([torch.cat(self.shuffle_single_dim(x_strip, dim=dims[1]), dim=dims[1]) for x_strip in x_strips], dim=dims[0])

    def transform(self, x, **kwargs):
        """
        Scale the input for BSR
        """
        return torch.cat([self.shuffle(x) for _ in range(self.num_scale)])

    def get_loss(self, logits, label):
        """
        Calculate the loss
        """
        return -self.loss(logits, label.repeat(self.num_scale)) if self.targeted else self.loss(logits, label.repeat(self.num_scale))


    
    def _register_model(self):


        def diverse_attn_map(module, input, output, attn_map_change_range):

            batch_size, num_heads, seq_length, _ = output.shape
            s_output = output * 1.0

            attn_map_noise = torch.empty_like(s_output)

            for head in range(num_heads):

                M = torch.tensor(np.random.uniform(1 - attn_map_change_range, 1 + attn_map_change_range, (seq_length, seq_length)), dtype=torch.float32).to(output.device)
                
                noisy_attn = s_output[:, head, :, :] * M 
                
                normalized_attn = torch.softmax(noisy_attn, dim=-1)

                attn_map_noise[:, head, :, :] = normalized_attn

            return attn_map_noise



        def cross_iter_emb_momentum(module, input, output, scale, mom_emb_decay): 

            s = scale 
            s_output = output * s


            module_id = id(module)
            if module_id not in accumulated_features:
                accumulated_features[module_id] = s_output.clone()
            else:
                accumulated_features[module_id] = mom_emb_decay * accumulated_features[module_id].clone().detach() + s_output
                 
            return accumulated_features[module_id]
        


        if self.model_name in ['vit_base_patch16_224' ]: 

            for i in [0,1,4,9,11]: 
                self.model.blocks[i].attn.attn_drop.register_forward_hook(partial(diverse_attn_map, attn_map_change_range=25))  # default: 25


            if self.mom_attn_emb:
                for i in range(12):
                    self.model.blocks[i].attn.register_forward_hook(partial(cross_iter_emb_momentum, scale=0.8, mom_emb_decay=0.3)) # default: scale=0.8, mom_emb_decay=0.3

            if self.mom_mlp_emb:
                for i in range(12):
                    self.model.blocks[i].mlp.register_forward_hook(partial(cross_iter_emb_momentum, scale=0.8, mom_emb_decay=0.3))
