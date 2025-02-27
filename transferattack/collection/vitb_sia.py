import torch
import torch.nn.functional as F
from functools import partial

from ..utils import *
from ..gradient.mifgsm import MIFGSM

import scipy.stats as st
accumulated_features = {} 

class VITB_SIA(MIFGSM):
    """
    SIA(Structure Invariant Attack)
    'Structure Invariant Transformation for better Adversarial Transferability'(https://arxiv.org/abs/2309.14700)
    
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
    
    Example script:
        python main.py --attack sia --output_dir adv_data/sia/resnet18
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_scale=5, num_block=3, targeted=False, random_start=False, 
                norm='linfty', loss='crossentropy', device=None, attack='SIA', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)
        self.num_scale = num_scale
        self.num_block = num_block
        self.kernel = self.gkern()
        self.op = [self.vertical_shift, self.horizontal_shift, self.vertical_flip, self.horizontal_flip, self.rotate180, self.scale, self.add_noise]

        self.model_name = model_name
        self.model = self.model[1]              


        self._register_model()                  
        self.model = wrap_model(self.model.eval().cuda()) # wrap the model again

    def vertical_shift(self, x):
        _, _, w, _ = x.shape
        step = np.random.randint(low = 0, high=w, dtype=np.int32)
        return x.roll(step, dims=2)

    def horizontal_shift(self, x):
        _, _, _, h = x.shape
        step = np.random.randint(low = 0, high=h, dtype=np.int32)
        return x.roll(step, dims=3)

    def vertical_flip(self, x):
        return x.flip(dims=(2,))

    def horizontal_flip(self, x):
        return x.flip(dims=(3,))

    def rotate180(self, x):
        return x.rot90(k=2, dims=(2,3))
    
    def scale(self, x):
        return torch.rand(1)[0] * x

    def add_noise(self, x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16/255,16/255), 0, 1)

    def gkern(self, kernel_size=3, nsig=3):
        x = np.linspace(-nsig, nsig, kernel_size)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        stack_kernel = np.stack([kernel, kernel, kernel])
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return torch.from_numpy(stack_kernel.astype(np.float32)).to(self.device)

    def blur(self, x):
        return F.conv2d(x, self.kernel, stride=1, padding='same', groups=3)

    def blocktransform(self, x, choice=-1):
        _, _, w, h = x.shape
        y_axis = [0,] + np.random.choice(list(range(1, h)), self.num_block-1, replace=False).tolist() + [h,]
        x_axis = [0,] + np.random.choice(list(range(1, w)), self.num_block-1, replace=False).tolist() + [w,]
        y_axis.sort()
        x_axis.sort()
        
        x_copy = x.clone()
        for i, idx_x in enumerate(x_axis[1:]):
            for j, idx_y in enumerate(y_axis[1:]):
                chosen = choice if choice >= 0 else np.random.randint(0, high=len(self.op), dtype=np.int32)
                x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y] = self.op[chosen](x_copy[:, :, x_axis[i]:idx_x, y_axis[j]:idx_y])

        return x_copy

    def transform(self, x, **kwargs):
        """
        Scale the input for BSR
        """
        return torch.cat([self.blocktransform(x) for _ in range(self.num_scale)])

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




