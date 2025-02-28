from functools import partial

import torch

from ..gradient.mifgsm import MIFGSM
from ..utils import *


class GNS(MIFGSM):
    """
    TGR (Token Gradient Regularization)
    'Transferable Adversarial Attacks on Vision Transformers with Token Gradient Regularization (CVPR 2023)'(https://arxiv.org/abs/2303.15754)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1.0, mlp_gamma=0.25 (we follow mlp_gamma=0.5 in official code)

    Example script:
        python main.py --attack=tgr --input_dir=./data --output_dir=./results/tgr/vit --model vit_base_patch16_224 --batchsize 1

    NOTE:
        1) The code only support batchsize = 1.
    """



    def __init__(self, **kwargs):
        self.model_name = kwargs['model_name']  
        super().__init__(**kwargs)              

        self.model = self.model[1]              
        
        self.u = 0.6

        self._register_model()                  

        self.model = wrap_model(self.model.eval().cuda()) # wrap the model again


    def _register_model(self):
        def attn_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            return (out_grad, )

        def attn_cait_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]

            B, H, W, C = grad_in[0].shape
            return (out_grad, )

        def q_tgr(module, grad_in, grad_out, gamma):
            # cait Q only uses class token
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            return (out_grad, grad_in[1], grad_in[2])

        def v_tgr(module, grad_in, grad_out, name):
            
            is_high_pytorch = False
            if len(grad_in[0].shape) == 2:                
                grad_in = list(grad_in)                   
                is_high_pytorch = True
                grad_in[0] = grad_in[0].unsqueeze(0)      

            if self.model_name in ['visformer_small']:
                B, C, H, W = grad_in[0].shape
                c_mus = torch.mean(abs(grad_in[0]), dim=[0, 2, 3]).cpu().numpy()
                mu = np.mean(c_mus)
                std = np.std(c_mus)
                muustd = mu + self.u * std
                c_factor = c_mus > muustd
                c_temp = np.tanh(abs((c_mus-mu)/std))
                c_factor = np.array(c_factor).astype(np.float32)
                c_factor[c_factor == False] = c_temp[c_factor == False]
                grad_in[0][:, :, :, :] = grad_in[0][:, :, :, :] * torch.from_numpy(c_factor).to(grad_in[0].device).view(1, C, 1, 1)
                
                                    
            if self.model_name in ['vit_base_patch16_224', 'pit_b_224', 'cait_s24_224']:
                c = grad_in[0].shape[2]
                c_mus = torch.mean(abs(grad_in[0]), dim=[0, 1]).cpu().numpy()
                mu = np.mean(c_mus)
                std = np.std(c_mus)
                muustd = mu + self.u * std
                c_factor = c_mus > muustd
                c_temp = np.tanh(abs((c_mus-mu)/std))
                c_factor = np.array(c_factor).astype(np.float32)
                c_factor[c_factor == False] = c_temp[c_factor == False]
                grad_in[0][:, :, :] = grad_in[0][:, :, :] * torch.from_numpy(c_factor).to(grad_in[0].device).view(1, 1, c)

            mask = torch.ones_like(grad_in[0])
            out_grad = mask * grad_in[0][:]

            if is_high_pytorch:
                out_grad = out_grad.squeeze(0)                    

            # return (out_grad, grad_in[1])
            for i in range(len(grad_in)):                         
                if i == 0:
                    return_dics = (out_grad,)                     
                else:
                    return_dics = return_dics + (grad_in[i],)     
            return return_dics


            # return (out_grad, grad_in[1])


        def mlp_tgr(module, grad_in, grad_out, gamma):
            mask = torch.ones_like(grad_in[0]) * gamma
            out_grad = mask * grad_in[0][:]
            for i in range(len(grad_in)):
                if i == 0:
                    return_dics = (out_grad,)
                else:
                    return_dics = return_dics + (grad_in[i],)
            return return_dics

        attn_tgr_hook = partial(attn_tgr, gamma=0.5)
        attn_cait_tgr_hook = partial(attn_cait_tgr, gamma=0.5)
        v_tgr_hook = v_tgr
        q_tgr_hook = partial(q_tgr, gamma=0.5)
        mlp_tgr_hook = partial(mlp_tgr, gamma=0.5)

        if self.model_name in ['vit_base_patch16_224', 'deit_base_distilled_patch16_224']:
            for i in range(12):

                self.model.blocks[i].attn.attn_drop.register_backward_hook(
                    attn_tgr_hook)
                self.model.blocks[i].attn.qkv.register_backward_hook(
                    partial(v_tgr_hook, name=f"model.blocks[{i}].attn.qkv"))
                self.model.blocks[i].mlp.register_backward_hook(mlp_tgr_hook)
        elif self.model_name == 'pit_b_224':
            for block_ind in range(13):
                if block_ind < 3:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 9 and block_ind >= 3:
                    transformer_ind = 1
                    used_block_ind = block_ind - 3
                elif block_ind < 13 and block_ind >= 9:
                    transformer_ind = 2
                    used_block_ind = block_ind - 9
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(
                    attn_tgr_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(
                    partial(v_tgr_hook, name=f"model.transformers[{transformer_ind}].blocks[{used_block_ind}].attn.qkv"))
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(
                    mlp_tgr_hook)
                
        elif self.model_name == 'pit_ti_224':
            for block_ind in range(12):
                if block_ind < 2:
                    transformer_ind = 0
                    used_block_ind = block_ind
                elif block_ind < 8 and block_ind >= 2:
                    transformer_ind = 1
                    used_block_ind = block_ind - 2
                elif block_ind < 12 and block_ind >= 8:
                    transformer_ind = 2
                    used_block_ind = block_ind - 8
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.attn_drop.register_backward_hook(
                    attn_tgr_hook)
                self.model.transformers[transformer_ind].blocks[used_block_ind].attn.qkv.register_backward_hook(
                    partial(v_tgr_hook, name=f"model.transformers[{transformer_ind}].blocks[{used_block_ind}].attn.qkv"))
                self.model.transformers[transformer_ind].blocks[used_block_ind].mlp.register_backward_hook(
                    mlp_tgr_hook)               

        elif self.model_name == 'cait_s24_224':
            for block_ind in range(26):
                if block_ind < 24:
                    self.model.blocks[block_ind].attn.attn_drop.register_backward_hook(
                        attn_tgr_hook)
                    self.model.blocks[block_ind].attn.qkv.register_backward_hook(
                        partial(v_tgr_hook, name=f"model.blocks[{block_ind}].attn.qkv"))
                    self.model.blocks[block_ind].mlp.register_backward_hook(
                        mlp_tgr_hook)
                elif block_ind > 24:
                    self.model.blocks_token_only[block_ind-24].attn.attn_drop.register_backward_hook(
                        attn_cait_tgr_hook)
                    self.model.blocks_token_only[block_ind -
                                                 24].attn.q.register_backward_hook(q_tgr_hook)
                    self.model.blocks_token_only[block_ind -
                                                    24].attn.k.register_backward_hook(partial(v_tgr_hook, name=f"model.blocks_token_only[{block_ind-24}].attn.k"))
                    self.model.blocks_token_only[block_ind -
                                                    24].attn.v.register_backward_hook(partial(v_tgr_hook, name=f"model.blocks_token_only[{block_ind-24}].attn.v"))
                    self.model.blocks_token_only[block_ind -
                                                 24].mlp.register_backward_hook(mlp_tgr_hook)
                    



