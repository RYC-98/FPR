import torch

from ..utils import *
from ..attack import Attack
from functools import partial
accumulated_features = {} 


class VITB_VMI(Attack):
    """
    VMI-FGSM Attack
    'Enhancing the transferability of adversarial attacks through variance tuning (CVPR 2021)'(https://arxiv.org/abs/2103.15571)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        beta (float): the relative value for the neighborhood.
        num_neighbor (int): the number of samples for estimating the gradient variance.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        
    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, beta=1.5, num_neighbor=20, epoch=10, decay=1.
    """
    
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, beta=1.5, num_neighbor=5, epoch=10, decay=1., targeted=False, 
                random_start=False, norm='linfty', loss='crossentropy', device=None, attack='VMI-FGSM', **kwargs):
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.radius = beta * epsilon
        self.epoch = epoch
        self.decay = decay
        self.num_neighbor = num_neighbor

        self.model_name = model_name
        self.model = self.model[1]              


        self._register_model()                  
        self.model = wrap_model(self.model.eval().cuda()) # wrap the model again


    def get_variance(self, data, delta, label, cur_grad, momentum, **kwargs):
        """
        Calculate the gradient variance    
        """
        grad = 0
        for _ in range(self.num_neighbor):
            # Obtain the output
            # This is inconsistent for transform!
            logits = self.get_logits(self.transform(data+delta+torch.zeros_like(delta).uniform_(-self.radius, self.radius).to(self.device), momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad += self.get_grad(loss, delta)

        return grad / self.num_neighbor - cur_grad

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for VMI-FGSM

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        momentum, variance = 0, 0
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad+variance, momentum)

            # Calculate the variance
            variance = self.get_variance(data, delta, label, grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, self.alpha)

        return delta.detach()
    
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


            for i in range(12):
                self.model.blocks[i].attn.register_forward_hook(partial(cross_iter_emb_momentum, scale=0.8, mom_emb_decay=0.3)) # default: scale=0.8, mom_emb_decay=0.3

            for i in range(12):
                self.model.blocks[i].mlp.register_forward_hook(partial(cross_iter_emb_momentum, scale=0.8, mom_emb_decay=0.3))




