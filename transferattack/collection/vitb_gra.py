from functools import partial

import torch

from ..gradient.mifgsm import MIFGSM
from ..utils import *



accumulated_features = {} 


class VITB_GRA(MIFGSM):



    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False,  random_start=False, norm='linfty', loss='crossentropy', device=None, 
                 attack='mda', **kwargs):        
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack)

        self.model_name = model_name
        self.model = self.model[1]             


        self._register_model()                  
        self.model = wrap_model(self.model.eval().cuda()) # wrap the model again


        self.radius = 3.5 * epsilon
        self.num_neighbor = 5
        self.epoch = epoch
        self.decay = decay

    def forward(self, data, label, **kwargs):
        """
        The attack procedure for GRA

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

        # Initialize the attenuation factor for decay indicator
        eta = 0.94

        # Initialize the decay indicator
        M = torch.full_like(delta, 1 / eta)

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data+delta, momentum=momentum))

            # Calculate the loss
            loss = self.get_loss(logits, label)

            # Calculate the current gradients
            grad = self.get_grad(loss, delta)

            # Calculate the average gradients
            samgrad = self.get_average_gradient(data, delta, label, momentum)

            # Calculate the cosine similarity
            s = self.get_cosine_similarity(grad, samgrad)

            # Calculate the global weighted gradient
            current_grad = s * grad + (1 - s) * samgrad

            # Save the previous perturbation
            last_momentum = momentum

            # Calculate the momentum
            momentum = self.get_momentum(current_grad, momentum)

            # Update decay indicator
            M = self.get_decay_indicator(M, delta, momentum, last_momentum, eta)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, momentum, M * self.alpha)

        return delta.detach()



    def get_average_gradient(self, data, delta, label, momentum, **kwargs):
        """
        Calculate the average gradient of the samples
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

        return grad / self.num_neighbor

    def get_cosine_similarity(self, cur_grad, sam_grad, **kwargs):
        """
        Calculate cosine similarity to find the score
        """

        cur_grad = cur_grad.view(cur_grad.size(0), -1)
        sam_grad = sam_grad.view(sam_grad.size(0), -1)

        cos_sim = torch.sum(cur_grad * sam_grad, dim=1) / (
                    torch.sqrt(torch.sum(cur_grad ** 2, dim=1)) * torch.sqrt(torch.sum(sam_grad ** 2, dim=1)))
        cos_sim = cos_sim.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        return cos_sim

    def get_decay_indicator(self, M, delta, cur_noise, last_noise, eta, **kwargs):
        """
        Define the decay indicator
        """
    
        if isinstance(last_noise, int):
            last_noise = torch.full(cur_noise.shape, last_noise)
        else:
            last_noise = last_noise

        if torch.cuda.is_available():
            last_noise = last_noise.cuda()

        last = last_noise.sign()
        cur = cur_noise.sign()
        eq_m = (last == cur).float()
        di_m = torch.ones_like(delta) - eq_m
        M = M * (eq_m + di_m * eta)

        return M

    

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




