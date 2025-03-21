'''

Thanks for the contribution of Xiaosen Wang, etc, our code is modified from https://github.com/Trustworthy-AI-Group/TransferAttack

'''
import os
import argparse
import tqdm
import torch

import transferattack
from transferattack.utils import *

def get_parser():
    parser = argparse.ArgumentParser(description='Generating transferable adversaria examples')
    parser.add_argument('-e', '--eval', action='store_true', help='attack/evluation')
    parser.add_argument('--batchsize', default=2, type=int, help='the bacth size')
    parser.add_argument('--output_dir', default='./adv_data/our_vit', type=str, help='the path to store the adversarial patches')
    parser.add_argument('--attack', default='vitb', type=str, help='the attack algorithm', choices=transferattack.attack_zoo.keys())
    # vitb, deitb, pitt, caits, gra, vitb_gra, etc,  
    parser.add_argument('--model', default='vit_base_patch16_224' , type=str, help='the source surrogate model')
    # 'vit_base_patch16_224', 'cait_s24_224',  'pit_ti_224', 'deit_base_distilled_patch16_224'
    parser.add_argument('--epoch', default=10, type=int, help='the iterations for updating the adversarial patch')
    parser.add_argument('--eps', default=16 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--alpha', default=1.6 / 255, type=float, help='the stepsize to update the perturbation')
    parser.add_argument('--momentum', default=0., type=float, help='the decay factor for momentum based attack')
    parser.add_argument('--ensemble', action='store_true', help='enable ensemble attack')
    parser.add_argument('--random_start', default=False, type=bool, help='set random start')
    parser.add_argument('--input_dir', default='./data', type=str, help='the path for custom benign images, default: untargeted attack data')
    parser.add_argument('--targeted', action='store_true', help='targeted attack')
    parser.add_argument('--GPU_ID', default='0', type=str)
    return parser.parse_args()


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = AdvDataset(input_dir=args.input_dir, output_dir=args.output_dir, targeted=args.targeted, eval=args.eval)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=4)

    # attack
    if not args.eval:
        if args.attack in transferattack.attack_zoo: 
            if args.ensemble or len(args.model.split(','))>1: 
                args.model = args.model.split(',')            

            attacker = transferattack.attack_zoo[args.attack.lower()](model_name = args.model, targeted = args.targeted)
        else:
            raise Exception("Unspported attack algorithm {}".format(args.attack))

        for batch_idx, [images, labels, filenames] in tqdm.tqdm(enumerate(dataloader)):
            perturbations = attacker(images, labels)
            # break
            save_images(args.output_dir, images + perturbations.cpu(), filenames)
    
    # eval
    else:
        asr = dict()
        res = '|'
        for model_name, model in load_pretrained_model(cnn_model_paper, vit_model_paper): 

            model = wrap_model(model.eval().cuda())
            for p in model.parameters():
                p.requires_grad = False
            correct, total = 0, 0
            for images, labels, _ in dataloader:
                if args.targeted:
                    labels = labels[1]
                pred = model(images.cuda())
                correct += (labels.numpy() == pred.argmax(dim=1).detach().cpu().numpy()).sum()
                total += labels.shape[0]
            if args.targeted: # correct: pred == target_label
                asr[model_name] = (correct / total) * 100
            else: # correct: pred == original_label
                asr[model_name] = (1 - correct / total) * 100
            print(model_name, asr[model_name])
            res += ' {:.1f} |'.format(asr[model_name])

        print(asr)
        print(res)
        # with open('results_eval.txt', 'a') as f:
        #     f.write(args.output_dir + res + '\n')


if __name__ == '__main__':
    main()

