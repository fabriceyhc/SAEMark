# This script is used to judge the sigma(insignificant features) / sigma(significant features) ratio

import torch
import numpy as np

def get_mask(device, restricted):
    mask = [(0 if i in restricted else 1)for i in range(16384)]
    return torch.asarray(mask).to(device)
    
def gather_residual_activations(model, target_layer, inputs, attention_mask):
    target_act = None
    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal target_act # make sure we can modify the target_act from the outer scope
        target_act = outputs[0]
        return outputs
    
    # Move inputs to the same device as the model
    inputs = inputs.to(model.device)

    handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
    _ = model(inputs, attention_mask)
    handle.remove()
    return target_act

def calc_ratio(model, tokenizer, prompt, path_to_params, device, feature_mask):
    torch.cuda.set_device(device=device)
    from jump_relu_sae import JumpReLUSAE

    torch.set_grad_enabled(False)

    # set the tokenizer_padding side to right
    tokenizer.padding_side = "right"

    # Use the tokenizer to convert it to tokens. 
    # Note that this implicitly add a special "Beginning of Sequence" or <bos> token to the start
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True).to(device)

    # use the attention mask to calculate the length
    attention_mask = inputs['attention_mask']
    lengths = torch.sum(attention_mask, dim=1)

    params = np.load(path_to_params)
    pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}

    sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
    sae.load_state_dict(pt_params)

    # get input_ids from batch encoding
    inputs = inputs['input_ids']
    inputs = inputs.to(device)  # assume that inputs is a PyTorch tensor

    target_act = gather_residual_activations(model, 20, inputs, attention_mask)
    sae.cuda()
    sae_acts = sae.encode(target_act.to(torch.float32))

    # set the res list
    res = list()

    # go for each to calculate max_ind, ratio
    for i, sae_act in enumerate(sae_acts):
        # ignore the <bos> token of sae_act
        # and ignore the padding space
        cleared_sae_act = sae_act[1:lengths[i]]
        # multiply the cleared_sae_act by feature_mask
        cleared_sae_act = cleared_sae_act * feature_mask
        # get the max_ind of sae_act
        _, inds = cleared_sae_act.max(-1)
        # change the inds to ordinary array
        inds = inds.tolist()
        index = {}
        for ind in inds:
            if ind not in index:
                index[ind] = 1
            else:
                index[ind] += 1
        max_freq = max(index.values())
        max_inds = [k for k, v in index.items() if v == max_freq]
        if len(max_inds) > 1:
            max_ind = max(max_inds)
        else:
            max_ind = max_inds[0]
        # remove the repeat elements in inds
        inds = set(inds)
        # calculate the ratio
        col_sum = torch.sum(cleared_sae_act, dim=0)
        tot_sum_sig = torch.tensor(0, dtype=float).to(device)
        for ind in inds:
            tot_sum_sig += col_sum[ind]
        tot_sum_sig = tot_sum_sig.item()
        tot_sum = col_sum.sum()
        tot_sum = tot_sum.item()
        res.append([max_ind, round(tot_sum_sig / tot_sum, 4)])

    del sae_acts
    torch.cuda.empty_cache()
    return res

