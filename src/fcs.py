# This script is used to judge the sigma(insignificant features) / sigma(significant features) ratio

import torch
import numpy as np

# Module-level SAE cache: (path_to_params, device_str) → JumpReLUSAE
# Avoids reloading the 500MB+ SAE from disk on every sentence.
_sae_cache: dict = {}

def get_mask(device, restricted):
    mask = [(0 if i in restricted else 1)for i in range(16384)]
    return torch.asarray(mask).to(device)

def gather_residual_activations(model, target_layer, inputs, attention_mask):
    target_act = None
    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal target_act
        target_act = outputs[0]
        return outputs

    inputs = inputs.to(model.device)
    attention_mask = attention_mask.to(model.device)

    handle = model.model.layers[target_layer].register_forward_hook(gather_target_act_hook)
    _ = model(inputs, attention_mask)
    handle.remove()
    # transformers v5: decoder layer hook returns (T, d_model) without batch dim
    if target_act.dim() == 2:
        target_act = target_act.unsqueeze(0)
    return target_act

def calc_ratio(model, tokenizer, prompt, path_to_params, device, feature_mask, target_layer=20):
    torch.cuda.set_device(device=device)
    from jump_relu_sae import JumpReLUSAE

    torch.set_grad_enabled(False)

    cache_key = (str(path_to_params), str(device))
    if cache_key not in _sae_cache:
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}
        sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1])
        sae.load_state_dict(pt_params)
        sae.to(device)
        _sae_cache[cache_key] = sae
    sae = _sae_cache[cache_key]

    tokenizer.padding_side = "right"

    res = list()

    # Process each sequence individually: transformers v5 decoder-layer hooks return
    # (T, d_model) without the batch dimension regardless of batch size, so batched
    # calls silently return only the first sequence's activations.
    for single_prompt in prompt:
        inputs = tokenizer(
            [single_prompt], return_tensors="pt",
            add_special_tokens=True, truncation=True,
        ).to(device)
        attention_mask = inputs['attention_mask']
        length = int(torch.sum(attention_mask).item())

        target_act = gather_residual_activations(
            model, target_layer, inputs['input_ids'], attention_mask
        )
        # target_act: (1, T, d_model)
        sae_act = sae.encode(target_act[0].to(torch.float32))  # (T, d_sae)

        # strip BOS token; sae_act[1:length] covers real tokens only
        cleared = sae_act[1:length] * feature_mask

        if cleared.shape[0] == 0:
            res.append([0, 0.0])
            continue

        _, inds = cleared.max(-1)
        inds_list = inds.tolist()
        index = {}
        for ind in inds_list:
            index[ind] = index.get(ind, 0) + 1
        max_freq = max(index.values())
        max_inds = [k for k, v in index.items() if v == max_freq]
        max_ind = max(max_inds) if len(max_inds) > 1 else max_inds[0]

        inds_set = set(inds_list)
        col_sum = torch.sum(cleared, dim=0)
        tot_sum_sig = sum(col_sum[ind].item() for ind in inds_set)
        tot_sum = col_sum.sum().item()
        ratio = round(tot_sum_sig / tot_sum, 4) if tot_sum != 0 else 0.0
        res.append([max_ind, ratio])

    return res
