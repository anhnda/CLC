import tqdm
import torch
import logging

from flatquant.utils import cleanup_memory
from flatquant.quant_utils import WeightQuantizer

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def find_qlayers(module, layers=[torch.nn.Linear, ], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


@torch.no_grad()
def rtn_fwrd(model, dev, args):
    '''
    Round-to-nearest weight quantization used by the FlatQuant pipeline.
    '''
    assert args.w_groupsize == -1, "Groupsize not supported in RTN!"
    layers = model.model.layers
    torch.cuda.empty_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = find_qlayers(layer, layers=[torch.nn.Linear])

        for name in subset:
            layer_weight_bits = args.w_bits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue

            quantizer = WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=not(args.w_asym), mse=args.rtn_mse
            )
            W = subset[name].weight.data
            w_dtype = W.dtype
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(w_dtype)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer

    cleanup_memory(verbose=True)
    return quantizers
