# import debugpy
# debugpy.listen(("0.0.0.0", 8901))  # 将端口设置为5678
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()  # 让程序在此等待，直到调试器连接


import math
import time

import torch
import torch.nn as nn
import transformers

from gptq_data_utils import get_gptq_loaders

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


DEBUG = False 
DEV = torch.device('cuda:0')
# DEV = torch.device('cpu')

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def find_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    
    return res

def quantize(x, scale, zero, maxq):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

def gptq2normal(base_model, gptq_model):
    print('init', base_model.state_dict()['backbone.layers.47.mixer.out_proj.weight'])
    for name, param in gptq_model.named_parameters():
        if 'mixer.out_proj' in name or 'mixer.in_proj' in name:
            base_model.state_dict()[name].copy_(param)
    print('after', base_model.state_dict()['backbone.layers.47.mixer.out_proj.weight'])
    return base_model

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True, 
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        trits=False
    ):
        self.maxq = torch.tensor(2 ** bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink 
        if trits:
            self.maxq = torch.tensor(-1) 

    def find_params(self, x, weight=False):
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmin < 0
            if torch.any(tmp):
                xmin[tmp] = -xmax[tmp]
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        if self.maxq < 0:
          self.scale = xmax
          self.zero = xmin
        else:
          self.scale = (xmax - xmin) / self.maxq
          if self.sym:
              self.zero = torch.full_like(self.scale, (self.maxq + 1) / 2)
          else:
              self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid 
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / self.maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else self.zero
                q = quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)
                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1)) 
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def quantize(self, x):
        if self.ready():
            return quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


# Assumes layer is perfectly divisible into 1024 * 1024 blocks
class Quant3Linear(nn.Module): 

    def __init__(self, infeatures, outfeatures, faster=False):
        super().__init__()
        self.register_buffer('zeros', torch.zeros((outfeatures, 1)))
        self.register_buffer('scales', torch.zeros((outfeatures, 1)))
        self.register_buffer('bias', torch.zeros(outfeatures))
        self.register_buffer(
            'qweight', torch.zeros((infeatures // 32 * 3, outfeatures), dtype=torch.int)
        )
        self.faster = faster

    def pack(self, linear, scales, zeros):
        self.zeros = zeros * scales
        self.scales = scales.clone()
        if linear.bias is not None:
            self.bias = linear.bias.clone()

        intweight = torch.round((linear.weight.data + self.zeros) / self.scales).to(torch.int)
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)
        qweight = np.zeros(
            (intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32
        )
        i = 0
        row = 0
        while row < qweight.shape[0]:
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i))
            i += 10
            qweight[row] |= intweight[i] << 30
            row += 1
            qweight[row] |= (intweight[i] >> 2) & 1
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 1)
            i += 10
            qweight[row] |= intweight[i] << 31
            row += 1
            qweight[row] |= (intweight[i] >> 1) & 0x3
            i += 1
            for j in range(i, i + 10):
                qweight[row] |= intweight[j] << (3 * (j - i) + 2)
            i += 10
            row += 1

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight) 

    def forward(self, x):
        if x.shape[-1] == x.numel():
            outshape = list(x.shape)
            y = self.bias.clone()
            outshape[-1] = self.bias.numel()
            dtype = x.dtype
            if self.faster:
                x = x.half()
                quant_cuda.vecquant3matmul_faster(x, self.qweight, y, self.scales, self.zeros)
            else:
                x = x.float()
                quant_cuda.vecquant3matmul(x, self.qweight, y, self.scales, self.zeros)
            y = y.to(dtype)
            return y.reshape(outshape)
        raise ValueError('Only supports a single token currently.')

def make_quant3(module, names, name='', faster=False):
    if isinstance(module, Quant3Linear):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in names:
            setattr(
                module, attr, Quant3Linear(tmp.in_features, tmp.out_features, faster=faster)
            )
    for name1, child in module.named_children():
        make_quant3(child, names, name + '.' + name1 if name != '' else name1, faster=faster)

def llama_pack3(model, quantizers):
    layers = find_layers(model)
    layers = {n: layers[n] for n in quantizers}
    make_quant3(model, quantizers)
    qlayers = find_layers(model, [Quant3Linear])
    print('Packing ...')
    for name in qlayers:
        # print(name)
        quantizers[name] = quantizers[name].cpu()
        qlayers[name].pack(layers[name], quantizers[name].scale, quantizers[name].zero)
    print('Done.')
    return model


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        # print("W", W)
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        # print('h', H)
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        # print('full', full)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    
    return quantizers


@torch.no_grad()
def mamba_sequential(model, dataloader, dev):
    print('Starting ...')

    layers = model.backbone.layers

    model.backbone.embedding = model.backbone.embedding.to(dev)
    model.backbone.norm_f = model.backbone.norm_f.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.d_model), dtype=dtype, device=dev
    )
    
    cache = {'i': 0}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, residual=None, **kwargs):
            # print('inp', inp)
            inps[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.backbone.embedding = model.backbone.embedding.cpu()
    model.backbone.norm_f = model.backbone.norm_f.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer, layers=[nn.Linear])
        # print('full', full)
        

        if args.true_sequential:
            sequential = [
                ['mixer.out_proj'], 
                ['mixer.in_proj']
            ]
        else:
            sequential = [list(full.keys())]
       
        for names in sequential:
            # print('sequential', sequential, names, i)
            subset = {n: full[n] for n in names}
            

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )
            # print('gptq', gptq)
            def add_batch(name):
                def tmp(_, inp, out):
                    # print(f'{name} add_batch inp', inp)
                    # print(f'{name} add_batch out', out)
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                # print('for name in subset', name)
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            # print('handles', handles)
            # print('after gptq', i, gptq, )
            for j in range(args.nsamples):
                # print(f'inps[{j}].unsqueeze(0)', inps[j].unsqueeze(0))
                outs[j] = layer(inps[j].unsqueeze(0))[0]
                # print(f'outs[{j}]', outs[j])
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')
                gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups
                )
                quantizers['backbone.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                gptq[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0))[0]

        layers[i] = layer.cpu()
        del layer
        del gptq 
        torch.cuda.empty_cache()

        # print('outs', outs)
        # print('inps', inps)

        inps, outs = outs, inps
    
    return quantizers


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    ) 
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 8, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--save', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--eval', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )

    args = parser.parse_args()

    # model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype='auto').to(DEV)
    # model.seqlen = 2048
    # model.eval()
    # dataloader, testloader = get_gptq_loaders(
    #     args.dataset, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, 
    #     model=args.model
    # )

    model = MambaLMHeadModel.from_pretrained(args.model)
    # print('init', model.state_dict()['backbone.layers.47.mixer.out_proj.weight'])
    model.eval()
    dataloader, testloader = get_gptq_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, seqlen=2048, 
        model="EleutherAI/gpt-neox-20b"
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        # quantizers = llama_sequential(model, dataloader, DEV)
        quantizers = mamba_sequential(model, dataloader, DEV)
        # print(time.time() - tick)

    if args.save:
        base_model = MambaLMHeadModel.from_pretrained(args.model).to('cpu')
        model = gptq2normal(base_model, model.to('cpu'))
        torch.save(model.state_dict(), args.save)

    # if args.eval:
    #     base_model = MambaLMHeadModel.from_pretrained(args.model).to('cpu')

    #     tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    #     # for k in model.state_dict():
    #     #     print('before', k)
    #     print('before', model.state_dict()['backbone.layers.47.mixer.out_proj.weight'])
    #     llama_pack3(model, quantizers)
    #     model.to(DEV)
    #     # for k in model.state_dict():
    #     #     print('after', k)
    #     print('after', model.state_dict()['backbone.layers.47.mixer.out_proj.qweight'])
    #     exit()
    #     res = {}
    #     ppl_res = evaluate_ckpt_ppl(
    #         model = model,
    #         tokenizer = tokenizer,
    #         ppl_datasets = ['wikitext2', 'ptb', 'c4'],
    #         max_length = 2048
    #     )
    #     res.update(ppl_res)
    #     task_res = evaluate_ckpt_task(
    #         model = model,
    #         tokenizer = tokenizer,
    #         tasks = ['boolq','piqa','hellaswag','winogrande','arc_easy','arc_challenge','openbookqa'],
    #         num_fewshot = 0,
    #         batch_size = 16,
    #         max_length = 2048
    #         )
    #     res.update(task_res)
    #     with Path(f'eval_result/{args.model.replace("/", "_")}.json').open('w') as f:
    #         json.dump(res, f, indent=4)

    
    # CUDA_VISIBLE_DEVICES=0 python gptq.py pretrained/mamba2-780m c4 --wbits 4 --true-sequential --act-order --save gptq_mamba/mamba2_780M_4bit_seq/pytorch_model.bin
    # CUDA_VISIBLE_DEVICES=1 python gptq.py pretrained/mamba2-1.3b c4 --wbits 4 --true-sequential --act-order --save gptq_mamba/mamba2_1.3b_4bit_seq/pytorch_model.bin
    # CUDA_VISIBLE_DEVICES=2 python gptq.py pretrained/mamba2-2.7b c4 --wbits 4 --true-sequential --act-order --save gptq_mamba/mamba2_3b_4bit_seq/pytorch_model.bin