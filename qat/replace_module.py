import torch.nn as nn

# from .outlier_quantizer import (
#     BinaryXnorExceptOutliersLinear,
#     BinaryXnorExceptOutliersLinearHessian,
# )
from .utils_quant import QuantizeLinear
from .learnable_binarizer import BinaryLinearWscales
from .base_binarizer import BinaryInterface

def replace_with_outlier_binarylinear(model, binarization_method, outlier_fraction, keep_parts):
    module_name_dict = {name: module for name, module in model.named_modules()}
    for name, module in module_name_dict.items():
        if any(kp in name for kp in keep_parts):
            continue
        if isinstance(module, nn.Linear):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            if binarization_method == "xnor_outlier":
                qlinear = BinaryXnorExceptOutliersLinear(
                    module.weight, module.bias, outlier_fraction
                )
            elif binarization_method == "xnor_outlier_hessian":
                qlinear = BinaryXnorExceptOutliersLinearHessian(
                    module.weight, module.bias, outlier_fraction
                )
            else:
                raise NotImplementedError
            setattr(father, name[ind + 1 :], qlinear)
            print(f"replace layer {name} with {qlinear}")
            # qlinear.global_name = args.model_name + name

    return model


def replace_with_learnable_binarylinear(model, scaling_pattern, keep_parts):
    module_name_dict = {name: module for name, module in model.named_modules()}
    for name, module in module_name_dict.items():
        if any(kp in name for kp in keep_parts):
            continue
        if isinstance(module, nn.Linear):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            qlinear = BinaryLinearWscales(
                module.weight, module.bias, scaling_pattern
            )
            print(name, module.weight.shape, qlinear.weight.shape)
            setattr(father, name[ind + 1 :], qlinear)

    return model


def replace_with_quantizelinear(model, w_bits, scaling_pattern, keep_parts):
    module_name_dict = {name: module for name, module in model.named_modules()}
    for name, module in module_name_dict.items():
        if any(kp in name for kp in keep_parts):
            continue
        if isinstance(module, nn.Linear):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            qlinear = QuantizeLinear(module.weight.shape[1], module.weight.shape[0], w_bits = w_bits, scaling_pattern=scaling_pattern)
            print(name, module.weight.shape, qlinear.weight.shape)
            setattr(father, name[ind + 1 :], qlinear)
            print(f"replace layer {name} ({module.weight.shape}) with {qlinear}")
            
    return model


def binarylinear_to_regularlinear(model):
    module_name_dict = {name: module for name, module in model.named_modules()}
    for name, module in module_name_dict.items():
        if isinstance(module, BinaryInterface):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            linear = module.to_regular_linear()
            setattr(father, name[ind + 1 :], linear)
            print(f"replace layer {name} with {linear}")

    return model


def check_para_state(model):
    trainable_params, all_param = 0, 0
    for key, param in model.named_parameters():
        trainable = 'T*'
        all_param += param.numel()
        if param.requires_grad:
            trainable = 'T+'
            trainable_params += param.numel()
        else:
            trainable = 'T-'
        print(trainable, key)
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    for name, module in model.named_modules():
        binarized = 'B*'
        if isinstance(module, nn.Linear):
            binarized = 'B-'
        elif isinstance(module, BinaryInterface):
            binarized = 'B+'
        print(binarized, name)

def check_outlinear_frac(model):
    tot_bit=0
    tot_params=0
    for name, module in model.named_modules():
        if isinstance(module, BinaryInterface):
            module.gen_outlier_mask()
            # print(module.outlier_nbits)
            tot_bit+=(module.outlier_nbits+1)*module.weight.numel()
            tot_params+=module.weight.numel()
    print(f"mean_bit: {tot_bit/tot_params} frac: {tot_bit/tot_params/16}")