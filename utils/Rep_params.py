import torch
import torch.nn as nn
import torch.nn.functional as F


# 3*3卷积和BN融合
def merge_conv3x3_bn(conv_w, conv_b, bn_mean, bn_var, bn_gamma, bn_beta, bn_eps=1e-7):
    fused_w = (bn_gamma.view(-1, 1, 1, 1)*conv_w) / torch.sqrt(bn_var + bn_eps).view(-1, 1, 1, 1)
    fuse_b = bn_gamma * (conv_b - bn_mean) / torch.sqrt(bn_var + bn_eps) + bn_beta
    return fused_w, fuse_b

# 结构重新参数化
def rep_params(model, model_plain, opt, device):
    state_dict_model = model.state_dict()
    state_dict_model_plain = model_plain.state_dict()
    enumerate
    for key, value in state_dict_model_plain.items():
        if key in state_dict_model:
            state_dict_model_plain[key] = state_dict_model[key]

    for m in range(opt['m_block']):
        key_weight_model_plain = 'M_block.{}.weight'.format(2*m+2)
        key_bias_model_plain = 'M_block.{}.bias'.format(2*m+2)
        
        branch1_conv3x3_weight = state_dict_model['M_block.{}.branch1.0.weight'.format(2*m+2)]
        branch1_conv3x3_bias = state_dict_model['M_block.{}.branch1.0.bias'.format(2*m+2)]
        branch1_bn_weight = state_dict_model['M_block.{}.branch1.1.weight'.format(2*m+2)]
        branch1_bn_bias = state_dict_model['M_block.{}.branch1.1.bias'.format(2*m+2)]
        branch1_bn_mean = state_dict_model['M_block.{}.branch1.1.running_mean'.format(2*m+2)]
        branch1_bn_var = state_dict_model['M_block.{}.branch1.1.running_var'.format(2*m+2)]
        branch1_conv1x1_weight = state_dict_model['M_block.{}.branch1.2.weight'.format(2*m+2)]
        branch1_conv1x1_bias = state_dict_model['M_block.{}.branch1.2.bias'.format(2*m+2)]

        branch2_conv3x3_weight = state_dict_model['M_block.{}.branch2.0.weight'.format(2*m+2)]
        branch2_conv3x3_bias = state_dict_model['M_block.{}.branch2.0.bias'.format(2*m+2)]
        branch2_bn_weight = state_dict_model['M_block.{}.branch2.1.weight'.format(2*m+2)]
        branch2_bn_bias = state_dict_model['M_block.{}.branch2.1.bias'.format(2*m+2)]
        branch2_bn_mean = state_dict_model['M_block.{}.branch2.1.running_mean'.format(2*m+2)]
        branch2_bn_var = state_dict_model['M_block.{}.branch2.1.running_var'.format(2*m+2)]
        branch2_conv1x1_weight = state_dict_model['M_block.{}.branch2.2.weight'.format(2*m+2)]
        branch2_conv1x1_bias = state_dict_model['M_block.{}.branch2.2.bias'.format(2*m+2)]

        # 合并两条分支上的BN层和conv3x3
        branch1_fuse_w, branch1_fuse_b = merge_conv3x3_bn(branch1_conv3x3_weight, branch1_conv3x3_bias, branch1_bn_mean, branch1_bn_var, branch1_bn_weight, branch1_bn_bias)
        branch2_fuse_w, branch2_fuse_b = merge_conv3x3_bn(branch2_conv3x3_weight, branch2_conv3x3_bias, branch2_bn_mean, branch2_bn_var, branch2_bn_weight, branch2_bn_bias)
            
        # 两条分支下的3*3 + 1*1的合并
        branch1_w = torch.rand((opt["c_channel"], opt["c_channel"], 3, 3)).to(device)
        branch1_b = torch.rand([opt["c_channel"]]).to(device)
        branch2_w = torch.rand((opt["c_channel"], opt["c_channel"], 3, 3)).to(device)
        branch2_b = torch.rand([opt["c_channel"]]).to(device)

        for i in range(opt['c_channel']):
            branch1_w[i,...] = torch.sum(branch1_fuse_w * branch1_conv1x1_weight[i,...].unsqueeze(1), dim=0)
            branch1_b[i] = branch1_conv1x1_bias[i] + torch.sum(branch1_fuse_b * branch1_conv1x1_weight[i,...].squeeze(1).squeeze(1))

            branch2_w[i,...] = torch.sum(branch2_fuse_w * branch2_conv1x1_weight[i,...].unsqueeze(1), dim=0)
            branch2_b[i] = branch2_conv1x1_bias[i] + torch.sum(branch2_fuse_b * branch2_conv1x1_weight[i,...].squeeze(1).squeeze(1))

        # 残差分支合并
        branch_residual = torch.zeros(opt["c_channel"], opt["c_channel"], 3, 3, device=device)
        for i in range(opt["c_channel"]):
            branch_residual[i, i, 1, 1] = 1.0
        
        # 融合所有weight和bias
        weight = branch1_w + branch2_w + branch_residual
        bias = branch1_b + branch2_b
        state_dict_model_plain[key_weight_model_plain] = weight
        state_dict_model_plain[key_bias_model_plain] = bias
    model_plain.load_state_dict(state_dict_model_plain)
    return model_plain
