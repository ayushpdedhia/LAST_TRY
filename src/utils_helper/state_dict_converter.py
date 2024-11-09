import torch
import torch.nn as nn

def convert_state_dict(old_state_dict):
    """
    Convert and resize state dict to match new architecture
    """
    new_state_dict = {}
    
    # Architecture channel sizes
    encoder_channels = [3, 64, 128, 256, 512, 512, 512, 512, 512]
    decoder_channels = [512, 512, 512, 512, 256, 128, 64, 3]
    
    # Initialize batch norm for all layers (will be overwritten if exists in weights)
    def init_bn_params(size):
        return {
            'weight': torch.ones(size),
            'bias': torch.zeros(size),
            'running_mean': torch.zeros(size),
            'running_var': torch.ones(size)
        }
    
    # Handle encoders
    for i in range(1, 9):
        enc_prefix = f'enc{i}'
        
        # Convert and resize conv weights
        if f'{enc_prefix}.0.weight' in old_state_dict:
            weight = old_state_dict[f'{enc_prefix}.0.weight']
            bias = old_state_dict[f'{enc_prefix}.0.bias']
            
            # Resize if needed
            if weight.size(0) != encoder_channels[i]:
                weight = nn.functional.interpolate(
                    weight.unsqueeze(0), 
                    size=(encoder_channels[i], weight.size(1)), 
                    mode='bilinear'
                ).squeeze(0)
                bias = bias[:encoder_channels[i]] if bias.size(0) > encoder_channels[i] else torch.cat([bias, bias.mean().repeat(encoder_channels[i] - bias.size(0))])
            
            new_state_dict[f'encoder{i}.pconv.weight'] = weight
            new_state_dict[f'encoder{i}.pconv.bias'] = bias

        # Initialize batch norm parameters for all encoders (except encoder1)
        if i > 1:  # encoder1 has bn=False
            bn_params = init_bn_params(encoder_channels[i])
            
            # If parameters exist in old state dict, process them
            if f'{enc_prefix}.1.weight' in old_state_dict:
                old_bn_weight = old_state_dict[f'{enc_prefix}.1.weight']
                old_bn_bias = old_state_dict[f'{enc_prefix}.1.bias']
                old_bn_mean = old_state_dict[f'{enc_prefix}.1.running_mean']
                old_bn_var = old_state_dict[f'{enc_prefix}.1.running_var']
                
                # If sizes don't match, interpolate or truncate
                if old_bn_weight.size(0) != encoder_channels[i]:
                    scale_factor = encoder_channels[i] / old_bn_weight.size(0)
                    if scale_factor > 1:
                        # Repeat values for upscaling
                        bn_params['weight'] = old_bn_weight.repeat(int(scale_factor))[:encoder_channels[i]]
                        bn_params['bias'] = old_bn_bias.repeat(int(scale_factor))[:encoder_channels[i]]
                        bn_params['running_mean'] = old_bn_mean.repeat(int(scale_factor))[:encoder_channels[i]]
                        bn_params['running_var'] = old_bn_var.repeat(int(scale_factor))[:encoder_channels[i]]
                    else:
                        # Take subset for downscaling
                        bn_params['weight'] = old_bn_weight[:encoder_channels[i]]
                        bn_params['bias'] = old_bn_bias[:encoder_channels[i]]
                        bn_params['running_mean'] = old_bn_mean[:encoder_channels[i]]
                        bn_params['running_var'] = old_bn_var[:encoder_channels[i]]
                else:
                    bn_params['weight'] = old_bn_weight
                    bn_params['bias'] = old_bn_bias
                    bn_params['running_mean'] = old_bn_mean
                    bn_params['running_var'] = old_bn_var

            new_state_dict[f'encoder{i}.batchnorm.weight'] = bn_params['weight']
            new_state_dict[f'encoder{i}.batchnorm.bias'] = bn_params['bias']
            new_state_dict[f'encoder{i}.batchnorm.running_mean'] = bn_params['running_mean']
            new_state_dict[f'encoder{i}.batchnorm.running_var'] = bn_params['running_var']
    
    # Handle decoders
    for i in range(1, 9):
        dec_prefix = f'dec{9-i}'
        
        if f'{dec_prefix}.0.weight' in old_state_dict:
            weight = old_state_dict[f'{dec_prefix}.0.weight']
            bias = old_state_dict[f'{dec_prefix}.0.bias']
            
            # Resize if needed
            if weight.size(0) != decoder_channels[i-1]:
                weight = nn.functional.interpolate(
                    weight.unsqueeze(0), 
                    size=(decoder_channels[i-1], weight.size(1)), 
                    mode='bilinear'
                ).squeeze(0)
                bias = bias[:decoder_channels[i-1]] if bias.size(0) > decoder_channels[i-1] else torch.cat([bias, bias.mean().repeat(decoder_channels[i-1] - bias.size(0))])
            
            new_state_dict[f'decoder{i}.pconv.weight'] = weight
            new_state_dict[f'decoder{i}.pconv.bias'] = bias

        # Initialize batch norm parameters for decoders (except decoder8)
        if i < 8:  # decoder8 has bn=False
            bn_params = init_bn_params(decoder_channels[i-1])
            
            # If parameters exist in old state dict, process them
            if f'{dec_prefix}.1.weight' in old_state_dict:
                old_bn_weight = old_state_dict[f'{dec_prefix}.1.weight']
                old_bn_bias = old_state_dict[f'{dec_prefix}.1.bias']
                old_bn_mean = old_state_dict[f'{dec_prefix}.1.running_mean']
                old_bn_var = old_state_dict[f'{dec_prefix}.1.running_var']
                
                # If sizes don't match, interpolate or truncate
                if old_bn_weight.size(0) != decoder_channels[i-1]:
                    scale_factor = decoder_channels[i-1] / old_bn_weight.size(0)
                    if scale_factor > 1:
                        bn_params['weight'] = old_bn_weight.repeat(int(scale_factor))[:decoder_channels[i-1]]
                        bn_params['bias'] = old_bn_bias.repeat(int(scale_factor))[:decoder_channels[i-1]]
                        bn_params['running_mean'] = old_bn_mean.repeat(int(scale_factor))[:decoder_channels[i-1]]
                        bn_params['running_var'] = old_bn_var.repeat(int(scale_factor))[:decoder_channels[i-1]]
                    else:
                        bn_params['weight'] = old_bn_weight[:decoder_channels[i-1]]
                        bn_params['bias'] = old_bn_bias[:decoder_channels[i-1]]
                        bn_params['running_mean'] = old_bn_mean[:decoder_channels[i-1]]
                        bn_params['running_var'] = old_bn_var[:decoder_channels[i-1]]
                else:
                    bn_params['weight'] = old_bn_weight
                    bn_params['bias'] = old_bn_bias
                    bn_params['running_mean'] = old_bn_mean
                    bn_params['running_var'] = old_bn_var

            new_state_dict[f'decoder{i}.batchnorm.weight'] = bn_params['weight']
            new_state_dict[f'decoder{i}.batchnorm.bias'] = bn_params['bias']
            new_state_dict[f'decoder{i}.batchnorm.running_mean'] = bn_params['running_mean']
            new_state_dict[f'decoder{i}.batchnorm.running_var'] = bn_params['running_var']
    
    # Initialize convfinal if it exists in the model
    new_state_dict['convfinal.weight'] = torch.randn(3, 3, 1, 1) * 0.01
    new_state_dict['convfinal.bias'] = torch.zeros(3)
    
    return new_state_dict