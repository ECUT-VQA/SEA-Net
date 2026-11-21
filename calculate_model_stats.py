import torch
from models.stairIQA_resnet import resnet50, resnet50_imdt
from thop import profile


def calculate_flops_manual(model, input_size=(3, 224, 224)):
    """
    手动计算模型的完整FLOPS
    """
    print("\n===== 手动计算FLOPS =====")

    total_flops = 0

    # 基础ResNet50 FLOPS约4.1G (前向)
    # 这里基于thop计算的实际值
    base_resnet_flops = 4.1e9  # 4.1 G
    print(f"基础ResNet50 FLOPS (前向): {base_resnet_flops / 1e9:.2f} G")

    # EEA模块 FLOPS计算
    # 输入: [1, 2048, 7, 7]
    eea_in_channels = 2048
    eea_h, eea_w = 7, 7
    eea_out = 7 * 7 * 2048  # 100352

    # 1. depthwise卷积: k=3, groups=2048
    # FLOPs = out_h * out_w * in_channels * k_h * k_w
    eea_conv1_flops = eea_h * eea_w * eea_in_channels * 3 * 3
    print(f"EEA depthwise conv: {eea_conv1_flops / 1e6:.2f} M")

    # 2. BatchNorm (elementwise ops，通常忽略或计为微小值)
    eea_bn_flops = eea_out * 2  # mean和variance计算

    # 3. 通道注意力: 1x1卷积
    # Conv1: 2048 -> 128
    eea_attn1_flops = eea_out * eea_in_channels * 1 * 1 // 16
    # Conv2: 128 -> 2048
    eea_attn2_flops = eea_out * (eea_in_channels // 16) * 1 * 1

    eea_total = eea_conv1_flops + eea_bn_flops + eea_attn1_flops + eea_attn2_flops
    print(f"EEA总FLOPS: {eea_total / 1e6:.2f} M")

    # DAA模块 FLOPS计算
    # 输入: [1, 2048, 7, 7]
    daa_in_channels = 2048
    daa_h, daa_w = 7, 7

    # 1. AdaptiveAvgPool2d(4x4)
    daa_pool1_flops = daa_in_channels * 7 * 7 * 4 * 4

    # 2. Conv: 2048 -> 256 (k=3)
    daa_conv_flops = 4 * 4 * daa_in_channels * 256 * 3 * 3

    # 3. AdaptiveAvgPool2d(1x1)
    daa_pool2_flops = 256 * 4 * 4

    # 4. Linear: 256 -> 4
    daa_fc1_flops = 256 * 4

    # 5. Linear: 4 -> 256
    daa_fc2_flops = 4 * 256

    # 6. Linear: 256 -> 2048
    daa_fc3_flops = 256 * daa_in_channels

    daa_total = daa_pool1_flops + daa_conv_flops + daa_pool2_flops + daa_fc1_flops + daa_fc2_flops + daa_fc3_flops
    print(f"DAA总FLOPS: {daa_total / 1e6:.2f} M")

    # AFM模块 (主要是elementwise操作，FLOPS很小)
    # sigmoid操作: 5个参数
    afm_flops = 5
    print(f"AFM总FLOPS: {afm_flops}")

    # ECA模块
    # Conv1d: 2048 -> 2048 (k=3)
    eca_flops = 2048 * 2048 * 3
    print(f"ECA总FLOPS: {eca_flops / 1e6:.2f} M")

    # Quality回归头
    # Linear: 2048 -> 128
    quality_fc1_flops = 2048 * 128
    # Linear: 128 -> 128
    quality_fc2_flops = 128 * 128
    # Linear: 128 -> 1
    quality_fc3_flops = 128 * 1
    quality_total = quality_fc1_flops + quality_fc2_flops + quality_fc3_flops
    print(f"Quality回归头FLOPS: {quality_total / 1e6:.2f} M")

    # Hyper结构 FLOPS
    # 假设输入64x56x56 (layer1输出)
    hyper1_1_in, hyper1_1_out = 64, 256
    hyper_conv1 = 56 * 56 * 64 * 64 * 1 * 1  # 降维
    hyper_conv2 = 56 * 56 * 16 * 16 * 3 * 3  # depthwise
    hyper_conv3 = 56 * 56 * 16 * 256 * 1 * 1  # 升维
    hyper1_1_flops = hyper_conv1 + hyper_conv2 + hyper_conv3
    print(f"Hyper1_1 FLOPS: {hyper1_1_flops / 1e6:.2f} M")

    # Hyper2_1: 256 -> 512
    hyper2_1_in, hyper2_1_out = 256, 512
    h2, w2 = 56, 56
    h2_out, w2_out = 28, 28
    h2_conv1 = h2 * w2 * 256 * 64 * 1 * 1
    h2_conv2 = h2_out * w2_out * 16 * 16 * 3 * 3
    h2_conv3 = h2_out * w2_out * 16 * 512 * 1 * 1
    hyper2_1_flops = h2_conv1 + h2_conv2 + h2_conv3
    print(f"Hyper2_1 FLOPS: {hyper2_1_flops / 1e6:.2f} M")

    # Hyper3_1: 512 -> 1024
    hyper3_1_flops = 28 * 28 * 512 * 128 * 1 * 1 + 14 * 14 * 128 * 128 * 3 * 3 + 14 * 14 * 128 * 1024 * 1 * 1
    print(f"Hyper3_1 FLOPS: {hyper3_1_flops / 1e6:.2f} M")

    # Hyper4_1: 1024 -> 2048
    hyper4_1_flops = 14 * 14 * 1024 * 256 * 1 * 1 + 7 * 7 * 256 * 256 * 3 * 3 + 7 * 7 * 256 * 2048 * 1 * 1
    print(f"Hyper4_1 FLOPS: {hyper4_1_flops / 1e6:.2f} M")

    # 其他Hyper分支 (简化估算)
    hyper_other_flops = hyper2_1_flops * 3 + hyper3_1_flops * 2 + hyper4_1_flops * 1
    print(f"其他Hyper分支 FLOPS: {hyper_other_flops / 1e6:.2f} M")

    hyper_total = hyper1_1_flops + hyper2_1_flops + hyper3_1_flops + hyper4_1_flops + hyper_other_flops
    print(f"Hyper结构总FLOPS: {hyper_total / 1e6:.2f} M")

    # 总FLOPS (前向)
    total_forward = base_resnet_flops + eea_total + daa_total + afm_flops + eca_flops + quality_total + hyper_total
    total_forward_g = total_forward / 1e9

    # 总FLOPS (前向+反向)
    total_backward = total_forward * 2
    total_backward_g = total_backward / 1e9

    print(f"\n===== 完整FLOPS统计 =====")
    print(f"前向FLOPS: {total_forward_g:.2f} G")
    print(f"前向+反向FLOPS: {total_backward_g:.2f} G")

    return total_forward, total_forward_g


def calculate_model_stats(model_name='resnet50', pretrained=False, input_size=(3, 224, 224)):
    """
    计算模型的参数量和FLOPS
    """
    if model_name == 'resnet50':
        model = resnet50(pretrained=pretrained)
    elif model_name == 'resnet50_imdt':
        model = resnet50_imdt(pretrained=pretrained)
    else:
        raise ValueError(f"未知模型名称: {model_name}")

    device = torch.device('cpu')
    model.to(device)

    print(f"\n===== 模型: {model_name} =====")

    # 计算参数量
    print("\n===== 参数量 =====")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f} M")

    # 使用thop计算FLOPS
    print("\n===== THOP FLOPS计算 =====")
    input_tensor = torch.randn(1, *input_size)
    flops_forward, params = profile(model, inputs=(input_tensor,), verbose=False)
    print(f"THOP前向FLOPS: {flops_forward / 1e9:.2f} G")
    print(f"THOP前向+反向FLOPS: {flops_forward * 2 / 1e9:.2f} G")

    # 手动计算完整FLOPS
    total_forward, total_forward_g = calculate_flops_manual(model, input_size)

    print(f"\n===== 最终结果 =====")
    print(f"参数量: {total_params / 1e6:.2f} M")
    print(f"FLOPS (前向): {total_forward_g:.2f} G")
    print(f"FLOPS (前向+反向): {total_forward_g * 2:.2f} G")

    return flops_forward, total_params


if __name__ == '__main__':
    print("=" * 50)
    print("计算模型参数量和FLOPS")
    print("=" * 50)

    calculate_model_stats('resnet50')
