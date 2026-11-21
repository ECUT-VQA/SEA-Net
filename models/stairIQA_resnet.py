import torchvision.models as models
import torch
from torch import nn


class EdgeEnhancementAttention(nn.Module):
    """
    边缘增强注意力模块 (Edge Enhancement Attention, EEA)
    专注于增强边缘结构信息，抑制无效噪声
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction = reduction

        # 1. 可学习的边缘特征提取
        self.edge_detector = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.edge_norm = nn.BatchNorm2d(in_channels)
        self.edge_act = nn.ReLU(inplace=True)

        # 2. 通道自适应注意力
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 3. 可学习的特征融合
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.edge_detector.weight, mode='fan_out', nonlinearity='relu')
        for m in self.attention.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        B, C, H, W = x.size()

        # 步骤1: 提取边缘特征（可学习）
        edge_feat = self.edge_detector(x)
        edge_feat = self.edge_norm(edge_feat)
        edge_feat = self.edge_act(edge_feat)

        # 步骤2: 计算边缘特征的全局统计
        edge_global = self.avg_pool(edge_feat)

        # 步骤3: 生成通道注意力权重
        attn_weights = self.attention(edge_global)

        # 步骤4: 应用注意力权重
        enhanced_feat = x * attn_weights

        # 步骤5: 可学习的特征融合
        out = self.fusion_weight * enhanced_feat + (1 - self.fusion_weight) * x

        return out


class DistortionAwareAdapter(nn.Module):
    """
    可训练失真感知适配器（参考 HyperIQA 设计）
    提取全局失真纹理、退化模式、局部畸变信息
    并生成五层层级重要性权重
    """

    def __init__(self, in_channels=64, reduction=16):
        super().__init__()
        self.in_channels = in_channels
        
        # 1. 全局失真纹理提取
        self.global_texture_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 2. 退化模式检测
        self.degradation_pattern_detector = nn.Sequential(
            nn.AdaptiveAvgPool2d((16, 16)),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels // (2 * reduction), kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // (2 * reduction)),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 3. 局部畸变信息提取（高频分量）
        self.local_distortion_extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels // reduction, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // reduction),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # 4. 特征融合与层级权重生成器
        self.global_texture_dim = in_channels // reduction
        self.degradation_pattern_dim = in_channels // (2 * reduction)
        self.local_distortion_dim = in_channels // reduction
        
        total_dim = self.global_texture_dim + self.degradation_pattern_dim + self.local_distortion_dim
        
        # 层级重要性权重生成器（输出5个层级的权重）
        self.layer_weight_generator = nn.Sequential(
            nn.Linear(total_dim, total_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(total_dim * 2, 10),
            nn.ReLU(),
            nn.Linear(10, 5),
            nn.Softmax(dim=1)  # 确保权重和为1
        )
        
        # 失真类型分类器（可选，用于辅助监督）
        self.distortion_classifier = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, 4)  # 模糊、噪声、压缩、混合
        )

    def forward(self, x):
        """
        输入: x - 浅层特征图 [B, C, H, W]
        输出: 
            layer_weights - 五层层级重要性权重 [B, 5]
            distortion_type - 失真类型预测 [B, 4]
        """
        # 1. 提取全局失真纹理特征
        global_texture = self.global_texture_extractor(x)  # [B, global_dim]
        
        # 2. 提取退化模式特征
        degradation_pattern = self.degradation_pattern_detector(x)  # [B, pattern_dim]
        
        # 3. 提取局部畸变信息
        local_distortion = self.local_distortion_extractor(x)  # [B, local_dim]
        
        # 4. 特征融合
        fused_distortion_feature = torch.cat([
            global_texture,
            degradation_pattern,
            local_distortion
        ], dim=1)  # [B, total_dim]
        
        # 5. 生成五层层级重要性权重
        layer_weights = self.layer_weight_generator(fused_distortion_feature)  # [B, 5]
        
        # 6. 预测失真类型（可选）
        distortion_type = self.distortion_classifier(fused_distortion_feature)  # [B, 4]
        
        return layer_weights, distortion_type


class ECA(nn.Module):
    def __init__(self, channels=2048, k_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if x.dim() == 4:
            b, c, _, _ = x.size()
            v = x.mean(dim=(2, 3))
            v1 = v.view(b, 1, c)
            w = self.conv(v1)
            w = self.sigmoid(w).view(b, c)
            out_vec = v * w
            return out_vec
        elif x.dim() == 2:
            b, c = x.size()
            v1 = x.view(b, 1, c)
            w = self.conv(v1)
            w = self.sigmoid(w).view(b, c)
            return x * w
        else:
            return x


class resnet50(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(resnet50, self).__init__()
        if pretrained:
            resnet50_features = nn.Sequential(*list(models.resnet50(weights='DEFAULT').children()))
        else:
            resnet50_features = nn.Sequential(*list(models.resnet50().children()))

        # 五级阶梯特征提取器
        self.feature_extraction_stem = torch.nn.Sequential()
        self.feature_extraction1 = torch.nn.Sequential()  # Stage 1
        self.feature_extraction2 = torch.nn.Sequential()  # Stage 2
        self.feature_extraction3 = torch.nn.Sequential()  # Stage 3
        self.feature_extraction4 = torch.nn.Sequential()  # Stage 4

        for x in range(0, 4):
            self.feature_extraction_stem.add_module(str(x), resnet50_features[x])

        for x in range(4, 5):
            self.feature_extraction1.add_module(str(x), resnet50_features[x])

        for x in range(5, 6):
            self.feature_extraction2.add_module(str(x), resnet50_features[x])

        for x in range(6, 7):
            self.feature_extraction3.add_module(str(x), resnet50_features[x])

        for x in range(7, 8):
            self.feature_extraction4.add_module(str(x), resnet50_features[x])

        # Hyper 分支（多尺度特征增强）
        self.hyper1_1 = self.hyper_structure1(64, 256)
        self.hyper2_1 = self.hyper_structure2(256, 512)
        self.hyper3_1 = self.hyper_structure2(512, 1024)
        self.hyper4_1 = self.hyper_structure2(1024, 2048)

        self.hyper2_2 = self.hyper_structure2(256, 512)
        self.hyper3_2 = self.hyper_structure2(512, 1024)
        self.hyper4_2 = self.hyper_structure2(1024, 2048)

        self.hyper3_3 = self.hyper_structure2(512, 1024)
        self.hyper4_3 = self.hyper_structure2(1024, 2048)

        self.hyper4_4 = self.hyper_structure2(1024, 2048)

        # 失真感知适配器（在浅层嵌入）
        self.distortion_adapter = DistortionAwareAdapter(in_channels=64)

        # 边缘特征增强 + 冗余特征剔除
        self.eea = EdgeEnhancementAttention(2048)
        self.eca = ECA(2048, 3)

        # 质量回归头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.quality = self.quality_regression(2048, 128, 1)

    def hyper_structure1(self, in_channels, out_channels):
        hyper_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        return hyper_block

    def hyper_structure2(self, in_channels, out_channels):
        hyper_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        return hyper_block

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(middle_channels, middle_channels),
            nn.GELU(),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x):
        # Stage 0: 浅层特征提取
        x_stem = self.feature_extraction_stem(x)  # [B, 64, H/4, W/4]
        
        # 失真感知适配器：提取失真特征并生成层级权重
        layer_weights, distortion_type = self.distortion_adapter(x_stem)  # [B, 5], [B, 4]
        
        # Stage 1
        x_hyper1 = self.hyper1_1(x_stem)
        x_stage1 = self.feature_extraction1(x_stem)

        # Stage 2
        x_hyper1 = self.hyper2_1(x_hyper1 + x_stage1)
        x_hyper2 = self.hyper2_2(x_stage1)
        x_stage2 = self.feature_extraction2(x_stage1)

        # Stage 3
        x_hyper1 = self.hyper3_1(x_hyper1 + x_stage2)
        x_hyper2 = self.hyper3_2(x_hyper2 + x_stage2)
        x_hyper3 = self.hyper3_3(x_stage2)
        x_stage3 = self.feature_extraction3(x_stage2)

        # Stage 4
        x_hyper1 = self.hyper4_1(x_hyper1 + x_stage3)
        x_hyper2 = self.hyper4_2(x_hyper2 + x_stage3)
        x_hyper3 = self.hyper4_3(x_hyper3 + x_stage3)
        x_hyper4 = self.hyper4_4(x_stage3)
        x_stage4 = self.feature_extraction4(x_stage3)

        # 收集五层特征（Stage1~Stage5 对应五个分支）
        # x_stage4: Stage5 主分支特征
        # x_hyper1~x_hyper4: Stage1~Stage4 的 Hyper 增强特征
        
        # 自适应加权融合（根据失真感知适配器生成的权重）
        # layer_weights[:, 0]: Stage1 权重
        # layer_weights[:, 1]: Stage2 权重
        # layer_weights[:, 2]: Stage3 权重
        # layer_weights[:, 3]: Stage4 权重
        # layer_weights[:, 4]: Stage5 权重
        
        # 将权重转换为适合特征融合的形状 [B, 5, 1, 1, 1]
        w = layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, 5, 1, 1, 1]
        
        # 融合五个分支的特征
        # 注意：这里我们使用五个分支的最终特征（都经过 Hyper 增强并上采样到相同维度）
        fused_feature = (
            w[:, 0] * x_hyper1 +
            w[:, 1] * x_hyper2 +
            w[:, 2] * x_hyper3 +
            w[:, 3] * x_hyper4 +
            w[:, 4] * x_stage4
        )

        # 边缘特征增强 + 冗余特征剔除
        fused_feature = self.eea(fused_feature)

        # 质量预测
        x = self.avgpool(fused_feature)
        x = self.eca(x)
        x = torch.flatten(x, 1)
        quality_score = self.quality(x)

        return quality_score, layer_weights, distortion_type


class resnet50_imdt(torch.nn.Module):
    def __init__(self, pretrained=True):
        super(resnet50_imdt, self).__init__()
        if pretrained:
            resnet50_features = nn.Sequential(*list(models.resnet50(weights='DEFAULT').children()))
        else:
            resnet50_features = nn.Sequential(*list(models.resnet50().children()))

        self.feature_extraction_stem = torch.nn.Sequential()
        self.feature_extraction1 = torch.nn.Sequential()
        self.feature_extraction2 = torch.nn.Sequential()
        self.feature_extraction3 = torch.nn.Sequential()
        self.feature_extraction4 = torch.nn.Sequential()

        for x in range(0, 4):
            self.feature_extraction_stem.add_module(str(x), resnet50_features[x])

        for x in range(4, 5):
            self.feature_extraction1.add_module(str(x), resnet50_features[x])

        for x in range(5, 6):
            self.feature_extraction2.add_module(str(x), resnet50_features[x])

        for x in range(6, 7):
            self.feature_extraction3.add_module(str(x), resnet50_features[x])

        for x in range(7, 8):
            self.feature_extraction4.add_module(str(x), resnet50_features[x])

        self.hyper1_1 = self.hyper_structure1(64, 256)
        self.hyper2_1 = self.hyper_structure2(256, 512)
        self.hyper3_1 = self.hyper_structure2(512, 1024)
        self.hyper4_1 = self.hyper_structure2(1024, 2048)

        self.hyper2_2 = self.hyper_structure2(256, 512)
        self.hyper3_2 = self.hyper_structure2(512, 1024)
        self.hyper4_2 = self.hyper_structure2(1024, 2048)

        self.hyper3_3 = self.hyper_structure2(512, 1024)
        self.hyper4_3 = self.hyper_structure2(1024, 2048)

        self.hyper4_4 = self.hyper_structure2(1024, 2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.quality1 = self.quality_regression(2048, 128, 1)
        self.quality2 = self.quality_regression(2048, 128, 1)
        self.quality3 = self.quality_regression(2048, 128, 1)
        self.quality4 = self.quality_regression(2048, 128, 1)
        self.quality5 = self.quality_regression(2048, 128, 1)
        self.quality6 = self.quality_regression(2048, 128, 1)
        self.eca = ECA(2048, 3)
        self.branch_alpha = nn.Parameter(torch.zeros(5))

    def hyper_structure1(self, in_channels, out_channels):
        hyper_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        return hyper_block

    def hyper_structure2(self, in_channels, out_channels):
        hyper_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.Conv2d(in_channels // 4, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        return hyper_block

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )
        return regression_block

    def forward(self, x):
        x = self.feature_extraction_stem(x)

        x_hyper1 = self.hyper1_1(x)
        x = self.feature_extraction1(x)

        x_hyper1 = self.hyper2_1(x_hyper1 + x)
        x_hyper2 = self.hyper2_2(x)
        x = self.feature_extraction2(x)

        x_hyper1 = self.hyper3_1(x_hyper1 + x)
        x_hyper2 = self.hyper3_2(x_hyper2 + x)
        x_hyper3 = self.hyper3_3(x)
        x = self.feature_extraction3(x)

        x_hyper1 = self.hyper4_1(x_hyper1 + x)
        x_hyper2 = self.hyper4_2(x_hyper2 + x)
        x_hyper3 = self.hyper4_3(x_hyper3 + x)
        x_hyper4 = self.hyper4_4(x)
        x = self.feature_extraction4(x)

        w = torch.sigmoid(self.branch_alpha)
        x = 2.0 * (w[0] * x + w[1] * x_hyper1 + w[2] * x_hyper2 + w[3] * x_hyper3 + w[4] * x_hyper4)

        x = self.avgpool(x)
        x = self.eca(x)
        x = torch.flatten(x, 1)

        x1 = self.quality1(x)
        x2 = self.quality2(x)
        x3 = self.quality3(x)
        x4 = self.quality4(x)
        x5 = self.quality5(x)
        x6 = self.quality6(x)

        return x1, x2, x3, x4, x5, x6


if __name__ == '__main__':
    model = resnet50()
    print("Model Architecture:")
    print(model)
    
    # Test forward pass
    input_tensor = torch.randn(2, 3, 320, 320)
    quality_score, layer_weights, distortion_type = model(input_tensor)
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Quality score shape: {quality_score.shape}")
    print(f"Layer weights shape: {layer_weights.shape}")
    print(f"Layer weights sum: {layer_weights.sum(dim=1)}")
    print(f"Distortion type shape: {distortion_type.shape}")
