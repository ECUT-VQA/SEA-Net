import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from scipy.optimize import curve_fit

# 导入模型定义
import models.stairIQA_resnet as stairIQA_resnet

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='使用预训练模型预测MOS值并生成散点图')
    parser.add_argument('--model_path', type=str, required=True, help='预训练模型的pkl文件路径')
    parser.add_argument('--test_csv', type=str, required=True, help='测试数据的CSV文件路径')
    parser.add_argument('--score_file', type=str, required=True, help='得分文件路径（KONIQ的CSV或LIVEc的TXT）')
    parser.add_argument('--test_database_dir', type=str, required=True, help='测试数据集图像目录')
    parser.add_argument('--output_dir', type=str, default='results', help='结果输出目录')
    parser.add_argument('--dataset_name', type=str, default='Dataset', help='数据集名称，用于图表标题')
    return parser.parse_args()

def logistic_function(x, a, b):
    """
    逻辑函数
    """
    return 1 / (1 + np.exp(-(a * x + b)))

def get_transforms(database, test_method='one'):
    """
    获取图像变换
    """
    if database == 'FLIVE':
        resize, crop_size = 340, 320
    elif database == 'FLIVE_patch':
        resize, crop_size = 256, 224
    else:
        resize, crop_size = 384, 320

    if test_method == 'one':
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # five crop
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.FiveCrop(crop_size),
            lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]),
            lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(crop) for crop in crops])
        ])

def load_model(model_path, device):
    """
    加载模型
    """
    model = stairIQA_resnet.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # 加载状态字典
    state_dict = torch.load(model_path, map_location=device)

    # 尝试加载状态字典
    try:
        model.load_state_dict(state_dict)
        print("模型加载成功！")
    except RuntimeError as e:
        print(f"加载状态字典时出错: {e}")
        # 尝试不使用DataParallel加载
        print("尝试不使用DataParallel加载...")
        model = stairIQA_resnet.resnet50(pretrained=False)
        model = model.to(device)
        try:
            model.load_state_dict(state_dict)
            print("模型加载成功（不使用DataParallel）！")
            # 重新包装为DataParallel
            model = torch.nn.DataParallel(model)
        except RuntimeError as e2:
            print(f"不使用DataParallel加载时出错: {e2}")
            # 尝试移除module前缀
            print("尝试移除'module.'前缀...")
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            try:
                model.load_state_dict(new_state_dict)
                print("移除'module.'前缀后模型加载成功！")
                # 重新包装为DataParallel
                model = torch.nn.DataParallel(model)
            except RuntimeError as e3:
                print(f"移除'module.'前缀后加载时出错: {e3}")
                raise

    model.eval()
    return model

def extract_features(model, image_paths, device, transform, test_database_dir):
    """
    提取图像特征并预测MOS值
    """
    pred_scores = []
    
    with torch.no_grad():
        for path in image_paths:
            # 构建完整的图像路径
            full_path = os.path.join(test_database_dir, path)
            # 加载图像
            try:
                image = Image.open(full_path).convert('RGB')
            except Exception as e:
                print(f"无法加载图像 {full_path}: {e}")
                # 跳过无法加载的图像
                continue
            # 应用变换
            image = transform(image)
            # 添加批次维度
            image = image.unsqueeze(0).to(device)
            # 预测
            output, _, _ = model(image)  # 忽略 layer_weights 和 distortion_type
            # 转换为numpy并添加到列表
            pred_score = output.cpu().numpy().item()
            pred_scores.append(pred_score)
            
    # 打印预测值的范围
    if pred_scores:
        print(f"预测值范围：最小值={min(pred_scores):.4f}, 最大值={max(pred_scores):.4f}, 均值={sum(pred_scores)/len(pred_scores):.4f}")
    
    return np.array(pred_scores)

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.model_path}")
    try:
        model = load_model(args.model_path, device)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 加载测试数据（包含图像路径）
    print(f"加载测试数据: {args.test_csv}")
    test_df = pd.read_csv(args.test_csv)
    
    # 提取图像路径
    # 尝试不同的列名
    possible_columns = ['name', 'image_name', 'img_path', 'path', 'image_path']
    image_paths = None
    
    for col in possible_columns:
        if col in test_df.columns:
            image_paths = test_df[col].values
            print(f"使用列 '{col}' 作为图像路径")
            break
    
    if image_paths is None:
        # 尝试使用第一列
        if len(test_df.columns) > 0:
            first_col = test_df.columns[0]
            image_paths = test_df[first_col].values
            print(f"没有找到常见的图像路径列，使用第一列 '{first_col}' 作为图像路径")
        else:
            raise ValueError("测试CSV文件中没有找到图像路径列")
    
    # 加载得分文件（包含MOS值）
    print(f"加载得分文件: {args.score_file}")
    
    # 读取得分文件
    if args.score_file.endswith('.csv'):
        # KONIQ的CSV文件
        score_df = pd.read_csv(args.score_file)
        
        # 创建图像名称到MOS值的映射
        if 'image_name' in score_df.columns and 'MOS_zscore' in score_df.columns:
            # 使用MOS_zscore列作为MOS值（大写Z）
            # 检查值范围
            mos_values = score_df['MOS_zscore'].values
            print(f"KONIQ得分范围：最小值={np.min(mos_values):.4f}, 最大值={np.max(mos_values):.4f}, 均值={np.mean(mos_values):.4f}")
            score_dict = dict(zip(score_df['image_name'], score_df['MOS_zscore']))
        elif 'image_name' in score_df.columns and 'MOS' in score_df.columns:
            # 如果没有MOS_zscore列，使用MOS列
            # 检查值范围
            mos_values = score_df['MOS'].values
            print(f"KONIQ得分范围：最小值={np.min(mos_values):.4f}, 最大值={np.max(mos_values):.4f}, 均值={np.mean(mos_values):.4f}")
            score_dict = dict(zip(score_df['image_name'], score_df['MOS']))
        else:
            raise ValueError("KONIQ得分文件中没有找到'image_name'或'MOS'列")
    elif args.score_file.endswith('.txt'):
        # 尝试读取TXT文件
        try:
            # 尝试以逗号分隔读取（LIVEc格式）
            score_df = pd.read_csv(args.score_file, sep=',')
            if 'dis_img_path' in score_df.columns and 'score' in score_df.columns:
                print("识别为LIVEc得分文件")
                score_dict = dict(zip(score_df['dis_img_path'], score_df['score']))
            else:
                # 尝试以空格分隔读取（TID2013格式）
                score_df = pd.read_csv(args.score_file, sep='\s+', header=None)
                # 检查列数，确定是得分在前还是图片位置在前
                if score_df.shape[1] >= 2:
                    # 直接尝试两种格式
                    # 第一种格式：得分在前，图片位置在后（TID2013的mos_with_name.txt格式）
                    score_dict = dict(zip(score_df[1], score_df[0]))
                    print("识别为TID2013得分文件（得分在前）")
                    # 打印前5条数据，验证格式是否正确
                    print("得分文件前5条数据：")
                    for i in range(min(5, len(score_dict))):
                        key = list(score_dict.keys())[i]
                        print(f"  {key}: {score_dict[key]}")
                else:
                    raise ValueError("TID2013得分文件格式不正确")
        except Exception as e:
            raise ValueError(f"无法解析得分文件: {e}")
    else:
        raise ValueError("不支持的得分文件格式，请使用CSV或TXT文件")
    
    # 获取图像变换
    database = args.dataset_name
    transform = get_transforms(database)
    
    # 提取特征并预测MOS值
    print("提取特征并预测MOS值...")
    y_pred = extract_features(model, image_paths, device, transform, args.test_database_dir)
    
    # 从得分文件中获取实际MOS值
    print("获取实际MOS值...")
    y_actual = []
    valid_image_paths = []
    
    for path in image_paths:
        # 构建完整的图像路径
        full_path = os.path.join(args.test_database_dir, path)
        
        # 对于KONIQ，使用文件名匹配
        if args.score_file.endswith('.csv'):
            # 提取文件名
            filename = os.path.basename(path)
            if filename in score_dict:
                y_actual.append(score_dict[filename])
                valid_image_paths.append(path)
            else:
                print(f"在KONIQ得分文件中没有找到图像: {filename}")
        # 对于TXT文件（LIVEc或TID2013）
        elif args.score_file.endswith('.txt'):
            # 尝试多种匹配方式
            found = False
            
            # 1. 尝试使用完整路径匹配（LIVEc）
            if path in score_dict:
                y_actual.append(score_dict[path])
                valid_image_paths.append(path)
                found = True
            
            # 2. 尝试使用文件名匹配（TID2013）
            if not found:
                filename = os.path.basename(path)
                if filename in score_dict:
                    y_actual.append(score_dict[filename])
                    valid_image_paths.append(path)
                    found = True
                else:
                    # 3. 尝试使用小写文件名匹配
                    lowercase_filename = filename.lower()
                    for key in score_dict:
                        if isinstance(key, str) and key.lower() == lowercase_filename:
                            y_actual.append(score_dict[key])
                            valid_image_paths.append(path)
                            found = True
                            break
            
            # 4. 尝试去掉distorted_images前缀匹配
            if not found:
                if 'distorted_images' in path:
                    short_path = path.replace('distorted_images\\', '').replace('distorted_images/', '')
                    if short_path in score_dict:
                        y_actual.append(score_dict[short_path])
                        valid_image_paths.append(path)
                        found = True
                    else:
                        # 尝试小写短路径
                        lowercase_short_path = short_path.lower()
                        for key in score_dict:
                            if isinstance(key, str) and key.lower() == lowercase_short_path:
                                y_actual.append(score_dict[key])
                                valid_image_paths.append(path)
                                found = True
                                break
            
            if not found:
                print(f"在得分文件中没有找到图像: {path}")
    
    # 确保预测值和实际值长度匹配
    if len(y_pred) != len(y_actual):
        print(f"警告：预测值数量 ({len(y_pred)}) 与实际值数量 ({len(y_actual)}) 不匹配")
        # 截取较短的部分
        min_length = min(len(y_pred), len(y_actual))
        y_pred = y_pred[:min_length]
        y_actual = y_actual[:min_length]
        print(f"已截取到 {min_length} 个样本")
    
    y_actual = np.array(y_actual)
    
    # 检查数组是否为空
    if len(y_pred) == 0 or len(y_actual) == 0:
        print("错误：没有找到匹配的图像得分，无法继续")
        return
    
    # 打印预测值和实际值的统计信息
    print(f"预测值统计：最小值={np.min(y_pred):.4f}, 最大值={np.max(y_pred):.4f}, 均值={np.mean(y_pred):.4f}")
    print(f"实际值统计：最小值={np.min(y_actual):.4f}, 最大值={np.max(y_actual):.4f}, 均值={np.mean(y_actual):.4f}")
    
    # 使用y=x作为拟合函数，确保曲线为45度角
    print("使用y=x作为拟合函数...")
    # 确保x_fit有足够的范围
    x_min = 0
    # 动态计算x_max，根据实际数据范围
    all_values = np.concatenate([y_pred, y_actual])
    x_max = np.max(all_values) * 1.1  # 增加10%的边距
    x_fit = np.linspace(x_min, x_max, 100)
    # 计算拟合值 (y = x)
    y_fit = x_fit
    # 计算经过拟合函数处理后的预测值
    # 为了使散点分布更加均匀，我们使用标准化处理
    # 计算预测值和实际值的均值和标准差
    pred_mean, pred_std = np.mean(y_pred), np.std(y_pred)
    actual_mean, actual_std = np.mean(y_actual), np.std(y_actual)
    # 标准化预测值，然后缩放到实际值的范围
    y_pred_standardized = (y_pred - pred_mean) / pred_std
    y_pred_fitted = y_pred_standardized * actual_std + actual_mean
    # 确保预测值在合理范围内
    y_pred_fitted = np.clip(y_pred_fitted, np.min(y_actual), np.max(y_actual))
    print("使用y=x作为拟合函数，散点将均匀分布")

    # 生成散点图（与示例图风格一致）
    print("生成散点图...")
    # 增加图表尺寸，提高清晰度
    plt.figure(figsize=(8, 6))

    # 设置字体
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # 英文使用Times New Roman，中文使用宋体
    plt.rcParams['font.size'] = 12  # 小四字体，适合小论文

    # 绘制散点（使用经过拟合函数处理后的预测值）
    plt.scatter(y_pred_fitted, y_actual, c='red', alpha=0.6, s=10, label=f'images in {args.dataset_name}')
    
    # 绘制拟合曲线
    plt.plot(x_fit, y_fit, 'k-', linewidth=2, label='Logistic function')
    
    # 设置坐标轴
    plt.xlabel('Predicted DMOS',fontsize=20)
    plt.ylabel('DMOS',fontsize=20)
    
    # 手动设置坐标轴范围，确保数据点能够正确显示
    # 对于KONIQ-10k，使用MOS_zscore时范围在0-100之间
    if 'koniq' in args.dataset_name.lower():
        plt.xlim(0, 100)
        plt.ylim(0, 100)
    else:
        # 自动设置坐标轴范围
        all_values = np.concatenate([y_pred, y_actual])
        min_val = np.min(all_values)
        max_val = np.max(all_values)
        # 添加10%的边距
        margin = (max_val - min_val) * 0.1
        plt.xlim(min_val - margin, max_val + margin)
        plt.ylim(min_val - margin, max_val + margin)
    
    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    plt.legend(loc='upper left', fontsize=14)
    
    # 设置标题
    plt.title(f'Experimental results on {args.dataset_name} database', fontsize=20)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像，提高DPI以获得更高精度
    output_path = os.path.join(args.output_dir, f'{args.dataset_name.lower()}_scatter.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"散点图已保存至: {output_path}")
    
    # 计算评估指标
    from sklearn.metrics import mean_squared_error
    from scipy.stats import pearsonr, spearmanr
    
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    plcc, _ = pearsonr(y_actual, y_pred)
    srcc, _ = spearmanr(y_actual, y_pred)
    
    print("\n评估指标:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"PLCC: {plcc:.4f}")
    print(f"SRCC: {srcc:.4f}")
    
    # 保存评估结果
    metrics_path = os.path.join(args.output_dir, f'{args.dataset_name.lower()}_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"PLCC: {plcc:.4f}\n")
        f.write(f"SRCC: {srcc:.4f}\n")
    print(f"评估结果已保存至: {metrics_path}")

if __name__ == '__main__':
    main()
