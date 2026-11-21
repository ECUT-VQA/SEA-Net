import os
import argparse
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

import models.stairIQA_resnet as stairIQA_resnet
from IQADataset import IQA_dataloader
from utils import performance_fit


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Cross-Dataset Generalization Test for StairAMBQ")
    parser.add_argument('--model_path', help='Path of trained model.', default='', type=str)
    parser.add_argument('--test_database', default='LIVE_challenge', type=str,
                        help='Test dataset: LIVE_challenge, SPAQ, BID, FLIVE, FLIVE_patch, Koniq10k')
    parser.add_argument('--train_database', default='Koniq10k', type=str,
                        help='Training dataset: Koniq10k, FLIVE, etc.')
    parser.add_argument('--test_database_dir', type=str, required=True,
                        help='Directory of test dataset images')
    parser.add_argument('--test_csv', type=str, required=True,
                        help='CSV file containing test image names and MOS scores')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--test_method', default='five', type=str,
                        help='Test method: one (center crop) or five (five crop)')
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    return args


def stack_to_tensor(crops):
    """Stack crops to tensor."""
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])


def normalize_stack(crops):
    """Normalize stacked crops."""
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return torch.stack([norm(crop) for crop in crops])


def get_transforms(database, test_method):
    """Get transforms based on database and test method."""
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
            stack_to_tensor,
            normalize_stack
        ])


def load_model(model_path, device):
    """Load trained model."""
    model = stairIQA_resnet.resnet50(pretrained=False)
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)

    # 打印状态字典的前5个键，用于调试
    print("State dict keys:")
    for i, key in enumerate(list(state_dict.keys())[:5]):
        print(f"  {i+1}. {key}")

    # 检查模型的状态字典键
    print("\nModel state dict keys:")
    model_state_dict = model.state_dict()
    for i, key in enumerate(list(model_state_dict.keys())[:5]):
        print(f"  {i+1}. {key}")

    # 尝试加载状态字典
    try:
        model.load_state_dict(state_dict)
        print("\nModel loaded successfully!")
    except RuntimeError as e:
        print(f"\nError loading state dict: {e}")
        # 尝试不使用DataParallel加载
        print("\nTrying to load without DataParallel...")
        model = stairIQA_resnet.resnet50(pretrained=False)
        model = model.to(device)
        try:
            model.load_state_dict(state_dict)
            print("Model loaded successfully without DataParallel!")
            # 重新包装为DataParallel
            model = torch.nn.DataParallel(model)
        except RuntimeError as e2:
            print(f"Error loading without DataParallel: {e2}")
            # 尝试移除module前缀
            print("\nTrying to remove 'module.' prefix...")
            new_state_dict = {}
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            try:
                model.load_state_dict(new_state_dict)
                print("Model loaded successfully after removing 'module.' prefix!")
                # 重新包装为DataParallel
                model = torch.nn.DataParallel(model)
            except RuntimeError as e3:
                print(f"Error loading after removing 'module.' prefix: {e3}")
                raise

    model.eval()
    return model


def test_model(model, test_loader, device, test_method):
    """Test model on test dataset."""
    pred_scores = []
    true_scores = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            if test_method == 'one':
                images = images.to(device)
                outputs, _, _ = model(images)  # 忽略 layer_weights 和 distortion_type
                # 移除多余的维度
                pred_scores.extend(outputs.cpu().numpy().flatten())
            else:  # five crop
                bs, ncrops, c, h, w = images.size()
                images = images.to(device)
                outputs, _, _ = model(images.view(-1, c, h, w))  # 忽略 layer_weights 和 distortion_type
                outputs = outputs.view(bs, ncrops, -1).mean(1)
                # 移除多余的维度
                pred_scores.extend(outputs.cpu().numpy().flatten())

            # 确保labels的形状正确
            true_scores.extend(labels.numpy().flatten())

            if (batch_idx + 1) % 10 == 0:
                print(f'Processed {batch_idx + 1}/{len(test_loader)} batches')

    # 转换为numpy数组并确保形状正确
    pred_scores = np.array(pred_scores)
    true_scores = np.array(true_scores)
    
    print(f'\nPred scores shape: {pred_scores.shape}')
    print(f'True scores shape: {true_scores.shape}')

    return pred_scores, true_scores


def main():
    args = parse_args()

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    print(f'Loading model from: {args.model_path}')
    print(f'Training Dataset: {args.train_database}')
    print(f'Testing on dataset: {args.test_database}')
    print(f'Test method: {args.test_method}')

    # Load model
    model = load_model(args.model_path, device)
    print('Model loaded successfully!')

    # Get transforms
    test_transform = get_transforms(args.test_database, args.test_method)

    # Create test dataset
    test_dataset = IQA_dataloader(
        data_dir=args.test_database_dir,
        csv_path=args.test_csv,
        transform=test_transform,
        database=args.test_database
    )

    # Create test dataloader
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Windows系统上设置为0以避免多进程问题
        pin_memory=True
    )

    print(f'Test dataset size: {len(test_dataset)}')

    # Test model
    print('\nStarting testing...')
    pred_scores, true_scores = test_model(model, test_loader, device, args.test_method)

    # Calculate performance metrics
    test_srcc, test_plcc, test_krcc, test_rmse = performance_fit(pred_scores, true_scores)

    # Print results
    print('\n' + '='*50)
    print('Cross-Dataset Generalization Results')
    print('='*50)
    print(f'Training Dataset: {args.train_database}')
    print(f'Test Dataset: {args.test_database}')
    print(f'Test Method: {args.test_method}')
    print('-'*50)
    print(f'SRCC: {test_srcc:.4f}')
    print(f'PLCC: {test_plcc:.4f}')
    print(f'KRCC: {test_krcc:.4f}')
    print(f'RMSE: {test_rmse:.4f}')
    print('='*50)

    # Save results
    results_file = f'results_cross_dataset_{args.train_database}_to_{args.test_database}.txt'
    with open(results_file, 'w') as f:
        f.write(f'Cross-Dataset Generalization Results\n')
        f.write(f'Training Dataset: {args.train_database}\n')
        f.write(f'Test Dataset: {args.test_database}\n')
        f.write(f'Test Method: {args.test_method}\n')
        f.write(f'SRCC: {test_srcc:.4f}\n')
        f.write(f'PLCC: {test_plcc:.4f}\n')
        f.write(f'KRCC: {test_krcc:.4f}\n')
        f.write(f'RMSE: {test_rmse:.4f}\n')

    print(f'\nResults saved to: {results_file}')


if __name__ == '__main__':
    main()
