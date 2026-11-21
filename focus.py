import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import models.stairIQA_resnet as stairIQA_resnet


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Generate attention heatmap for AFM module")
    parser.add_argument('--model_path', help='Path of trained model.', required=True, type=str)
    parser.add_argument('--image_path', help='Path of input image.', required=True, type=str)
    parser.add_argument('--output_dir', help='Directory to save heatmaps.', default='heatmaps', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    return args


def load_model(model_path, device):
    """Load trained model."""
    model = stairIQA_resnet.resnet50(pretrained=False)
    model = model.to(device)

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)

    # 检查状态字典中的键是否包含"module."前缀
    first_key = next(iter(state_dict.keys()))
    if first_key.startswith('module.'):
        # 移除"module."前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path, device):
    """Preprocess input image."""
    transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(320),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    original_image = np.array(image)
    image = transform(image).unsqueeze(0).to(device)
    return image, original_image


def get_afm_features(model, image):
    """Get features from AFM module."""
    # Hook to capture features
    features = {}

    def hook_fn(module, input, output):
        features['feature_extraction4'] = output

    # Register hook
    handle = model.feature_extraction4.register_forward_hook(hook_fn)

    # Forward pass to get features
    with torch.no_grad():
        # Get features from different branches
        x = model.feature_extraction_stem(image)
        x_hyper1 = model.hyper1_1(x)
        x = model.feature_extraction1(x)

        x_hyper1 = model.hyper2_1(x_hyper1 + x)
        x_hyper2 = model.hyper2_2(x)
        x = model.feature_extraction2(x)

        x_hyper1 = model.hyper3_1(x_hyper1 + x)
        x_hyper2 = model.hyper3_2(x_hyper2 + x)
        x_hyper3 = model.hyper3_3(x)
        x = model.feature_extraction3(x)

        x_hyper1 = model.hyper4_1(x_hyper1 + x)
        x_hyper2 = model.hyper4_2(x_hyper2 + x)
        x_hyper3 = model.hyper4_3(x_hyper3 + x)
        x_hyper4 = model.hyper4_4(x)
        x = model.feature_extraction4(x)

    # Remove hook
    handle.remove()

    return {
        'x': x,  # 主分支
        'x_hyper1': x_hyper1,  # 边缘关注分支
        'x_hyper2': x_hyper2,
        'x_hyper3': x_hyper3,
        'x_hyper4': x_hyper4
    }


def generate_heatmap(feature_map):
    """Generate heatmap from feature map."""
    # Calculate mean across channels
    heatmap = torch.mean(feature_map, dim=1).squeeze().cpu().numpy()
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    # Resize to original image size
    heatmap = cv2.resize(heatmap, (320, 320))
    # Apply colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def overlay_heatmap(original_image, heatmap, alpha=0.5):
    """Overlay heatmap on original image."""
    # Resize original image to match heatmap size
    original_resized = cv2.resize(original_image, (320, 320))
    # Convert to float32 for blending
    original_resized = original_resized.astype(np.float32)
    heatmap = heatmap.astype(np.float32)
    # Blend images
    overlay = cv2.addWeighted(original_resized, 1 - alpha, heatmap, alpha, 0)
    overlay = np.uint8(overlay)
    return overlay


def main():
    args = parse_args()

    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Set device
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    print(f'Loading model from: {args.model_path}')
    print(f'Processing image: {args.image_path}')

    # Load model
    model = load_model(args.model_path, device)
    print('Model loaded successfully!')

    # Preprocess image
    image, original_image = preprocess_image(args.image_path, device)

    # Get AFM features
    afm_features = get_afm_features(model, image)

    # Generate heatmaps for each branch
    branches = ['x', 'x_hyper1', 'x_hyper2', 'x_hyper3', 'x_hyper4']
    branch_names = ['Main Branch', 'Edge Focus Branch (Hyper 1)', 'Hyper Branch 2', 'Hyper Branch 3', 'Hyper Branch 4']

    plt.figure(figsize=(15, 10))

    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.resize(original_image, (320, 320)))
    plt.title('Original Image')
    plt.axis('off')

    # Heatmaps for each branch
    for i, (branch, name) in enumerate(zip(branches, branch_names)):
        feature = afm_features[branch]
        heatmap = generate_heatmap(feature)
        overlay = overlay_heatmap(original_image, heatmap)

        plt.subplot(2, 3, i + 2)
        plt.imshow(overlay)
        plt.title(name)
        plt.axis('off')

    # Save the figure
    output_path = os.path.join(args.output_dir, f'afm_heatmap_{os.path.basename(args.image_path)}.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'\nHeatmap saved to: {output_path}')

    # Also save individual heatmaps
    for branch, name in zip(branches, branch_names):
        feature = afm_features[branch]
        heatmap = generate_heatmap(feature)
        overlay = overlay_heatmap(original_image, heatmap)

        branch_output_path = os.path.join(args.output_dir, f'{branch}_heatmap_{os.path.basename(args.image_path)}.png')
        plt.imsave(branch_output_path, overlay)

    print('\nIndividual heatmaps saved successfully!')
    print('\nEdge Focus Branch (Hyper 1) highlights edge information in the image.')


if __name__ == '__main__':
    main()
