import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from PIL import Image
from torchvision import transforms
import models.stairIQA_resnet as stairIQA_resnet

try:
    import tkinter as tk
    from tkinter import filedialog

    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False


def _stack_to_tensor(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])


def _normalize_stack(crops):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return torch.stack([norm(crop) for crop in crops])


def _select_output(out, trained_database):
    if not isinstance(out, tuple):
        return out
    mapping = {
        'FLIVE': 0,
        'FLIVE_patch': 1,
        'LIVE_challenge': 2,
        'Koniq10k': 3,
        'SPAQ': 4,
        'BID': 5,
    }
    idx = mapping.get(str(trained_database).strip(), None)
    if idx is None:
        return torch.stack([o.squeeze(-1) for o in out], dim=1).mean(1, keepdim=True)
    return out[idx]


def logistic_func(X, b1, b2, b3, b4):
    X = np.asarray(X, dtype=np.float64)
    beta = max(abs(b4), 1e-8)
    z = (X - b3) / beta
    z = np.clip(z, -50.0, 50.0)
    return b2 + (b1 - b2) / (1 + np.exp(-z))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='')
    parser.add_argument('--snapshot', type=str, default='')
    parser.add_argument('--resize', type=int, default=384)
    parser.add_argument('--crop_size', type=int, default=320)
    parser.add_argument('--test_method', type=str, default='five', choices=['one', 'five'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--calibrate_csv', type=str, default='')
    parser.add_argument('--calibrate_dir', type=str, default='')
    parser.add_argument('--trained_database', type=str, default='')
    return parser.parse_args()


def load_model(snapshot, device):
    state = torch.load(snapshot, map_location=device)
    from collections import OrderedDict
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if any(str(k).startswith('module.') for k in state.keys()):
        state = OrderedDict((str(k).replace('module.', ''), v) for k, v in state.items())
    is_imdt = any(str(k).startswith('quality1') for k in state.keys())
    if is_imdt:
        model = stairIQA_resnet.resnet50_imdt(pretrained=True)
    else:
        model = stairIQA_resnet.resnet50(pretrained=True)
    has_eca = any('eca' in str(k) for k in state.keys())
    if hasattr(model, 'eca') and not has_eca:
        model.eca = nn.Identity()
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # input image_path and pkl_path
    image_path = "E:/IQA代码/dataset/LIVE_Challenge/Images/100.bmp"
    snapshot_path = "snapshots/eca/train-ind-LIVE_challenge-stairIQA_resnet-exp_id-2.pkl"
    if (not image_path) or (not os.path.exists(image_path)):
        if _TK_AVAILABLE:
            root = tk.Tk()
            root.withdraw()
            image_path = filedialog.askopenfilename(title='Select image',
                                                    filetypes=[('Images', '*.jpg;*.jpeg;*.png;*.bmp')])
        else:
            image_path = input('Image path: ').strip()
    if (not snapshot_path) or (not os.path.exists(snapshot_path)):
        if _TK_AVAILABLE:
            root = tk.Tk()
            root.withdraw()
            snapshot_path = filedialog.askopenfilename(title='Select model snapshot (.pkl)',
                                                       filetypes=[('Pickle', '*.pkl')])
        else:
            snapshot_path = input('Snapshot (.pkl) path: ').strip()

    model = load_model(snapshot_path, device)

    if args.test_method == 'one':
        tfm = transforms.Compose([
            transforms.Resize(args.resize),
            transforms.CenterCrop(args.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img = Image.open(image_path).convert('RGB')
        img = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(img)
            out = _select_output(out, args.trained_database)
            score = out.item()
        output_text = f"Predicted score: {score:.4f}"
    else:
        tfm = transforms.Compose([
            transforms.Resize(args.resize),
            transforms.FiveCrop(args.crop_size),
            _stack_to_tensor,
            _normalize_stack,
        ])
        img = Image.open(image_path).convert('RGB')
        crops = tfm(img).to(device)
        ncrops, c, h, w = crops.size()
        with torch.no_grad():
            out = model(crops.view(-1, c, h, w))
            out = _select_output(out, args.trained_database)
            score = out.view(1, ncrops, -1).mean(1).item()
        if args.calibrate_csv:
            df = pd.read_csv(args.calibrate_csv)
            ys = df['mos'].to_numpy(dtype=np.float64)
            outs = []
            base_dir = args.calibrate_dir if args.calibrate_dir else os.path.dirname(image_path)
            for fn in df['image']:
                p = os.path.join(base_dir, str(fn).strip())
                img2 = Image.open(p).convert('RGB')
                crops2 = tfm(img2).to(device)
                n2, c2, h2, w2 = crops2.size()
                with torch.no_grad():
                    o_all = model(crops2.view(-1, c2, h2, w2))
                    o_sel = _select_output(o_all, args.trained_database)
                    o = o_sel.view(1, n2, -1).mean(1).item()
                outs.append(o)
            outs = np.asarray(outs, dtype=np.float64)
            beta = [np.max(ys), np.min(ys), np.mean(outs), 0.5]
            popt, _ = curve_fit(logistic_func, outs, ys, p0=beta, maxfev=1000000)
            fitted = logistic_func(score, *popt)
            output_text = f"Predicted score (raw): {score:.4f}  Fitted(MOS-like): {fitted:.4f}"
        else:
            output_text = f"Predicted score: {score:.4f}"

    print(output_text)


if __name__ == "__main__":
    main()
