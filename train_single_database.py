import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
from torch.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn

import IQADataset
import models.stairIQA_resnet as stairIQA_resnet
from utils import performance_fit, L1RankLoss
import platform


def _stack_to_tensor(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])


def _normalize_stack(crops):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return torch.stack([norm(crop) for crop in crops])


def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="In the wild Image Quality Assessment")
    parser.add_argument('--gpu', help="GPU device id to use [0]", default=0, type=int)
    parser.add_argument('--num_epochs', help='Maximum number of training epochs.', default=30, type=int)
    parser.add_argument('--batch_size', help='Batch size.', default=40, type=int)
    parser.add_argument('--resize', help='resize.', type=int)
    parser.add_argument('--crop_size', help='crop_size.', type=int)
    parser.add_argument('--lr', type=float, default=0.00001)
    # parser.add_argument('--decay_ratio', type=float, default=0.9)
    # parser.add_argument('--decay_interval', type=float, default=10)
    parser.add_argument('--snapshot', help='Path of model snapshot.', default='', type=str)
    parser.add_argument('--results_path', type=str)
    parser.add_argument('--database_dir', type=str)
    parser.add_argument('--model', default='ResNet', type=str)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--print_samples', type=int, default=50)
    parser.add_argument('--database', default='FLIVE', type=str)
    parser.add_argument('--test_method', default='five', type=str,
                        help='use the center crop or five crop to test the image')
    parser.add_argument('--exp_id', default=None, type=int,
                        help='Specific experiment ID to run (0-9). If not set, runs all.')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    gpu = args.gpu
    model = args.model
    cudnn.enabled = True
    cudnn.benchmark = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    # decay_interval = args.decay_interval
    # decay_ratio = args.decay_ratio
    snapshot = args.snapshot
    database = args.database
    print_samples = args.print_samples
    results_path = args.results_path
    database_dir = args.database_dir
    resize = args.resize
    crop_size = args.crop_size
    train_filename_list = ''
    test_filename_list = ''

    best_all = np.zeros([10, 4])
    # If exp_id is provided, only run that one. Otherwise,run all 10 rounds.
    exp_ids = [args.exp_id] if args.exp_id is not None else range(10)
    for exp_id in exp_ids:

        print('The current exp_id is ' + str(exp_id))
        if not os.path.exists(snapshot):
            os.makedirs(snapshot)
        trained_model_file = os.path.join(snapshot,
                                          'train-ind-{}-{}-exp_id-{}.pkl'.format(database, args.model, exp_id))

        print('The save model name is ' + trained_model_file)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

        if database == 'Koniq10k':
            train_filename_list = 'csvfiles/Koniq10k_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/Koniq10k_test_' + str(exp_id) + '.csv'
        elif database == 'FLIVE':
            train_filename_list = 'csvfiles/FLIVE_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/FLIVE_test_' + str(exp_id) + '.csv'
        elif database == 'FLIVE_patch':
            train_filename_list = 'csvfiles/FLIVE_patch_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/FLIVE_patch_test_' + str(exp_id) + '.csv'
        elif database == 'LIVE_challenge':
            train_filename_list = 'csvfiles/LIVE_challenge_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/LIVE_challenge_test_' + str(exp_id) + '.csv'
        elif database == 'SPAQ':
            train_filename_list = 'csvfiles/SPAQ_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/SPAQ_test_' + str(exp_id) + '.csv'
        elif database == 'BID':
            train_filename_list = 'csvfiles/BID_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/BID_test_' + str(exp_id) + '.csv'
        elif database == 'LIVE':
            train_filename_list = 'csvfiles/LIVE_train_' + str(exp_id) + '.csv'
            test_filename_list = 'csvfiles/LIVE_test_' + str(exp_id) + '.csv'
        print(train_filename_list)
        print(test_filename_list)

        # load the network
        if args.model == 'stairIQA_resnet':
            model = stairIQA_resnet.resnet50(pretrained=True)

        transformations_train = transforms.Compose([transforms.Resize(resize), transforms.RandomCrop(crop_size),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])
        if args.test_method == 'one':
            transformations_test = transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(crop_size),
                                                       transforms.ToTensor(),
                                                       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                            std=[0.229, 0.224, 0.225])])
        elif args.test_method == 'five':
            transformations_test = transforms.Compose([
                transforms.Resize(resize),
                transforms.FiveCrop(crop_size),
                _stack_to_tensor,
                _normalize_stack,
            ])

        train_dataset = IQADataset.IQA_dataloader(database_dir, train_filename_list, transformations_train, database)
        test_dataset = IQADataset.IQA_dataloader(database_dir, test_filename_list,
                                                 transformations_test, database)

        _workers = 0 if platform.system().lower() == 'windows' else 8
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                                                   num_workers=_workers)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                                  num_workers=_workers)

        if args.multi_gpu:
            model = torch.nn.DataParallel(model)
            model = model.to(device)
        else:
            model = model.to(device)

        # criterion = nn.MSELoss().to(device)
        criterion = L1RankLoss(l1_w=1.0, rank_w=1.0, hard_thred=1, use_margin=False).to(device)
        param_num = 0
        for param in model.parameters():
            param_num += int(np.prod(param.shape))
        print('Trainable params: %.2f million' % (param_num / 1e6))

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_interval, gamma=decay_ratio)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1.7e-5)
        # 初始化混合精度训练的梯度缩放器
        scaler = GradScaler('cuda')
        print("Ready to train network")

        best_test_criterion = -1  # SRCC min
        best = np.zeros(4)

        n_train = len(train_dataset)
        n_test = len(test_dataset)

        for epoch in range(num_epochs):
            # train
            model.train()

            batch_losses = []
            batch_losses_each_disp = []
            session_start_time = time.time()
            for i, (image, mos) in enumerate(train_loader):
                image = image.to(device)
                mos = mos[:, np.newaxis]
                mos = mos.to(device)

                # mos_output = model(image)
                with autocast('cuda'):
                    mos_output, _, _ = model(image)  # 忽略 layer_weights 和 distortion_type
                    loss = criterion(mos_output, mos)

                batch_losses.append(loss.item())
                batch_losses_each_disp.append(loss.item())

                optimizer.zero_grad()  # clear gradients for next train
                # 使用梯度缩放
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                # torch.autograd.backward(loss)
                # optimizer.step()

                if (i + 1) % print_samples == 0:
                    session_end_time = time.time()
                    avg_loss_epoch = sum(batch_losses_each_disp) / print_samples
                    print('Epoch: %d/%d | Step: %d/%d | Training loss: %.4f' %
                          (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size,
                           avg_loss_epoch))
                    batch_losses_each_disp = []
                    print('CostTime: {:.4f}'.format(session_end_time - session_start_time))
                    session_start_time = time.time()

            avg_loss = sum(batch_losses) / (len(train_dataset) // batch_size)
            print('Epoch %d averaged training loss: %.4f' % (epoch + 1, avg_loss))

            # 失真感知适配器会动态生成层级权重，无需手动打印

            scheduler.step()
            lr_current = scheduler.get_last_lr()
            print('The current learning rate is {:.06f}'.format(lr_current[0]))

            # Test 
            model.eval()
            y_output = np.zeros(n_test)
            y_test = np.zeros(n_test)

            with torch.no_grad():
                for i, (image, mos) in enumerate(test_loader):
                    if args.test_method == 'one':
                        image = image.to(device)
                        y_test[i] = mos.item()
                        mos = mos.to(device)
                        outputs, _, _ = model(image)  # 忽略 layer_weights 和 distortion_type
                        y_output[i] = outputs.item()


                    elif args.test_method == 'five':
                        bs, ncrops, c, h, w = image.size()
                        y_test[i] = mos.item()
                        image = image.to(device)
                        mos = mos.to(device)

                        outputs, _, _ = model(image.view(-1, c, h, w))  # 忽略 layer_weights 和 distortion_type
                        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
                        y_output[i] = outputs_avg.item()

                test_PLCC, test_SRCC, test_KRCC, test_RMSE = performance_fit(y_test, y_output)
                print("Test results: SRCC={:.4f}, KRCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SRCC, test_KRCC,
                                                                                                test_PLCC, test_RMSE))

                if test_SRCC > best_test_criterion:
                    print("Update best model using best_val_criterion ")
                    torch.save(model.state_dict(), trained_model_file)
                    best[0:4] = [test_SRCC, test_KRCC, test_PLCC, test_RMSE]
                    best_test_criterion = test_SRCC  # update best val SRCC

                    print(
                        "The best Test results: SRCC={:.4f}, KRCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SRCC,
                                                                                                           test_KRCC,
                                                                                                           test_PLCC,
                                                                                                           test_RMSE))
        print(database)
        best_all[exp_id, :] = best
        print("The best Val results: SRCC={:.4f}, KRCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best[0], best[1],
                                                                                                best[2], best[3]))
        print(
            '*************************************************************************************************************************')
    best_median = np.median(best_all, 0)
    best_mean = np.mean(best_all, 0)
    best_std = np.std(best_all, 0)
    print(
        '*************************************************************************************************************************')
    print(best_all)
    print("The median val results: SRCC={:.4f}, KRCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best_median[0],
                                                                                              best_median[1],
                                                                                              best_median[2],
                                                                                              best_median[3]))
    print(
        "The mean val results: SRCC={:.4f}, KRCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best_mean[0], best_mean[1],
                                                                                          best_mean[2], best_mean[3]))
    print("The std val results: SRCC={:.4f}, KRCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best_std[0], best_std[1],
                                                                                           best_std[2], best_std[3]))
    print(
        '*************************************************************************************************************************')
