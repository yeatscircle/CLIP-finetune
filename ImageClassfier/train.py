import torch
import math
import os
import torch.nn as nn
import dataset
from torch.utils.data import DataLoader
import Config
# 用于显示进度条
from tqdm import tqdm
# 绘制评估曲线
from torch.utils.tensorboard import SummaryWriter
import model
import UtilityFunction


def trainer(train_loader, valid_loader, model, config, device, rest_net_flag=False):
    # 对于分类任务, 我们常用cross-entropy评估模型表现.
    criterion = nn.CrossEntropyLoss()
    # 初始化优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    # 模型存储位置
    save_path = config['save_path'] if rest_net_flag else config['resnet_save_path']

    writer = SummaryWriter()
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    for epoch in range(n_epochs):
        model.train()
        loss_record = []
        train_accs = []
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            # 稳定训练的技巧
            if config['clip_flag']:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            optimizer.step()
            step += 1
            acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()
            l_ = loss.detach().item()
            loss_record.append(l_)
            train_accs.append(acc.detach().item())
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': f'{l_:.5f}', 'acc': f'{acc:.5f}'})

        mean_train_acc = sum(train_accs) / len(train_accs)
        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)
        writer.add_scalar('ACC/train', mean_train_acc, step)
        model.eval()  # 设置模型为评估模式
        loss_record = []
        test_accs = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
                acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()

            loss_record.append(loss.item())
            test_accs.append(acc.detach().item())

        mean_valid_acc = sum(test_accs) / len(test_accs)
        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(
            f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f},acc: {mean_train_acc:.4f} Valid loss: {mean_valid_loss:.4f},acc: {mean_valid_acc:.4f} ')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        writer.add_scalar('ACC/valid', mean_valid_acc, step)
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), save_path)  # 保存最优模型
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return


def main():
    # 设计随机种子
    UtilityFunction.all_seed(Config.config['seed'])

    # 创建数据集
    train_dataset = dataset.sportsDataset(os.path.join(Config.config['dataset_dir'], 'train'),
                                          transformer=Config.train_tfm)
    valid_dataset = dataset.sportsDataset(os.path.join(Config.config['dataset_dir'], 'valid'),
                                          transformer=Config.test_tfm)
    # 装载数据
    train_loader = DataLoader(train_dataset, batch_size=Config.config['batch_size'], shuffle=True,
                              num_workers=Config.config['num_workers'], pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.config['batch_size'], shuffle=True,
                              num_workers=Config.config['num_workers'], pin_memory=True)

    my_model = model.CLIPClassifier().to(Config.device)

    trainer(train_loader, valid_loader, my_model, Config.config, Config.device)


if __name__ == '__main__':
    main()
