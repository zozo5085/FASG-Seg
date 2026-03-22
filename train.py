import sys
import os
import argparse
import ssl
import time
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import clip
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random

# path inject
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config.configs import cfg_from_file
from model.model import FASGSeg
from utils.test_mIoU import mean_iou_gpu_from_paths
from utils.preprocess import val_preprocess, preprocess, read_file_list, prepare_dataset_cls_tokens

ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] 隨機種子已固定為: {seed}")


def custom_collate_fn(batch):
    imgs, labels, metas, filenames, pseudo_classes = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    return imgs, labels, metas, filenames, pseudo_classes


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='config/voc_train_ori_cfg.yaml', type=str)
    args = parser.parse_args()
    return args


class Train(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_filenames, _, self.train_images, self.train_labels, _, _, _, self.pseudo_classes = read_file_list(
            cfg)

    def __getitem__(self, idx):
        with open(self.train_images[idx], 'rb') as f:
            value_buf = f.read()
        with open(self.train_labels[idx], 'rb') as f:
            label_buf = f.read()
        img, label, img_metas = preprocess(self.cfg, value_buf, label_buf, return_meta=True, unlabeled=False)
        return img, label, img_metas, self.train_images[idx], self.pseudo_classes[idx]

    def __len__(self):
        return len(self.train_images)


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1 - epoch / num_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_single_gpu():
    set_seed(42)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    clip_model, _ = clip.load("ViT-B/16")
    clip_model = clip_model.to(device)

    args = get_parser()
    cfg = cfg_from_file(args.cfg_file)
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    log = open('experiments/log_voc_rectification.txt', mode='a')

    train_filenames, val_filenames, train_images, train_labels, val_images, val_labels, results_iou, pseudo_classes = read_file_list(
        cfg)
    cls_name_token, classes = prepare_dataset_cls_tokens(cfg)
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT, map_location="cpu").to(device)

    train_data = Train(cfg)
    train_loader = DataLoader(dataset=train_data, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True,
                              batch_size=cfg.TRAIN.BATCH_SIZE, collate_fn=custom_collate_fn)

    model = FASGSeg(cfg=cfg, clip_model=clip_model, rank=device, zeroshot_weights=text_weight).to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR, momentum=0.9,
                          weight_decay=0.0005)

    max_epoch = cfg.TRAIN.MAX_EPOCH
    stop_epoch = cfg.TRAIN.EPOCH if cfg.TRAIN.EPOCH >= 0 else max_epoch
    c_num = cfg.DATASET.NUM_CLASSES
    best_iou = 0.0
    pd_thresh = 0.5  # PD Trick 參數

    for epoch in range(max_epoch):
        idx = 0
        model.train()
        running_loss = 0.0
        lr = adjust_learning_rate_poly(optimizer, epoch, max_epoch, cfg.TRAIN.LR, power=0.9)
        loop = tqdm(train_loader, desc=f"Epoch {epoch} Training")

        for img, label, img_metas, filenames, pseudo_class in loop:
            time.sleep(0.08)
            gt_cls = []
            batch_size = img.shape[0]
            for i in range(batch_size):
                temp = [int(tensor) if isinstance(tensor, int) else int(tensor.item()) for tensor in pseudo_class[i]]
                gt_cls.append(temp)

            if sum([len(t) for t in gt_cls]) == 0:
                continue

            img = img.to(device)
            output, loss = model(img, gt_cls, text_weight, cls_name_token, training=True, img_metas=img_metas)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            loop.set_postfix(lr=f"{lr:.6f}", loss=f"{loss.item():.4f}", avg=f"{running_loss / (idx + 1):.4f}")
            idx += 1

        print(f'epoch {epoch} finish, lr:{lr:.6f}', file=log)

        model.eval()
        success_num = 0
        current_results_iou = []

        with torch.no_grad():
            for v_idx in tqdm(range(len(val_images)), desc="Validating"):
                with open(val_images[v_idx], 'rb') as f:
                    value_buf = f.read()
                img = val_preprocess(cfg, value_buf).unsqueeze(dim=0).to(device)
                label = Image.open(val_labels[v_idx])
                ori_shape = tuple((label.size[1], label.size[0]))
                gt_cls = []
                shape = img.shape[2:]

                output = model(img, gt_cls, text_weight, cls_name_token, training=False)

                # ==========================================================
                # 🔥 加入原版能暴增分數的 PD Trick (推論外掛)
                # ==========================================================
                N, C, H, W = output.shape
                _output = F.softmax(output * 10, dim=1)
                max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
                selected_cls = (max_cls_conf < pd_thresh)[:, :, None, None].expand(N, C, H, W)
                output[selected_cls] = -100.0

                output = F.interpolate(output, shape, None, 'bilinear', False).reshape(1, c_num, shape[0], shape[1])
                output = F.interpolate(output, ori_shape, None, 'bilinear', False).reshape(1, c_num, ori_shape[0],
                                                                                           ori_shape[1])

                output = F.softmax(output, dim=1)
                output = torch.argmax(output, dim=1).squeeze(dim=0)

                save_path = os.path.join(cfg.SAVE_DIR, val_filenames[v_idx] + '.pt')
                torch.save(output.cpu(), save_path)
                current_results_iou.append(save_path)
                success_num += 1

            iou = mean_iou_gpu_from_paths(current_results_iou, val_labels, num_classes=c_num, ignore_index=255,
                                          reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL)
            avg = iou['mIoU']

            print(f"Epoch {epoch} strict mIoU: {avg:.4f}")
            print(f'epoch={epoch} strict_miou={avg:.4f}', file=log)
            log.flush()

            if avg > best_iou:
                best_iou = avg
                torch.save(model.state_dict(), os.path.join(cfg.SAVE_DIR, 'best_weight.pth'))
                print(f"*** New Best Strict Saved: {best_iou:.4f} ***")

        if epoch == stop_epoch:
            break
    log.close()


if __name__ == '__main__':
    train_single_gpu()