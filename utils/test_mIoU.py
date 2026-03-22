import torch
import numpy as np
from PIL import Image


@torch.no_grad()
def fast_hist_gpu(pred: torch.Tensor, label: torch.Tensor, num_classes: int, ignore_index: int):

    assert pred.device == label.device, (pred.device, label.device)
    assert pred.dtype == torch.long and label.dtype == torch.long

    mask = (label != ignore_index)
    pred = pred[mask]
    label = label[mask]

    pred = pred.clamp(min=0, max=num_classes - 1)
    label = label.clamp(min=0, max=num_classes - 1)

    idx = label * num_classes + pred
    hist = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return hist


@torch.no_grad()
def mean_iou_gpu_from_tensors(
    preds: list,
    labels: list,
    num_classes: int,
    ignore_index: int = 255,
    reduce_zero_label: bool = True,
    device: str = "cuda:0",
):

    dev = torch.device(device)
    hist = torch.zeros((num_classes, num_classes), device=dev, dtype=torch.long)

    for pred, lab in zip(preds, labels):
        if not isinstance(pred, torch.Tensor):
            pred = torch.as_tensor(pred)
        if not isinstance(lab, torch.Tensor):
            lab = torch.as_tensor(lab)

        pred = pred.to(dev, non_blocking=True).long()
        lab = lab.to(dev, non_blocking=True).long()

        if reduce_zero_label:

            lab = lab.clone()
            bg_mask = (lab == 0)
            ignore_mask = (lab == ignore_index)

            lab = lab - 1
            lab[bg_mask] = ignore_index
            lab[ignore_mask] = ignore_index

        hist += fast_hist_gpu(pred, lab, num_classes=num_classes, ignore_index=ignore_index)

    intersect = torch.diag(hist).float()
    union = (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist)).float().clamp(min=1.0)
    iou = intersect / union
    miou = float(iou.mean().item())
    return {"IoU": iou, "mIoU": miou}


@torch.no_grad()
def mean_iou_gpu_from_paths(
    pred_paths: list,
    label_paths: list,
    num_classes: int,
    ignore_index: int = 255,
    reduce_zero_label: bool = True,
    device: str = "cuda:0",
):
    dev = torch.device(device)
    hist = torch.zeros((num_classes, num_classes), device=dev, dtype=torch.long)

    for pred_p, lab_p in zip(pred_paths, label_paths):
        pred = torch.load(pred_p, map_location="cpu").long()
        lab_img = Image.open(lab_p)
        lab_np = np.array(lab_img, dtype=np.int64)
        lab = torch.from_numpy(lab_np).long()

        pred = pred.to(dev, non_blocking=True)
        lab = lab.to(dev, non_blocking=True)

        if reduce_zero_label:
            lab = lab.clone()
            bg_mask = (lab == 0)
            ignore_mask = (lab == ignore_index)

            lab = lab - 1
            lab[bg_mask] = ignore_index
            lab[ignore_mask] = ignore_index

        hist += fast_hist_gpu(pred, lab, num_classes=num_classes, ignore_index=ignore_index)

    intersect = torch.diag(hist).float()
    union = (hist.sum(dim=1) + hist.sum(dim=0) - torch.diag(hist)).float().clamp(min=1.0)
    iou = intersect / union
    miou = float(iou.mean().item())
    return {"IoU": iou, "mIoU": miou}