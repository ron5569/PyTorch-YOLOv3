
from __future__ import division
import sys
import warnings
 
if not sys.warnoptions:
    warnings.simplefilter("ignore")

from models import *
from utils.logger import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


def print_to_tensorboard(patth, prefix):
    print("\n---- Evaluating Model ----")
    # Evaluate the model on the validation set
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=patth,
        iou_thres=0.5,
        conf_thres=0.5,
        nms_thres=0.5,
        img_size=opt.img_size,
        batch_size=8,
    )
    evaluation_metrics = [
        (f"{prefix}_precision", precision.mean()),
        (f"{prefix}_recall", recall.mean()),
        (f"{prefix}_mAP", AP.mean()),
        (f"{prefix}_f1", f1.mean()),
    ]
    logger.list_of_scalars_summary(evaluation_metrics, epoch)
    # Print class APs and mAP
    ap_table = [["Index", "Class name", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
    print(AsciiTable(ap_table).table)
    print(f"---- mAP {AP.mean()}")



def do_transfer():

    def get_yolo_layers(model):
        return [i for i, x in enumerate(model.module_defs) if x['type'] == 'yolo']  # [82, 94, 106] for yolov3
    yolo_layers = get_yolo_layers(model)
    nf = int(model.module_defs[yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)
    for p in model.parameters():
        if p.numel() == nf or p.shape[0] == nf:  # train (yolo biases)
            p.requires_grad = True
        else:  # freeze layer
            p.requires_grad = False


def load_model_custom(weights):
    # load model
    try:
        chkpt = torch.load(weights, map_location=device)
        state_dict = model.state_dict()
        # for k,v in chkpt.items():
        #     if state_dict[k].numel() == v.numel():
        #         print("what so ever!")
        #     else:
        #         print(k,v )
        chkpt2 = {k: v for k, v in chkpt.items() if state_dict[k].numel() == v.numel()}
        model.load_state_dict(chkpt2, strict=False)
    except KeyError as e:
        s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
            "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
        raise KeyError(s) from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom1.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/rgo.data", help="path to data config file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="interval evaluations on validation set")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--transfer", default=True, help="transfer")
    opt = parser.parse_args()
    print(opt)

    logger = Logger("logs")

    available_else_cpu = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(available_else_cpu)

    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(opt.data_config)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    model.apply(weights_init_normal)

    # If specified we start from checkpoint
    if opt.transfer and opt.pretrained_weights.endswith(".pth"):
        print("Do transfer learning")
        load_model_custom(opt.weights_path)
        do_transfer()

    elif opt.pretrained_weights:
        if opt.pretrained_weights.endswith(".pth"):
            load = torch.load(opt.pretrained_weights, map_location=available_else_cpu)
            model.load_state_dict(load)
        else:
            model.load_darknet_weights(opt.pretrained_weights)


    # Get dataloader
    dataset = ListDataset(train_path, augment=True, multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    for epoch in range(opt.epochs):
        model.train()
        start_time = time.time()
        for batch_i, (_, imgs, targets) in enumerate(dataloader):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # ----------------
            #   Log progress
            # ----------------

            log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, opt.epochs, batch_i, len(dataloader))

            metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

            # Log metrics at each YOLO layer
            for i, metric in enumerate(metrics):
                formats = {m: "%.6f" for m in metrics}
                formats["grid_size"] = "%2d"
                formats["cls_acc"] = "%.2f%%"
                row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
                metric_table += [[metric, *row_metrics]]

                # Tensorboard logging
                tensorboard_log = []
                for j, yolo in enumerate(model.yolo_layers):
                    for name, metric in yolo.metrics.items():
                        if name != "grid_size":
                            tensorboard_log += [(f"{name}_{j+1}", metric)]
                tensorboard_log += [("loss", loss.item())]
                #logger.list_of_scalars_summary(tensorboard_log, batches_done)

            log_str += AsciiTable(metric_table).table
            log_str += f"\nTotal loss {loss.item()}"

            # Determine approximate time left for epoch
            epoch_batches_left = len(dataloader) - (batch_i + 1)
            time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
            log_str += f"\n---- ETA {time_left}"

            print(log_str)

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            print_to_tensorboard(train_path, "train")
            print_to_tensorboard(valid_path, "val")

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_%d.pth" % epoch)
