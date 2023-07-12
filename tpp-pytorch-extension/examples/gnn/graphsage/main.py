################################################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                                  #
#                                                                                              #
# For information on the license, see the LICENSE file.                                        #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/                       #
# SPDX-License-Identifier: BSD-3-Clause                                                        #
################################################################################################
# Author: Sasikanth Avancha (Intel Corp.)                                                      #
# Original version:                                                                            #
# https://github.com/dmlc/dgl/blob/master/examples/pytorch/ogb/ogbn-products/graphsage/main.py #
################################################################################################

import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.nn.pytorch as dglnn
from dgl.nn.pytorch.conv import SAGEConv
import time
import argparse
import tqdm
from ogb.nodeproppred import DglNodePropPredDataset
from tpp_pytorch_extension.gnn.graphsage import fused_graphsage
from tpp_pytorch_extension.gnn.common import gnn_utils
import tpp_pytorch_extension as ppx
import os, psutil
from contextlib import contextmanager


@contextmanager
def opt_impl(enable=True, use_bf16=False):
    try:
        global SAGEConv
        orig_SAGEConv = SAGEConv
        try:
            if enable:
                if use_bf16:
                    SAGEConv = fused_graphsage.SAGEConvOptBF16
                else:
                    SAGEConv = fused_graphsage.SAGEConvOpt
            yield
        finally:
            SAGEConv = orig_SAGEConv
    except ImportError as e:
        pass


def block(model):
    for m in model.modules():
        if hasattr(m, "maybe_block_params"):
            m.maybe_block_params()


class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        if not args.opt_mlp:
            self.dropout = nn.Dropout(dropout)
            dropout = 0
            self.activation = activation

        self.layers = nn.ModuleList()
        self.layers.append(
            SAGEConv(
                in_feats, n_hidden, "mean", feat_drop=dropout, activation=activation
            )
        )
        for i in range(1, n_layers - 1):
            self.layers.append(
                SAGEConv(
                    n_hidden, n_hidden, "mean", feat_drop=dropout, activation=activation
                )
            )
        self.layers.append(SAGEConv(n_hidden, n_classes, "mean"))

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_dst = h[: block.num_dst_nodes()]
            h = layer(block, (h, h_dst))
            if not args.opt_mlp:
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)
        return h

    def inference(self, g, x, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        if device.type != "cpu":
            for l, layer in enumerate(self.layers):
                y = th.zeros(
                    g.num_nodes(),
                    self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                ).to(device)

                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.NodeDataLoader(
                    g,
                    th.arange(g.num_nodes()),
                    sampler,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=args.num_workers,
                )

                for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                    block = blocks[0].int().to(device)

                    h = x[input_nodes]
                    h_dst = h[: block.num_dst_nodes()]
                    h = layer(block, (h, h_dst))
                    if l != len(self.layers) - 1:
                        h = self.activation(h)
                        h = self.dropout(h)

                    y[output_nodes] = h

                x = y
            return y
        else:
            if args.use_bf16:
                x = x.to(th.bfloat16)

            for l, layer in enumerate(self.layers):
                x = layer(g, (x, x))
                if not args.opt_mlp:
                    if l != len(self.layers) - 1:
                        x = self.activation(x)
                        x = self.dropout(x)
            return x


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, test_nid, device):
    """
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : Validation node ids
    test_nid: Test node ids
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, device).to(th.float32)
    model.train()
    return (
        compute_acc(pred[val_nid], labels[val_nid]),
        compute_acc(pred[test_nid], labels[test_nid]),
        pred,
    )


def load_subtensor(nfeat, labels, seeds, input_nodes):
    """
    Extracts features and labels for a set of nodes.
    """
    if args.opt_mlp:
        batch_inputs = gnn_utils.gather_features(nfeat, input_nodes)
    else:
        batch_inputs = nfeat[input_nodes]

    batch_labels = labels[seeds]

    return batch_inputs, batch_labels


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    th.save(state, filename, _use_new_zipfile_serialization=True)


#### Entry point
def run(args, device, data):
    # Unpack data
    train_nid, val_nid, test_nid, in_feats, labels, n_classes, nfeat, g = data

    # Create PyTorch DataLoader for constructing blocks
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )

    if args.dataset == "ogbn-papers100M":
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            train_nid,
            sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
            use_cpu_worker_affinity=args.cpu_worker_aff,
            persistent_workers=args.cpu_worker_aff,
            formats=["csc"],
        )
    else:
        dataloader = dgl.dataloading.NodeDataLoader(
            g,
            train_nid,
            sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=args.num_workers,
            persistent_workers=args.cpu_worker_aff,
            use_cpu_worker_affinity=args.cpu_worker_aff,
        )

    # Define model and optimizer
    with opt_impl(args.opt_mlp, args.use_bf16):
        model = SAGE(
            in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout
        )
        model = model.to(device)
    block(model)
    loss_fcn = nn.CrossEntropyLoss()

    if args.opt_mlp:
        no_decay = ["bias"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.wd,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = ppx.optim.AdamW(
            optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    avgb = 0
    best_eval_acc = 0
    best_test_acc = 0
    if args.opt_mlp:
        ppx.manual_seed(args.seed)
    record_shapes = False
    with th.autograd.profiler.profile(
        enabled=args.profile, record_shapes=record_shapes
    ) as prof:
        if prof and args.opt_mlp:
            ppx.reset_debug_timers()
        for epoch in range(args.num_epochs):
            batch_fwd_time = AverageMeter()
            batch_bwd_time = AverageMeter()
            data_time = AverageMeter()
            gather_time = AverageMeter()

            tic = time.time()

            end = time.time()
            for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
                if args.opt_mlp and epoch == 0 and step == 0:
                    cores = int(os.environ["OMP_NUM_THREADS"])
                    gnn_utils.affinitize_cores(cores, args.num_workers)
                # measure data loading time
                t0 = time.time()
                data_time.update(t0 - end)

                # Load the input features as well as output labels
                batch_inputs, batch_labels = load_subtensor(
                    nfeat, labels, seeds, input_nodes
                )
                t1 = time.time()
                gather_time.update(t1 - t0)
                t2 = time.time()

                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs).to(th.float32)
                t3 = time.time()
                batch_fwd_time.update(t3 - t2)

                loss = loss_fcn(batch_pred, batch_labels)

                optimizer.zero_grad()

                t4 = time.time()
                loss.backward()
                t5 = time.time()
                batch_bwd_time.update(t5 - t4)

                optimizer.step()

                end = time.time()

                if step % args.log_every == 0:
                    acc = compute_acc(batch_pred, batch_labels)
                    # Record step loss, training acc, dataloading time (DL), gather time (GT)
                    # forward time (FWD) and backprop time (BWD)
                    print(
                        "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Acc {:.2f} | "
                        "DL (s) {data_time.val:.3f} ({data_time.avg:.3f}) | "
                        "GT (s) {gather_time.val:.3f} ({gather_time.avg:.3f}) | "
                        "FWD (s) {batch_fwd_time.val:.3f} ({batch_fwd_time.avg:.3f}) | "
                        "BWD (s) {batch_bwd_time.val:.3f} ({batch_bwd_time.avg:.3f}) | ".format(
                            epoch,
                            step,
                            loss.item(),
                            acc.item(),
                            data_time=data_time,
                            gather_time=gather_time,
                            batch_fwd_time=batch_fwd_time,
                            batch_bwd_time=batch_bwd_time,
                        )
                    )

            toc = time.time()

            print("Epoch Time(s): {:.4f}".format(toc - tic))

            if epoch >= 1:
                avgb += toc - tic

            if epoch % args.eval_every == 0 and epoch != 0:
                if args.dataset == "ogbn-products":
                    eval_acc, test_acc, pred = evaluate(
                        model, g, nfeat, labels, val_nid, test_nid, device
                    )

                    if args.save_pred:
                        np.savetxt(
                            args.save_pred + "%02d" % epoch,
                            pred.argmax(1).cpu().numpy(),
                            "%d",
                        )
                    print("Eval Acc {:.4f}".format(eval_acc))
                    if eval_acc > best_eval_acc:
                        best_eval_acc = eval_acc
                        best_test_acc = test_acc
                    print(
                        "Best Eval Acc {:.4f} Test Acc {:.4f}".format(
                            best_eval_acc, best_test_acc
                        )
                    )
                elif args.dataset == "ogbn-papers100M":
                    # On low- to medium-sized memory systems (< 512GB DRAM), online inference for ogb-papers100M is not possible
                    # so, we save the model and run inference separately
                    save_checkpoint(
                        {"state_dict": model.state_dict()},
                        filename="ogbp100M" + str(epoch) + ".pth.tar",
                    )

    if prof and args.opt_mlp:
        ppx.print_debug_timers(0)

    if prof:
        with open("gsage.prof", "w") as prof_f:
            prof_f.write(
                prof.key_averages(group_by_input_shape=record_shapes).table(
                    sort_by="cpu_time_total"
                )
            )
        if ppx.extend_profiler:
            with open("gsage_nested.prof", "w") as prof_f:
                prof_f.write(
                    prof.nested_key_averages().table(sort_by=None, row_limit=1000)
                )
            with open("gsage_top_level.prof", "w") as prof_f:
                prof_f.write(
                    prof.nested_key_averages(only_top_level=True).table(
                        sort_by="cpu_time_total"
                    )
                )

    print("Avg epoch time: {}".format(avgb / (epoch)))
    if args.dataset == "ogbn-products":
        return best_test_acc, avgb / (epoch)
    elif args.dataset == "ogbn-papers100M":
        return avgb / (epoch)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("CPU training")
    argparser.add_argument("--num-epochs", type=int, default=20)
    argparser.add_argument("--num-hidden", type=int, default=256)
    argparser.add_argument("--num-layers", type=int, default=3)
    argparser.add_argument("--fan-out", type=str, default="5,10,15")
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--val-batch-size", type=int, default=10000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.003)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of sampling processes. Use 0 for no extra process.",
    )
    argparser.add_argument("--save-pred", type=str, default="")
    argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument(
        "--opt_mlp",
        action="store_true",
        help="Whether to use optimized MLP impl when available",
    )
    argparser.add_argument(
        "--use_bf16",
        action="store_true",
        help="Whether to use BF16 datatype when available",
    )
    argparser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    argparser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    argparser.add_argument(
        "--profile",
        action="store_true",
        help="Whether to profile or not",
    )
    argparser.add_argument(
        "--cpu-worker-aff",
        action="store_true",
        help="Whether to affinitize dataloader workers or not",
    )
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="default dataset name",
    )

    args = argparser.parse_args()

    device = th.device("cpu")

    # load data
    data = DglNodePropPredDataset(name=args.dataset)
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    graph, labels = data[0]
    n_classes = data.num_classes

    if args.dataset == "ogbn-products":
        nfeat = graph.ndata.pop("feat").to(device)
        if args.use_bf16:
            nfeat = nfeat.to(th.bfloat16)
        labels = labels[:, 0].to(device)
        in_feats = nfeat.shape[1]
        n_classes = (labels.max() + 1).item()
    elif args.dataset == "ogbn-papers100M":
        # For higher accuracy, add reverse edges
        ed = graph.edges()
        graph = dgl.add_edges(graph, ed[1], ed[0])
        labels = labels[:, 0].long()
        in_feats = graph.ndata["feat"].shape[1]
        nfeat = graph.ndata["feat"]
        if args.use_bf16:
            nfeat = nfeat.to(th.bfloat16)
        del data, ed

    # Create csr/coo/csc formats before launching sampling processes.
    # This avoids creating certain formats in each data loader process, which saves momory and CPU.
    # Works only for ogbn-products. Crashes when applied to ogbn-papers

    if args.dataset != "ogbn-papers100M":
        graph.create_formats_()
    # Pack data
    data = train_idx, val_idx, test_idx, in_feats, labels, n_classes, nfeat, graph

    # Run 10 times
    test_accs = []
    epoch_time = []
    if not args.profile:
        if args.dataset == "ogbn-products":
            for i in range(1):
                acc, et = run(args, device, data)
                test_accs.append(acc)
                epoch_time.append(et)
                print(
                    "Average test accuracy:", np.mean(test_accs), "±", np.std(test_accs)
                )
                print(
                    "Average epoch time:", np.mean(epoch_time), "±", np.std(epoch_time)
                )
        elif args.dataset == "ogbn-papers100M":
            for i in range(1):
                et = run(args, device, data)
                epoch_time.append(et)
                print(
                    "Average epoch time:", np.mean(epoch_time), "±", np.std(epoch_time)
                )

    else:
        run(args, device, data)
