import argparse
import logging
import random
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
try:
    from transformers import (ConstantLRSchedule, WarmupLinearSchedule, WarmupConstantSchedule)
except:
    from transformers import get_constant_schedule, get_constant_schedule_with_warmup,  get_linear_schedule_with_warmup

from modeling import modeling_greaselm_new_experiments
from utils import data_utils
from utils import optimization_utils
from utils import parser_utils
from utils import utils


DECODER_DEFAULT_LR = {
    'csqa': 1e-3,
    'obqa': 3e-4,
    'medqa_usmle': 1e-3,
}

import numpy as np

import socket, os, subprocess

logger = logging.getLogger(__name__)


def load_data(args, devices, kg):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and args.cuda:
        torch.cuda.manual_seed(args.seed)


    #########################################################
    # Construct the dataset
    #########################################################
    dataset = data_utils.GreaseLM_DataLoader(args.train_statements, args.train_adj,
        args.dev_statements, args.dev_adj,
        args.test_statements, args.test_adj,
        batch_size=args.batch_size, eval_batch_size=args.eval_batch_size,
        device=devices,
        model_name=args.encoder,
        max_node_num=args.max_node_num, max_seq_length=args.max_seq_len,
        is_inhouse=args.inhouse, inhouse_train_qids_path=args.inhouse_train_qids,
        subsample=args.subsample, n_train=args.n_train, debug=args.debug, cxt_node_connects_all=args.cxt_node_connects_all, kg=kg)

    return dataset


def construct_model(args, kg):
    ########################################################
    #   Load pretrained concept embeddings
    ########################################################
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = np.concatenate(cp_emb, 1)
    cp_emb = torch.tensor(cp_emb, dtype=torch.float)

    concept_num, concept_in_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))
    if args.random_ent_emb:
        cp_emb = None
        freeze_ent_emb = False
        concept_in_dim = args.gnn_dim
    else:
        freeze_ent_emb = args.freeze_ent_emb

    ##########################################################
    #   Build model
    ##########################################################

    if kg == "cpnet":
        n_ntype = 4
        n_etype = 38
    elif kg == "ddb":
        n_ntype = 4
        n_etype = 34
    else:
        raise ValueError("Invalid KG.")
    if args.cxt_node_connects_all:
        n_etype += 2
    model = modeling_greaselm_new_experiments.GreaseLM(args, args.encoder, k=args.k, n_ntype=n_ntype, n_etype=n_etype, n_concept=concept_num,
        concept_dim=args.gnn_dim,
        concept_in_dim=concept_in_dim,
        n_attention_head=args.att_head_num, fc_dim=args.fc_dim, n_fc_layer=args.fc_layer_num,
        p_emb=args.dropouti, p_gnn=args.dropoutg, p_fc=args.dropoutf,
        pretrained_concept_emb=cp_emb, freeze_ent_emb=freeze_ent_emb,
        init_range=args.init_range, ie_dim=args.ie_dim, info_exchange=args.info_exchange, ie_layer_num=args.ie_layer_num, sep_ie_layers=args.sep_ie_layers, layer_id=args.encoder_layer)
    return model


def sep_params(model, loaded_roberta_keys):
    """Separate the parameters into loaded and not loaded."""
    loaded_params = dict()
    not_loaded_params = dict()
    params_to_freeze = []
    small_lr_params = dict()
    large_lr_params = dict()
    for n, p in model.named_parameters():
        if n in loaded_roberta_keys:
            loaded_params[n] = p
            params_to_freeze.append(p)
            small_lr_params[n] = p
        else:
            not_loaded_params[n] = p
            large_lr_params[n] = p

    return loaded_params, not_loaded_params, params_to_freeze, small_lr_params, large_lr_params


def count_parameters(loaded_params, not_loaded_params):
    num_params = sum(p.numel() for p in not_loaded_params.values() if p.requires_grad)
    num_fixed_params = sum(p.numel() for p in not_loaded_params.values() if not p.requires_grad)
    num_loaded_params = sum(p.numel() for p in loaded_params.values())
    print('num_trainable_params:', num_params)
    print('num_fixed_params:', num_fixed_params)
    print('num_loaded_params:', num_loaded_params)
    print('num_total_params:', num_params + num_fixed_params + num_loaded_params)


def calc_loss_and_acc(logits, labels, loss_type, loss_func):
    bs = labels.size(0)

    if loss_type == 'margin_rank':
        num_choice = logits.size(1)
        flat_logits = logits.view(-1)
        correct_mask = F.one_hot(labels, num_classes=num_choice).view(-1)  # of length batch_size*num_choice
        correct_logits = flat_logits[correct_mask == 1].contiguous().view(-1, 1).expand(-1, num_choice - 1).contiguous().view(-1)  # of length batch_size*(num_choice-1)
        wrong_logits = flat_logits[correct_mask == 0]
        y = wrong_logits.new_ones((wrong_logits.size(0),))
        loss = loss_func(correct_logits, wrong_logits, y)  # margin ranking loss
    elif loss_type == 'cross_entropy':
        loss = loss_func(logits, labels)
    loss *= bs

    n_corrects = (logits.argmax(1) == labels).sum().item()

    return loss, n_corrects


def calc_eval_accuracy(eval_set, model, loss_type, loss_func, debug, save_test_preds, preds_path):
    """Eval on the dev or test set - calculate loss and accuracy"""
    total_loss_acm = 0.0
    n_samples_acm = n_corrects_acm = 0
    model.eval()
    if save_test_preds:
        utils.check_path(preds_path)
        f_preds = open(preds_path, 'w')
    with torch.no_grad():
        for qids, labels, *input_data in tqdm(eval_set, desc="Dev/Test batch"):
            bs = labels.size(0)
            logits, _ = model(*input_data)

            loss, n_corrects = calc_loss_and_acc(logits, labels, loss_type, loss_func)

            total_loss_acm += loss.item()
            n_corrects_acm += n_corrects
            n_samples_acm += bs
            if save_test_preds:
                predictions = logits.argmax(1) #[bsize, ]
                for qid, pred in zip(qids, predictions):
                    print ('{},{}'.format(qid, chr(ord('A') + pred.item())), file=f_preds)
                    f_preds.flush()
            if debug:
                break
    if save_test_preds:
        f_preds.close()
    return total_loss_acm / n_samples_acm, n_corrects_acm / n_samples_acm


def train(args, resume, has_test_split, devices, kg):
    print("args: {}".format(args))

    if resume:
        args.save_dir = os.path.dirname(args.resume_checkpoint)
    if not args.debug:
        log_path = os.path.join(args.save_dir, 'log.csv')
        utils.check_path(log_path)

        # Set up tensorboard
        tb_dir = os.path.join(args.save_dir, "tb")
        if not resume:
            with open(log_path, 'w') as fout:
                fout.write('epoch,step,dev_acc,test_acc,best_dev_acc,final_test_acc,best_dev_epoch\n')

            if os.path.exists(tb_dir):
                shutil.rmtree(tb_dir)
        tb_writer = SummaryWriter(tb_dir)

        config_path = os.path.join(args.save_dir, 'config.json')
        utils.export_config(args, config_path)

        model_path = os.path.join(args.save_dir, 'model.pt')

    dataset = load_data(args, devices, kg)
    train_dataloader = dataset.train()
    dev_dataloader = dataset.dev()
    if has_test_split:
        test_dataloader = dataset.test()

    model = construct_model(args, kg)
    model.lmgnn.mp.resize_token_embeddings(len(dataset.tokenizer))

    # Get the names of the loaded LM parameters
    loading_info = model.lmgnn.loading_info
    # loaded_roberta_keys = [k.replace("roberta.", "lmgnn.mp.") for k in loading_info["all_keys"]]
    def _rename_key(key):
        if key.startswith("roberta."):
            return key.replace("roberta.", "lmgnn.mp.")
        else:
            return "lmgnn.mp." + key

    loaded_roberta_keys = [_rename_key(k) for k in loading_info["all_keys"]]

    # Separate the parameters into loaded and not loaded
    loaded_params, not_loaded_params, params_to_freeze, small_lr_params, large_lr_params = sep_params(model, loaded_roberta_keys)

    # print non-loaded parameters
    print('Non-loaded parameters:')
    for name, param in not_loaded_params.items():
        if param.requires_grad:
            print('\t{:45}\ttrainable\t{}\tdevice:{}'.format(name, param.size(), param.device))
        else:
            print('\t{:45}\tfixed\t{}\tdevice:{}'.format(name, param.size(), param.device))

    # Count parameters
    count_parameters(loaded_params, not_loaded_params)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    #########################################################
    # Create an optimizer
    #########################################################
    grouped_parameters = [
        {'params': [p for n, p in small_lr_params.items() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.encoder_lr},
        {'params': [p for n, p in small_lr_params.items() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.encoder_lr},
        {'params': [p for n, p in large_lr_params.items() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay, 'lr': args.decoder_lr},
        {'params': [p for n, p in large_lr_params.items() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr': args.decoder_lr},
    ]
    optimizer = optimization_utils.OPTIMIZER_CLASSES[args.optim](grouped_parameters)

    #########################################################
    # Optionally loading from a checkpoint
    #########################################################
    if resume:
        print("loading from checkpoint: {}".format(args.resume_checkpoint))
        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')
        last_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        model.load_state_dict(checkpoint["model"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])
        best_dev_epoch = checkpoint["best_dev_epoch"]
        best_dev_acc = checkpoint["best_dev_acc"]
        final_test_acc = checkpoint["final_test_acc"]
    else:
        last_epoch = -1
        global_step = 0
        best_dev_epoch = best_dev_acc = final_test_acc = 0


    #########################################################
    # Create a scheduler
    #########################################################
    if args.lr_schedule == 'fixed':
        try:
            scheduler = ConstantLRSchedule(optimizer)
        except:
            scheduler = get_constant_schedule(optimizer)
    elif args.lr_schedule == 'warmup_constant':
        try:
            scheduler = WarmupConstantSchedule(optimizer, warmup_steps=args.warmup_steps, last_epoch=last_epoch)
        except:
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, last_epoch=last_epoch)
    elif args.lr_schedule == 'warmup_linear':
        max_steps = int(args.n_epochs * (dataset.train_size() / args.batch_size))
        try:
            scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=max_steps, last_epoch=last_epoch)
        except:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=max_steps, last_epoch=last_epoch)
    if resume:
        scheduler.load_state_dict(checkpoint["scheduler"])

    model.to(devices[1])
    model.lmgnn.concept_emb.to(devices[0])

    # Construct the loss function
    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    else:
        raise ValueError("Invalid value for args.loss.")

    #############################################################
    #   Training
    #############################################################

    print()
    print('-' * 71)

    total_loss_acm = 0.0
    n_samples_acm = n_corrects_acm = 0
    total_time = 0
    model.train()
    # If all the parameters are frozen in the first few epochs, just skip those epochs.
    if len(params_to_freeze) >= len(list(model.parameters())) - 1:
        args.unfreeze_epoch = 0
    if last_epoch + 1 <= args.unfreeze_epoch:
        utils.freeze_params(params_to_freeze)
    for epoch_id in trange(last_epoch + 1, args.n_epochs, desc="Epoch"):
        if epoch_id == args.unfreeze_epoch:
            utils.unfreeze_params(params_to_freeze)
        if epoch_id == args.refreeze_epoch:
            utils.freeze_params(params_to_freeze)
        model.train()

        for qids, labels, *input_data in tqdm(train_dataloader, desc="Batch"):
            # labels: [bs]
            start_time = time.time()
            optimizer.zero_grad()
            bs = labels.size(0)
            for a in range(0, bs, args.mini_batch_size):
                b = min(a + args.mini_batch_size, bs)
                logits, _ = model(*[x[a:b] for x in input_data])
                # logits: [bs, nc]

                loss, n_corrects = calc_loss_and_acc(logits, labels[a:b], args.loss, loss_func)

                total_loss_acm += loss.item()
                loss = loss / bs
                loss.backward()
                n_corrects_acm += n_corrects
                n_samples_acm += (b - a)

            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scheduler.step()
            # Gradients are accumulated and not back-proped until a batch is processed (not a mini-batch).
            optimizer.step()

            total_time += (time.time() - start_time)

            if (global_step + 1) % args.log_interval == 0:
                ms_per_batch = 1000 * total_time / args.log_interval
                print('| step {:5} |  lr: {:9.7f} | total loss {:7.4f} | ms/batch {:7.2f} |'.format(global_step, scheduler.get_lr()[0], total_loss_acm / n_samples_acm, ms_per_batch))

                if not args.debug:
                    tb_writer.add_scalar('Train/Lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('Train/Loss', total_loss_acm / n_samples_acm, global_step)
                    tb_writer.add_scalar('Train/Acc', n_corrects_acm / n_samples_acm, global_step)
                    tb_writer.add_scalar('Train/ms_per_batch', ms_per_batch, global_step)
                    tb_writer.flush()
                wandb.log({"lr": scheduler.get_lr()[0], "train_loss": total_loss_acm / n_samples_acm, "train_acc": n_corrects_acm / n_samples_acm, "ms_per_batch": ms_per_batch}, step=global_step)

                total_loss_acm = 0.0
                n_samples_acm = n_corrects_acm = 0
                total_time = 0
            global_step += 1 # Number of batches processed up to now

        # Save checkpoints and evaluate after every epoch
        model.eval()
        preds_path = os.path.join(args.save_dir, 'dev_e{}_preds.csv'.format(epoch_id))
        dev_total_loss, dev_acc = calc_eval_accuracy(dev_dataloader, model, args.loss, loss_func, args.debug, not args.debug, preds_path)
        if has_test_split:
            preds_path = os.path.join(args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
            test_total_loss, test_acc = calc_eval_accuracy(test_dataloader, model, args.loss, loss_func, args.debug, not args.debug, preds_path)
        else:
            test_acc = 0

        print('-' * 71)
        print('| epoch {:3} | step {:5} | dev_acc {:7.4f} | test_acc {:7.4f} |'.format(epoch_id, global_step, dev_acc, test_acc))
        print('-' * 71)

        if not args.debug:
            tb_writer.add_scalar('Dev/Acc', dev_acc, global_step)
            tb_writer.add_scalar('Dev/Loss', dev_total_loss, global_step)
            if has_test_split:
                tb_writer.add_scalar('Test/Acc', test_acc, global_step)
                tb_writer.add_scalar('Test/Loss', test_total_loss, global_step)
            tb_writer.flush()

        if dev_acc >= best_dev_acc:
            best_dev_acc = dev_acc
            final_test_acc = test_acc
            best_dev_epoch = epoch_id
        if not args.debug:
            with open(log_path, 'a') as fout:
                fout.write('{:3},{:5},{:7.4f},{:7.4f},{:7.4f},{:7.4f},{:3}\n'.format(epoch_id, global_step, dev_acc, test_acc, best_dev_acc, final_test_acc, best_dev_epoch))

        wandb.log({"dev_acc": dev_acc, "dev_loss": dev_total_loss, "best_dev_acc": best_dev_acc, "best_dev_epoch": best_dev_epoch}, step=global_step)
        if has_test_split:
            wandb.log({"test_acc": test_acc, "test_loss": test_total_loss, "final_test_acc": final_test_acc}, step=global_step)

        # Save the model checkpoint
        if args.save_model:
            model_state_dict = model.state_dict()
            del model_state_dict["lmgnn.concept_emb.emb.weight"]
            checkpoint = {"model": model_state_dict, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch_id, "global_step": global_step, "best_dev_epoch": best_dev_epoch, "best_dev_acc": best_dev_acc, "final_test_acc": final_test_acc, "config": args}
            print('Saving model to {}.{}'.format(model_path, epoch_id))
            torch.save(checkpoint, model_path +".{}".format(epoch_id))
        model.train()
        start_time = time.time()
        if epoch_id > args.unfreeze_epoch and epoch_id - best_dev_epoch >= args.max_epochs_before_stop:
            break

        if args.debug:
            break

    if not args.debug:
        tb_writer.close()


def evaluate(args, has_test_split, devices, kg):
    assert args.load_model_path is not None
    load_model_path = args.load_model_path
    print("loading from checkpoint: {}".format(load_model_path))
    checkpoint = torch.load(load_model_path, map_location='cpu')

    train_statements = args.train_statements
    dev_statements = args.dev_statements
    test_statements = args.test_statements
    train_adj = args.train_adj
    dev_adj = args.dev_adj
    test_adj = args.test_adj
    debug = args.debug
    inhouse = args.inhouse

    args = utils.import_config(checkpoint["config"], args)
    args.train_statements = train_statements
    args.dev_statements = dev_statements
    args.test_statements = test_statements
    args.train_adj = train_adj
    args.dev_adj = dev_adj
    args.test_adj = test_adj
    args.inhouse = inhouse

    dataset = load_data(args, devices, kg)
    dev_dataloader = dataset.dev()
    if has_test_split:
        test_dataloader = dataset.test()
    model = construct_model(args, kg)
    model.lmgnn.mp.resize_token_embeddings(len(dataset.tokenizer))

    model.load_state_dict(checkpoint["model"], strict=False)
    epoch_id = checkpoint['epoch']

    model.to(devices[1])
    model.lmgnn.concept_emb.to(devices[0])
    model.eval()

    if args.loss == 'margin_rank':
        loss_func = nn.MarginRankingLoss(margin=0.1, reduction='mean')
    elif args.loss == 'cross_entropy':
        loss_func = nn.CrossEntropyLoss(reduction='mean')
    else:
        raise ValueError("Invalid value for args.loss.")

    print ('inhouse?', args.inhouse)

    print ('args.train_statements', args.train_statements)
    print ('args.dev_statements', args.dev_statements)
    print ('args.test_statements', args.test_statements)
    print ('args.train_adj', args.train_adj)
    print ('args.dev_adj', args.dev_adj)
    print ('args.test_adj', args.test_adj)

    model.eval()
    # Evaluation on the dev set
    preds_path = os.path.join(args.save_dir, 'dev_e{}_preds.csv'.format(epoch_id))
    dev_total_loss, dev_acc = calc_eval_accuracy(dev_dataloader, model, args.loss, loss_func, debug, not debug, preds_path)
    if has_test_split:
        # Evaluation on the test set
        preds_path = os.path.join(args.save_dir, 'test_e{}_preds.csv'.format(epoch_id))
        test_total_loss, test_acc = calc_eval_accuracy(test_dataloader, model, args.loss, loss_func, debug, not debug, preds_path)
    else:
        test_acc = 0

    print('-' * 71)
    print('dev_acc {:7.4f}, test_acc {:7.4f}'.format(dev_acc, test_acc))
    print('-' * 71)


def get_devices(use_cuda):
    """Get the devices to put the data and the model based on whether to use GPUs and, if so, how many of them are available."""
    if torch.cuda.device_count() >= 2 and use_cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:1")
        print("device0: {}, device1: {}".format(device0, device1))
    elif torch.cuda.device_count() == 1 and use_cuda:
        device0 = torch.device("cuda:0")
        device1 = torch.device("cuda:0")
    else:
        device0 = torch.device("cpu")
        device1 = torch.device("cpu")
    return device0, device1


def main(args):
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(name)s:%(funcName)s():%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.WARNING)

    has_test_split = True
    devices = get_devices(args.cuda)
    kg = "cpnet"
    model = construct_model(kg)
    print(model)
