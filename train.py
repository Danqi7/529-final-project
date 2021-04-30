import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchvision import models
#from torchvision.models.vgg import VGG
from vgg import VGG

import matplotlib.pyplot as plt

import os
import time
import numpy as np
import argparse
import PIL

from utils import load_data, plot_and_save, save_model_info
from positional_embeddings import gaussian_pos_embedding
from fcn import FCN32s, FCN8s

# Set the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Hyperparameters
image_size = 224
image_c = 3
n_class = 1
GAUSSIAN_SIGMA = 90

def train(model, optim, loss_function, train_loader, sample_test, params, test_params):
    # Params
    batch_size = params["batch_size"]
    num_epochs = params["num_epochs"]
    num_iters_per_print = params['num_iters_per_print']
    num_epoch_per_eval = params['num_epoch_per_eval']
    # save_file = params['save_file']

    # Print some info about train data
    num_data = len(train_loader) * batch_size
    num_iters_per_epoch = num_data / batch_size
    print('Total number of Iterations: ', num_iters_per_epoch * num_epochs)
    print("num_data: ", num_data, "\nnum_iters_per_epoch: ", num_iters_per_epoch)

    losses = []
    precisions = []
    recalls = []
    maes = []
    fmeasures = []

    #best_mae = 1
    model.train()  # Set model to train mode
    for e in range(num_epochs):
        for i, data in enumerate(train_loader, 0):
            x = data[0].to(device)  # b x C x W x H
            y = data[1].to(device)  # b x 1 x W x H

            # Forward & backward pass
            model.zero_grad()

            output = model(x)  # b x 1 x W x H
            loss = loss_function(output, y)

            loss.backward()
            optim.step()

            # Bookkeeping
            if i % num_iters_per_print == 0 or i == num_iters_per_epoch-1:
              print(
                  "[%d/%d][%d/%d]\tLoss_D: %.4f\t"
                  % (
                      e,
                      num_epochs,
                      i,
                      len(train_loader),
                      loss
                  )
              )
              losses.append(loss.item())

            # Eval on sample testing on the end batch of epoch
            if (e % num_epoch_per_eval == 0 and i == num_iters_per_epoch - 1) or (e == num_epochs-1 and i == num_iters_per_epoch - 1):
                print(
                    "[%d/%d][%d/%d]\tEvaluating..."
                    % (
                        e,
                        num_epochs,
                        i,
                        len(train_loader)
                    )
                )
                pr, rc, fm, mae = eval_sample(model, sample_test, test_params)
                precisions.append(pr)
                recalls.append(rc)
                fmeasures.append(fm)
                maes.append(mae)

                # # Save best model
                # if mae > best_mae and save_file != "":
                #     best_mae = mae
                #     torch.save(model, save_file + "/model.pt")

    #wandb.log({'losses': losses, 'class_accs': class_accs, "attr_accs": attr_accs})
    return losses, precisions, recalls, fmeasures, maes


def eval_sample(model, test_data, params):
    '''
    Evaluate SOD model on test data
    If prob > threshold, then 1

    Output:
        precision: tp / tp + fp
        recall: tp / tp + fn
        f_measure: 
        MAE: 
    '''
    tp = 0
    fp = 0
    fn = 0
    batch_mae = []
    batch_size = []

    threshold = 0.5
    belta_sq = 0.3

    model.eval()  # set model to eval mode for bn, dropout behave properly
    with torch.no_grad():
        x = test_data[0].to(device)  # b x C x W x H
        y = test_data[1].to(device)  # b x 1 x W x H

        output = model(x).cpu().numpy()  # b x 1 x W x H
        pred_mask = np.copy(output)
        y = y.cpu().numpy()
        pred_mask[output > threshold] = 1
        pred_mask[output <= threshold] = 0

        tp += np.sum(pred_mask[y == 1] == 1)
        fp += np.sum(pred_mask[y == 0] == 1)
        fn += np.sum(pred_mask[y == 1] == 0)

        mae = np.mean(np.abs(pred_mask-y))
        batch_mae.append(mae)
        batch_size.append(x.shape[0])

        print(
            "tp:%d\t fp:%d\t fn:%d\t mae:%f\t"
            % (
                tp,
                fp,
                fn,
                mae
            )
        )

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = (1+belta_sq) * precision * recall / \
        (belta_sq * precision + recall)
    MAE = np.sum(np.array(batch_mae) * np.array(batch_size)) / \
        np.sum(batch_size)

    return precision, recall, f_measure, MAE


def eval(model, test_loader, params):
    '''
    Evaluate SOD model on test data
    If prob > threshold, then 1

    Output:
        precision: tp / tp + fp
        recall: tp / tp + fn
        f_measure: 
        MAE: 
    '''
    tp = 0
    fp = 0
    fn = 0
    batch_mae = []
    batch_size = []

    threshold = 0.5
    belta_sq = 0.3

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            x = data[0].to(device)  # b x C x W x H
            y = data[1].to(device)  # b x 1 x W x H

            output = model(x).cpu().numpy()  # b x 1 x W x H
            pred_mask = np.copy(output)
            y = y.cpu().numpy()
            pred_mask[output > threshold] = 1
            pred_mask[output <= threshold] = 0

            tp += np.sum(pred_mask[y == 1] == 1)
            fp += np.sum(pred_mask[y == 0] == 1)
            fn += np.sum(pred_mask[y == 1] == 0)

            mae = np.mean(np.abs(pred_mask-y))
            batch_mae.append(mae)
            batch_size.append(x.shape[0])

            print(
                "[%d/%d]\ttp:%d\t fp:%d\t fn:%d\t mae:%f\t"
                % (
                    i,
                    len(test_loader),
                    tp,
                    fp,
                    fn,
                    mae
                )
            )

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = (1+belta_sq) * precision * recall / \
        (belta_sq * precision + recall)
    MAE = np.sum(np.array(batch_mae) * np.array(batch_size)) / \
        np.sum(batch_size)

    return precision, recall, f_measure, MAE


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_files", type=str, default='./models/',
                        help="Where to store the trained model")
    parser.add_argument("--batch_size", default=16,
                        type=int, help="batch size")
    parser.add_argument("--num_epochs", default=1,
                        type=int, help="epochs to run")
    parser.add_argument("--positional_encoding", default=False, action="store_true",
                        help="Whether to add positional encoding at encoder")
    parser.add_argument("--pos_inject_layer", default=0, type=int,
                        help="Which layer at the encoder to inject the positional encoding")
    parser.add_argument("--pos_embed_type", default="Gaussian", type=str, help="Gaussian|Random|")

    args = parser.parse_args()

    print(args)
    batch_size = args.batch_size

    if not os.path.exists(args.store_files):
        os.makedirs(args.store_files)

    # Data
    train_data, test_data = load_data()
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True)
    
    # Model
    fcn_model = FCN8s(
        n_class=n_class, positional_encoding=args.positional_encoding, pos_inject_layer=args.pos_inject_layer, pos_embed_type=args.pos_embed_type)
    #n_class, h, w = 1, 224, 224
    # input = torch.autograd.Variable(torch.randn(batch_size, 3, h, w))
    # #input = next(iter(train_data))[0].reshape((1,3,224,224))
    # print('input.shape: ', input.shape)
    # output = fcn_model(input)
    # print('output shape: ', output.shape)
    # assert output.size() == torch.Size([batch_size, n_class, h, w])
    # print('Check Pass')

    # Train Model
    fcn_model = fcn_model.to(device)

    num_iters_per_print = 10
    num_epochs = args.num_epochs
    num_epoch_per_eval = 1
    save_file = ''
    params = {
        "batch_size": args.batch_size,
        "num_iters_per_print": num_iters_per_print,
        "num_epoch_per_eval": num_epoch_per_eval,
        "num_epochs": args.num_epochs,
        "save_file": args.store_files
    }
    eval_params = {
        "belta_sq": 0.3,
        "threshold": 0.5
    }
    learning_rate = 1e-4
    optim = torch.optim.Adam(fcn_model.parameters(), lr=learning_rate)
    loss_function = nn.BCELoss()

    # Train
    print(fcn_model)  # Show model architecture
    validation_loader = torch.utils.data.DataLoader(
        test_data, batch_size=100, shuffle=True)
    sample_validation = next(iter(validation_loader))
    start_time = time.time()
    losses, precisions, recalls, fmeasures, maes = train(
        fcn_model, optim, loss_function, train_loader, sample_validation, params, eval_params)

    elapsed_time = time.time() - start_time
    print('Training Finished in : ', time.strftime(
        "%H:%M:%S", time.gmtime(elapsed_time)))

    # Evaluate
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True)
    eval_params = {
        "belta_sq": 0.3,
        "threshold": 0.5
    }
    pr, rc, fm, mae = eval(fcn_model, test_loader, params)

    results = (np.mean(losses), pr, rc, fm, mae)
    print("Test Precision: %.4f, \t Recall: %.4f, \t F-measure: %.4f, \t MAE: %.4f " %
          (pr, rc, fm, mae))

    # Create model directory
    dir_name = args.store_files + "pos_encoding%d_injectlayer%d_type%s_encoder%.1f" % (
        args.positional_encoding, args.pos_inject_layer, args.pos_embed_type, start_time)
    os.mkdir(dir_name)
    print('Saving model to dir: ', dir_name)

    model_save_name = 'fcn.pt'
    path = dir_name + "/" + model_save_name
    # Save Model
    torch.save(fcn_model.state_dict(), path)

    # Plot
    plot_and_save(losses, "loss", dir_name,
                  "train", freq=num_iters_per_print)
    plot_and_save(precisions, "precision", dir_name,
                  "validation", freq=num_epoch_per_eval)
    plot_and_save(recalls, "recall", dir_name,
                  "validation", freq=num_epoch_per_eval)
    plot_and_save(fmeasures, "f-measure", dir_name,
                  "validation", freq=num_epoch_per_eval)
    plot_and_save(maes, "MAE", dir_name,
                  "validation", freq=num_epoch_per_eval)

    # Save model info
    save_model_info(fcn_model, results, params, elapsed_time, dir_name)

    # Visualize