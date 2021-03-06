import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim
from torchvision import models
#from torchvision.models.vgg import VGG
from vgg import VGG

import matplotlib.pyplot as plt

import os
import copy
import time
from datetime import date
import numpy as np
import argparse
import PIL

from utils import plot_and_save, save_model_info, save_PRCurve_fmeasures
from data_utils import load_data
from positional_embeddings import gaussian_pos_embedding
from fcn import FCN32s, FCN8s
from evaluations import eval, eval_sample, visualize

# Set the device to use
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Hyperparameters
image_size = 224
image_c = 3
n_class = 1
GAUSSIAN_SIGMA = 90
torch.manual_seed(0)

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

    best_fmeasure = 0
    best_model = model
    model.train()  # Set model to train mode
    for e in range(num_epochs):
        epoch_loss = 0
        for i, data in enumerate(train_loader, 0):
            model.train()
            x = data[0].to(device)  # b x C x W x H
            y = data[1].to(device)  # b x 1 x W x H

            # Forward & backward pass
            model.zero_grad()

            output = model(x)  # b x 1 x W x H
            loss = loss_function(output, y)

            loss.backward()
            optim.step()

            epoch_loss += loss

            # Bookkeeping
            if i % num_iters_per_print == 0 or i == num_iters_per_epoch-1:
            #   print(
            #       "[%d/%d][%d/%d]\tLoss: %.4f\t"
            #       % (
            #           e,
            #           num_epochs,
            #           i,
            #           len(train_loader),
            #           loss
            #       )
            #   )
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

                # Best Model
                if fm > best_fmeasure:
                    best_model = copy.deepcopy(model)
                    best_fmeasure = fm
            
            # Clear Cache
            torch.cuda.empty_cache()
        #print('Epoch %d Average Train Loss: %.4f'%(e, epoch_loss/(i+1)))

    return losses, precisions, recalls, fmeasures, maes, best_fmeasure, best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--store_files", type=str, default='./models/'+date.today().strftime("%d_%m") + '/',
                        help="Where to store the trained model")
    parser.add_argument("--batch_size", default=22,
                        type=int, help="batch size")
    parser.add_argument("--num_epochs", default=1,
                        type=int, help="epochs to run")
    parser.add_argument("--pretrained", default=False, action='store_true')
    parser.add_argument("--pretrained_model", default='vgg11', type=str, help="vgg11|vgg16")
    parser.add_argument("--residual_level", default=32, type=int, help="Up to which level of residual connection, 8 means going back to w/8")
    parser.add_argument("--decoder_kernel", default=3, type=int, help="transposed conv kernel size 3 or 4")
    parser.add_argument("--decoder_bn", default=False, action="store_true", help="Whether to use BatchNorm in decoder")
    parser.add_argument("--positional_encoding", default=False, action="store_true",
                        help="Whether to add positional encoding at encoder")
    parser.add_argument("--pos_inject_layer", default=0, type=int,
                        help="Which layer at the encoder to inject the positional encoding")
    parser.add_argument("--pos_embed_type", default="Gaussian", type=str, help="Gaussian|Random|HV|H|V|H+V|")
    parser.add_argument("--pos_inject_side", default="encoder", type=str, help="Where to inject pos encoding, encoder|decoder")
    parser.add_argument("--save_model", default=False, action="store_true", help="Whether to save model checkpoint")

    args = parser.parse_args()

    print(args)
    batch_size = args.batch_size

    if not os.path.exists(args.store_files):
        os.makedirs(args.store_files)

    # Data
    #train_data, test_data = load_data('DUTS')
    _, train_data = load_data('MSRAB')
    _, test_data = load_data('DUTS')
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=False)
    
    # Model
    if args.residual_level == 8:
        fcn_model = FCN8s(
            n_class=n_class,
            pretrained_model=args.pretrained_model,
            decoder_kernel=args.decoder_kernel,
            decoder_bn = args.decoder_bn,
            pretrained=args.pretrained,
            positional_encoding=args.positional_encoding,
            pos_inject_layer=args.pos_inject_layer,
            pos_inject_side=args.pos_inject_side,
            pos_embed_type=args.pos_embed_type)
    elif args.residual_level == 32:
        fcn_model = FCN32s(
            n_class=n_class,
            pretrained_model=args.pretrained_model,
            decoder_kernel=args.decoder_kernel,
            decoder_bn=args.decoder_bn,
            pretrained=args.pretrained,
            positional_encoding=args.positional_encoding,
            pos_inject_layer=args.pos_inject_layer,
            pos_inject_side=args.pos_inject_side,
            pos_embed_type=args.pos_embed_type)


    # Train Model
    fcn_model = fcn_model.to(device)

    num_iters_per_print = 100
    num_epochs = args.num_epochs
    num_epoch_per_eval = 1
    save_file = 'test_model/'
    learning_rate = 1e-4
    params = {
        "batch_size": args.batch_size,
        "num_iters_per_print": num_iters_per_print,
        "num_epoch_per_eval": num_epoch_per_eval,
        "num_epochs": args.num_epochs,
        "save_file": args.store_files,
        "learning_rate": learning_rate,
    }
    eval_params = {
        "belta_sq": 0.3,
        "threshold": 0.5
    }
    
    optim = torch.optim.Adam(fcn_model.parameters(), lr=learning_rate)
    loss_function = nn.BCELoss()

    # Train
    print(fcn_model)  # Show model architecture
    validation_loader = torch.utils.data.DataLoader(
        test_data, batch_size=200, shuffle=False)
    sample_validation = next(iter(validation_loader))
    start_time = time.time()
    losses, precisions, recalls, fmeasures, maes, best_fmeasure, best_model = train(
        fcn_model, optim, loss_function, train_loader, sample_validation, params, eval_params)

    elapsed_time = time.time() - start_time
    print('Training Finished in : ', time.strftime(
        "%H:%M:%S", time.gmtime(elapsed_time)))

    # Evaluate
    all_results = {}
    plot_results = {} # PR Curve + f-measure vs thresholds
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)
    eval_params = {
        "belta_sq": 0.3,
        "threshold": 0.5,
    }
    # Compare best with final model
    fpr, frc, ffm, fmae, ffmax, ffmax_thresh, _ = eval(fcn_model, test_loader, params)
    pr, rc, fm, mae, fmax, fmax_thresh, plot_result = eval(best_model, test_loader, params)

    results = (np.mean(losses), pr, rc, fm, mae)
    all_results['DUTS'] = (pr, rc, fm, mae, fmax, fmax_thresh)
    plot_results['DUTS'] = plot_result
    print("Final Test Precision: %.4f, \t Recall: %.4f, \t F-measure: %.4f, \t MAE: %.4f, \t Max F-measure: %.4f at thresh %.2f" %
          (fpr, frc, ffm, fmae, ffmax, ffmax_thresh))
    print("Best Test Precision: %.4f, \t Recall: %.4f, \t F-measure: %.4f, \t MAE: %.4f, \t Max F-measure: %.4f at thresh %.2f" %
          (pr, rc, fm, mae, fmax, fmax_thresh))

    # Evaluate zero-shot on HKU-IS Data
    _, HKU_test_data = load_data('HKU')
    HKU_test_loader = torch.utils.data.DataLoader(
        HKU_test_data, batch_size=batch_size, shuffle=False)
    pr, rc, fm, mae, fmax, fmax_thresh, plot_result = eval(
        best_model, HKU_test_loader, params)
    all_results['HKU'] = (pr, rc, fm, mae, fmax, fmax_thresh)
    plot_results['HKU'] = plot_result

    # Evalate zero-shot on ECSSD Data
    _, ECSSD_test_data = load_data('ECSSD')
    ECSSD_test_loader = torch.utils.data.DataLoader(
        ECSSD_test_data, batch_size=batch_size, shuffle=False)
    pr, rc, fm, mae, fmax, fmax_thresh, plot_result = eval(
        best_model, ECSSD_test_loader, params)
    all_results['ECSSD'] = (pr, rc, fm, mae, fmax, fmax_thresh)
    plot_results['ECSSD'] = plot_result

    # Evaluate on PASCAL-S Data
    _, PSCALS_test_data = load_data('PASCALS')
    PSCALS_test_loader = torch.utils.data.DataLoader(
        PSCALS_test_data, batch_size=batch_size, shuffle=False)
    pr, rc, fm, mae, fmax, fmax_thresh, plot_result = eval(
        best_model, PSCALS_test_loader, params)
    all_results['PASCALS'] = (pr, rc, fm, mae, fmax, fmax_thresh)
    plot_results['PASCALS'] = plot_result

    # Evaluate on DUT-OMRON Data
    _, OMRON_test_data = load_data('OMRON')
    OMRON_test_loader = torch.utils.data.DataLoader(
        OMRON_test_data, batch_size=batch_size, shuffle=False)
    pr, rc, fm, mae, fmax, fmax_thresh, plot_result = eval(
        best_model, OMRON_test_loader, params)
    all_results['OMRON'] = (pr, rc, fm, mae, fmax, fmax_thresh)
    plot_results['OMRON'] = plot_result

    print(all_results)

    # Average F-measure, MAE cross all datasets
    num_datasets = 0
    total_pr = 0
    total_rc = 0
    total_fm = 0
    total_mae = 0
    for i, (k, v) in enumerate(all_results.items()):
        (vpr, vrc, vfm, vmae, fmax, fmax_thresh) = v
        num_datasets += 1
        total_pr += vpr
        total_rc += vrc
        total_fm += vfm
        total_mae += vmae
    average_result = (total_pr/num_datasets, total_rc/num_datasets,
                      total_fm/num_datasets, total_mae/num_datasets)
    print("Average F-measure: %.4f, \t MAE: %.4f " %
          (total_fm/num_datasets, total_mae/num_datasets))

    # Create model directory
    dir_name = args.store_files + \
        "residule%d__pretrained%d_posencoding%d__type%s_%s%.1f_fm%.2f_MAE%.2f" % (args.residual_level,
                                                                                  args.pretrained,
                                                                                  args.positional_encoding,
                                                                                  args.pos_embed_type,
                                                                                  args.pos_inject_side,
                                                                                  start_time,
                                                                                  total_fm/num_datasets, total_mae/num_datasets)
    os.mkdir(dir_name)
    print('Saving model to dir: ', dir_name)



    # Save Model
    model_save_name = 'fcn.pt'
    path = dir_name + "/" + model_save_name
    if args.save_model == True:
        torch.save(fcn_model.state_dict(), path)
    
    # Save PR Curves + F-measures vs Tresholds
    save_PRCurve_fmeasures(plot_results, dir_name)

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
    save_model_info(fcn_model, results, all_results, average_result, params, elapsed_time, dir_name)

    

    for i, (k, v) in enumerate(all_results.items()):
        (vpr, vrc, vfm, vmae, vfmax, vfmax_thresh) = v
        print("%s : F-measure: %.4f, MAE: %.4f \nMax F-measure: %.4f at tresh %.2f" %
              (k, vfm, vmae, vfmax,vfmax_thresh))
    
    # Visualize
    visualize(best_model, dir_name)

    # Visualize weight 64 x 1 x 4 x 3
    # [out_channel x in_channel x 3 x 3]
    #print('save_file: ', save_file)
    #print(fcn_model.state_dict().keys())

    weights = fcn_model.state_dict()['pretrained_net.features.0.weight'] # 64 x 4 x 3 x 3
    encoding_weights = weights[:, -1, :, :] # 64 x 1 x 3 x 3
    # upsample
    scale_factor = 24
    m = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
    
    encoding_weights = m(encoding_weights.view(
        encoding_weights.shape[0], 1, 3, 3))
    encoding_weights = encoding_weights.view(
        encoding_weights.shape[0], 1, 3*scale_factor, 3*scale_factor)
    weight_images = vutils.make_grid(encoding_weights, padding=2, normalize=True)
    vutils.save_image(weight_images, dir_name +
                      "/input_weights.png")
