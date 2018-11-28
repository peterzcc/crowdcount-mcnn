import os
import torch
import numpy as np

from mcnn.crowd_count import CrowdCounter
from mcnn import network
from mcnn.exr_data_loader import ExrImageDataLoader
from mcnn import utils
import argparse


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

def main():
    parser = argparse.ArgumentParser(description='mcnn worldexp.')
    parser.add_argument('--preload', type=int, default=0)
    parser.add_argument('--data', type=str, default="/mnt/m2/mzcc/crowd_data/worldexpo", help='data path')
    parser.add_argument('--model', type=str, default="./saved_models/mcnn_worldexpo_2000.h5", help='model path')
    args = parser.parse_args()
    method = 'mcnn'
    dataset_name = 'minibus'
    output_dir = './saved_models/'

    data_path = args.data

    gt_path = None
    model_path = args.model

    output_dir = './output/'
    model_name = os.path.basename(model_path).split('.')[0]
    file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)


    net = CrowdCounter()

    trained_model = os.path.join(model_path)
    network.load_net(trained_model, net)
    net.cuda()
    net.eval()
    mae = 0.0
    mse = 0.0

    #load test data
    data_loader = ExrImageDataLoader(data_path, None, mask_path=None,
                                     shuffle=False, gt_downsample=True, pre_load=args.preload)

    for blob in data_loader:
        im_data = blob['data']
        gt_data = blob['gt_density']
        density_map = net(im_data, gt_data)
        density_map = density_map.data.cpu().numpy()
        gt_count = np.asscalar(np.sum(gt_data))
        et_count = np.asscalar(np.sum(density_map))
        log_text = 'fname: %s gt_cnt: %4.1f, et_cnt: %4.1f' % (blob['fname'], gt_count, et_count)
        print(log_text)
        mae += abs(gt_count-et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))
        if vis:
            utils.display_results(im_data, gt_data, density_map)
        if save_output:
            utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')

    mae = mae/data_loader.get_num_samples()
    mse = np.sqrt(mse/data_loader.get_num_samples())
    print('\nMAE: %0.2f, MSE: %0.2f' % (mae,mse))

    f = open(file_results, 'w')
    f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
    f.close()


if __name__ == "__main__":
    main()