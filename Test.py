import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from lib.Network import Network
from torch.nn.parallel import DistributedDataParallel as DDP
from util.data_val import test_dataset, create_dataloader
import torch.distributed as dist
import torch.multiprocessing as mp


def main(local_rank, world_size):
    # Initialize the distributed environment
    dist.init_process_group(backend="nccl", init_method='env://', world_size=world_size, rank=local_rank)

    # Set the device for the current process
    torch.cuda.set_device(local_rank)

    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=384, help='testing size')
    parser.add_argument('--pth_path', type=str,
                        default='/boot/FEDER/snapshot/12.31/output_federmambamhat_withresloss_withmhat1*0.1_withCIAF_14_bs7_384_lr-4_192_GPU23_valCOD10K/Net_epoch_best.pth')
    parser.add_argument('--test_dataset_path', type=str, default='/data/data_cos/COD/TestDataset')
    opt = parser.parse_args()

    for _data_name in ['COD10K', 'CAMO', 'CHAMELEON', 'NC4K']:
        data_path = opt.test_dataset_path + '/{}/'.format(_data_name)
        save_path = './res/{}_3/{}/'.format(opt.pth_path.split('/')[-2], _data_name)
        os.makedirs(save_path, exist_ok=True)

        model = Network(channels=192).cuda()
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        model.load_state_dict(torch.load(opt.pth_path))
        model.eval()



        image_root = '{}/Imgs/'.format(data_path)
        gt_root = '{}/GT/'.format(data_path)
        test_loader = test_dataset(image_root, gt_root, opt.testsize)
        mae_sum = 0
        num = 0

        for i in range(test_loader.size):
            image, gt, name, _ = test_loader.load_data()
            print('> {} - {}'.format(_data_name, name))

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            result = model(image)

            res = F.interpolate(result[4], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            cv2.imwrite(save_path + name, res * 255)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        print('MAE: {}.'.format(mae))


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"  # 指定使用2、3号GPU
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1216'

    # Number of GPUs
    world_size = torch.cuda.device_count()

    # Use torch.multiprocessing to spawn processes
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)