import os
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime

from torchvision.utils import make_grid
from lib.Network import Network
from util.data_val import get_loader, test_dataset
from util.utils import clip_gradient, adjust_lr, get_coef,cal_ual
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from torch import optim
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

def reduce_tensor(tensor: torch.Tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / (weit.sum(dim=(2, 3)) + 1e-6)


    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1+ 1e-6)
    return (wbce + wiou).mean()

def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()

def train(train_loader, model, optimizer, epoch, save_path, writer,opt,scaler):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    total_step = len(train_loader)
    step = 0

    try:
        for i, (images, gts, edges) in enumerate(train_loader, start=1):


            images = images.cuda()
            gts = gts.cuda()
            edges = edges.cuda()


            preds = model(images)
            mse_loss = nn.MSELoss()
            ual_coef = get_coef(iter_percentage=i/total_step, method='cos')
            ual_loss = cal_ual(seg_logits=preds[4], seg_gts=gts)
            ual_loss *= ual_coef
            loss_init = structure_loss(preds[0], gts) * 0.0625 + structure_loss(preds[1], gts) * 0.125 + structure_loss(
                preds[2], gts) * 0.25 + \
                        structure_loss(preds[3], gts) * 0.5
            loss_final = structure_loss(preds[4], gts)

            loss_edge = dice_loss(preds[5], edges) * 0.125 + dice_loss(preds[6], edges) * 0.25 + \
                        dice_loss(preds[7], edges) * 0.5
            loss_recontrusted = mse_loss(preds[1] * images + preds[8], images) * 0.125 + mse_loss(preds[2] * images + preds[9],
                                                                          images) * 0.25 + mse_loss(
                preds[3] * images + preds[10], images) * 0.5 + mse_loss(preds[4] * images + preds[11], images)
            loss = loss_init + loss_final + loss_edge + 2 * ual_loss

            optimizer.zero_grad()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            clip_gradient(optimizer, opt.clip)
            scaler.update()

            step += 1
            epoch_step += 1
            loss_all += loss.data



            if i % 100 == 0 or i == total_step or i == 1:
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f} Loss3: {:0.4f} Loss4: {:0.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data, loss_recontrusted.data, loss_edge.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f},Loss3: {:.4f} Loss4: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_init.data,loss_final.data,loss_recontrusted.data,loss_edge.data))

                writer.add_scalars('Loss_Statistics',
                                   {'Loss_init': loss_init.data, 'Loss_final': loss_final.data,
                                     'Loss_total': loss.data,'Loss_recontrusted': loss_recontrusted.data},
                                   global_step=step)


        loss_all /= epoch_step
        
        if dist.get_rank() == 0:
            logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
            writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
            # if epoch % 80 == 0:
            #     torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        if dist.get_rank() == 0:
            print('Keyboard Interrupt: save model and exit.')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
            print('Save checkpoints successfully!')
        raise




def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_mae, best_epoch
    model.eval()

    with torch.no_grad():
        mae_sum = 0
        f_measure_sum = 0
        e_measure_sum = 0
        structure_measure_sum = 0
        mae_sum_edge = 0
        for i in range(test_loader.size):
            image, gt,  name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
                
            # Use local_rank for device placement
            local_rank = dist.get_rank()
            image = image.cuda(local_rank)
                
                
            result = model(image)
                    
            res = F.upsample(result[4], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])


        mae = mae_sum / test_loader.size

        print('Epoch: {}, MAE: {}, bestMAE: {}, bestEpoch: {}.'.format(epoch, mae, best_mae, best_epoch))

        if epoch == 1:
            best_mae = mae
            best_epoch = 1
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                print('Save state_dict successfully! Best epoch:{}.'.format(epoch))
        logging.info(
            '[Val Info]:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


def init_ddp(local_rank):
    torch.cuda.set_device(local_rank)
    os.environ['RANK'] = str(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')



def get_ddp_generator(seed=3047):
    local_rank = dist.get_rank()
    g = torch.Generator()
    g.manual_seed(seed + local_rank)
    return g


def main(local_rank, opt):
    global best_mae, best_epoch
    best_mae = 1
    best_epoch = 1
    init_ddp(local_rank)
    model = Network(channels=192).cuda()


    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    num_gpus = torch.cuda.device_count()
    scaler = GradScaler()
    if num_gpus > 1:
        print('Using {} GPUs'.format(num_gpus))
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],find_unused_parameters=True)



    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load),strict=False)
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if dist.get_rank() == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              edge_root=opt.train_root + 'Edge/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              
                              num_workers=0)
    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)


    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}'.format(opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                                                         opt.decay_rate, opt.load, save_path, opt.decay_epoch))


    writer = SummaryWriter(save_path + 'summary')

    # learning rate schedule
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-5)
    print("Start train...")

    for epoch in range(1, opt.epoch):

        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)

        cosine_schedule.step()
        writer.add_scalar('learning_rate', cosine_schedule.get_lr()[0], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))
        train_loader.sampler.set_epoch(epoch)
        train(train_loader, model, optimizer, epoch, save_path, writer, opt,scaler)
        if epoch % 5 == 0 or epoch > 10: # 训练时的epoch
            val(val_loader, model, epoch, save_path, writer)

    dist.destroy_process_group()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=3000, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=3, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=384,help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=80, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0,1', help='train use gpu')
    parser.add_argument('--train_root', type=str, default='/home/chunming/data_cos/COD/TrainDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='/home/chunming/data_cos/CDS2K/CDS2K/Positive/',
                        help='the test rgb images root')
    parser.add_argument('--model', type=str, default='train')
    parser.add_argument('--save_path', type=str,default='/home/chunming/FEDER/snapshot/1.17/outputcds2k2_federpvt50*0.1_withresloss_withmhat1*0.1_withCIAF_4_bs3_512_lr-4_192_GPU23_val/', help='the path to save model and log')
    opt = parser.parse_args()

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1288'
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    world_size = torch.cuda.device_count()
    os.environ['WORLD_SIZE'] = str(world_size)
    cudnn.benchmark = True
    best_mae = 1
    best_epoch = 0
    if opt.model == 'train':
        mp.spawn(fn=main, args=(opt,), nprocs=world_size)




