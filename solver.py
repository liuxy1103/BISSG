# -*- coding: utf-8 -*-
# @Time : 2021/4/8
# @Author : Xiaoyu Liu
# @Email : liuxyu@mail.ustc.edu.cn
# @Software: PyCharm

import time
import torch
import numpy as np
from utils.utils import ensure_dir
from tensorboardX import SummaryWriter
from torch.nn.functional import interpolate
from model.metric import dc_score, MeanIoU
from model.loss import MSE_loss, DiceLoss, BCE_loss  # 引入不同loss
from model.loss_spixel import *
from model.loss_embedding import *
from model.loss_egnn import *
from train_util import *
from skimage import io
from sklearn.decomposition import PCA
import os
import torch.nn as nn
from skimage.metrics import adapted_rand_error as adapted_rand_ref
from skimage.metrics import variation_of_information as voi_ref
from utils.water import *
import copy
from graph_partition import *
from lib.evaluate.CVPPP_evaluate import BestDice, AbsDiffFGLabels, SymmetricBestDice
from connectivity import enforce_connectivity
from tqdm import tqdm
from postprocessing import merge_small_object, merge_func
from data.data_segmentation import relabel

import warnings
warnings.filterwarnings("ignore")

class Solver():
    def __init__(self, gpus, model, criterion, metric, batch_size, epochs, lr,
                 trainset, valset, train_loader, val_loader, logs_dir,
                 patience, checkpoint_dir=None, weight=None,
                 resume=False, resume_path=None, log=None, args=None):
        self.args = args
        self.log = log
        self.device, device_ids = self._prepare_device(gpus)
        log.info("Available Devices: %s" % device_ids)
        self.model = model.to(self.device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=[args.local_rank])

        self.cri_name = criterion
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(model.to(self.device), device_ids=device_ids)
        self.weight = weight
        if criterion == "MSE_loss" or criterion == "DiceLoss":
            self.criterion = eval(criterion + "()")
        else:
            self.criterion = eval(criterion)
        self.criterion_aff = nn.BCELoss()
        self.metric = eval(metric)

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        egnn_params = list(map(id, model.module.egnn.parameters()))
        base_params = filter(lambda p: id(p) not in egnn_params,
                             model.parameters())
        # optimizer = torch.optim.SGD([
        #     {'params': base_params},
        #     {'params': net.conv5.parameters(), 'lr': lr * 100},
        #     {'params': net.conv4.parameters(), 'lr': lr * 100},
        #     , lr=lr, momentum=0.9)

        self.optimizer = torch.optim.Adam([
            {'params': base_params, 'lr': lr * 0.1},
            {'params': model.module.egnn.parameters()}]
            , lr=self.lr, betas=(0.9, 0.999))
        #self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr, betas=(0.9, 0.999))
        self.best_val_loss = np.inf
        self.aver_SBD = 0
        self.best_voi_mean = np.inf

        self.len_train = len(trainset)
        self.len_val = len(valset)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.patience = patience
        self.checkpoint_dir = checkpoint_dir
        ensure_dir(checkpoint_dir)
        self.resume_path = resume_path
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                       patience=patience, factor=0.5)
        # self.StepLR = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)

        if resume is True:
            self.start_epoch = self._resume_checkpoint(resume_path)
        else:
            self.start_epoch = 0
            log.info("Start a new training!")

        ensure_dir(logs_dir)
        self.writer = SummaryWriter(logs_dir)

    def train_epoch(self, epoch):
        save_path = './train_fragments/' + str(epoch)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.model.train()
        loss_t = 0
        loss_egnn_t=0
        loss_emb_t = 0
        loss_bound_t = 0
        loss_binary_t = 0
        for batch_idx, (data, target,fg,boundary,weightmap, affs_mask) in enumerate(self.train_loader): #imgs, label, fg, lb_affs, weightmap, affs_mask


            data, target,boundary,weightmap, affs_mask = data.to(self.device), target.to(self.device).long(),\
                                    boundary.to(self.device).long(),weightmap.to(self.device).float(),\
                                                         affs_mask.to(self.device).float()

            self.optimizer.zero_grad()
            # print(data.dtype,data.shape,target.shape)
            out_boundary,x_emb,binary_seg,edge_feat_list,node_feat_list,segments,adj = self.model(data.float(),mode='train',fg = fg)

            if adj is None:
                continue
            # boundary loss
            out_boundary = torch.sigmoid(out_boundary)  ###配合下面的这种loss来使用
            boundary_loss = self.criterion_aff(out_boundary, boundary.float()).to(self.device)

            # embedding loss
            emb_loss = discriminative_loss(x_emb, target[:, 0, :], alpha=self.args.alpha, beta=self.args.beta,
                                           gama=self.args.gama)
            # semantic loss
            binary_loss = self.criterion(binary_seg, torch.gt(target[:, 0, :], 0), weight_rate=self.weight).to(
                self.device)  # convert instance to semantic

            with torch.no_grad():
                io.imsave(save_path + '/' + str(batch_idx) + '_target' + '.png', (target.cpu().numpy()[0, 0]))
                io.imsave(save_path + '/' + str(batch_idx) + '_raw' + '.png', (data.cpu().numpy()[0, 0]))
                io.imsave(save_path + '/' + str(batch_idx) + '_segments' + '.png', (segments[0, 0]))
                pred_mask = F.softmax(binary_seg, dim=1)
                pred_mask = torch.argmax(pred_mask, dim=1).squeeze(0)
                pred_mask_b = pred_mask.data.cpu().numpy()
                pred_mask_b = pred_mask_b.astype(np.uint8)
                # pred_mask_b = remove_samll_object(pred_mask_b)
                io.imsave(save_path + '/' + str(batch_idx) + '_fg' + '.png', pred_mask_b)

            # pca = PCA(n_components=3)
            # x_emb = np.transpose(out_emb[0].detach().cpu().numpy(), [1, 2, 0])
            # h,w,c = x_emb.shape
            # x_emb = x_emb.reshape(-1, 32)
            # new_emb = pca.fit_transform(x_emb)
            # new_emb = new_emb.reshape(h, w, 3)
            # save_path = os.path.join('./save_patch', str(epoch),str(batch_idx))
            # print(save_path)
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # io.imsave(os.path.join(save_path,'segments.tif'),segments[0,0].astype(np.uint16))
            # io.imsave(os.path.join(save_path,'emb.tif'),new_emb)
            # io.imsave(os.path.join(save_path, 'target.tif'), target[0,0].cpu().detach().numpy().astype(np.uint16))

            # egnn loss

            egnn_loss = Graph_MALIS(edge_feat_list,node_feat_list, target[:, 0, :], segments, adj)

            print('all parts of losses:',self.args.loss_spixel * boundary_loss, self.args.loss_embedding * emb_loss,
                  self.args.loss_binary * binary_loss, self.args.loss_gnn * egnn_loss)
            loss = self.args.loss_spixel * boundary_loss + self.args.loss_embedding * emb_loss \
                    + self.args.loss_gnn * egnn_loss

            loss.backward()
            loss_t = loss_t + loss.item()  # tensor——>scalar
            loss_egnn_t = loss_egnn_t +egnn_loss.item()
            loss_emb_t = loss_emb_t+emb_loss.item()
            loss_bound_t = loss_bound_t+boundary_loss.item()
            loss_binary_t = loss_binary_t + binary_loss.item()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            self.log.info('Train Epoch: {} {} Loss: {:.6f}'.format(
                epoch,
                self._progress(batch_idx, self.len_train),
                loss.item()))
        # self.StepLR.step()
        self.writer.add_scalar("train/Train_loss", loss_t / self.len_train, epoch)
        self.writer.add_scalar("train/Train_EGNN_loss", loss_egnn_t / self.len_train, epoch)
        self.writer.add_scalar("train/boundary_loss", loss_bound_t / self.len_train, epoch)
        self.writer.add_scalar("train/Train_Emb_loss", loss_emb_t / self.len_train, epoch)
        self.writer.add_scalar("train/Train_Binary_loss", loss_binary_t / self.len_train, epoch)
        self.writer.add_scalar("train/Lr", self.optimizer.param_groups[0]["lr"], epoch)

    def validate_epoch(self, epoch):
        save_path = './val_fragments/'+str(epoch)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        loss_v = 0
        loss_b = 0
        loss_egnn_v = 0
        loss_emb_v = 0
        loss_binary_v = 0
        diffFG_v1 = 0
        bestDice_v1 = 0
        SBD_v1 = 0
        voi1 = []

        diffFG_v2 = 0
        bestDice_v2 = 0
        SBD_v2 = 0
        voi2 = []

        pbar = tqdm(total=len(self.val_loader))
        for batch_idx, (data, target,fg,boundary,weightmap, affs_mask) in enumerate(self.val_loader): #imgs, label, fg, lb_affs, weightmap, affs_mask

            with torch.no_grad():
                data, target, boundary, weightmap, affs_mask = data.to(self.device), target.to(self.device).long(), \
                                                               boundary.to(self.device).long(), \
                                                               weightmap.to(self.device).float(), \
                                                               affs_mask.to(self.device).float()

                out_boundary,x_emb,binary_seg,edge_feat_list,_,segments,adj = self.model(data.float(),fg = fg)
                if adj is None:
                    continue

                # spixel loss
                out_boundary = torch.sigmoid(out_boundary)  ###配合下面的这种loss来使用
                boundary_loss = self.criterion_aff(out_boundary, boundary.float()).to(self.device)
                io.imsave(save_path + '/' + str(batch_idx) + '_boundary_gt' + '.png', 1-0.5*(boundary.cpu().numpy()[0,0] + boundary.cpu().numpy()[0,1]))
                color_gt = draw_fragments_2d(target.cpu().numpy()[0, 0])
                io.imsave(save_path + '/' + str(batch_idx) + '_target' + '.png', color_gt)
                # egnn loss
                #egnn_loss = egnn_BCE(edge_feat_list, target[:, 0, :], segments, adj)
                # embedding loss
                emb_loss = discriminative_loss(x_emb, target[:, 0, :], alpha=self.args.alpha, beta=self.args.beta,
                                               gama=self.args.gama)

                # semantic loss
                binary_loss = self.criterion(binary_seg, torch.gt(target[:, 0, :], 0), weight_rate=self.weight).to(
                    self.device)  # convert instance to semantic
                # print(self.args.loss_spixel * boundary_loss, self.args.loss_embedding * emb_loss,
                #       self.args.loss_binary * binary_loss, self.args.loss_gnn * egnn_loss)
                # loss = self.args.loss_spixel * boundary_loss + self.args.loss_embedding * emb_loss \
                #        + self.args.loss_binary * binary_loss + self.args.loss_gnn * egnn_loss
                print(self.args.loss_spixel * boundary_loss, self.args.loss_embedding * emb_loss,
                      self.args.loss_binary * binary_loss)
                loss = self.args.loss_spixel * boundary_loss + self.args.loss_embedding * emb_loss + self.args.loss_binary * binary_loss

                loss_v = loss_v + loss.item()
                loss_b = loss_b + boundary_loss.item()
                #loss_egnn_v = loss_egnn_v+ egnn_loss.item()
                loss_emb_v = loss_emb_v+ emb_loss.item()
                loss_binary_v = loss_binary_v + binary_loss.item()

            out_boundary = out_boundary[0]
            boundary = 1.0 - 0.5 * (out_boundary[0] + out_boundary[1])
            boundary = boundary.cpu().numpy()
            io.imsave(save_path + '/' + str(batch_idx) + '_boundary' + '.png', boundary)
            segments = segments[0,0]
            color_frag = draw_fragments_2d(segments)
            cv2.imwrite(save_path + '/' + str(batch_idx)+'_0'+ '.png', color_frag)

            gt_ins = target[0, 0].cpu().numpy()
            gt_ins = gt_ins.max()-gt_ins
            gt_ins = gt_ins.astype(np.uint16)

            SBD1 = SymmetricBestDice(segments.astype(np.uint16), gt_ins)
            diffFG1 = AbsDiffFGLabels(segments.astype(np.uint16), gt_ins)
            bestDice1 = 0
            diffFG_v1= diffFG_v1 + abs(diffFG1)
            bestDice_v1 = bestDice_v1 + bestDice1
            SBD_v1 = SBD_v1 + SBD1
            voi_split1, voi_merge1 = voi_ref(gt_ins, segments, ignore_labels=(0))
            voi_sum1 = voi_split1 + voi_merge1
            voi1.append(voi_sum1)

            fragments2 = self.agglo(segments, edge_feat_list, adj)
            fragments2 = fragments2 * fg.squeeze().cpu().numpy()

            color_frag2 = draw_fragments_2d(fragments2)
            cv2.imwrite(save_path + '/' + str(batch_idx) +'_1'+ '.png', color_frag2)
            # gt = draw_fragments_2d(target[0,0].cpu().numpy())
            # cv2.imwrite(save_path + '/' + str(batch_idx) + '_target' + '.png', gt)

            x_emb = x_emb.squeeze(0)
            x_emb = np.array(x_emb.cpu())
            shape = x_emb.shape
            # f = h5py.File(spixel_path + '/embedding.hdf', 'w')
            # ds = f.create_dataset('main', data=x_emb, dtype=x_emb.dtype, compression="lzf")
            # ds[:] = x_emb
            pca = PCA(n_components=3)
            x_emb = np.transpose(x_emb, [1, 2, 0])
            x_emb = x_emb.reshape(-1, 16)
            new_emb = pca.fit_transform(x_emb)
            new_emb = new_emb.reshape(shape[-2], shape[-1], 3)
            io.imsave(save_path + '/' + str(batch_idx) + '_embedding.tif', new_emb*fg.squeeze().cpu().numpy()[:,:,np.newaxis].astype(new_emb.dtype))

            #evaluate
            fragments2 = merge_func(fragments2)
            fragments2 = relabel(fragments2)
            
            SBD2 = SymmetricBestDice(fragments2.astype(np.uint16), gt_ins)
            diffFG2 = AbsDiffFGLabels(fragments2.astype(np.uint16), gt_ins)
            bestDice2 = 0
            diffFG_v2 = diffFG_v2 + abs(diffFG2)
            bestDice_v2 = bestDice_v2 + bestDice2
            SBD_v2 = SBD_v2 + SBD2
            voi_split2, voi_merge2 = voi_ref(gt_ins, fragments2, ignore_labels=(0))
            voi_sum2 = voi_split2 + voi_merge2
            voi2.append(voi_sum2)

            pbar.update(1)
        pbar.close()

        aver_loss = loss_v / (self.len_val)
        loss_b = loss_b / (self.len_val)
        #loss_egnn_v = loss_egnn_v/(self.len_val)
        loss_binary=0
        loss_emb_v = loss_emb_v/(self.len_val)
        loss_binary_v = loss_binary/(self.len_val)

        aver_diffFG1 = diffFG_v1 / (self.len_val)
        aver_bestDice1 = bestDice_v1 / (self.len_val)
        aver_SBD1 = SBD_v1/(self.len_val)
        voi_mean1 = np.mean(voi1)

        aver_diffFG2 = diffFG_v2 / (self.len_val)
        aver_bestDice2 = bestDice_v2 / (self.len_val)
        aver_SBD2 = SBD_v2/(self.len_val)
        voi_mean2 = np.mean(voi2)

        self.log.info('Validation Loss: {:.6f}'.format(
            aver_loss))
        self.log.info('Validation aver_diffFG1: {:.6f}'.format(
            aver_diffFG1))
        self.log.info('Validation aver_bestDice1: {:.6f}'.format(
            aver_bestDice1))
        self.log.info('Validation aver_SBD1: {:.6f}'.format(
            aver_SBD1))
        self.log.info('Validation voi_mean1: {:.6f}'.format(
            voi_mean1))

        self.log.info('Validation aver_diffFG2: {:.6f}'.format(
            aver_diffFG2))
        self.log.info('Validation aver_bestDice2: {:.6f}'.format(
            aver_bestDice2))
        self.log.info('Validation aver_SBD2: {:.6f}'.format(
            aver_SBD2))
        self.log.info('Validation voi_mean2: {:.6f}'.format(
            voi_mean2))

        self._save_checkpoint(epoch)

        self._best_model(aver_loss,aver_SBD2,voi_mean2)

        self.writer.add_scalar("val/Validation_loss", aver_loss, epoch)
        self.writer.add_scalar("val/boundary_loss", loss_b, epoch)
        #self.writer.add_scalar("val/EGNN_loss", loss_egnn_v, epoch)
        self.writer.add_scalar("val/Emb_loss", loss_emb_v, epoch)
        self.writer.add_scalar("val/Binary_loss", loss_binary_v, epoch)
        self.writer.add_scalar("val/aver_diffFG1", aver_diffFG1, epoch)
        self.writer.add_scalar("val/aver_bestDice1", aver_bestDice1, epoch)
        self.writer.add_scalar("val/aver_SBD1", aver_SBD1, epoch)
        self.writer.add_scalar("val/voi_mean1", voi_mean1, epoch)

        self.writer.add_scalar("val/aver_diffFG2", aver_diffFG2, epoch)
        self.writer.add_scalar("val/aver_bestDice2", aver_bestDice2, epoch)
        self.writer.add_scalar("val/aver_SBD2", aver_SBD2, epoch)
        self.writer.add_scalar("val/voi_mean2", voi_mean2, epoch)


        if self.lr_scheduler is not None:
            self.lr_scheduler.step(voi_mean2)
        self.log.info("Current learning rate is {}".format(self.optimizer.param_groups[0]["lr"]))




    def agglo(self, segments, edge_feat_list, adj):

        #segments = np.array(segments[0, 0, :])
        # print(torch.unique(segments))
        layers = edge_feat_list.shape[0]
        # edge_list = edge_feat_list[layers-1,:]
        edge_list = torch.mean(edge_feat_list, 0)
        edge_list = F.softmax(edge_list, dim=0)
        edge_list = torch.argmax(edge_list, 0)
        n_nodes = edge_list.shape[0]
        edge_list = edge_list.float() * adj.float()

        to_merge1 = {}
        for i in range(n_nodes):
            for j in range(i + 1):
                if edge_list[i, j] == 1 and i!=0 and j!=0:
                    to_merge1[(i, j)] = 1
        print('num of merge-pairs:', len(to_merge1.keys()))
        if len(to_merge1.keys())==0:
            return segments

        merge_map = {}
        inverse1, pack1 = np.unique(segments, return_inverse=True)
        pack1 = pack1.reshape(segments.shape)
        labels = copy.deepcopy(inverse1)
        labels = labels.reshape((-1, 1))
        to_merge1 = np.array(list(to_merge1.keys()))
        to_merge1 = np.vstack((to_merge1, np.hstack((labels, labels))))
        for v1, v2 in to_merge1:
            merge_map.setdefault(v1, v1)
            merge_map.setdefault(v2, v2)
            while v1 != merge_map[v1]:
                v1 = merge_map[v1]
            while v2 != merge_map[v2]:
                v2 = merge_map[v2]
            if v1 > v2:
                v1, v2 = v2, v1
            merge_map[v2] = v1
        merge_map[0] = 0
        next_label = 1
        for v in sorted(merge_map.keys()):
            if v == 0:
                continue
            if merge_map[v] == v:
                merge_map[v] = next_label
                next_label += 1
            else:
                merge_map[v] = merge_map[merge_map[v]]

        for idx, val in enumerate(inverse1):
            if val in merge_map:
                val = merge_map[val]  # merge to small id
                inverse1[idx] = val
        segments = inverse1[pack1]
        return segments

    # torch.cuda.empty_cache()
    def train(self):
        for epoch in range(self.start_epoch, self.epochs):
            # if epoch==30:
            #     for k, v in self.model.named_parameters():
            #         # print(k)
            #         if '_seg' in k or '_emb' in k:
            #             v.requires_grad = True
            #     print('following parameters are training:')
            #     for name, param in self.model.named_parameters():
            #         if param.requires_grad:
            #             print(name)
            start_time = time.time()
            self.train_epoch(epoch)

            self.model.eval()
            with torch.no_grad():
                if epoch>20:
                    self.validate_epoch(epoch)
                    # self.test_epoch(epoch)
            self._save_checkpoint(epoch)
            end_time = time.time()
            self.log.info('Epoch: {} Spend Time: {:.3f}s'.format(epoch, end_time - start_time))

    def _progress(self, batch_idx, len_set):
        base = '[{}/{} ({:.0f}%)]'
        current = batch_idx
        total = int(len_set / self.batch_size)
        return base.format(current, total, 100.0 * current / total)

    def _best_model(self, aver_loss,aver_SBD, voi_mean):
        if aver_loss < self.best_val_loss:
            self.best_val_loss = aver_loss
        if aver_SBD > self.aver_SBD:
            self.aver_SBD = aver_SBD
        if voi_mean < self.best_voi_mean:
            self.best_voi_mean = voi_mean
        self.log.info(
            "Best val loss: {:.6f} , Best mean aver_SBD: {:.4f}, Best mean voi_mean: {:.4f} ".format(self.best_val_loss,
                                                                                                     self.aver_SBD,
                                                                                                     self.best_voi_mean))

    def _prepare_device(self, n_gpu_use):
        """
        setup GPU device if available, move model into configured device
        """
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            self.log.warning("Warning: There\'s no GPU available on this machine,"
                             "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            self.log.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                             "on this machine.".format(n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device(self.args.local_rank if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _metric_score(self, output, target):
        output = output.squeeze(0).squeeze(0).cpu().numpy()
        target = target.squeeze(0).squeeze(0).cpu().numpy()
        score = self.metric(output, target)
        return score

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.log.warning("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1

        self.model.load_state_dict(checkpoint['state_dict'])

        self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.log.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

        return self.start_epoch

    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        """
        arch = type(self.model).__name__  # model name
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        filename = str(self.checkpoint_dir) + '/checkpoint-epoch{}.pth'.format(str(epoch))
        torch.save(state, filename)
        self.log.info("Saving checkpoint: {} ...".format(filename))
