import os
import numpy as np
from tensorboardX import SummaryWriter
import torch
import time
import datetime
import csv
import shutil
import random
import torch.utils.data as data
import math

from croppingModel import RegionFeatureExtractor,CroppingGraph
from croppingModel import cropping_rank_loss, cropping_regression_loss, score_feature_correlation
from cropping_dataset import GAICDataset
from config import cfg
from test import evaluate_on_FLMS, evaluate_on_GAICD

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device('cuda:{}'.format(cfg.gpu_id))
torch.cuda.set_device(cfg.gpu_id)
MOS_MEAN = 2.95
SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def create_dataloader():
    dataset = GAICDataset(split='train')
    if cfg.keep_aspect_ratio:
        assert cfg.batch_size == 1, 'batch size must be 1 when keeping image aspect ratio'
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size,
                                             shuffle=True, num_workers=cfg.num_workers,
                                             drop_last=False, worker_init_fn=random.seed(SEED))
    print('training set has {} samples, {} batches'.format(len(dataset), len(dataloader)))
    return dataloader

class Trainer:
    def __init__(self, feature_extractor, cropping_gnn):
        self.extractor = feature_extractor
        self.gnn = cropping_gnn
        self.epoch = 0
        self.iters = 0
        self.max_epoch = cfg.max_epoch
        self.writer = SummaryWriter(log_dir=cfg.log_dir)
        self.optimizer, self.lr_scheduler = self.get_optimizer()
        self.train_loader = create_dataloader()
        self.eval_results = []
        self.best_results = {'srcc': 0, 'acc5': 0., 'acc10': 0.,
                             'FLMS_iou':0., 'FLMS_disp':1.}

    def get_optimizer(self):
        params = [
            {'params': self.extractor.parameters(), 'lr': cfg.lr},
            {'params': self.gnn.parameters(),       'lr': cfg.lr}
        ]
        optimizer = torch.optim.Adam(
            params, weight_decay=cfg.weight_decay
        )
        # warm_up_with_cosine_lr
        warm_up_epochs = 5
        warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * (
                    math.cos((epoch - warm_up_epochs) / (self.max_epoch - warm_up_epochs) * math.pi) + 1)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
        return optimizer, lr_scheduler

    def run(self):
        print(("========  Begin Training  ========="))
        self.lr_scheduler.step()
        for epoch in range(self.max_epoch):
            self.epoch = epoch
            self.train()
            if (epoch+1) % cfg.eval_freq == 0 or epoch == (self.max_epoch-1):
                self.eval()
                self.record_eval_results()
            self.lr_scheduler.step()

    def train(self):
        self.extractor.train()
        self.gnn.train()
        start = time.time()
        batch_idx  = 0
        running_reg_loss  = 0.
        running_rank_loss = 0.
        running_adj_corr  = 0.
        running_total_loss = 0.
        total_batch = len(self.train_loader)
        data_iter  = iter(self.train_loader)
        view_per_image = 64

        while batch_idx < total_batch:
            try:
                # torch.autograd.set_detect_anomaly(True)
                batch_idx += 1
                self.iters += 1
                batch_data  = next(data_iter)
                im = batch_data[0].to(device)
                rois = batch_data[1].to(device)
                gt_scores = batch_data[2].to(device)

                random_ID = list(range(0, rois.shape[1]))
                random.shuffle(random_ID)
                chosen_ID = random_ID[:view_per_image]
                rois = rois[:,chosen_ID]
                gt_scores = gt_scores[:,chosen_ID]
                region_feat = self.extractor(im, rois)

                if random.uniform(0,1) <= 0.3:
                    batch_data2 = next(data_iter)
                    im2 = batch_data2[0].to(device)
                    rois2 = batch_data2[1].to(device)
                    gt_scores2 = batch_data2[2].to(device)

                    random_ID = list(range(0, rois2.shape[1]))
                    random.shuffle(random_ID)
                    chosen_ID = random_ID[:view_per_image]

                    rois2 = rois2[:,chosen_ID]
                    gt_scores2 = gt_scores2[:, chosen_ID]
                    region_feat2 = self.extractor(im2, rois2)
                    region_feat  = torch.cat([region_feat, region_feat2], dim=0)
                    gt_scores    = torch.cat([gt_scores,   gt_scores2], dim=-1)
                    random_ID    = list(range(0, region_feat.shape[0]))
                    random.shuffle(random_ID)
                    chosen_ID    = random_ID[:view_per_image]

                    region_feat  = region_feat[chosen_ID]
                    gt_scores    = gt_scores[:,chosen_ID]

                adj, pre_scores  = self.gnn(region_feat)
                loss_reg  = cropping_regression_loss(pre_scores, gt_scores, MOS_MEAN)
                loss_rank = cropping_rank_loss(pre_scores, gt_scores)
                adj_corr  = score_feature_correlation(gt_scores, adj)
                total_loss= loss_reg + loss_rank - adj_corr

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                running_reg_loss  += loss_reg.item()
                running_rank_loss += loss_rank.item()
                running_adj_corr  += adj_corr.item()
                running_total_loss+= total_loss.item()
            except StopIteration:
                data_iter = iter(self.train_loader)

            if batch_idx % cfg.display_freq == 0:
                avg_reg_loss  = running_reg_loss  / batch_idx
                avg_rank_loss = running_rank_loss / batch_idx
                avg_adj_corr  = running_adj_corr  / batch_idx
                avg_total_loss= running_total_loss/ batch_idx

                cur_lr = self.optimizer.param_groups[0]['lr']
                self.writer.add_scalar('train/reg_loss', avg_reg_loss, self.iters)
                self.writer.add_scalar('train/rank_loss', avg_rank_loss, self.iters)
                self.writer.add_scalar('train/adj_corr',  avg_adj_corr, self.iters)
                self.writer.add_scalar('train/total_loss', avg_total_loss, self.iters)
                self.writer.add_scalar('train/lr', cur_lr, self.iters)

                time_per_batch = (time.time() - start) / (batch_idx + 1.)
                last_batches = (self.max_epoch - self.epoch - 1) * total_batch + (total_batch - batch_idx - 1)
                last_time = int(last_batches * time_per_batch)
                time_str = str(datetime.timedelta(seconds=last_time))

                print('=== epoch:{}/{}, step:{}/{} | Total_Loss:{:.4f} | Adj_Corr: {:.4f} | lr:{:.6f} | estimated last time:{} ==='.format(
                    self.epoch, self.max_epoch, batch_idx, total_batch, avg_total_loss, avg_adj_corr, cur_lr, time_str
                ))

    def eval(self):
        srcc, acc5, acc10 = evaluate_on_GAICD(self.extractor, self.gnn)
        iou, disp = evaluate_on_FLMS(self.extractor, self.gnn)
        self.eval_results.append([self.epoch, srcc, acc5, acc10, iou, disp])
        epoch_result = {'srcc': srcc, 'acc5': acc5, 'acc10': acc10,
                        'FLMS_iou': iou, 'FLMS_disp': disp}
        for m in self.best_results.keys():
            update = False
            if ('disp' not in m) and (epoch_result[m] > self.best_results[m]):
                update = True
            elif ('disp' in m) and (epoch_result[m] < self.best_results[m]):
                update = True
            if update:
                self.best_results[m] = epoch_result[m]
                checkpoint_path = os.path.join(cfg.checkpoint_dir, 'extractor-best-{}.pth'.format(m))
                torch.save(self.extractor.state_dict(), checkpoint_path)

                checkpoint_path = os.path.join(cfg.checkpoint_dir, 'gnn-best-{}.pth'.format(m))
                torch.save(self.gnn.state_dict(), checkpoint_path)
                print('Update best {} model, best {}={:.4f}'.format(m, m, self.best_results[m]))
            if 'FLMS' in m:
                self.writer.add_scalar('eval_FLMS/{}'.format(m),  epoch_result[m], self.epoch)
            else:
                self.writer.add_scalar('eval_GAICD/{}'.format(m), epoch_result[m], self.epoch)
                if m == 'srcc':
                    self.writer.add_scalar('eval_GAICD/best-srcc', self.best_results[m], self.epoch)

        # if self.epoch % cfg.save_freq == 0:
        #     checkpoint_path = os.path.join(cfg.checkpoint_dir, 'epoch-{}.pth'.format(self.epoch))
        #     torch.save(self.model.state_dict(), checkpoint_path)

    def record_eval_results(self):
        csv_path = os.path.join(cfg.exp_path, '..', '{}.csv'.format(cfg.exp_name))
        header = ['epoch', 'srcc', 'acc5', 'acc10',
                  'FLMS_iou', 'FLMS_disp']
        rows = [header]
        for i in range(len(self.eval_results)):
            new_results = []
            for j in range(len(self.eval_results[i])):
                new_results.append(round(self.eval_results[i][j], 3))
            self.eval_results[i] = new_results
        rows += self.eval_results
        metrics = [[] for i in header]
        for result in self.eval_results:
            for i, r in enumerate(result):
                metrics[i].append(r)
        for name, m in zip(header, metrics):
            if name == 'epoch':
                continue
            index = m.index(max(m))
            if 'disp' in name:
                index = m.index(min(m))
            title = 'best {}(epoch-{})'.format(name, index)
            row = [l[index] for l in metrics]
            row[0] = title
            rows.append(row)
        with open(csv_path, 'w') as f:
            cw = csv.writer(f)
            cw.writerows(rows)
        print('Save result to ', csv_path)

if __name__ == '__main__':
    cfg.create_path()
    for file in os.listdir('./'):
        if file.endswith('.py'):
            shutil.copy(file, cfg.exp_path)
            print('backup', file)
    FeatureExtractor = RegionFeatureExtractor(loadweight=True).to(device)
    GNN = CroppingGraph().to(device)
    trainer = Trainer(FeatureExtractor, GNN)
    trainer.run()