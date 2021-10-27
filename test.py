import os
import numpy as np
import torch
from tqdm import tqdm
import pickle
from scipy.stats import spearmanr
import random
import cv2
import json
from cropping_dataset import FLMSDataset, GAICDataset
from config import cfg
from croppingModel import RegionFeatureExtractor,CroppingGraph

device = torch.device('cuda:{}'.format(cfg.gpu_id))
torch.cuda.set_device(cfg.gpu_id)
SEED = 0
random.seed(SEED)

save_dir = './results'
os.makedirs(save_dir, exist_ok=True)

def compute_acc(gt_scores, pr_scores):
    assert (len(gt_scores) == len(pr_scores)), '{} vs. {}'.format(len(gt_scores), len(pr_scores))
    sample_cnt = 0
    acc4_5  = [0 for i in range(4)]
    acc4_10 = [0 for i in range(4)]
    for i in range(len(gt_scores)):
        gts, preds = gt_scores[i], pr_scores[i]
        id_gt = sorted(range(len(gts)), key=lambda j : gts[j], reverse=True)
        id_pr = sorted(range(len(preds)), key=lambda j : preds[j], reverse=True)
        for k in range(4):
            temp_acc4_5  = 0.
            temp_acc4_10 = 0.
            for j in range(k+1):
                if gts[id_pr[j]] >= gts[id_gt[4]]:
                    temp_acc4_5 += 1.0
                if gts[id_pr[j]] >= gts[id_gt[9]]:
                    temp_acc4_10 += 1.0
            acc4_5[k]  += (temp_acc4_5 / (k+1.0))
            acc4_10[k] += ((temp_acc4_10) / (k+1.0))
        sample_cnt += 1
    acc4_5  = [i / sample_cnt for i in acc4_5]
    acc4_10 = [i / sample_cnt for i in acc4_10]
    # print('acc4_5', acc4_5)
    # print('acc4_10', acc4_10)
    avg_acc4_5  = sum(acc4_5)  / len(acc4_5)
    avg_acc4_10 = sum(acc4_10) / len(acc4_10)
    return avg_acc4_5, avg_acc4_10

def compute_iou_and_disp(gt_crop, pre_crop, im_w, im_h):
    ''''
    :param gt_crop: [[x1,y1,x2,y2]]
    :param pre_crop: [[x1,y1,x2,y2]]
    :return:
    '''
    gt_crop = gt_crop[gt_crop[:,0] >= 0]
    zero_t  = torch.zeros(gt_crop.shape[0])
    over_x1 = torch.maximum(gt_crop[:,0], pre_crop[:,0])
    over_y1 = torch.maximum(gt_crop[:,1], pre_crop[:,1])
    over_x2 = torch.minimum(gt_crop[:,2], pre_crop[:,2])
    over_y2 = torch.minimum(gt_crop[:,3], pre_crop[:,3])
    over_w  = torch.maximum(zero_t, over_x2 - over_x1)
    over_h  = torch.maximum(zero_t, over_y2 - over_y1)
    inter   = over_w * over_h
    area1   = (gt_crop[:,2] - gt_crop[:,0]) * (gt_crop[:,3] - gt_crop[:,1])
    area2   = (pre_crop[:,2] - pre_crop[:,0]) * (pre_crop[:,3] - pre_crop[:,1])
    union   = area1 + area2 - inter
    iou     = inter / union
    disp    = (torch.abs(gt_crop[:, 0] - pre_crop[:, 0]) + torch.abs(gt_crop[:, 2] - pre_crop[:, 2])) / im_w + \
              (torch.abs(gt_crop[:, 1] - pre_crop[:, 1]) + torch.abs(gt_crop[:, 3] - pre_crop[:, 3])) / im_h
    iou_idx = torch.argmax(iou, dim=-1)
    dis_idx = torch.argmin(disp, dim=-1)
    index   = dis_idx if (iou[iou_idx] == iou[dis_idx]) else iou_idx
    return iou[index].item(), disp[index].item()

def evaluate_on_GAICD(extracor, gnn, save_results=False):
    extracor.eval()
    gnn.eval()
    print('='*5, 'Evaluating on GAICD dataset', '='*5)
    srcc_list = []
    gt_scores = []
    pr_scores = []
    count = 0
    test_dataset = GAICDataset(split='test')
    test_loader  = torch.utils.data.DataLoader(
                        test_dataset, batch_size=1,
                        shuffle=False, num_workers=cfg.num_workers,
                        drop_last=False)
    if save_results:
        image_results = dict()
        result_dir    = os.path.join(save_dir, 'GAICD')
        os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader)):
            im = batch_data[0].to(device)
            rois = batch_data[1].to(device)
            scores = batch_data[2].cpu().numpy().reshape(-1)
            width = batch_data[3].item()
            height = batch_data[4].item()
            image_file = batch_data[5][0]
            image_name = os.path.basename(image_file)
            count += im.shape[0]

            region_feat = extracor(im, rois)
            _,pre_scores = gnn(region_feat)

            pre_scores = pre_scores.cpu().detach().numpy().reshape(-1)
            srcc_list.append(spearmanr(scores, pre_scores)[0])
            gt_scores.append(scores)
            pr_scores.append(pre_scores)

            if save_results:
                pre_index = np.argmax(pre_scores)
                cand_crop = rois.squeeze().cpu().detach().numpy().reshape(-1,4)
                cand_crop[:, 0::2] *= (float(width) / im.shape[-1])
                cand_crop[:, 1::2] *= (float(height) / im.shape[-2])
                cand_crop = cand_crop.astype(np.int32)
                pred_crop = cand_crop[pre_index] # x1,y1,x2,y2
                image_results[image_name] = pred_crop.tolist()
                # save predicted best crop
                src_img   = cv2.imread(image_file)
                crop_img  = src_img[pred_crop[1] : pred_crop[3], pred_crop[0] : pred_crop[2]]
                result_file = os.path.join(result_dir, image_name)
                cv2.imwrite(result_file, crop_img)
    if save_results:
        with open(os.path.join(save_dir, 'GAICD.json'), 'w') as f:
            json.dump(image_results, f)

    srcc = sum(srcc_list) / len(srcc_list)
    acc5, acc10 = compute_acc(gt_scores, pr_scores)
    print('Test on GAICD {} images, SRCC={:.3f}, acc5={:.3f}, acc10={:.3f}'.format(
        count, srcc, acc5, acc10
    ))
    return srcc, acc5, acc10

def get_pdefined_anchor():
    # get predefined boxes(x1, y1, x2, y2)
    pdefined_anchors = np.array(pickle.load(open(cfg.predefined_pkl, 'rb'), encoding='iso-8859-1')).astype(np.float32)
    print('num of pre-defined anchors: ', pdefined_anchors.shape)
    return pdefined_anchors

def evaluate_on_FLMS(extractor, gnn, save_results=False):
    print('=' * 5, f'Evaluating on FLMS', '=' * 5)
    extractor.eval()
    gnn.eval()
    pdefined_anchors = get_pdefined_anchor() # n,4, (x1,y1,x2,y2)

    accum_disp = 0
    accum_iou  = 0
    crop_cnt = 0
    alpha = 0.75
    alpha_cnt = 0
    cnt = 0

    if save_results:
        image_results = dict()
        result_dir    = os.path.join(save_dir, 'FLMS')
        os.makedirs(result_dir, exist_ok=True)

    with torch.no_grad():
        test_dataset= FLMSDataset()
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                                  shuffle=False, num_workers=cfg.num_workers,
                                                  drop_last=False)
        for batch_idx, batch_data in enumerate(tqdm(test_loader)):
            im = batch_data[0].to(device)
            gt_crop = batch_data[1] # x1,y1,w,h
            width = batch_data[2].item()
            height = batch_data[3].item()
            image_file = batch_data[4][0]
            image_name = os.path.basename(image_file)

            rois = np.zeros((len(pdefined_anchors), 4), dtype=np.float32)
            rois[:, 0::2] = pdefined_anchors[:, 0::2] * im.shape[-1]
            rois[:, 1::2] = pdefined_anchors[:, 1::2] * im.shape[-2]
            rois = torch.from_numpy(rois).unsqueeze(0).to(device)  # 1,n,4

            region_feat = extractor(im, rois)
            adj, scores = gnn(region_feat)
            scores = scores.reshape(-1)
            scores = scores.cpu().detach().numpy()
            idx = np.argmax(scores)

            pred_x1 = int(pdefined_anchors[idx][0] * width)
            pred_y1 = int(pdefined_anchors[idx][1] * height)
            pred_x2 = int(pdefined_anchors[idx][2] * width)
            pred_y2 = int(pdefined_anchors[idx][3] * height)
            pred_crop = torch.tensor([[pred_x1, pred_y1, pred_x2, pred_y2]])
            gt_crop = gt_crop.reshape(-1, 4)
            iou, disp = compute_iou_and_disp(gt_crop, pred_crop, width, height)
            if iou >= alpha:
                alpha_cnt += 1
            accum_iou += iou
            accum_disp += disp
            cnt += 1

            if save_results:
                image_results[image_name] = [pred_x1, pred_y1, pred_x2, pred_y2]
                src_img   = cv2.imread(image_file)
                pred_crop = src_img[pred_y1: pred_y2, pred_x1 : pred_x2]
                result_file = os.path.join(result_dir, image_name)
                cv2.imwrite(result_file, pred_crop)
    if save_results:
        with open(os.path.join(save_dir, 'FLMS.json'), 'w') as f:
            json.dump(image_results, f)
    avg_iou = accum_iou / cnt
    avg_disp = accum_disp / (cnt * 4.0)
    avg_recall = float(alpha_cnt) / cnt
    print('Test on {} images, IoU={:.4f}, Disp={:.4f}, recall={:.4f}(iou>={:.2f})'.format(
        cnt, avg_iou, avg_disp, avg_recall, alpha
    ))
    return avg_iou, avg_disp


if __name__ == '__main__':
    extractor = RegionFeatureExtractor(loadweight=False)
    extractor_weight = './pretrained_model/extractor-best-srcc.pth'
    extractor.load_state_dict(torch.load(extractor_weight))
    extractor = extractor.to(device).eval()

    gnn = CroppingGraph()
    gnn_weight = './pretrained_model/gnn-best-srcc.pth'
    gnn.load_state_dict(torch.load(gnn_weight))
    gnn = gnn.eval().to(device)
    evaluate_on_GAICD(extractor, gnn, save_results=True)
    evaluate_on_FLMS(extractor, gnn, save_results=True)


