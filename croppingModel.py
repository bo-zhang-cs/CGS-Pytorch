import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from roi_align.modules.roi_align import RoIAlignAvg
from rod_align.modules.rod_align import RoDAlignAvg
import warnings
warnings.filterwarnings("ignore")

class vgg_base(nn.Module):
    def __init__(self, loadweights=True):
        super(vgg_base, self).__init__()
        vgg = models.vgg16(pretrained=loadweights)
        self.feature3 = nn.Sequential(vgg.features[:23])
        self.feature4 = nn.Sequential(vgg.features[23:30])
        self.feature5 = nn.Sequential(vgg.features[30:])
        #flops, params = profile(self.feature, input_size=(1, 3, 256,256))

    def forward(self, x):
        f3 = self.feature3(x)
        f4 = self.feature4(f3)
        f5 = self.feature5(f4)
        return f3, f4, f5

class RegionFeatureExtractor(nn.Module):
    def __init__(self, loadweight = True):
        super(RegionFeatureExtractor, self).__init__()
        alignsize = 9
        reddim = 32
        downsample = 4
        dim_in = 512

        self.Feat_ext = vgg_base(loadweight)
        self.DimRed = nn.Conv2d(1536, reddim, kernel_size=1, padding=0)
        self.downsample2 = nn.UpsamplingBilinear2d(scale_factor=1.0/2.0)
        self.upsample2 = nn.UpsamplingBilinear2d(scale_factor=2.0)
        self.RoIAlign = RoIAlignAvg(alignsize, alignsize, 1.0/2**downsample)
        self.RoDAlign = RoDAlignAvg(alignsize, alignsize, 1.0/2**downsample)
        self.FC_region = nn.Sequential(
                        nn.Conv2d(reddim*2, 1024, kernel_size=alignsize, padding=0),
                        nn.ReLU(True),
                        nn.Conv2d(1024, dim_in, kernel_size=1),
                        nn.ReLU(True),
                        nn.Flatten(1))
        self.FC_region.apply(weights_init)

    def forward(self, im_data, crops):
        # print(im_data.shape, im_data.dtype, im_data.device, crops.shape, crops.dtype, crops.device)
        B, N, _ = crops.shape
        if crops.shape[-1] == 4:
            index = torch.arange(B).view(-1, 1).repeat(1, N).reshape(B, N, 1).to(crops.device)
            crops = torch.cat((index, crops),dim=-1).contiguous()
        if crops.dim() == 3:
            crops = crops.flatten(0,1)

        f3,f4,f5 = self.Feat_ext(im_data)
        f3 = F.interpolate(f3, size=f4.shape[2:], mode='bilinear', align_corners=True)
        f5 = F.interpolate(f5, size=f4.shape[2:], mode='bilinear', align_corners=True)
        cat_feat = torch.cat((f3,f4,0.5*f5),1)
        red_feat = self.DimRed(cat_feat)

        RoI_feat = self.RoIAlign(red_feat, crops)
        RoD_feat = self.RoDAlign(red_feat, crops)
        fuse_feat = torch.cat((RoI_feat, RoD_feat), 1)
        region_feature = self.FC_region(fuse_feat)
        return region_feature


class CroppingGraph(nn.Module):
    def __init__(self):
        super(CroppingGraph, self).__init__()
        dim_in  = 512
        dim_out = 256
        self.Wm = nn.Linear(dim_in, dim_out, bias=False)
        self.Wn = nn.Linear(dim_in, dim_out, bias=False)
        self.Wr = nn.Linear(dim_in, dim_out, bias=False)
        self.feature_trans = nn.Linear(dim_in, dim_out, bias=False)
        self.feature_rg    = nn.Linear(dim_out, dim_out)
        self.feature_lg    = nn.Linear(dim_out, dim_out)
        self.prediction    = nn.Linear(dim_out, 1)

    def forward(self, x):
        if x.dim() > 2:
            x = x.squeeze()
        assert x.dim() == 2, x.dim()
        xm = self.Wm(x)
        xn = self.Wn(x)
        # n,n,d
        diff = xm[:,None,:] - xn[None,:,:]
        diff = torch.pow(diff, 2)
        # n,n
        dist = torch.sqrt(torch.sum(diff, dim=-1)) / 2
        exps = torch.exp(-dist)
        eye_t = torch.eye(dist.shape[0]).to(dist.device)
        one_t = torch.ones_like(dist)
        exps = exps / (x.shape[0] / 64.)
        adj  = exps * (one_t - eye_t) + eye_t

        # n,d
        xr   = self.Wr(x)
        xr   = torch.mm(adj, xr)
        xl   = self.feature_trans(x)
        # fuse relation feature and local feature
        weight = torch.sigmoid(self.feature_rg(xr) + self.feature_lg(xl))
        feat   = (1 - weight) * xr + weight * xl
        score  = self.prediction(feat)
        return adj,score

def xavier(param):
    torch.nn.init.xavier_uniform_(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def cropping_regression_loss(pre_score, gt_score, score_mean):
    if pre_score.dim() > 1:
        pre_score = pre_score.reshape(-1)
    if gt_score.dim() > 1:
        gt_score  = gt_score.reshape(-1)
    assert pre_score.shape == gt_score.shape, '{} vs. {}'.format(pre_score.shape, gt_score.shape)
    l1_loss = F.smooth_l1_loss(pre_score, gt_score, reduction='none')
    weight  = torch.exp((gt_score - score_mean).clip(min=0,max=100))
    reg_loss= torch.mean(weight * l1_loss)
    # reg_loss  = F.smooth_l1_loss(pre_score, gt_score, reduction='mean')
    return reg_loss

def cropping_rank_loss(pre_score, gt_score):
    '''
    :param pre_score:
    :param gt_score:
    :return:
    '''
    if pre_score.dim() > 1:
        pre_score = pre_score.reshape(-1)
    if gt_score.dim() > 1:
        gt_score  = gt_score.reshape(-1)
    assert pre_score.shape == gt_score.shape, '{} vs. {}'.format(pre_score.shape, gt_score.shape)
    N = pre_score.shape[0]
    pair_num = N * (N-1) / 2
    pre_diff = pre_score[:,None] - pre_score[None,:]
    gt_diff  = gt_score[:,None]  - gt_score[None,:]
    indicat  = -1 * torch.sin(gt_diff) * (pre_diff - gt_diff)
    diff     = torch.maximum(indicat, torch.zeros_like(indicat))
    rank_loss= torch.sum(diff) / pair_num
    return rank_loss

def score_feature_correlation(gt_score, feat_adj):
    '''
    :param gt_score: n
    :param feat_adj: n,n
    :return:
    '''
    if gt_score.dim() > 1:
        gt_score = gt_score.reshape(-1)

    score_diff = torch.pow(gt_score[:,None] - gt_score[None,:],2)
    # n,n
    score_adj  = torch.exp(-score_diff / 2)
    score_adj  = score_adj - score_adj.mean()
    feat_adj   = feat_adj  - feat_adj.mean()
    corr_numer = torch.sum(score_adj * feat_adj)
    corr_demon = torch.pow(score_adj, 2).sum() * torch.pow(feat_adj, 2).sum()
    corr_demon = torch.sqrt(corr_demon + 1e-12)
    corr       = corr_numer / corr_demon
    return corr

if __name__ == '__main__':
    net = RegionFeatureExtractor(loadweight=False)
    net = net.eval().cuda()
    roi = torch.randint(0, 224, (1,64,4)).float().cuda()
    img = torch.randn((1, 3, 256, 256)).cuda()
    print(roi.shape, img.shape)
    out = net(img, roi)
    print(out.shape, out)
    # print(out.shape)
    # gnn = CroppingGraph().cuda()
    # adj,score = gnn(out)
    # print(adj.shape,adj)
    # print(score.shape, score)

    # gt_score = torch.tensor([1.,2.]).cuda()
    # pr_score = torch.randn(2,1).cuda()
    # print('rank loss', cropping_rank_loss(pr_score, gt_score))
    # print('reg  loss', cropping_regression_loss(pr_score, gt_score, 3))
    # print('corr',      score_feature_correlation(pr_score, adj))

