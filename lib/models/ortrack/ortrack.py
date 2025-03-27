"""
Basic ORTrack model.
"""
import os
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head

from lib.utils.box_ops import box_xyxy_to_cxcywh

import numpy as np
from scipy.stats import multivariate_normal
import cv2
from lib.models.ortrack.deit import deit_tiny_patch16_224, deit_tiny_patch16_224_distill
from lib.models.ortrack.vision_transformer import vit_tiny_patch16_224, vit_tiny_distilled_patch16_224
from lib.models.ortrack.eva import eva02_tiny_patch14_224, eva02_tiny_patch14_224_distill

class ORTrack(nn.Module):
    """ This is the base class for ORTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_sz_t = int(box_head.feat_template_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
            self.feat_len_t = int(self.feat_sz_t ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        self.intensity = []
        self.randomMask = False

    def random_masking(self, N, H, W, D, mask_ratio, device):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        """
        # N, H, W, D = x.shape  # batch, height, width, dim
        len_keep = int(H*W * (1 - mask_ratio))

        noise = torch.rand(N, H, W, device=device)  # noise in [0, 1]
        noise_vec = torch.reshape(noise,(N,H*W))

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise_vec, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, H, W],device=device)
        mask_vec = torch.reshape(mask,(N,H*W))
        mask_vec[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask_vec = torch.gather(mask_vec, dim=1, index=ids_restore)
        mask = torch.reshape(mask_vec, (N,H,W))

        return mask


    def simulate_ihhomogenous_Poisson_process(self, intensity):
        # Number of points to simulate
        num_points = np.random.poisson(intensity.max()*np.prod(intensity.shape), 1)[0]

        # Generate random points according to an inhomogeneous Poisson process
        x_points = (np.floor(np.random.uniform(0, intensity.shape[1], num_points))).astype(np.int32)
        y_points = (np.floor(np.random.uniform(0, intensity.shape[0], num_points))).astype(np.int32)

        # Accept or reject points based on the rate function
        accept_prob = intensity[x_points,y_points] / intensity.max()
        accepted_points = np.random.rand(num_points) < accept_prob

        x_points = x_points[accepted_points]
        y_points = y_points[accepted_points]
        return x_points, y_points

    def random_masking_CoxProcess(self, intensity, N, H, W, mask_ratio, device):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        """
        # N, H, W, D = x.shape  # batch, height, width, dim
        len_keep = int(H*W * (1 - mask_ratio))
        poisson_mean = int(H*W * (mask_ratio))
        poisson_samples = np.random.poisson(poisson_mean, N)
        masks = []
        for i in range(N):
            inhPoisson_intensity = poisson_samples[i] * intensity
            x_points, y_points = self.simulate_ihhomogenous_Poisson_process(inhPoisson_intensity)
            mask = torch.ones([1, H, W],device=device)
            mask[:,y_points,x_points] = 0
            masks.append(mask)

        masks = torch.cat(masks,dim=0)

        return masks

    def masking_CoxProcess(self, N, intensity, block_sz, mask_ratio, device):
        H, W = intensity.shape
        h = int(H/block_sz)
        w = int(W/block_sz)
        assert H % block_sz==0 & W % block_sz==0, 'H/block_sz is not int!'

        intensity = cv2.resize(intensity,dsize=(h, w))
        intensity = intensity/intensity.sum()

        mask = self.random_masking_CoxProcess(intensity, N, int(h), int(w), mask_ratio, device)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')
        return mask


    def masking(self, template, block_sz, mask_ratio, device):
        N,  D, H, W = template.shape
        h = H/block_sz
        w = W/block_sz
        assert H % block_sz==0 & W % block_sz==0, 'H/block_sz is not int!'

        mask = self.random_masking(N, int(h), int(w), D, mask_ratio, device)
        mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest')
        # mask.to(template.device)
        return mask


    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                is_distill=False,
                ):

        if not is_distill:
            if self.training and self.randomMask == True:
                mask = self.masking(template, 16, 0.3, template.device)
                mask = mask.repeat(1,template.shape[1],1,1)
            elif self.training:
                if len(self.intensity) == 0:
                    template_r = int(template.shape[-1]/2)
                    sigma = 64
                    x, y = np.mgrid[-template_r:template_r:1, -template_r:template_r:1]
                    pos = np.dstack((x, y))
                    intensity = multivariate_normal([0.0, 0.], [[sigma*template_r, 0.0], [0.0, sigma*template_r]]).pdf(pos)
                    intensity = intensity/intensity.sum()
                else:
                    intensity = self.intensity
                mask = self.masking_CoxProcess(template.shape[0], intensity, 16, 0.3, template.device)
                mask = mask.repeat(1,template.shape[1],1,1)

        x, aux_dict = self.backbone(z=template, x=search)

        if self.training and not is_distill:
            x1, aux_dict1 = self.backbone(z=template * mask, x=search)
            sim_loss = torch.nn.functional.mse_loss(x[:, :self.feat_len_t],x1[:, :self.feat_len_t].detach())
        else:
            sim_loss = 0

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        out['sim_loss'] = sim_loss
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_ortrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('ORTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_patch16_224':
        backbone = deit_tiny_patch16_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_distilled_patch16_224':
        backbone = deit_tiny_patch16_224_distill(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224':
        backbone = vit_tiny_patch16_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_distilled_patch16_224':
        backbone = vit_tiny_distilled_patch16_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'eva02_tiny_patch14_224':
        backbone = eva02_tiny_patch14_224(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'eva02_tiny_distilled_patch14_224':
        backbone = eva02_tiny_patch14_224_distill(num_classes=0, pretrained=True)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError


    if cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_patch16_224' or cfg.MODEL.BACKBONE.TYPE == 'deit_tiny_distilled_patch16_224' \
           or cfg.MODEL.BACKBONE.TYPE == 'eva02_tiny_patch14_224' or cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224' \
            or cfg.MODEL.BACKBONE.TYPE == 'eva02_tiny_distilled_patch14_224' or cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_distilled_patch16_224':
        pass
    else:
        backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = ORTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'ORTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
