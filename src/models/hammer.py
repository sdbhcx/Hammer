"""
Hammer.

Author: Lei Yao (rayyohhust@gmail.com)
Please cite our work if the code is helpful to you.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration

from src.models.losses import ca_loss
from src.models.point_model.pointnet2 import (
    PointNetSetAbstractionMsg, 
    PointNetFeaturePropagation
)
from src.models.decoder.transformer import CrossAttnBlock
    

class SwapAxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.transpose(1, 2)
    

class PointBackbone(nn.Module):
    def __init__(self, emb_dim, normal_channel, additional_channel, N_p, latent_out_dim):
        super().__init__()
        self.N_p = N_p
        self.normal_channel = normal_channel
        # downsampling layers
        self.sa1 = PointNetSetAbstractionMsg(
            npoint=512,
            radius_list=[0.1, 0.2, 0.4],
            nsample_list=[32, 64, 128],
            in_channel=3,
            mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]]
        )
        self.sa2 = PointNetSetAbstractionMsg(
            npoint=128,
            radius_list=[0.4, 0.8],
            nsample_list=[64, 128],
            in_channel=128+128+64,
            mlp_list=[[128, 128, 256], [128, 196, 256]]
        )
        self.sa3 = PointNetSetAbstractionMsg(
            npoint=self.N_p,
            radius_list=[0.2, 0.4],
            nsample_list=[16, 32],
            in_channel=256+256,
            mlp_list=[[128, 128, 256], [128, 196, 256]]
        )

        self.attend = CrossAttnBlock(
            embedding_dim=emb_dim,
            num_heads=8,
            mlp_dim=1024,
            activation=nn.GELU,
            attention_downsample_rate=2,
        )
        self.proj = nn.Sequential(
            nn.Linear(latent_out_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # upsampling layers
        self.fp3 = PointNetFeaturePropagation(
            in_channel=512+emb_dim, mlp=[768, 512]
        )
        self.fp2 = PointNetFeaturePropagation(
            in_channel=832, mlp=[768, 512]
        )
        self.fp1 = PointNetFeaturePropagation(
            in_channel=518+additional_channel, mlp=[512, 512]
        )

        self.feat_proj = nn.Sequential(
            nn.Linear(2*emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )
        self.soft_weights = nn.Linear(emb_dim, 1, bias=False)

    def forward(self, xyz, hidden_states):
        if self.normal_channel:
            feat_0 = xyz
            xyz_0 = xyz[:, :3, :]
        else:
            feat_0 = xyz
            xyz_0 = xyz

        xyz_1, feat_1 = self.sa1(xyz_0, feat_0) # [B, 3, npoint_sa1] --- [B, 320, npoint_sa1]
        xyz_2, feat_2 = self.sa2(xyz_1, feat_1) # [B, 3, npoint_sa2] --- [B, 512, npoint_sa2]
        xyz_3, feat_3 = self.sa3(xyz_2, feat_2) # [B, 3, N_p] --- [B, 512, N_p]
        feat_3_geom = feat_3 # [B, 512, N_p] pre-fusion geometry, kept for structural supervision

        h = self.proj(hidden_states)

        feat_3 = self.attend(feat_3.transpose(1, 2), h)
        feat_3 = feat_3.transpose(1, 2) # [B, emb_dim, N_p]

        feat_2 = self.fp3(xyz_2, xyz_3, feat_2, feat_3)
        feat_1 = self.fp2(xyz_1, xyz_2, feat_1, feat_2)
        feat_0 = self.fp1(xyz_0, xyz_1, torch.cat([xyz_0, feat_0], 1), feat_1)

        feat_0 = feat_0.transpose(1, 2)
        scores = self.soft_weights(h)
        weights = F.softmax(scores, dim=1)
        avg_h = torch.sum(weights * h, dim=1, keepdim=True)
        avg_h = avg_h.repeat(1, feat_0.shape[1], 1)
        feat_0 = torch.cat([feat_0, avg_h], dim=-1)
        feat_0 = self.feat_proj(feat_0)

        feat_1 = feat_1.transpose(1, 2)
        feat_2 = feat_2.transpose(1, 2)
        feat_3 = feat_3.transpose(1, 2)

        return [feat_3, feat_2, feat_1, feat_0], feat_3_geom
    

class LiftTo3D(nn.Module):
    def __init__(self, depth, embedding_dim):
        super().__init__()
        self.depth = depth
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(
                CrossAttnBlock(
                    embedding_dim=embedding_dim,
                    num_heads=8,
                    mlp_dim=1024,
                    activation=nn.GELU,
                    attention_downsample_rate=2,
                )
            )
        
    def forward(self, cont_token, point_embedding):
        assert len(point_embedding) == self.depth, \
            "Point embedding length must match depth"
        for i in range(self.depth):
            cont_token = self.layers[i](cont_token, point_embedding[i])
        return cont_token
    

class AffordDecoder(nn.Module):
    def __init__(self, embedding_dim, downsample_rate=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        hidden_dim = embedding_dim // downsample_rate

        self.cross_attn = CrossAttnBlock(
            embedding_dim=embedding_dim,
            num_heads=8,
            mlp_dim=1024,
            activation=nn.GELU,
            attention_downsample_rate=2,
        )
        
        self.out_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            SwapAxes(),
            nn.GroupNorm(4, hidden_dim),
            nn.ReLU(),
            SwapAxes(),
            nn.Linear(hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, cont_token):
        src = self.cross_attn(src, cont_token)
        affordance = self.out_head(src)
        masks = self.sigmoid(affordance)
        return masks


class SegModel(nn.Module):
    def __init__(self, config, **kwargs):
        super(SegModel, self).__init__()
        self.config = config
        self.initialize_seg_modules(self.config, kwargs)

    def initialize_seg_modules(self, config, kwargs):
        additional_channel = kwargs.get("additional_channel", 0)
        point_embd_dim = kwargs.get("point_embd_dim", 512)
        point_N_p = kwargs.pop("point_N_p", 64)
        point_normal_channel = kwargs.pop("point_normal_channel", False)
        self.point_backbone = PointBackbone(
            emb_dim=point_embd_dim,
            normal_channel=point_normal_channel,
            additional_channel=additional_channel,
            N_p=point_N_p,
            latent_out_dim=kwargs["vlm_out_dim"]
        )
        for param in self.point_backbone.parameters():
            param.requires_grad = True

        self.lift_to_3d = LiftTo3D(
            depth=4,
            embedding_dim=point_embd_dim
        )

        self.afford_decoder = AffordDecoder(
            embedding_dim=point_embd_dim,
            downsample_rate=8
        )

        # Projection layer for text embeddings
        in_dim = config.hidden_size
        vlm_out_dim = kwargs["vlm_out_dim"]
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, vlm_out_dim),
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            param.requires_grad = True


class Hammer(nn.Module):
    def __init__(self, config, **kwargs):
        super(Hammer, self).__init__()
        self.cont_token_idx = kwargs.pop("cont_token_idx")
        vlm_out_dim = kwargs["vlm_out_dim"]
        point_embd_dim = kwargs.get("point_embd_dim", 512)
        point_normal_channel = kwargs.pop("point_normal_channel", False)
        if point_normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        kwargs["additional_channel"] = additional_channel

        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            kwargs["model"],
            torch_dtype=kwargs["torch_dtype"],
            attn_implementation=kwargs["attention"]
        )
        self.seg_model = SegModel(config, **kwargs)

        self.projection = nn.Sequential(
            nn.Linear(vlm_out_dim, point_embd_dim),   
            nn.ReLU(),
            nn.Linear(point_embd_dim, point_embd_dim),
            nn.ReLU()
        )
        
        self.ce_loss_weight = kwargs.pop("ce_loss_weight", 0)
        self.mask_loss_weight = kwargs.pop("mask_loss_weight", 0)
        self.struct_loss_weight = kwargs.pop("struct_loss_weight", 0.0)
        self.struct_decay_steps = kwargs.pop("struct_decay_steps", 0)
        self.struct_group_k = kwargs.pop("struct_group_k", 4)
        self._struct_step = 0

    @torch.autocast(device_type='cuda', dtype=torch.float32)    
    def get_point_embs(self, points, hidden_states):
        point_embedding = self.seg_model.point_backbone(points, hidden_states)
        
        return point_embedding

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    @staticmethod
    def gram_loss(feat_3_geom, gt_affords, k_group=4):
        """
        Align the second-order structure of pre-fusion geometric features with the
        intent structure aggregated from ground-truth affordance labels.

        Args:
            feat_3_geom: pre-fusion geometric point features of shape (B, C, N)
            gt_affords: ground-truth affordance probabilities of shape (B, M, 1)
            k_group: groups of consecutive points collapsed into one cell
        Returns:
            mse loss between the predicted and target similarity matrices
        """
        B, C, N = feat_3_geom.shape
        N_g = N // k_group
        if N_g < 2:
            return feat_3_geom.new_zeros(())

        feat = feat_3_geom.reshape(B, C, N_g, k_group).mean(dim=-1)
        feat = F.normalize(feat, dim=1)
        pred_sim = torch.bmm(feat, feat.transpose(1, 2))

        gt = gt_affords.reshape(B, -1).unsqueeze(1)
        gt_g = F.interpolate(gt, size=(N_g,), mode="nearest")
        positive = (gt_g > 0.5).float().squeeze(1)
        target_sim = torch.bmm(positive.unsqueeze(1), positive.unsqueeze(2))
        target_sim = target_sim / target_sim.mean().clamp_min(1e-8)

        return F.mse_loss(pred_sim, target_sim)

    def _struct_weight(self):
        """Cosine decay of the structural loss weight down to zero."""
        weight = self.struct_loss_weight
        if weight <= 0:
            return 0.0
        decay_steps = self.struct_decay_steps
        if decay_steps <= 0:
            return weight
        t = min(self._struct_step / decay_steps, 1.0)
        return weight * 0.5 * (1.0 + math.cos(math.pi * t))

    def model_forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,
        attention_masks: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_grid_thw: torch.FloatTensor,
        points: torch.FloatTensor,
        gt_affords: torch.FloatTensor,
        offset: torch.LongTensor,
        inference: bool = False,
        **kwargs
    ):
        B = points.shape[0]
        assert B == len(offset) - 1, "Batch size mismatch with offset length"

        output = self.vlm.forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            attention_mask=attention_masks,
            use_cache=False,
            output_hidden_states=True,
        )
        output_hidden_states = output.hidden_states[-1]
        cont_token_mask = input_ids == self.cont_token_idx

        hidden_states = []
        assert len(self.seg_model.text_hidden_fcs) == 1
        hidden_states.append(self.seg_model.text_hidden_fcs[0](output_hidden_states))
        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)

        pred_embeddings = last_hidden_state[cont_token_mask]
        cont_token_counts = cont_token_mask.int().sum(-1)
        cont_token_offset = cont_token_counts.cumsum(-1)
        cont_token_offset = torch.cat(
            [
                torch.zeros(1, device=cont_token_offset.device).long(), 
                cont_token_offset
            ], 
            dim=0
        )
        cont_token_offset = cont_token_offset[offset]

        pred_embeddings_list = []
        for i in range(len(cont_token_offset) - 1):
            start_i, end_i = cont_token_offset[i], cont_token_offset[i + 1]
            pred_embeddings_list.append(pred_embeddings[start_i:end_i])
        pred_embeddings = torch.stack(pred_embeddings_list, dim=0)

        point_embeddings, feat_3_geom = self.get_point_embs(points, last_hidden_state)

        cont = self.projection(pred_embeddings)
        cont = self.seg_model.lift_to_3d(cont, point_embeddings)
        pred_afford = self.seg_model.afford_decoder(point_embeddings[-1], cont)
        loss_ca = ca_loss(pred_afford, gt_affords)
        loss_ce = output.loss * self.ce_loss_weight
        loss_ca = loss_ca * self.mask_loss_weight

        struct_weight = self._struct_weight()
        if struct_weight > 0 and not inference:
            loss_struct = self.gram_loss(
                feat_3_geom, gt_affords, self.struct_group_k
            ) * struct_weight
            self._struct_step += 1
        else:
            loss_struct = loss_ce.new_zeros(())
        total_loss = loss_ce + loss_ca + loss_struct
        return {
            "loss": total_loss,
            "ce_loss": loss_ce,
            "ca_loss": loss_ca,
            "struct_loss": loss_struct,
            "struct_weight": struct_weight,
            "pred_afford": pred_afford
        }

