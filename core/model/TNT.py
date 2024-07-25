# TNT model
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader

# from core.model.backbone.vectornet import VectorNetBackbone
from core.model.backbone.vectornet_v2 import VectorNetBackbone
from core.model.layers.target_prediction import TargetPred
# from core.model.layers.target_prediction_v2 import TargetPred
from core.model.layers.motion_etimation import MotionEstimation
from core.model.layers.scoring_and_selection import TrajScoreSelection, distance_metric
from core.loss import TNTLoss

from core.dataloader.argoverse_loader_v2 import GraphData, ArgoverseInMem


class TNT(nn.Module):
    def __init__(self,
                 in_channels=8,
                 horizon=50,
                 num_subgraph_layers=3,
                 num_global_graph_layer=4,
                 subgraph_width=128,
                 global_graph_width=128,
                 with_aux=False,
                 aux_width=128,
                 target_pred_hid=128,
                 m=50,
                 motion_esti_hid=128,
                 score_sel_hid=64,
                 temperature=0.01,
                 k=5,
                 device=torch.device("cpu")
                 ):
        """
        TNT algorithm for trajectory prediction
        :param in_channels: int, the number of channels of the input node features
        :param horizon: int, the prediction horizon (prediction length)
        :param num_subgraph_layers: int, the number of subgraph layer
        :param num_global_graph_layer: the number of global interaction layer
        :param subgraph_width: int, the channels of the extrated subgraph features
        :param global_graph_width: int, the channels of extracted global graph feature
        :param with_aux: bool, with aux loss or not
        :param aux_width: int, the hidden dimension of aux recovery mlp
        :param n: int, the number of sampled target candidate
        :param target_pred_hid: int, the hidden dimension of target prediction
        :param m: int, the number of selected candidate
        :param motion_esti_hid: int, the hidden dimension of motion estimation
        :param score_sel_hid: int, the hidden dimension of score module
        :param temperature: float, the temperature when computing the score
        :param k: int, final output trajectories
        :param lambda1: float, the weight of candidate prediction loss
        :param lambda2: float, the weight of motion estimation loss
        :param lambda3: float, the weight of trajectory scoring loss
        :param device: the device for computation
        :param multi_gpu: the multi gpu setting
        """
        super(TNT, self).__init__()
        self.horizon = horizon
        self.m = m
        self.k = k

        self.with_aux = with_aux

        self.device = device

        # feature extraction backbone
        self.backbone = VectorNetBackbone(
            in_channels=in_channels,
            num_subgraph_layres=num_subgraph_layers,
            subgraph_width=subgraph_width,
            num_global_graph_layer=num_global_graph_layer,
            global_graph_width=global_graph_width,
            with_aux=with_aux,
            aux_mlp_width=aux_width,
            device=device
        )

        self.target_pred_layer = TargetPred(
            in_channels=global_graph_width,
            hidden_dim=target_pred_hid,
            m=m,
            device=device
        )
        self.motion_estimator = MotionEstimation(
            in_channels=global_graph_width,
            horizon=horizon,
            hidden_dim=motion_esti_hid
        )
        self.traj_score_layer = TrajScoreSelection(
            feat_channels=global_graph_width,
            horizon=horizon,
            hidden_dim=score_sel_hid,
            temper=temperature,
            device=self.device
        )
        self._init_weight()

    def forward(self, data):
        """
        output prediction for training
        :param data: observed sequence data
        :return: dict{
                        "target_prob":  the predicted probability of each target candidate,
                        "offset":       the predicted offset of the target position from the gt target candidate,
                        "traj_with_gt": the predicted trajectory with the gt target position as the input,
                        "traj":         the predicted trajectory without the gt target position,
                        "score":        the predicted score for each predicted trajectory,
                     }
        """
        n = int(data.candidate_len_max[0].cpu().numpy()) # 所有csv中最多的采样点数量

        target_candidate = data.candidate.view(-1, n, 2)   # [bs, max_cand, 2] 
        batch_size, _, _ = target_candidate.size()
        candidate_mask = data.candidate_mask.view(-1, n) # [bs, max_cand]

        # feature encoding
        global_feat, aux_out, aux_gt = self.backbone(data)# global_feat[bs, time_step_len, global_graph_width] time_s_l是pad之后的obj_num+seg_num
        target_feat = global_feat[:, 0].unsqueeze(1) # [bs,1,feat_channels][64,1,64] 取batch中每个sample的agent项 

        # 1.cand打分并生成offset。predict prob. for each target candidate, and corresponding offest
        # [bx,max_cand],[bs,max_cand,2] <-          [bs,1,D]+[bs,max_cand,2]+[bs,max_cand]
        target_prob, offset = self.target_pred_layer(target_feat, target_candidate, candidate_mask) 

        # 2. 轨迹生成。predict the trajectory given the target gt (Teacher Forcing，这里的target gt就是真实agent 的最终点)
        target_gt = data.target_gt.view(-1, 1, 2) # [bs*2] -> [bs, 1, 2]
        traj_with_gt = self.motion_estimator(target_feat, target_gt) # [bs,1,D]+[bs,1,2]-> ->[bs,1,horizon*2]

        # 3. 轨迹打分。predict the trajectories for the M most-likely predicted target, and the score
        _, indices = target_prob.topk(self.m, dim=1)# [bx,max_cand] -> [bs, 50] 
        batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(self.m)]).T # ->(bs, 50)
        target_pred_se, offset_pred_se = target_candidate[batch_idx, indices], offset[batch_idx, indices] # [bs,50,2] [bs,50,2] <-[bs,max_cand,2][(bs,50),(bs, 50)]

        trajs = self.motion_estimator(target_feat, target_pred_se + offset_pred_se)# [bs, 50, 60]<-[bs,1,D]+[bs,50,2] # 每个candiadate都算出一个轨迹

        score = self.traj_score_layer(target_feat, trajs)# [bs, 50] <- [bs,1,D] + [bs,50,60]

        return {
            "target_prob": target_prob, # [bx,max_cand]
            "offset": offset, # [bs,max_cand,2]
            "traj_with_gt": traj_with_gt, # [bs,1,horizon*2]
            "traj": trajs, # [bs,50,horizon*2]
            "score": score # [bs,50]
        }, aux_out, aux_gt

    def inference(self, data):
        """
        predict the top k most-likely trajectories
        :param data: observed sequence data
        :return:
        """
        n = data.candidate_len_max[0]
        target_candidate = data.candidate.view(-1, n, 2)    # [batch_size, N, 2]
        batch_size, _, _ = target_candidate.size()

        global_feat, _, _ = self.backbone(data)     # [batch_size, time_step_len, global_graph_width=feature]
        target_feat = global_feat[:, 0].unsqueeze(1)

        # predict the prob. of target candidates and selected the most likely M candidate
        target_prob, offset_pred = self.target_pred_layer(target_feat, target_candidate)
        _, indices = target_prob.topk(self.m, dim=1)
        batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(self.m)]).T
        target_pred_se, offset_pred_se = target_candidate[batch_idx, indices], offset_pred[batch_idx, indices]

        # # DEBUG
        # gt = data.y.unsqueeze(1).view(batch_size, -1, 2).cumsum(axis=1)

        # trajectory estimation for the m predicted target location
        traj_pred = self.motion_estimator(target_feat, target_pred_se + offset_pred_se)

        # score the predicted trajectory and select the top k trajectory
        score = self.traj_score_layer(target_feat, traj_pred) # [bs,m,100] -> [bs,m]
        score_sort = score.topk(self.k, dim=-1).values # [bs,k]
        return self.traj_selection(traj_pred, score).view(batch_size, self.k, self.horizon, 2),score_sort # [bs,k,50,2]

    def candidate_sampling(self, data):
        """
        sample candidates given the test data
        :param data:
        :return:
        """
        raise NotImplementedError

    # todo: determine appropiate threshold
    def traj_selection(self, traj_in, score, threshold=16):
        """
        select the top k trajectories according to the score and the distance
        :param traj_in: candidate trajectories, [batch, M, horizon * 2]
        :param score: score of the candidate trajectories, [batch, M]
        :param threshold: float, the threshold for exclude traj prediction
        :return: [batch_size, k, horizon * 2]
        """
        # re-arrange trajectories according the the descending order of the score
        _, batch_order = score.sort(descending=True)
        traj_pred = torch.cat([traj_in[i, order] for i, order in enumerate(batch_order)], dim=0).view(-1, self.m, self.horizon * 2)
        traj_selected = traj_pred[:, :self.k].clone()                                   # [batch_size, k, horizon * 2]

        # check the distance between them, NMS, stop only when enough trajs collected
        for batch_id in range(traj_pred.shape[0]):                              # one batch for a time
            traj_cnt = 1
            thres = threshold
            while traj_cnt < self.k:
                for j in range(1, self.m):
                    dis = distance_metric(traj_selected[batch_id, :traj_cnt], traj_pred[batch_id, j].unsqueeze(0))
                    if not torch.any(dis < thres):
                        traj_selected[batch_id, traj_cnt] = traj_pred[batch_id, j].clone()

                        traj_cnt += 1
                    if traj_cnt >= self.k:
                        break
                thres /= 2.0

        return traj_selected

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


if __name__ == "__main__":
    batch_size = 32
    DATA_DIR = "../../dataset/interm_tnt_n_s_0804_small"
    # DATA_DIR = "../../dataset/interm_tnt_n_s_0804"
    TRAIN_DIR = os.path.join(DATA_DIR, 'train_intermediate')
    # TRAIN_DIR = os.path.join(DATA_DIR, 'val_intermediate')
    # TRAIN_DIR = os.path.join(DATA_DIR, 'test_intermediate')

    dataset = ArgoverseInMem(TRAIN_DIR)
    data_iter = DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True)

    m, k = 50, 6
    pred_len = 50

    # device = torch.device("cuda:1")
    device = torch.device("cpu")

    model = TNT(in_channels=dataset.num_features,
                horizon=pred_len,
                m=m,
                k=k,
                with_aux=True,
                device=device).to(device)

    # train mode
    model.train()
    for i, data in enumerate(tqdm(data_iter)):
        loss, _ = model.loss(data.to(device))
        print("Training Pass! loss: {}".format(loss))

        if i == 2:
            break

    # eval mode
    model.eval()
    for i, data in enumerate(tqdm(data_iter)):
        pred = model(data.to(device))
        print("Evaluation Pass! Shape of out: {}".format(pred.shape))

        if i == 2:
            break
