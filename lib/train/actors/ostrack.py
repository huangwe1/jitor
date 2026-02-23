from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import jittor as jt
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate

from lib.utils.nt_xent import NTXentLoss


class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize
        self.cfg = cfg

        self.device = 'cuda' if jt.has_cuda else 'cpu'
        self.temperature = 0.5
        self.use_cosine_similarity = True
        self.nt_xent_criterion = NTXentLoss(self.device, self.bs,
                                            self.temperature, self.use_cosine_similarity)

    def __call__(self, data):
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], self.device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                            total_epochs=ce_start_epoch + ce_warm_epoch,
                                            ITERS_PER_EPOCH=1,
                                            base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]

        phrase_ids = data['phrase_ids'].permute(1, 0)
        phrase_attnmask = data['phrase_attnmask'].permute(1, 0)
        out_dict = self.net(template=template_list,
                            search=search_img,
                            phrase_ids=phrase_ids,
                            phrase_attnmask=phrase_attnmask,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        gt_bbox = gt_dict['search_anno'][-1]
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        pred_boxes = pred_dict['pred_boxes']
        if jt.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox).unsqueeze(1).repeat(1, num_queries, 1).view(-1, 4).clamp(min_v=0.0, max_v=1.0)

        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
        except:
            giou_loss, iou = jt.float32(0.0), jt.float32(0.0)

        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = jt.float32(0.0)

        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss

        ## Multi-Modal Alignment
        loss_cma_xl = 0.5 * self.nt_xent_criterion(pred_dict['vision_x_vectors'],
                                                     pred_dict['language_vectors'])
        loss_cma_zl = 0.5 * self.nt_xent_criterion(pred_dict['vision_z_vectors'],
                                                     pred_dict['language_vectors'])
        loss_cma = 0.5 * loss_cma_xl + 0.5 * loss_cma_zl
        loss = loss + loss_cma

        loss_ima = 0.5 * self.nt_xent_criterion(pred_dict['vision_x_vectors'],
                                                  pred_dict['vision_z_vectors'])
        loss_ima = 1.0 * loss_ima
        loss = loss + loss_ima

        if return_status:
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/cma": loss_cma.item(),
                      "Loss/ima": loss_ima.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
