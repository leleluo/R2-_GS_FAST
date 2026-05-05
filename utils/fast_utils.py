import random
import torch
from r2_gaussian.gaussian.render_query import render
from r2_gaussian.utils.loss_utils import l1_loss, ssim


# ----------------------------
# 1. 随机采样多视角
# ----------------------------
def sampling_cameras(viewpoint_stack, num_cams=10):
    """从训练相机列表中随机采样若干视角（不修改原列表）"""
    vs = list(viewpoint_stack)
    num_cams = min(num_cams, len(vs))
    return random.sample(vs, num_cams)


# ----------------------------
# 2. 单视角 photometric loss（L1 + 可选 SSIM）
# ----------------------------
def compute_photometric_loss(viewpoint_cam, rendered, lambda_dssim=0.0):
    """与主训练 loss 保持一致的 photometric loss（标量）"""
    gt_image = viewpoint_cam.original_image.to(rendered.device)
    loss_l1 = l1_loss(rendered, gt_image)
    if lambda_dssim > 0:
        loss_dssim = 1.0 - ssim(rendered, gt_image)
        return (1.0 - lambda_dssim) * loss_l1 + lambda_dssim * loss_dssim
    return loss_l1


# ----------------------------
# 3. 多视角一致性评分（VCD + VCP 核心）
#
# 流程：
#   ① 对每个视角普通渲染，得到预测图
#   ② 逐像素计算 L1 误差，用百分位阈值生成 pixel_error_map（0/1）
#   ③ 携带 pixel_error_map 再次渲染，CUDA kernel 内部统计：
#      - gaussian_cnt  : 每个高斯命中高误差像素的次数  → importance_score（VCD）
#   ④ 用整视角 photo_loss 加权累加，得到 pruning_score（VCP）
# ----------------------------
def compute_gaussian_score_r2gs(
    camlist,
    gaussians,
    pipe,
    lambda_dssim=0.0,
    DENSIFY=True,
    quantile=0.70,          # 取误差前 (1-quantile)*100% 的像素为高误差区域
):
    """
    参数：
        camlist      : 采样好的相机列表
        gaussians    : 当前高斯模型
        pipe         : 渲染管线参数
        lambda_dssim : SSIM 权重（与主训练一致即可，默认 0）
        DENSIFY      : True → 同时计算 importance_score；False → 只算 pruning_score
        quantile     : 百分位阈值，0.70 表示误差前 30% 的像素被标记为高误差

    返回：
        importance_score : [N] float tensor，VCD 致密化判据（DENSIFY=False 时为 None）
        pruning_score    : [N] float tensor，VCP 剪枝判据，归一化到 [0,1]
    """
    device = gaussians.get_xyz.device
    n_pts  = gaussians.get_xyz.shape[0]
    eps    = 1e-6

    # 跨视角累积量
    full_metric_counts = torch.zeros(n_pts, device=device)   # 命中高误差像素次数之和
    full_metric_score  = torch.zeros(n_pts, device=device)   # photo_loss 加权累积

    for cam in camlist:
        # ---- ① 第一次渲染：得到预测图 ----
        with torch.no_grad():
            pkg1 = render(cam, gaussians, pipe)
        rendered_image = pkg1["render"].detach()              # [C, H, W]

        # ---- ② 构造逐像素误差图 & 百分位阈值 ----
        gt_image = cam.original_image.to(device)             # [C, H, W]
        l1_map   = torch.abs(rendered_image - gt_image).mean(dim=0)   # [H, W]

        threshold       = torch.quantile(l1_map, quantile)
        pixel_error_map = (l1_map > threshold).float()       # [H, W]，0/1

        # 整视角 photo_loss（标量），用于 pruning_score 加权
        photo_loss = compute_photometric_loss(
            cam, rendered_image, lambda_dssim
        ).detach().clamp_min(0.0)

        # ---- ③ 第二次渲染：携带 pixel_error_map，CUDA 统计每个高斯的命中 ----
        with torch.no_grad():
            pkg2 = render(cam, gaussians, pipe,
                          pixel_error_map=pixel_error_map)

        # gaussian_cnt：该视角下每个高斯命中高误差像素的次数，形状 [N]
        gaussian_cnt = pkg2["gaussian_cnt"].detach()

        # ---- ④ 跨视角累加 ----
        if DENSIFY:
            full_metric_counts += gaussian_cnt

        # pruning_score：误差越大的视角权重越高
        full_metric_score += photo_loss * gaussian_cnt

    # ---- ⑤ 换算为最终分数 ----

    # importance_score：平均每个视角命中高误差像素次数（向下取整，对齐 FastGS）
    if DENSIFY:
        importance_score = torch.div(
            full_metric_counts, len(camlist), rounding_mode='floor'
        )
    else:
        importance_score = None

    # pruning_score：归一化到 [0, 1]，越大表示该高斯在多视角下持续贡献高误差
    if full_metric_score.max() > eps:
        pruning_score = (
            (full_metric_score - full_metric_score.min()) /
            (full_metric_score.max() - full_metric_score.min() + eps)
        )
    else:
        pruning_score = torch.zeros_like(full_metric_score)

    return importance_score, pruning_score
