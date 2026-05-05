/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include "utility.h"


std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
    const torch::Tensor& means3D,
    const torch::Tensor& opacity,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const int image_height,
    const int image_width,
    const torch::Tensor& campos,
    const bool prefiltered,
    const int mode,
    const torch::Tensor& pixel_error_map,   // ★ 新增
    const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }

  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, int_opts);

  // ===== 输入：pixel_error_map（来自 Python，可为空）=====
  const float* pixel_ptr = nullptr;
  bool use_error_map = (pixel_error_map.defined() && pixel_error_map.numel() > 0);

  // ===== 输出：责任 + 高误差像素计数（只在有 pixel_error_map 时才分配，节省显存）=====
  torch::Tensor gaussian_responsibility;
  torch::Tensor gaussian_err_count;
  float* resp_ptr = nullptr;
  float* cnt_ptr  = nullptr;
  if (use_error_map) {
    gaussian_responsibility = torch::zeros({P}, float_opts);
    gaussian_err_count      = torch::zeros({P}, float_opts);
    resp_ptr = gaussian_responsibility.contiguous().data_ptr<float>();
    cnt_ptr  = gaussian_err_count.contiguous().data_ptr<float>();
  } else {
    // 返回空 tensor，不占显存
    gaussian_responsibility = torch::zeros({0}, float_opts);
    gaussian_err_count      = torch::zeros({0}, float_opts);
  }

  torch::Tensor pem; // 用于保证 contiguous / reshape 后的 tensor 生命周期
  if (use_error_map) {
    if (!pixel_error_map.is_cuda()) {
      AT_ERROR("pixel_error_map must be a CUDA tensor");
    }
    if (pixel_error_map.scalar_type() != torch::kFloat32) {
      AT_ERROR("pixel_error_map must be float32");
    }

    // 允许 HxW 或 1xHxW
    if (pixel_error_map.ndimension() == 2) {
      if (pixel_error_map.size(0) != H || pixel_error_map.size(1) != W) {
        AT_ERROR("pixel_error_map must have shape (H, W)");
      }
      pem = pixel_error_map.contiguous();
    } else if (pixel_error_map.ndimension() == 3) {
      if (pixel_error_map.size(0) != 1 || pixel_error_map.size(1) != H || pixel_error_map.size(2) != W) {
        AT_ERROR("pixel_error_map must have shape (1, H, W) if 3D");
      }
      pem = pixel_error_map.view({H, W}).contiguous();
    } else {
      AT_ERROR("pixel_error_map must be 2D (H,W) or 3D (1,H,W)");
    }
    pixel_ptr = pem.data_ptr<float>();
  }
    // ===== 输入：cov3D_precomp（来自 Python，可为空）=====
  const float* cov_ptr = nullptr;
  bool use_cov = (cov3D_precomp.defined() && cov3D_precomp.numel() > 0);

  torch::Tensor cov_contig; // 保证 contiguous tensor 生命周期
  if (use_cov) {
    if (!cov3D_precomp.is_cuda()) {
      AT_ERROR("cov3D_precomp must be a CUDA tensor");
    }
    if (cov3D_precomp.scalar_type() != torch::kFloat32) {
      AT_ERROR("cov3D_precomp must be float32");
    }
    // 期望 shape: (P, 6) 或 (P*6,)
    cov_contig = cov3D_precomp.contiguous();
    cov_ptr = cov_contig.data_ptr<float>();
  }
    // ===== 输入：scales / rotations（可为空，但要保证 device/dtype 正确）=====
  const float* scales_ptr = nullptr;
  const float* rot_ptr    = nullptr;

  bool use_scales = (scales.defined() && scales.numel() > 0);
  bool use_rots   = (rotations.defined() && rotations.numel() > 0);

  torch::Tensor scales_contig;
  torch::Tensor rots_contig;

  if (use_scales) {
    if (!scales.is_cuda()) {
      AT_ERROR("scales must be a CUDA tensor");
    }
    if (scales.scalar_type() != torch::kFloat32) {
      AT_ERROR("scales must be float32");
    }
    scales_contig = scales.contiguous();
    scales_ptr = scales_contig.data_ptr<float>();
  }

  if (use_rots) {
    if (!rotations.is_cuda()) {
      AT_ERROR("rotations must be a CUDA tensor");
    }
    if (rotations.scalar_type() != torch::kFloat32) {
      AT_ERROR("rotations must be float32");
    }
    rots_contig = rotations.contiguous();
    rot_ptr = rots_contig.data_ptr<float>();
  }

  // 如果 cov_ptr==nullptr，说明要在 CUDA 里用 scales/rotations 去算 cov3D，二者必须存在
  if (cov_ptr == nullptr) {
    if (scales_ptr == nullptr) AT_ERROR("scales must be provided (CUDA float32) when cov3D_precomp is empty");
    if (rot_ptr == nullptr)    AT_ERROR("rotations must be provided (CUDA float32) when cov3D_precomp is empty");
  }

  // Buffers (unchanged)
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int rendered = 0;
  if (P != 0)
  {
    rendered = CudaRasterizer::Rasterizer::forward(
        geomFunc,
        binningFunc,
        imgFunc,
        P,
        W, H,
        means3D.contiguous().data_ptr<float>(),
        opacity.contiguous().data_ptr<float>(),
        scales_ptr,
        scale_modifier,
        rot_ptr,
        cov_ptr,
        viewmatrix.contiguous().data_ptr<float>(),
        projmatrix.contiguous().data_ptr<float>(),
        campos.contiguous().data_ptr<float>(),
        tan_fovx,
        tan_fovy,
        prefiltered,
        mode,
        pixel_ptr,   // 可能为 nullptr
        resp_ptr,    // 无 pixel_error_map 时为 nullptr，节省显存
        cnt_ptr,     // 同上
        out_color.contiguous().data_ptr<float>(),
        radii.contiguous().data_ptr<int>(),
        debug
    );
  }

  // ★ 返回 8 元组（多了 gaussian_err_count）
  return std::make_tuple(
      rendered,
      out_color,
      radii,
      gaussian_responsibility,
      gaussian_err_count,   // ★ 新增
      geomBuffer,
      binningBuffer,
      imgBuffer
  );
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int mode,
	const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dmu = torch::zeros({P, 1}, means3D.options()); 
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  // ===== 输入：cov3D_precomp（可为空）=====
  const float* cov_ptr = nullptr;
  bool use_cov = (cov3D_precomp.defined() && cov3D_precomp.numel() > 0);

  torch::Tensor cov_contig;
  if (use_cov) {
    if (!cov3D_precomp.is_cuda()) {
      AT_ERROR("cov3D_precomp must be a CUDA tensor");
    }
    if (cov3D_precomp.scalar_type() != torch::kFloat32) {
      AT_ERROR("cov3D_precomp must be float32");
    }
    cov_contig = cov3D_precomp.contiguous();
    cov_ptr = cov_contig.data_ptr<float>();
  }
   // ===== 输入：scales / rotations（可为空，但要保证 CUDA float32）=====
  const float* scales_ptr = nullptr;
  const float* rot_ptr    = nullptr;

  bool use_scales = (scales.defined() && scales.numel() > 0);
  bool use_rots   = (rotations.defined() && rotations.numel() > 0);

  torch::Tensor scales_contig;
  torch::Tensor rots_contig;

  if (use_scales) {
    if (!scales.is_cuda()) AT_ERROR("scales must be a CUDA tensor");
    if (scales.scalar_type() != torch::kFloat32) AT_ERROR("scales must be float32");
    scales_contig = scales.contiguous();
    scales_ptr = scales_contig.data_ptr<float>();
  }

  if (use_rots) {
    if (!rotations.is_cuda()) AT_ERROR("rotations must be a CUDA tensor");
    if (rotations.scalar_type() != torch::kFloat32) AT_ERROR("rotations must be float32");
    rots_contig = rotations.contiguous();
    rot_ptr = rots_contig.data_ptr<float>();
  }

  // backward 一般也需要 scales/rotations（它们参与梯度计算）
  // 如果你允许某些路径为空，那就按你的逻辑放宽；否则建议强制必须存在：
  if (scales_ptr == nullptr) AT_ERROR("scales must be provided as CUDA float32 tensor");
  if (rot_ptr == nullptr)    AT_ERROR("rotations must be provided as CUDA float32 tensor");
  if(P != 0)
  {   
	  CudaRasterizer::Rasterizer::backward(P, R,
	  W, H, 
	  means3D.contiguous().data<float>(),
	  scales_ptr,
	  scale_modifier,
	  rot_ptr,
	  cov_ptr,
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dmu.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  mode,
	  debug);
  }
  return std::make_tuple(dL_dmeans2D, dL_dopacity, dL_dmu, dL_dmeans3D, dL_dcov3D, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}

