/*!
 * Copyright (c) 2018 by TuSimple
 * \file proposal_target.cc
 * \brief C++ version proposal target
 * \author Tian Li, Yuntao Chen, Zehao Huang
 */

#include <algorithm>
#include <cstdlib>
#include "./proposal_mask_target-inl.h"
#include "../coco_api/common/maskApi.h"
using std::min;
using std::max;
using std::vector;
using std::begin;
using std::end;
using std::random_shuffle;
using std::log;

namespace mshadow{

template <typename DType>
void convertPoly2Mask(const Tensor<cpu, 2, DType> &rois,
                      const Tensor<cpu, 2, DType> &gt_polys,
                      Tensor<cpu, 4, DType> &mask_targets,
                      const vector<DType> &gt_assignment,
                      const vector<index_t> &kept_indexes,
                      const index_t mask_size,
                      const index_t fg_rois_this_image){
     /* !
     Converts a polygon to a pre-defined mask wrt to an roi
     *****Inputs****
     roi: The RoI bounding box
     poly: The polygon points the pre-defined format(see below)
     mask_size: The mask size
     *****Outputs****
     overlap: overlap of each box in boxes1 to each box in boxes2
     */
  for (index_t i = 0; i < fg_rois_this_image; ++i) {
    TensorContainer<cpu, 1, DType> poly(Shape1(gt_polys.size(1)), 0.f);
    TensorContainer<cpu, 1, DType> roi(Shape1(rois.size(1)), 0.f);
    Copy(poly, gt_polys[gt_assignment[kept_indexes[i]]]);
    Copy(roi, rois[i]);
    DType w = roi[3] - roi[1];
    DType h = roi[4] - roi[2];
    w = max((DType)1., w);
    h = max((DType)1., h);
    index_t category = static_cast<index_t>(poly[0]);
    index_t n_seg = static_cast<index_t>(poly[1]);

    if(!n_seg) return;

    index_t offset = 2 + n_seg;

    RLE* rles;
    rlesInit(&rles, n_seg);

    for(index_t k = 0; k < n_seg; k++) {
      index_t cur_len = poly[k + 2];
      double* xys = reinterpret_cast<double*>(malloc(sizeof(double) * cur_len));
      for(index_t j = 0; j < cur_len; j++){
        if (j % 2 == 0)
          xys[j] = (poly[offset+j+1] - roi[2]) * mask_size / h;
        else
          xys[j] = (poly[offset+j-1] - roi[1]) * mask_size / w;
      }
      rleFrPoly(rles + k, xys, cur_len / 2, mask_size, mask_size);
      free(xys);
      offset += cur_len;
    }

      // Decode RLE to mask
    byte* byte_mask = reinterpret_cast<byte*>(malloc(mask_size * mask_size * n_seg * sizeof(byte)));
    rleDecode(rles, byte_mask, n_seg);

    // Flatten mask
    for(index_t j = 0; j < mask_size; j++) {
      for(index_t k = 0; k < mask_size; k++) {
        DType cur_byte = 0;
        for(index_t l = 0; l < n_seg; l++){
          index_t offset = l * mask_size * mask_size + j * mask_size + k;
          if(byte_mask[offset] == 1){
            cur_byte = 1;
            break;
          }
        }
       mask_targets[i][category][j][k] = cur_byte;
      }   
    }
    // Check to make sure we don't have memory leak
    rlesFree(&rles, n_seg);
    free(byte_mask);
  }
} // convertPoly2Mask

template <typename DType>
inline void SampleROIMask(const Tensor<cpu, 2, DType> &all_rois,
                          const Tensor<cpu, 2, DType> &gt_boxes,
                          const Tensor<cpu, 2, DType> &gt_polys,
                          const Tensor<cpu, 1, DType> &bbox_mean,
                          const Tensor<cpu, 1, DType> &bbox_std,
                          const Tensor<cpu, 1, DType> &bbox_weight,
                          const index_t fg_rois_per_image,
                          const index_t rois_per_image,
                          const index_t num_classes,
                          const index_t mask_size,
                          const float fg_thresh,
                          const float bg_thresh_hi,
                          const float bg_thresh_lo,
                          const bool empty_flag,
                          Tensor<cpu, 2, DType> &&rois,
                          Tensor<cpu, 1, DType> &&labels,
                          Tensor<cpu, 2, DType> &&bbox_targets,
                          Tensor<cpu, 2, DType> &&bbox_weights,
                          Tensor<cpu, 4, DType> &&mask_targets) {

  vector<DType> max_overlaps(all_rois.size(0), 0.f);
  vector<DType> all_labels(all_rois.size(0), 0.f);
  vector<DType> gt_assignment(all_rois.size(0), 0.f);

  if (!empty_flag) {
    TensorContainer<cpu, 2, DType> IOUs(Shape2(all_rois.size(0), gt_boxes.size(0)), 0.f);
    BBoxOverlap(all_rois, gt_boxes, IOUs);
    for (index_t i = 0; i < IOUs.size(0); ++i) {
        DType max_value = IOUs[i][0];
        index_t max_index = 0;
        for (index_t j = 1; j < IOUs.size(1); ++j) {
          if (max_value < IOUs[i][j]) {
              max_value = IOUs[i][j];
              max_index = j;
          }
        }
        gt_assignment[i] = max_index;
        max_overlaps[i] = max_value;
        all_labels[i] = gt_boxes[max_index][4];
    }
  }

  vector<index_t> fg_indexes;
  vector<index_t> neg_indexes;
  for (index_t i = 0; i < max_overlaps.size(); ++i) {
    if (max_overlaps[i] >= fg_thresh) {
      fg_indexes.push_back(i);
    } else {
      neg_indexes.push_back(i);
    }
  }
  index_t fg_rois_this_image = min<index_t>(fg_rois_per_image, fg_indexes.size());
  if (fg_indexes.size() > fg_rois_this_image) {
    random_shuffle(begin(fg_indexes), end(fg_indexes));
    fg_indexes.resize(fg_rois_this_image);
  }

  vector<index_t> bg_indexes;
  for (index_t i = 0; i < max_overlaps.size(); ++i) {
    if (max_overlaps[i] >= bg_thresh_lo && max_overlaps[i] < bg_thresh_hi) {
        bg_indexes.push_back(i);
    }
  }
  index_t bg_rois_this_image = min<index_t>(rois_per_image - fg_rois_this_image, bg_indexes.size());
  if (bg_indexes.size() > bg_rois_this_image) {
      random_shuffle(begin(bg_indexes), end(bg_indexes));
      bg_indexes.resize(bg_rois_this_image);
  }

  vector<index_t> kept_indexes;

  for (index_t i = 0; i < fg_rois_this_image; ++i)
      kept_indexes.push_back(fg_indexes[i]);

  for (index_t i = 0; i < bg_rois_this_image; ++i)
      kept_indexes.push_back(bg_indexes[i]);

  if (kept_indexes.size() < rois_per_image) {
      index_t gap = rois_per_image - kept_indexes.size();
      random_shuffle(begin(neg_indexes), end(neg_indexes));
      neg_indexes.resize(gap);
      for (auto idx: neg_indexes)
          kept_indexes.push_back(idx);
  }

  for (index_t i = 0; i < kept_indexes.size(); ++i) {
    if (i < fg_rois_this_image)
      labels[i] = all_labels[kept_indexes[i]];
    Copy(rois[i], all_rois[kept_indexes[i]]);
  }


  TensorContainer<cpu, 2, DType> rois_tmp(Shape2(rois.size(0), 4));
  for (index_t i = 0; i < rois_tmp.size(0); ++i)
    Copy(rois_tmp[i], rois[i].Slice(1, 5));

  TensorContainer<cpu, 2, DType> gt_bboxes_tmp(Shape2(rois.size(0), 4));
  for (index_t i = 0; i < rois_tmp.size(0); ++i)
    Copy(gt_bboxes_tmp[i], gt_boxes[gt_assignment[kept_indexes[i]]].Slice(0, 4));

  TensorContainer<cpu, 2, DType> targets(Shape2(rois.size(0), 4));
  NonLinearTransformAndNormalization(rois_tmp, gt_bboxes_tmp, targets, bbox_mean, bbox_std);

  TensorContainer<cpu, 2, DType> bbox_target_data(Shape2(targets.size(0), 5));
  for (index_t i = 0; i < bbox_target_data.size(0); ++i) {
    bbox_target_data[i][0] = labels[i];
    Copy(bbox_target_data[i].Slice(1, 5), targets[i]);
  }

  ExpandBboxRegressionTargets(bbox_target_data, bbox_targets, bbox_weights, num_classes, bbox_weight);

  convertPoly2Mask(rois, gt_polys, mask_targets, gt_assignment, kept_indexes, mask_size, fg_rois_this_image);
}

template <typename DType>
void BBoxOverlap(const Tensor<cpu, 2, DType> &boxes,
                 const Tensor<cpu, 2, DType> &query_boxes,
                 Tensor<cpu, 2, DType> &overlaps) {
    const index_t n = boxes.size(0);
    const index_t k = query_boxes.size(0);
    for (index_t j = 0; j < k; ++j) {
        DType query_box_area = (query_boxes[j][2] - query_boxes[j][0] + 1.f) * (query_boxes[j][3] - query_boxes[j][1] + 1.f);
        for (index_t i = 0; i < n; ++i) {
            DType iw = min(boxes[i][3], query_boxes[j][2]) - max(boxes[i][1], query_boxes[j][0]) + 1.f;
            if (iw > 0) {
                DType ih = min(boxes[i][4], query_boxes[j][3]) - max(boxes[i][2], query_boxes[j][1]) + 1.f;
                if (ih > 0) {
                    DType box_area = (boxes[i][3] - boxes[i][1] + 1.f) * (boxes[i][4] - boxes[i][2] + 1.f);
                    DType union_area = box_area + query_box_area - iw * ih;
                    overlaps[i][j] = iw * ih / union_area;
                }
            }
        }
    }
}

template <typename DType>
void ExpandBboxRegressionTargets(const Tensor<cpu, 2, DType> &bbox_target_data,
                                 Tensor<cpu, 2, DType> &bbox_targets,
                                 Tensor<cpu, 2, DType> &bbox_weights,
                                 const index_t num_class,
                                 const Tensor<cpu, 1, DType> &bbox_weight) {
  index_t num_bbox = bbox_target_data.size(0);
  for (index_t i = 0; i < num_bbox; ++i) {
    if (bbox_target_data[i][0] > 0) {
      index_t cls = bbox_target_data[i][0];
      index_t start = 4 * cls;
      index_t end = start + 4;
      Copy(bbox_targets[i].Slice(start, end), bbox_target_data[i].Slice(1, 5));
      Copy(bbox_weights[i].Slice(start, end), bbox_weight);
    }
  }
}

template <typename DType>
void NonLinearTransformAndNormalization(const Tensor<cpu, 2, DType> &ex_rois,
                                        const Tensor<cpu, 2, DType> &gt_rois,
                                        Tensor<cpu, 2, DType> &targets,
                                        const Tensor<cpu, 1, DType> &bbox_mean,
                                        const Tensor<cpu, 1, DType> &bbox_std) {
  index_t num_roi = ex_rois.size(0);
  for (index_t i = 0; i < num_roi; ++i) {
      DType ex_width  = ex_rois[i][2] - ex_rois[i][0] + 1.f;
      DType ex_height = ex_rois[i][3] - ex_rois[i][1] + 1.f;
      DType ex_ctr_x  = ex_rois[i][0] + 0.5 * (ex_width - 1.f);
      DType ex_ctr_y  = ex_rois[i][1] + 0.5 * (ex_height - 1.f);
      DType gt_width  = gt_rois[i][2] - gt_rois[i][0] + 1.f;
      DType gt_height = gt_rois[i][3] - gt_rois[i][1] + 1.f;
      DType gt_ctr_x  = gt_rois[i][0] + 0.5 * (gt_width - 1.f);
      DType gt_ctr_y  = gt_rois[i][1] + 0.5 * (gt_height - 1.f);
      targets[i][0]   = (gt_ctr_x - ex_ctr_x) / (ex_width + 1e-14f);
      targets[i][1]   = (gt_ctr_y - ex_ctr_y) / (ex_height + 1e-14f);
      targets[i][2]   = log(gt_width / ex_width);
      targets[i][3]   = log(gt_height / ex_height);
      targets[i] -= bbox_mean;
      targets[i] /= bbox_std;
  }
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(ProposalMaskTargetParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ProposalMaskTargetOp<cpu, DType>(param);
  })
  return op;
}

template<>
Operator *CreateOp<gpu>(ProposalMaskTargetParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new ProposalMaskTargetOp<gpu, DType>(param);
  })
  return op;
}

Operator *ProposalMaskTargetProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                               std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(ProposalMaskTargetParam);

MXNET_REGISTER_OP_PROPERTY(ProposalMaskTarget, ProposalMaskTargetProp)
.describe("C++ version proposal target")
.add_argument("rois", "Symbol", "rois")
.add_argument("gt_boxes", "Symbol", "gtboxes")
.add_argument("gt_polys", "Symbol", "gtpolys")
.add_arguments(ProposalMaskTargetParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
