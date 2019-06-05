/*!
 * Copyright (c) 2018 by TuSimple
 * \file proposal_mask_target-inl.h
 * \brief C++ version proposal target
 * \author Tian Li, Yuntao Chen
 */
#ifndef MXNET_OPERATOR_PROPOSAL_MASK_TARGET_INL_H_
#define MXNET_OPERATOR_PROPOSAL_MASK_TARGET_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cmath>
#include <map>
#include <vector>
#include <string>
#include <memory>
#include "./operator_common.h"

#include <iostream>

namespace mxnet {
namespace op {

namespace proposal_mask_target_enum {
enum ProposalMaskTargetInputs {kRois, kGtBboxes, kGtPolys};
enum ProposalMaskTargetOutputs {kRoiOutput, kLabel, kBboxTarget, kBboxWeight, kMaskTarget};
}

struct ProposalMaskTargetParam : public dmlc::Parameter<ProposalMaskTargetParam> {
  index_t num_classes;
  index_t img_rois;
  index_t poly_len;
  index_t mask_size;
  float fg_fraction;
  float fg_thresh;
  float bg_thresh_hi;
  float bg_thresh_lo;
  bool proposal_without_gt;
  bool ohem;
  nnvm::Tuple<float> bbox_mean;
  nnvm::Tuple<float> bbox_std;
  nnvm::Tuple<float> bbox_weight;

  DMLC_DECLARE_PARAMETER(ProposalMaskTargetParam) {
    DMLC_DECLARE_FIELD(num_classes).describe("Number of classes for detection");
    DMLC_DECLARE_FIELD(img_rois).describe("Number of ROIs for one image");
    DMLC_DECLARE_FIELD(mask_size).describe("Size of mask target");
    DMLC_DECLARE_FIELD(fg_thresh).describe("Foreground IOU threshold");
    DMLC_DECLARE_FIELD(bg_thresh_hi).describe("Background IOU upper bound");
    DMLC_DECLARE_FIELD(bg_thresh_lo).describe("Background IOU lower bound");
    DMLC_DECLARE_FIELD(fg_fraction).set_default(0.25f).describe("Fraction of foreground proposals");
    DMLC_DECLARE_FIELD(proposal_without_gt).describe("Do not append ground-truth bounding boxes to output");
    DMLC_DECLARE_FIELD(ohem).set_default(false).describe("Do online hard sample mining");
    float tmp[] = {0.f, 0.f, 0.f, 0.f};
    DMLC_DECLARE_FIELD(bbox_mean).set_default(nnvm::Tuple<float>(tmp, tmp+4)).describe("Bounding box mean");
    tmp[0] = 0.1f; tmp[1] = 0.1f; tmp[2] = 0.2f; tmp[3] = 0.2f;
    DMLC_DECLARE_FIELD(bbox_std).set_default(nnvm::Tuple<float>(tmp, tmp+4)).describe("Bounding box std");
    tmp[0] = 1.f; tmp[1] = 1.f; tmp[2] = 1.f; tmp[3] = 1.f;
    DMLC_DECLARE_FIELD(bbox_weight).set_default(nnvm::Tuple<float>(tmp, tmp+4)).describe("Foreground bounding box weight");
  }
};

template<typename xpu, typename DType>
class ProposalMaskTargetOp : public Operator {
 public:
  explicit ProposalMaskTargetOp(ProposalMaskTargetParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 5);
    CHECK_EQ(req.size(), 5);
    CHECK_EQ(req[proposal_mask_target_enum::kRoiOutput], kWriteTo);
    CHECK_EQ(req[proposal_mask_target_enum::kLabel], kWriteTo);
    CHECK_EQ(req[proposal_mask_target_enum::kBboxTarget], kWriteTo);
    CHECK_EQ(req[proposal_mask_target_enum::kBboxWeight], kWriteTo);
    CHECK_EQ(req[proposal_mask_target_enum::kMaskTarget], kWriteTo);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    const index_t num_imgs = in_data[proposal_mask_target_enum::kRois].shape_[0];
    const index_t poly_len =in_data[proposal_mask_target_enum::kGtPolys].shape_[2];

    Tensor<xpu, 3, DType> xpu_rois      = in_data[proposal_mask_target_enum::kRois].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> xpu_gt_bboxes = in_data[proposal_mask_target_enum::kGtBboxes].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> xpu_gt_polys = in_data[proposal_mask_target_enum::kGtPolys].get<xpu, 3, DType>(s);

    TensorContainer<cpu, 3, DType> rois     (xpu_rois.shape_);
    TensorContainer<cpu, 3, DType> gt_bboxes(xpu_gt_bboxes.shape_);
    TensorContainer<cpu, 3, DType> gt_polys(xpu_gt_polys.shape_);

    Copy(rois, xpu_rois, s);
    Copy(gt_bboxes, xpu_gt_bboxes, s);
    Copy(gt_polys, xpu_gt_polys, s);

    std::vector<std::vector<Tensor<cpu, 1, DType>>> kept_rois;
    std::vector<std::vector<Tensor<cpu, 1, DType>>> kept_gtbboxes;
    std::vector<std::vector<Tensor<cpu, 1, DType>>> kept_gtpolys;

    // clean up bboxes
    for (index_t i = 0; i < num_imgs; ++i) {
        kept_gtbboxes.push_back(std::vector<Tensor<cpu, 1, DType>>());
        kept_gtpolys.push_back(std::vector<Tensor<cpu, 1, DType>>());
        for (index_t j = 0; j < gt_bboxes.size(1); ++j) {
            if (gt_bboxes[i][j][4] != -1) {
                kept_gtbboxes[i].push_back(gt_bboxes[i][j]);
                kept_gtpolys[i].push_back(gt_polys[i][j]);
            }
        }
        kept_rois.push_back(std::vector<Tensor<cpu, 1, DType>>());
        // all rois are kept
        for (index_t j = 0; j < rois.size(1); ++j) {
          kept_rois[i].push_back(rois[i][j]);
        }
        if (!param_.proposal_without_gt) {
          // all gt bboxes are appended
          for (index_t j = 0; j < kept_gtbboxes[i].size(); ++j) {
            kept_rois[i].push_back(kept_gtbboxes[i][j]);
          }
        }
    }

    const index_t _batch_rois = param_.img_rois * num_imgs;

    TensorContainer<cpu, 2, DType> cpu_output_rois(Shape2(_batch_rois, 5), 0.f);
    TensorContainer<cpu, 1, DType> cpu_labels(Shape1(_batch_rois), 0.f);
    TensorContainer<cpu, 2, DType> cpu_bbox_targets(Shape2(_batch_rois, param_.num_classes * 4), 0.f);
    TensorContainer<cpu, 2, DType> cpu_bbox_weights(Shape2(_batch_rois, param_.num_classes * 4), 0.f);
    TensorContainer<cpu, 4, DType> cpu_mask_targets(Shape4((index_t)(_batch_rois * param_.fg_fraction), param_.num_classes, param_.mask_size, param_.mask_size), -1.f);

    if (param_.ohem) {
        LOG(FATAL) << "OHEM not Implemented.";
    } else {
        index_t fg_rois_per_image = static_cast<index_t>(param_.img_rois * param_.fg_fraction);
        TensorContainer<cpu, 1, DType> bbox_mean(Shape1(4));
        TensorContainer<cpu, 1, DType> bbox_std(Shape1(4));
        TensorContainer<cpu, 1, DType> bbox_weight(Shape1(4));
        bbox_mean[0] = param_.bbox_mean[0];
        bbox_mean[1] = param_.bbox_mean[1];
        bbox_mean[2] = param_.bbox_mean[2];
        bbox_mean[3] = param_.bbox_mean[3];
        bbox_std[0] = param_.bbox_std[0];
        bbox_std[1] = param_.bbox_std[1];
        bbox_std[2] = param_.bbox_std[2];
        bbox_std[3] = param_.bbox_std[3];
        bbox_weight[0] = param_.bbox_weight[0];
        bbox_weight[1] = param_.bbox_weight[1];
        bbox_weight[2] = param_.bbox_weight[2];
        bbox_weight[3] = param_.bbox_weight[3];
        for (index_t i = 0; i < num_imgs; ++i) {
          bool empty_flag = false;
          index_t kept_gtbboxes_size_i = kept_gtbboxes[i].size();

          TensorContainer<cpu, 2, DType> kept_rois_i(Shape2(kept_rois[i].size(), rois.size(2)));
          for (index_t j = 0; j < kept_rois_i.size(0); ++j) {
              Copy(kept_rois_i[j], kept_rois[i][j]);
          }

          if (kept_gtbboxes_size_i == 0){
            kept_gtbboxes_size_i = 1;
            empty_flag = true;
          }

          TensorContainer<cpu, 2, DType> kept_gtbboxes_i(Shape2(kept_gtbboxes_size_i, rois.size(2)), 0.f);
          for (index_t j = 0; j < kept_gtbboxes[i].size(); ++j) {
            Copy(kept_gtbboxes_i[j], kept_gtbboxes[i][j]);
          }

          TensorContainer<cpu, 2, DType> kept_gtpolys_i(Shape2(kept_gtbboxes_size_i, poly_len), 0.f);
          for (index_t j = 0; j < kept_gtpolys[i].size(); ++j) {
            Copy(kept_gtpolys_i[j], kept_gtpolys[i][j]);
          }

          SampleROIMask(kept_rois_i,
                        kept_gtbboxes_i,
                        kept_gtpolys_i,
                        bbox_mean,
                        bbox_std,
                        bbox_weight,
                        fg_rois_per_image,
                        param_.img_rois,
                        param_.num_classes,
                        param_.mask_size,
                        param_.fg_thresh,
                        param_.bg_thresh_hi,
                        param_.bg_thresh_lo,
                        empty_flag,
                        cpu_output_rois.Slice(i * param_.img_rois, (i + 1) * param_.img_rois),
                        cpu_labels.Slice(i * param_.img_rois, (i + 1) * param_.img_rois),
                        cpu_bbox_targets.Slice(i * param_.img_rois, (i + 1) * param_.img_rois),
                        cpu_bbox_weights.Slice(i * param_.img_rois, (i + 1) * param_.img_rois),
                        cpu_mask_targets.Slice(i * fg_rois_per_image, (i + 1) * fg_rois_per_image));
        }
    }

    Tensor<xpu, 2, DType> xpu_output_rois  = out_data[proposal_mask_target_enum::kRoiOutput].get<xpu, 2, DType>(s);
    Tensor<xpu, 1, DType> xpu_labels       = out_data[proposal_mask_target_enum::kLabel].get<xpu, 1, DType>(s);
    Tensor<xpu, 2, DType> xpu_bbox_targets = out_data[proposal_mask_target_enum::kBboxTarget].get<xpu, 2, DType>(s);
    Tensor<xpu, 2, DType> xpu_bbox_weights = out_data[proposal_mask_target_enum::kBboxWeight].get<xpu, 2, DType>(s);
    Tensor<xpu, 4, DType> xpu_mask_targets = out_data[proposal_mask_target_enum::kMaskTarget].get<xpu, 4, DType>(s);

    Copy(xpu_output_rois, cpu_output_rois, s);
    Copy(xpu_labels, cpu_labels, s);
    Copy(xpu_bbox_targets, cpu_bbox_targets, s);
    Copy(xpu_bbox_weights, cpu_bbox_weights, s);
    Copy(xpu_mask_targets, cpu_mask_targets, s);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_grad.size(), 2);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3, DType> rois      = in_grad[proposal_mask_target_enum::kRois].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> gt_bboxes = in_grad[proposal_mask_target_enum::kGtBboxes].get<xpu, 3, DType>(s);

    rois = 0.f;
    gt_bboxes = 0.f;
  }

 private:
  ProposalMaskTargetParam param_;
};  // class ProposalMaskTargetOp

template<typename xpu>
Operator *CreateOp(ProposalMaskTargetParam param, int dtype);

#if DMLC_USE_CXX11
class ProposalMaskTargetProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> > &kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"rois", "gt_boxes", "gt_polys"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "label", "bbox_target", "bbox_weight", "mask_target"};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input:[rois, gt_boxes, gt_polys]";

    index_t num_imgs = in_shape->at(proposal_mask_target_enum::kRois)[0];
    index_t _batch_rois = num_imgs * param_.img_rois;

    auto output_rois_shape = Shape2(_batch_rois, 5);
    auto label_shape = Shape1(_batch_rois);
    auto bbox_target_shape = Shape2(_batch_rois, param_.num_classes * 4);
    auto bbox_weight_shape = Shape2(_batch_rois, param_.num_classes * 4);
    auto mask_target_shape = Shape4(static_cast<index_t>(_batch_rois * param_.fg_fraction), param_.num_classes, param_.mask_size, param_.mask_size);

    out_shape->clear();
    out_shape->push_back(output_rois_shape);
    out_shape->push_back(label_shape);
    out_shape->push_back(bbox_target_shape);
    out_shape->push_back(bbox_weight_shape);
    out_shape->push_back(mask_target_shape);
    aux_shape->clear();

    return true;
  }

  std::vector<int> DeclareBackwardDependency(
      const std::vector<int> &out_grad,
      const std::vector<int> &in_data,
      const std::vector<int> &out_data) const override {
    return {};
  }

  std::string TypeString() const override {
    return "ProposalMaskTarget";
  }

  OperatorProperty *Copy() const override {
    auto ptr = new ProposalMaskTargetProp();
    ptr->param_ = param_;
    return ptr;
  }

  Operator *CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator *CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  ProposalMaskTargetParam param_;
};  // class ProposalMaskTargetProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_proposal_mask_target_INL_H_
