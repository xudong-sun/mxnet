// post_detection_op
// PostDetectionOp

#ifndef MXNET_OPERATOR_PostDetectionOp_INL_H_
#define MXNET_OPERATOR_PostDetectionOp_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cmath>
#include <map>
#include <vector>
#include <string>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace post_detection_op_enum {
enum PostDetectionOpInputs {kRois, kScores, kBbox_deltas, kIm_info};
enum PostDetectionOpOutputs {kBatch_boxes, kBatch_boxes_rois};
enum PostDetectionOpResource {kTempSpace};
}

struct PostDetectionOpParam : public dmlc::Parameter<PostDetectionOpParam> {
  float thresh;
  int n_classes;
  float nms_thresh_lo, nms_thresh_hi;
  DMLC_DECLARE_PARAMETER(PostDetectionOpParam) {
  	DMLC_DECLARE_FIELD(thresh)
      .set_default(0.9).describe("Threshold.");
  	DMLC_DECLARE_FIELD(nms_thresh_lo)
      .set_default(0.3).describe("Lower bound of NMS.");
    DMLC_DECLARE_FIELD(nms_thresh_hi)
      .set_default(0.5).describe("Higher bound of NMS.");
  }
};

template<typename xpu, typename DType>
class PostDetectionOp : public Operator {
 public:
  explicit PostDetectionOp(PostDetectionOpParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(in_data.size(), 4);
    CHECK_EQ(out_data.size(), 2);
    CHECK_EQ(req.size(), 2);
    CHECK_EQ(req[post_detection_op_enum::kBatch_boxes], kWriteTo);
    CHECK_EQ(req[post_detection_op_enum::kBatch_boxes_rois], kWriteTo);

    int batch_size = in_data[post_detection_op_enum::kScores].size(0);
    int n_classes = in_data[post_detection_op_enum::kScores].size(1);

    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 2, DType> rois =
      in_data[post_detection_op_enum::kRois].get<xpu, 2, DType>(s);
    Tensor<xpu, 3, DType> scores = in_data[post_detection_op_enum::kScores].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> bbox_deltas = in_data[post_detection_op_enum::kBbox_deltas].get<xpu, 3, DType>(s);

    Tensor<xpu, 2, DType> im_info = in_data[post_detection_op_enum::kIm_info].get<xpu, 2, DType>(s);

    Tensor<xpu, 3, DType> batch_boxes = out_data[post_detection_op_enum::kBatch_boxes].get<xpu, 3, DType>(s);
    Tensor<xpu, 2, DType> batch_boxes_rois = out_data[post_detection_op_enum::kBatch_boxes_rois].get<xpu, 2, DType>(s);
#if defined(__CUDACC__)
    CHECK_EQ(s->blas_handle_ownership_, Stream<xpu>::OwnHandle)
        << "Must init CuBLAS handle in stream";
#endif
    PostDetctionForward(
      rois, scores, bbox_deltas, im_info,// inputs
      batch_boxes, batch_boxes_rois, // outputs
      this->param_
    );
    if (ctx.is_train) {
    	LOG(FATAL) << "Should use in test mode only.";
    }
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
	  LOG(FATAL) << "PostDetectionOp Backward Not Implemented.";
  }

 private:
  PostDetectionOpParam param_;
};  // class LSoftmaxOp

template<typename xpu>
Operator *CreateOp(PostDetectionOpParam param, int dtype);

#if DMLC_USE_CXX11
	class PostDetectionOpProp : public OperatorProperty {
	public:
	  void Init(const std::vector<std::pair<std::string, std::string> > &kwargs) override {
      // std::cout << "Init ... ";
	    param_.Init(kwargs);
      // std::cout << " Done" << std::endl;
	  }

    std::map<std::string, std::string> GetParams() const override {
      // std::cout << "GetParams" << std::endl;
 	   	return param_.__DICT__();
  	}

	  std::vector<std::string> ListArguments() const override {
    	return {"rois", "scores", "bbox_deltas", "im_info"};
  	}

	  std::vector<std::string> ListOutputs() const override {
    	return {"batch_boxes", "batch_boxes_rois"};
  	}

	  int NumOutputs() const override {
	    return 2;
	  }

	  int NumVisibleOutputs() const override {
	    return 2;
	  }

    bool InferShape(std::vector<TShape> *in_shape,
	                  std::vector<TShape> *out_shape,
	                  std::vector<TShape> *aux_shape) const override {
	    using namespace mshadow;
	    CHECK_EQ(in_shape->size(), 4) << "Input:[rois, scores, bbox_deltas]";
	    const TShape &roi_shape = in_shape->at(post_detection_op_enum::kRois);
	    const TShape &score_shape = in_shape->at(post_detection_op_enum::kScores);
			const TShape &bbox_deltas_shape = in_shape->at(post_detection_op_enum::kBbox_deltas);
			const TShape &im_info_shape = in_shape->at(post_detection_op_enum::kIm_info);


	    CHECK_EQ(roi_shape.ndim(), 2) << "roi_shape should be (batch_size*img_rois, 5)";
	    CHECK_EQ(score_shape.ndim(), 3) << "score_shape should be (batch_size, img_rois, num_cls)";
	    CHECK_EQ(bbox_deltas_shape.ndim(), 3) << "score_shape should be (batch_size, img_rois, 4 * num_cls)";
	    CHECK_EQ(im_info_shape.ndim(), 2) << "score_shape should be (batch_size, 3)";

	    const int batch_size = score_shape[0];
      const int img_rois = score_shape[1];
	    out_shape->clear();
	    out_shape->push_back(Shape3(batch_size, img_rois, 6));  // batch_boxes
      out_shape->push_back(Shape2(batch_size * img_rois, 5));  // batch_boxes_rois
	    aux_shape->clear();
	    return true;
	  }

 	  std::string TypeString() const override {
		  return "PostDetection";
		}

	  OperatorProperty *Copy() const override {
	    auto ptr = new PostDetectionOpProp();
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
  	PostDetectionOpParam param_;
	};
#endif // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif // MXNET_OPERATOR_PostDetectionOp_INL_H_
