#include "./post_detection_op-inl.h"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <memory>

namespace mshadow {

template <typename DType>
void nonlinear_clip(
  const DType* boxes, const DType* box_deltas,
  DType* _pred_boxes,
  const DType im_w, const DType im_h,
  const int B, const int N, const int C) {
  for( int j=0; j<B*N; j++) {
    // nonlinear
    DType w = boxes[j*5+1+2] - boxes[j*5+1+0] + 1.0;
    DType h = boxes[j*5+1+3] - boxes[j*5+1+1] + 1.0;
    DType cx = boxes[j*5+1+0] + 0.5 * (w - 1.0);
    DType cy = boxes[j*5+1+1] + 0.5 * (h - 1.0);
    for( int c=0; c<C; c++) {
      int i=j*C+c;
      DType pred_cx = box_deltas[i*4+0] * w + cx;
      DType pred_cy = box_deltas[i*4+1] * h + cy;
      DType pred_w = std::exp(box_deltas[i*4+2]) * w;
      DType pred_h = std::exp(box_deltas[i*4+3]) * h;

      _pred_boxes[i*4+0] = pred_cx - 0.5 * (pred_w - 1.0);
      _pred_boxes[i*4+1] = pred_cy - 0.5 * (pred_h - 1.0);
      _pred_boxes[i*4+2] = pred_cx + 0.5 * (pred_w - 1.0);
      _pred_boxes[i*4+3] = pred_cy + 0.5 * (pred_h - 1.0);

      // clip
      _pred_boxes[i*4+0] = std::max(std::min(_pred_boxes[i*4+0], im_w-1), (DType)0.0);
      _pred_boxes[i*4+1] = std::max(std::min(_pred_boxes[i*4+1], im_h-1), (DType)0.0);
      _pred_boxes[i*4+2] = std::max(std::min(_pred_boxes[i*4+2], im_w-1), (DType)0.0);
      _pred_boxes[i*4+3] = std::max(std::min(_pred_boxes[i*4+3], im_h-1), (DType)0.0);
    }
  }
}

template <typename DType>
void _fore_back_enhance(const DType* scores, DType* ehanced_score,
  const int B, const int N, const int C) {
  for(int i=0; i<B*N; i++) {
    DType max_val = 0.0;
    for(int c=0; c<C; c++) {
      DType cur_val = scores[i*C+c];
      if(cur_val > max_val) max_val = cur_val;
    }
    DType sum_val = 0.0;
    for(int c=1; c<C; c++) {
      DType cur_val = scores[i*C+c];
      if(cur_val >= max_val) {
        ehanced_score[i*C+c] = cur_val;
      } else {
        ehanced_score[i*C+c] = 0.0;
      }
      sum_val += ehanced_score[i*C+c];
    }
    ehanced_score[i*C+0] = scores[i*C+0];
    sum_val += ehanced_score[i*C+0];
    for(int c=0; c<C; c++) {
      ehanced_score[i*C+c] /= sum_val;
    }
  }
}

template <typename DType>
void weighted_nms(
  const DType* boxes, const DType* scores, const int* cls, const int* _order, const int n_box_this_batch,
  const float thresh_lo, const float thresh_hi,
  DType* keep_out, int &keep_num ) {

  DType* areas = new DType[n_box_this_batch];
  for(int i=0; i<n_box_this_batch; i++) {
    areas[i] = (boxes[4*i+2]-boxes[4*i+0]+(DType)1.0) *
               (boxes[4*i+3]-boxes[4*i+1]+(DType)1.0);
  }
  DType* ovr = new DType[n_box_this_batch];
  int n_keep = n_box_this_batch;
  int first_remaining_object = 0;
  keep_num = 0;
  std::vector<int> order(_order, _order + n_box_this_batch);
  std::vector<int> inds;
  inds.reserve(n_box_this_batch);
  while( n_keep > 0 ) {
    inds.clear();
    int i=order[0];
    DType x1 = boxes[4*i+0]; DType x2 = boxes[4*i+2];
    DType y1 = boxes[4*i+1]; DType y2 = boxes[4*i+3];
    DType tmp = 0.0;
    DType avg_x1 = 0.0, avg_x2 = 0.0, avg_y1 = 0.0, avg_y2 = 0.0;
    for(unsigned int j=0; j<order.size(); j++) {
      int oj = order[j];
      DType xx1 = std::max((DType)x1, boxes[4*oj+0]);
      DType xx2 = std::min((DType)x2, boxes[4*oj+2]);
      DType yy1 = std::max((DType)y1, boxes[4*oj+1]);
      DType yy2 = std::min((DType)y2, boxes[4*oj+3]);
      DType w = std::max((DType)0.0, xx2-xx1+(DType)1.0);
      DType h = std::max((DType)0.0, yy2-yy1+(DType)1.0);
      DType inter = w*h;
      ovr[j] = inter / (areas[i] + areas[oj] - inter); // iou
      if( ovr[j] <= thresh_lo ) {
        inds.push_back(oj);
      } else if( ovr[j] > thresh_hi ) {
        DType score_j = scores[j];
        tmp += score_j;
        avg_x1 += score_j * xx1;
        avg_x2 += score_j * xx2;
        avg_y1 += score_j * yy1;
        avg_y2 += score_j * yy2;
        n_keep --;
      }
    }
    if( tmp == 0.0 ) break;
    keep_out[keep_num*6+0] = avg_x1 / tmp;
    keep_out[keep_num*6+1] = avg_y1 / tmp;
    keep_out[keep_num*6+2] = avg_x2 / tmp;
    keep_out[keep_num*6+3] = avg_y2 / tmp;
    keep_out[keep_num*6+4] = scores[i];
    keep_out[keep_num*6+5] = cls[i];
    keep_num++;
    order.clear();
    order.swap(inds);
  } // while( n_keep > 0 )
  delete[] areas;
  delete[] ovr;
}

template <typename DType>
inline void PostDetctionForward(const Tensor<cpu, 2, DType> &rois,
                                const Tensor<cpu, 3, DType> &scores,
                                const Tensor<cpu, 3, DType> &bbox_deltas,
                                const Tensor<cpu, 2, DType> &im_info,
                                const Tensor<cpu, 3, DType> &batch_boxes,
                                const Tensor<cpu, 2, DType> &batch_boxes_rois,
                                const mxnet::op::PostDetectionOpParam &params_) {
  // std::cout << "-- PostDetctionForward --" << std::endl;
  const DType *in_rois = rois.dptr_;
  const DType *in_scrs = scores.dptr_;
  const DType *in_bbdt = bbox_deltas.dptr_;
  DType* out_batch_boxes = batch_boxes.dptr_;
  DType* out_batch_boxes_rois = batch_boxes_rois.dptr_;
  int B = bbox_deltas.size(0); // batches
  int N = bbox_deltas.size(1); // rois
  int C = scores.size(2); // classes
  DType thresh = params_.thresh;
  memset(out_batch_boxes, 0, (B*N*6) * sizeof(DType));
  memset(out_batch_boxes_rois, 0, (B*N*5) * sizeof(DType));

  DType im_h = im_info[0][0];
  DType im_w = im_info[0][1];
  // nonlinear_pred + clipboxes
  DType* _pred_boxes = new DType[N*B*(4*C)];
  nonlinear_clip(in_rois, in_bbdt,  // in
    _pred_boxes,                    // out
    im_w, im_h, B, N, C); // params

  // fore_back_enhance
  DType* _enhance_scores = new DType[N*B*C];
  _fore_back_enhance(in_scrs, _enhance_scores, B, N, C);

  // each single batch
  int* _keep = new int[N];
  int* _class = new int[N];
  DType* _keep_score = new DType[N]; // store the score of each box;

  for(int b=0; b<B; b++) {
    memset(_keep, 0, N*sizeof(int));  // init as 0
    int n_box_this_batch = 0;
    for( int c=1; c<C; c++) { // skip background
      for( int n=0; n<N; n++) {
        int idx=b*N*C+n*C+c;
        if(_enhance_scores[idx] > thresh) {
          _keep[n] = 1;
          _keep_score[n] = _enhance_scores[idx];
          _class[n] = c;
          n_box_this_batch += 1;
        }
      }
    }
    // prepare data for NMS
    DType* _boxes_batch = new DType[n_box_this_batch*4];
    DType* _score_batch = new DType[n_box_this_batch];
    DType* _score_batch_copy = new DType[n_box_this_batch];
    int* _class_batch = new int[n_box_this_batch];
    int* _order_batch = new int[n_box_this_batch];
    int box_batch_idx = 0;
    for( int n=0; n<N; n++) {
      if(_keep[n] == 1) {
        int keep_idx = box_batch_idx*4;
        int class_idx = _class[n];
        int actual_idx = b*(4*C*N)+n*4*C+4*class_idx;
        for(int bb=0; bb<4; bb++) {
          _boxes_batch[keep_idx+bb] = _pred_boxes[actual_idx+bb];
        }
        _score_batch[box_batch_idx] = _keep_score[n];
        _score_batch_copy[box_batch_idx] = _keep_score[n];
        _order_batch[box_batch_idx] = box_batch_idx;
        _class_batch[box_batch_idx] = _class[n];
        box_batch_idx ++;
      }
    }

    // argsort
    thrust::stable_sort_by_key(thrust::host,
        _score_batch_copy,
        _score_batch_copy + n_box_this_batch,
        _order_batch,
        thrust::greater<float>());

    // nms for this batch
    // prepare output variables
    int keep_num;
    DType* _keep_out = new DType[6*N];
    weighted_nms(
      _boxes_batch, _score_batch, _class_batch, _order_batch, n_box_this_batch,
      params_.nms_thresh_lo, params_.nms_thresh_hi,
      _keep_out, keep_num);

    for(int k=0; k<keep_num; k++) {
      int out_idx = b * N + k;
      out_batch_boxes_rois[out_idx * 5 + 0] = b;
      for(int l=0; l<6; l++) {
        out_batch_boxes[out_idx * 6 + l] = _keep_out[k * 6 + l]; // x1, y1, x2, y2, score, cls
        if (l < 4) {
          out_batch_boxes_rois[out_idx * 5 + l + 1] = _keep_out[k * 6 + l]; // b, x1, y1, x2, y2,
        }
      }
    }

    delete[] _keep_out;
    delete[] _boxes_batch;
    delete[] _score_batch;
    delete[] _score_batch_copy;
    delete[] _order_batch;
  } // end of iterate through batch

  delete[] _keep;
  delete[] _class;

  delete[] _pred_boxes;
  delete[] _enhance_scores;
}

template <typename DType>
inline void PostDetctionBackward() {
  LOG(FATAL) << "Not Implemented.";
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<cpu>(PostDetectionOpParam param, int dtype) {
  // std::cout << "CreateOp<cpu>" << std::endl;
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new PostDetectionOp<cpu, DType>(param);
  })
  return op;
}

Operator *PostDetectionOpProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                                std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(PostDetectionOpParam);

MXNET_REGISTER_OP_PROPERTY(PostDetection, PostDetectionOpProp)
.describe("PostDetction")
.add_argument("rois", "Symbol", "rois")
.add_argument("scores", "Symbol", "scores")
.add_argument("bbox_deltas", "Symbol", "bbox_deltas")
.add_arguments(PostDetectionOpParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
