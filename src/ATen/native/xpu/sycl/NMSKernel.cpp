#include <ATen/AccumulateType.h>
#include <comm/SYCLContext.h>
#include <comm/xpu_aten.h>

#include <ATen/native/mkldnn/xpu/detail/oneDNN.h>
#include <ATen/native/xpu/sycl/NMSKernel.h>
#include <torch/library.h>

using namespace at::native::onednn;

namespace at {
namespace native {
namespace xpu {

constexpr int nms_items_per_group = sizeof(unsigned long long) * 8;

template <typename scalar_t>
inline bool dev_iou(
    scalar_t const* const a,
    scalar_t const* const b,
    const float threshold) {
  scalar_t left = std::max(a[0], b[0]), right = std::min(a[2], b[2]);
  scalar_t top = std::max(a[1], b[1]), bottom = std::min(a[3], b[3]);
  scalar_t width = std::max(right - left, (scalar_t)0),
           height = std::max(bottom - top, (scalar_t)0);
  using acc_t = acc_type_device<scalar_t, kXPU>;
  acc_t area_inter = (acc_t)width * height;
  acc_t area_a = ((acc_t)a[2] - a[0]) * (a[3] - a[1]);
  acc_t area_b = ((acc_t)b[2] - b[0]) * (b[3] - b[1]);
  return (area_inter / (area_a + area_b - area_inter)) > threshold;
}

template <typename scalar_t, typename acc_t>
struct NMSKernelFunctor : public __SYCL_KER_CONFIG_CONVENTION__ {
  void operator()(sycl::nd_item<2> item) const {
    int row_start = item.get_group(0);
    int col_start = item.get_group(1);

    if (row_start > col_start)
      return;

    const int row_size = std::min(
        dets_num_ - row_start * nms_items_per_group, nms_items_per_group);
    const int col_size = std::min(
        dets_num_ - col_start * nms_items_per_group, nms_items_per_group);

    auto block_boxes =
        (scalar_t*)(slm_.template get_multi_ptr<sycl::access::decorated::no>()
                        .get()); // nms_items_per_group * 4
    if (item.get_local_id(1) < col_size) {
      block_boxes[item.get_local_id(1) * 4 + 0] = dets_sorted_ptr_
          [(nms_items_per_group * col_start + item.get_local_id(1)) * 4 + 0];
      block_boxes[item.get_local_id(1) * 4 + 1] = dets_sorted_ptr_
          [(nms_items_per_group * col_start + item.get_local_id(1)) * 4 + 1];
      block_boxes[item.get_local_id(1) * 4 + 2] = dets_sorted_ptr_
          [(nms_items_per_group * col_start + item.get_local_id(1)) * 4 + 2];
      block_boxes[item.get_local_id(1) * 4 + 3] = dets_sorted_ptr_
          [(nms_items_per_group * col_start + item.get_local_id(1)) * 4 + 3];
    }
    item.barrier(sycl_local_fence);

    if (item.get_local_id(1) < row_size) {
      const int cur_box_idx =
          nms_items_per_group * row_start + item.get_local_id(1);
      const scalar_t* cur_box = dets_sorted_ptr_ + cur_box_idx * 4;
      int i = 0;
      unsigned long long t = 0;
      int start = 0;
      if (row_start == col_start) {
        start = item.get_local_id(1) + 1;
      }
      for (i = start; i < col_size; i++) {
        if (dev_iou<scalar_t>(cur_box, block_boxes + i * 4, iou_threshold_)) {
          t |= 1ULL << i;
        }
      }
      const int col_blocks =
          (dets_num_ + nms_items_per_group - 1) / nms_items_per_group;
      mask_ptr_[cur_box_idx * col_blocks + col_start] = t;
    }
  }
  NMSKernelFunctor(
      int dets_num,
      float iou_threshold,
      scalar_t* dets_sorted_ptr,
      unsigned long long* mask_ptr)
      : dets_num_(dets_num),
        iou_threshold_(iou_threshold),
        dets_sorted_ptr_(dets_sorted_ptr),
        mask_ptr_(mask_ptr) {}

  void sycl_ker_config_convention(sycl::handler& cgh) {
    slm_ = sycl_local_acc_t<acc_t>(nms_items_per_group * 4, cgh);
  }

 private:
  int dets_num_;
  float iou_threshold_;
  scalar_t* dets_sorted_ptr_;
  unsigned long long* mask_ptr_;
  sycl_local_acc_t<acc_t> slm_;
};

Tensor nms_kernel(const Tensor& dets_sorted, float iou_threshold) {
  int dets_num = dets_sorted.size(0);
  int col_blocks = (dets_num + nms_items_per_group - 1) / nms_items_per_group;
  auto mask = at::empty(
      {dets_num * col_blocks}, dets_sorted.options().dtype(at::kLong));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      dets_sorted.scalar_type(),
      "nms_kernel",
      [&] {
        sycl::range<2> global_range{
            (size_t)col_blocks, (size_t)col_blocks * nms_items_per_group};
        sycl::range<2> local_range{1, (size_t)nms_items_per_group};
        using acc_t = acc_type_device<scalar_t, kXPU>;
        auto dets_sorted_ptr = dets_sorted.data_ptr<scalar_t>();
        auto mask_ptr = (unsigned long long*)mask.data_ptr<int64_t>();
        auto caller = NMSKernelFunctor<scalar_t, acc_t>(
            dets_num, iou_threshold, dets_sorted_ptr, mask_ptr);
        sycl_kernel_submit(
            global_range, local_range, at::xpu::getCurrentSYCLQueue(), caller);
      });
  return mask;
}

// template <typename scalar_t>
// struct TestLL : public __SYCL_KER_CONFIG_CONVENTION__ {
//   void operator()(sycl::nd_item<3> item) const {
//     auto global_x = item.get_global_id(0);
//     auto global_y = item.get_global_id(1);
//     auto global_z = item.get_global_id(2);

//     // auto _out = sycl::stream(65536, 256, cgh);
//     // out << "global_x is: " << global_x << " " << global_y << " " << global_z
//     //     << sycl::endl;
//     if (global_z < 1024) {
//       out_ptr[global_z] = act_ptr[global_z] + weight_ptr[global_z];
//     }
//   }

//   void sycl_ker_config_convention(sycl::handler& cgh) {
//     // local_data_ = sycl_local_acc_t<accscalar_t, 3>(
//     //     sycl::range<3>{
//     //         (size_t)block_row_, (size_t)local_size_, (size_t)vec_size},
//     //     cgh);
//     // _out = sycl::stream(65536, 256, cgh);
//   }

//   TestLL(scalar_t* act_ptr, scalar_t* weight_ptr, scalar_t* out_ptr)
//       : act_ptr(act_ptr), weight_ptr(weight_ptr), out_ptr(out_ptr) {
//     // auto& queue = at::xpu::getCurrentSYCLQueue();
//   }

//  private:
//   scalar_t* act_ptr;
//   scalar_t* weight_ptr;
//   scalar_t* out_ptr;
// };

// static Tensor test_ll(Tensor act, Tensor weight) {
//   sycl::range<3> global_range{(size_t)1, (size_t)1, (size_t)1024};
//   sycl::range<3> local_range{(size_t)1, (size_t)1, (size_t)32};

//   std::cout << "---- inside test_ll ----" << std::endl;

//   Tensor out = at::empty_like(act);

//   AT_DISPATCH_FLOATING_TYPES_AND2(
//       at::ScalarType::BFloat16,
//       at::ScalarType::Half,
//       act.scalar_type(),
//       "test_ll_kernel",
//       [&] {
//         auto& queue = at::xpu::getCurrentSYCLQueue();
//         auto act_ptr = act.data_ptr<scalar_t>();
//         auto weight_ptr = weight.data_ptr<scalar_t>();
//         auto out_ptr = out.data_ptr<scalar_t>();
//         auto kfn = TestLL<scalar_t>(act_ptr, weight_ptr, out_ptr);
//         std::cout << "---- start to submit ----" << std::endl;
//         sycl_kernel_submit(global_range, local_range, queue, kfn);
//       });

//   return out;
// }

// TORCH_LIBRARY_IMPL(onednn, XPU, m) {
//   m.impl(TORCH_SELECTIVE_NAME("onednn::test_ll"), TORCH_FN(test_ll));
// }

} // namespace xpu
} // namespace native
} // namespace at
