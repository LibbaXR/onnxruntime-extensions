// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"
#include "kernels/kernels.h"

#include <vector>
#include <cmath>

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi& ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

struct KernelOne {
  KernelOne(OrtApi api)
      : api_(api),
        ort_(api_) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
    const float* X = ort_.GetTensorData<float>(input_X);
    const float* Y = ort_.GetTensorData<float>(input_Y);

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_X);

    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    float* out = ort_.GetTensorMutableData<float>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // Do computation
    for (int64_t i = 0; i < size; i++) {
      out[i] = X[i] + Y[i];
    }
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};

struct KernelTwo {
  KernelTwo(OrtApi api)
      : api_(api),
        ort_(api_) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const float* X = ort_.GetTensorData<float>(input_X);

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_X);

    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    int32_t* out = ort_.GetTensorMutableData<int32_t>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // Do computation
    for (int64_t i = 0; i < size; i++) {
      out[i] = (int32_t)(round(X[i]));
    }
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};


struct CustomOpOne : Ort::CustomOpBase<CustomOpOne, KernelOne> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
    return new KernelOne(api);
  };

  const char* GetName() const { return "CustomOpOne"; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

} c_CustomOpOne;

struct CustomOpTwo : Ort::CustomOpBase<CustomOpTwo, KernelTwo> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
    return new KernelTwo(api);
  };

  const char* GetName() const { return "CustomOpTwo"; };

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; };

} c_CustomOpTwo;

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain("test.customop", &domain)) {
    return status;
  }
  else {
    if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomOpOne)) {
      return status;
    }

    if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomOpTwo)) {
      return status;
    }

    if (auto status = ortApi->AddCustomOpDomain(options, domain)) {
      return status;
    }
  }

  domain = nullptr;
  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

  auto custom_op_list = GetCustomOpSchemaList();
  for (auto it = custom_op_list; *it != nullptr; ++it) {
    auto obj_ptr = (*it)();
    // TODO: it doesn't make sense ORT needs non-const OrtCustomOp object, will fix in new ORT release
    OrtCustomOp * op_ptr = const_cast<OrtCustomOp *>(obj_ptr);
    if (auto status = ortApi->CustomOpDomain_Add(domain, op_ptr)) {
      return status;
    }
  }

#if defined(PYTHON_OP_SUPPORT)
  size_t count = 0;
  auto c_ops = FetchPyCustomOps(count);
  for (size_t n = 0; n < count; ++n){
    // TODO: it doesn't make sense ORT needs non-const OrtCustomOp object, will fix in new ORT release
    OrtCustomOp * op_ptr = const_cast<OrtCustomOp *>(c_ops+n);
    if (auto status = ortApi->CustomOpDomain_Add(domain, op_ptr)) {
      return status;
    }
  }
#endif

  return ortApi->AddCustomOpDomain(options, domain);
}