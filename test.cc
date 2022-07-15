#include "manasi_build.h"
#include "tensorflow/lite/c/common.h"
#include <manasi_byoc_library/Support.hpp>
#include <manasi_driver_library/manasi_api.h>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/c/builtin_op_data.h"

namespace manasi1 {

void ManasiBuilder::AddInput(TfLiteTensor &input_tensor) {
    TfLiteIntArray *tensor_dims = input_tensor.dims;
    std::string tensor_name = input_tensor.name;
    const byoc_library::TensorShape TensorShape = {tensor_dims->data[0],
                                                   tensor_dims->data[1],
                                                   tensor_dims->data[2],
                                                   tensor_dims->data[3]};
    const byoc_library::DataFormat DataFormat = byoc_library::DataFormat::NHWC;       // data layout
    const byoc_library::DataType DataType = GetTensorBYOCDataType(input_tensor.type); //数据的类型 uint8
    int zero_point = input_tensor.params.zero_point;
    float scale = input_tensor.params.scale;

    byoc_library::QuantizationInfo QuantizationInfo(zero_point, scale); //量化的信息

    const byoc_library::TensorInfo inputTensorInfo = byoc_library::TensorInfo(TensorShape,
                                                                              DataType,
                                                                              DataFormat,
                                                                              QuantizationInfo);
    bl::TensorAndId<bl::Operand> inputOperandAndId = byoc_library::AddInput(m_Network, inputTensorInfo);
    convertedTensors.emplace(tensor_name, inputOperandAndId.tensor);
}

void ManasiBuilder::AddOutput(TfLiteTensor &output_tensor) {
    std::string output_tensor_name = output_tensor.name;
    std::shared_ptr<bl::Operand> outputop = convertedTensors[output_tensor_name];
    bl::TensorAndId<bl::Output> output = bl::AddOutput(m_Network, *outputop, bl::DataFormat::NHWC88);
}

std::shared_ptr<manasi::byoc_library::Constant> ManasiBuilder::AddWeight(TfLiteTensor &weight_tensor, bool depth) {
    byoc_library::DataFormat DataFormat = bl::DataFormat::NHWC;

    const byoc_library::DataType DataType = GetTensorBYOCDataType(weight_tensor.type);
    unsigned int out_channels = weight_tensor.dims->data[0];
    unsigned int kernel_h = weight_tensor.dims->data[1];
    unsigned int kernel_w = weight_tensor.dims->data[2];
    unsigned int in_channels = weight_tensor.dims->data[3];

    byoc_library::TensorShape TensorShape{
        kernel_h,
        kernel_w,
        in_channels,
        out_channels,
    };

    byoc_library::QuantizationInfo QuantizationInfo(weight_tensor.params.zero_point, weight_tensor.params.scale);
    byoc_library::TensorInfo weightsInfo(TensorShape,
                                         DataType,
                                         DataFormat,
                                         QuantizationInfo);
    weightsInfo.m_DataFormat = depth ? bl::DataFormat::HWIM : bl::DataFormat::HWIO;

    int data_length = out_channels * kernel_h * kernel_w * in_channels;
    std::vector<uint8_t> weightsData(data_length);

    const uint8_t *data_uint8 = static_cast<const uint8_t *>(weight_tensor.data.uint8);

    if (depth) {
        memcpy(&weightsData[0], data_uint8, data_length * sizeof(unsigned char));
    } else {
        SwizzleOHWIToHWIO<uint8_t>(data_uint8, &weightsData[0], weight_tensor.dims);
    }
    return byoc_library::AddConstant(m_Network, weightsInfo, reinterpret_cast<const uint8_t *>(weightsData.data())).tensor;
}

std::shared_ptr<manasi::byoc_library::Constant> ManasiBuilder::AddBias(TfLiteTensor &input_tensor, TfLiteTensor &weight_tensor, TfLiteTensor &bias_tensor) {
    int zeropoint = bias_tensor.params.zero_point;
    float scale = input_tensor.params.scale * weight_tensor.params.scale; //直接用biastensor里的scale 有点问题
    TfLiteIntArray *bias_dims = bias_tensor.dims;

    const unsigned int numBiasElements = bias_dims->data[0]; //只有一个维度
    int data_length = numBiasElements;

    const int32_t *data_int32 = static_cast<const int32_t *>(bias_tensor.data.i32);
    std::vector<int32_t> zeroBiasData(numBiasElements);
    memcpy(&zeroBiasData[0], data_int32, data_length * sizeof(int32_t));

    const void *biasData = reinterpret_cast<void *>(zeroBiasData.data());

    const byoc_library::DataType biasDataType = byoc_library::DataType::INT32_QUANTIZED;
    const byoc_library::TensorShape biasTensorShape = {1, 1, 1, numBiasElements};

    byoc_library::TensorInfo biasInfo(biasTensorShape, biasDataType, byoc_library::DataFormat::NHWC, {zeropoint, scale});
    return byoc_library::AddConstant(m_Network, biasInfo, biasData).tensor;
}

std::pair<int, int> GetPadValue(int data, int kernel, int stride) {
    /*Get the pad tuple of value for SAME padding*/
    int out = int(std::ceil(float(data) / float(stride)));
    int pad = fmax(0, (out - 1) * stride + kernel - data);
    int pad_before = int(pad / 2);
    int pad_after = pad - pad_before;
    return std::make_pair(pad_before, pad_after);
}

void ManasiBuilder::AddActivation(TfLiteNode *node) {
    auto input_tensor = context->tensors[node->inputs->data[0]];
    auto input = convertedTensors[input_tensor.name];
    int type_ = kTfLiteBuiltinRelu; //kTfLiteBuiltinRelu6
    switch (type_) {
    case kTfLiteBuiltinRelu: {
        bl::ReluInfo reluInfo;
        int inputQuantizationScale = 1;
        int inputQuantizationOffset = 0;
        reluInfo.m_LowerBound = std::numeric_limits<uint8_t>::min();
        reluInfo.m_UpperBound = std::numeric_limits<uint8_t>::max();

        bl::AddRelu(m_Network, *input, reluInfo);
        break;
    }
    default:
        std::cout << ("activate Not supported");
    }
}

void ManasiBuilder::AddAddition(TfLiteNode *node) {
    auto input_tensor1 = context->tensors[node->inputs->data[0]];
    auto input_tensor2 = context->tensors[node->inputs->data[1]];

    auto outpt_tensor = context->tensors[node->outputs->data[0]];
    auto input1 = convertedTensors[input_tensor1.name];
    auto input2 = convertedTensors[input_tensor2.name];

    int out_zero_point = outpt_tensor.params.zero_point;
    float out_scale = outpt_tensor.params.scale;
    bl::QuantizationInfo outputQuantInfo(out_zero_point, out_scale);
    ManasiAddOperationResult output = bl::AddAddition(m_Network, *input1, *input2, outputQuantInfo);
    convertedTensors[outpt_tensor.name] = output.tensor;
}

void ManasiBuilder::AddPooling2d(TfLiteNode *node) {
    auto input_tensor = context->tensors[node->inputs->data[0]];
    auto input = convertedTensors[input_tensor.name];
    const TfLitePoolParams *params = reinterpret_cast<const TfLitePoolParams *>(node->builtin_data);

    // Compute padding value
    TfLitePadding padding_type = params->padding;
    unsigned int pad_to_begin[2] = {0, 0};
    unsigned int pad_to_end[2] = {0, 0};

    // pooling param
    unsigned int filter_size[2] = {params->filter_height, params->filter_width};

    if (padding_type == kTfLitePaddingSame) {
        pad_to_begin[0] = (filter_size[0] - 1) / 2;
        pad_to_begin[1] = (filter_size[1] - 1) / 2;
        pad_to_end[0] = (filter_size[0] - 1) - (filter_size[0] - 1) / 2;
        pad_to_end[1] = (filter_size[1] - 1) - (filter_size[1] - 1) / 2;
    }

    const bl::PoolingType poolingType = bl::PoolingType::MAX; // bl::PoolingType::AVG;
    int top = 0;
    int bottom = 0;
    int left = 0;
    int right = 0;
    const bl::Padding padding = bl::Padding(top, bottom, left, right);
    int PoolHeight = params->filter_height;
    int PoolWidth = params->filter_width;

    int StrideX = params->stride_width;
    int StrideY = params->stride_height;
    ;
    const bl::PoolingInfo poolingInfo(PoolHeight,
                                      PoolWidth,
                                      StrideX,
                                      StrideY,
                                      padding,
                                      poolingType);

    ManasiAddOperationResult output = bl::AddPooling(
        m_Network, *input, poolingInfo);
}

void ManasiBuilder::AddConvolution2d(TfLiteNode *node) {
    assert(node->inputs->size >= 2);
    assert(node->outputs->size == 1);
    auto input_tensor = context->tensors[node->inputs->data[0]];
    auto weight_tensor = context->tensors[node->inputs->data[1]];
    auto bias_tensor = context->tensors[node->inputs->data[2]];
    auto output_tensor = context->tensors[node->outputs->data[0]];

    auto input = convertedTensors[input_tensor.name];
    auto weights = AddWeight(weight_tensor, false);
    auto biases = AddBias(input_tensor, weight_tensor, bias_tensor);

    TfLiteConvParams *params = reinterpret_cast<TfLiteConvParams *>(node->builtin_data);
    const bl::Stride stride(params->stride_width, params->stride_height);

    TfLitePadding padding_p = params->padding;
    int stride_h = params->stride_height;
    int stride_w = params->stride_width;
    int dilation_h = params->dilation_height_factor;
    int dilation_w = params->dilation_width_factor;
    int kernel_h = weight_tensor.dims->data[1];
    int kernel_w = weight_tensor.dims->data[2];

    int dilated_kernel_h = dilation_h * (kernel_h - 1) + 1;
    int dilated_kernel_w = dilation_w * (kernel_w - 1) + 1;
    int input_h = input_tensor.dims->data[1];
    int input_w = input_tensor.dims->data[2];

    int pad_top = 0;
    int pad_bottom = 0;
    int pad_left = 0;
    int pad_right = 0;
    if (padding_p == kTfLitePaddingSame) {
        std::pair<int, int> padvalue = GetPadValue(input_h, dilated_kernel_h, stride_h);
        pad_top = padvalue.first;
        pad_bottom = padvalue.second;
        padvalue = GetPadValue(input_w, dilated_kernel_w, stride_w);
        pad_left = padvalue.first;
        pad_right = padvalue.second;
    }
    const bl::Padding pad(pad_top, pad_bottom, pad_left, pad_right);

    float quantizationScale = output_tensor.params.scale;
    int quantizationOffset = output_tensor.params.zero_point;

    const bl::QuantizationInfo quantizationInfo(quantizationOffset, quantizationScale);
    manasi::byoc_library::utils::Optional<bl::ConvolutionInfo> convolutionInfo = bl::ConvolutionInfo(pad, stride, quantizationInfo);
    if (!convolutionInfo.has_value()) {
        std::cout << ("convolutionInfo Not supported");
    }

    TfLiteIntArray *out_tensor_dims = output_tensor.dims;
    const byoc_library::TensorShape TensorShape = {out_tensor_dims->data[0],
                                                   out_tensor_dims->data[1],
                                                   out_tensor_dims->data[2],
                                                   out_tensor_dims->data[3]};          //输入数据的shape信息
    const byoc_library::DataFormat DataFormat = byoc_library::DataFormat::NHWC88;      // data layout
    const byoc_library::DataType DataType = GetTensorBYOCDataType(output_tensor.type); //数据的类型 uint8
    int out_zero_point = output_tensor.params.zero_point;
    float out_scale = output_tensor.params.scale;
    byoc_library::QuantizationInfo QuantizationInfo(out_zero_point, out_scale); //量化的信息
    byoc_library::TensorInfo outputTensorInfo = byoc_library::TensorInfo(TensorShape,
                                                                         DataType,
                                                                         DataFormat,
                                                                         QuantizationInfo);
    ManasiAddOperationResult output = bl::AddConvolution(m_Network, *input, *biases, *weights, convolutionInfo.value());
    convertedTensors[output_tensor.name] = output.tensor;
}

void ManasiBuilder::DepthwiseConvolution2d(TfLiteNode *node) {
    assert(node->inputs->size >= 2);
    assert(node->outputs->size == 1);
    auto input_tensor = context->tensors[node->inputs->data[0]];
    auto weight_tensor = context->tensors[node->inputs->data[1]];
    auto bias_tensor = context->tensors[node->inputs->data[2]];
    auto output_tensor = context->tensors[node->outputs->data[0]];
    auto input = convertedTensors[input_tensor.name];
    auto weights = AddWeight(weight_tensor, true);
    auto biases = AddBias(input_tensor, weight_tensor, bias_tensor);

    TfLiteDepthwiseConvParams *params = reinterpret_cast<TfLiteDepthwiseConvParams *>(node->builtin_data);

    TfLitePadding padding_p = params->padding;
    int stride_h = params->stride_height;
    int stride_w = params->stride_width;
    int dilation_h = params->dilation_height_factor;
    int dilation_w = params->dilation_width_factor;
    int kernel_h = weight_tensor.dims->data[1];
    int kernel_w = weight_tensor.dims->data[2];
    int output_channels = 0;
    int dilated_kernel_h = dilation_h * (kernel_h - 1) + 1;
    int dilated_kernel_w = dilation_w * (kernel_w - 1) + 1;
    int input_h = input_tensor.dims->data[1];
    int input_w = input_tensor.dims->data[2];

    int pad_top = 0;
    int pad_bottom = 0;
    int pad_left = 0;
    int pad_right = 0;
    if (padding_p == kTfLitePaddingSame) {
        std::pair<int, int> padvalue = GetPadValue(input_h, dilated_kernel_h, stride_h);
        pad_top = padvalue.first;
        pad_bottom = padvalue.second;
        padvalue = GetPadValue(input_w, dilated_kernel_w, stride_w);
        pad_left = padvalue.first;
        pad_right = padvalue.second;
    }
    const bl::Stride stride(params->stride_width, params->stride_height);
    const bl::Padding pad(pad_top, pad_bottom, pad_left, pad_right);

    float quantizationScale = output_tensor.params.scale;
    int quantizationOffset = output_tensor.params.zero_point;

    const bl::QuantizationInfo quantizationInfo(quantizationOffset, quantizationScale);
    manasi::byoc_library::utils::Optional<bl::ConvolutionInfo> convolutionInfo = bl::ConvolutionInfo(pad, stride, quantizationInfo);
    if (!convolutionInfo.has_value()) {
        std::cout << ("convolutionInfo Not supported");
    }

    TfLiteIntArray *out_tensor_dims = output_tensor.dims;
    const byoc_library::TensorShape TensorShape = {out_tensor_dims->data[0],
                                                   out_tensor_dims->data[1],
                                                   out_tensor_dims->data[2],
                                                   out_tensor_dims->data[3]};          //输入数据的shape信息
    const byoc_library::DataFormat DataFormat = byoc_library::DataFormat::NHWC88;      // data layout
    const byoc_library::DataType DataType = GetTensorBYOCDataType(output_tensor.type); //数据的类型 uint8
    int out_zero_point = output_tensor.params.zero_point;
    float out_scale = output_tensor.params.scale;
    byoc_library::QuantizationInfo QuantizationInfo(out_zero_point, out_scale); //量化的信息
    byoc_library::TensorInfo outputTensorInfo = byoc_library::TensorInfo(TensorShape,
                                                                         DataType,
                                                                         DataFormat,
                                                                         QuantizationInfo);
    ManasiAddOperationResult output = bl::AddDepthwiseConvolution(m_Network, *input, *biases, *weights, convolutionInfo.value());
    convertedTensors[output_tensor.name] = output.tensor;
}

void ManasiBuilder::BuildModule() {
    ManasiConfig m_ManasiConfig;
    m_Network = byoc_library::CreateNetwork(m_ManasiConfig.GetCapabilities());
    int tensor_id;

    for (int i = 0; i < input_tensors->size; i++) {
        tensor_id = input_tensors->data[i];
        TfLiteTensor input_tensor = context->tensors[tensor_id];
        if (input_tensor.allocation_type == kTfLiteMmapRo) {
            continue; // weight or bias tensor
        }
        AddInput(input_tensor);
    }

    TfLiteNode *node;
    TfLiteRegistration *reg;
    for (size_t i = 0; i < nodes->size(); i++) {
        context->GetNodeAndRegistration(context, i, &node, &reg);
        switch (reg->builtin_code) {
        case kTfLiteBuiltinConv2d: AddConvolution2d(node); break;
        case kTfLiteBuiltinDepthwiseConv2d: DepthwiseConvolution2d(node); break;
        case kTfLiteBuiltinAdd: AddAddition(node); break;
        default:
            std::cout << "datatype not supported!\n";
        }
    }

    // output
    for (int i = 0; i < output_tensors->size; i++) {
        tensor_id = output_tensors->data[i];
        TfLiteTensor output_tensor = context->tensors[tensor_id];
        AddOutput(output_tensor);
    }

    // Compile the Network.
    bl::CompilationOptions options;
    std::vector<std::unique_ptr<bl::CompiledNetwork>> compiledNetwork = bl::Compile(*m_Network, options);
    bl::ModelStream *modelStream = compiledNetwork.front()->GetModelStreamStruct();
    std::cout << "end" << std::endl;
};

} // namespace manasi1





#ifndef TVM_RELAY_BACKEND_CONTRIB_MANASI_BUILD_H_
#define TVM_RELAY_BACKEND_CONTRIB_MANASI_BUILD_H_

#include "tensorflow/lite/c/common.h"
#include "iostream"
#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <manasi_driver_library/manasi_api.h>
#include <manasi_byoc_library/Support.hpp>
#include <manasi_byoc_library/Optional.hpp>
using namespace manasi;

namespace manasi1 {

template <typename T>

void SwizzleOHWIToHWIO(const void *inputBuffer, void *outputBuffer,
                       TfLiteIntArray *out_tensor_dims) {
    // ARMNN_ASSERT(inputShape.GetNumDimensions() == 4);
    const T *typedInputData = reinterpret_cast<const T *>(inputBuffer);
    std::vector<T> output;
    int data_length = out_tensor_dims->data[0] * out_tensor_dims->data[1] * out_tensor_dims->data[2] * out_tensor_dims->data[3];
    output.reserve(data_length);

    uint32_t dimO = out_tensor_dims->data[0];
    uint32_t dimH = out_tensor_dims->data[1];
    uint32_t dimW = out_tensor_dims->data[2];
    uint32_t dimI = out_tensor_dims->data[3];

    for (unsigned int inH = 0; inH < dimH; inH++) {
        for (unsigned int inW = 0; inW < dimW; inW++) {
            for (unsigned int inI = 0; inI < dimI; inI++) {
                for (unsigned int inO = 0; inO < dimO; inO++) {
                    unsigned int flatIndex = (inO * dimH * dimW * dimI) + (inH * dimW * dimI) + (inW * dimI) + inI;
                    T elem = typedInputData[flatIndex];
                    output.push_back(elem);
                }
            }
        }
    }
    memcpy(outputBuffer, output.data(), data_length * sizeof(T));
}



extern "C" {

namespace bl = byoc_library;
using ManasiOperationId = uint32_t;
using ManasiInputOutputId = std::pair<ManasiOperationId, uint32_t>;
using ManasiAddOperationResult = bl::TensorAndId<bl::Operand>;
using ManasiNetworkPtr = std::shared_ptr<bl::Network>;
using ManasiCompiledNetworkPtr = std::unique_ptr<bl::CompiledNetwork>;
using ManasiConstantPtr = std::shared_ptr<bl::Constant>;
using ManasiOperandPtr = std::shared_ptr<bl::Operand>;

struct Network {
    Network(std::vector<uint8_t> serializedCompiledNetwork,
            std::unordered_map<uint32_t, uint32_t> inputSlotsToManasiInputs,
            std::unordered_map<uint32_t, uint32_t> outputSlotsToManasiOutputs) :
        m_SerializedCompiledNetwork(std::move(serializedCompiledNetwork)),
        m_InputSlotsToManasiInputs(std::move(inputSlotsToManasiInputs)),
        m_OutputSlotsToManasiOutputs(std::move(outputSlotsToManasiOutputs)) {
    }
    std::vector<uint8_t> m_SerializedCompiledNetwork;
    std::unordered_map<uint32_t, uint32_t> m_InputSlotsToManasiInputs;
    std::unordered_map<uint32_t, uint32_t> m_OutputSlotsToManasiOutputs;
};


struct ManasiConfig {
    ManasiConfig() {
    }
    bool m_PerfOnly = false;
    std::vector<char> GetCapabilities() {
        return byoc_library::GetRawHardwareCapabilities(byoc_library::ManasiVariant::MANASI_V1);
    }
};


struct ManasiOperand {
    uint32_t operationId;
    std::shared_ptr<bl::Operand> tensor;
    uint32_t outputIndex;
};


class ManasiBuilder {
public:
    ManasiBuilder(TfLiteContext *context, const TfLiteIntArray *input_tensors,
                  const TfLiteIntArray *output_tensors, const std::vector<int> *nodes) :
        context(context),
        input_tensors(input_tensors), output_tensors(output_tensors), nodes(nodes) {
    }
    ~ManasiBuilder() = default;

    void BuildModule();

private:
    void AddInput(TfLiteTensor &input_tensor);
    void AddOutput(TfLiteTensor &output_tensor);
    std::shared_ptr<manasi::byoc_library::Constant> AddWeight(TfLiteTensor &weight_tensor, bool depth);
    std::shared_ptr<manasi::byoc_library::Constant> AddBias(TfLiteTensor &input_tensor, TfLiteTensor &weight_tensor, TfLiteTensor &bias_tensor);
    void AddConvolution2d(TfLiteNode *node);
    void DepthwiseConvolution2d(TfLiteNode *node);
    void AddActivation(TfLiteNode *node);
    void AddAddition(TfLiteNode *node);
    void AddPooling2d(TfLiteNode *node);

    TfLiteContext *context;

    const TfLiteIntArray *input_tensors;
    const TfLiteIntArray *output_tensors;
    const std::vector<int> *nodes;

    std::shared_ptr<manasi::byoc_library::Network> m_Network;
    std::unordered_map<std::string, std::shared_ptr<bl::Operand>> convertedTensors;

    byoc_library::DataType GetTensorBYOCDataType(TfLiteType type_id) {
        switch (type_id) {
        case kTfLiteInt32:
            return byoc_library::DataType::INT32_QUANTIZED;
        case kTfLiteUInt8:
            return byoc_library::DataType::UINT8_QUANTIZED;
        case kTfLiteInt8:
            return byoc_library::DataType::INT8_QUANTIZED;
        default: {
            // LOG(ERROR) << "datatype not supported!\n";
            return byoc_library::DataType::INT32_QUANTIZED;
            break;
        }
        }
    }
};

} // extern "C"

} // namespace manasi1
#endif // TVM_RELAY_BACKEND_CONTRIB_MANASI_BUILD_H_

