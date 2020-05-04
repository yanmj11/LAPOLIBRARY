#ifndef _ARM_COMPUTE_NEABMCONVOLUTIONLAYER_H_

#define _ARM_COMPUTE_NEARMCONVOLUTIONLAYER_H_
#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/core/NEON/kernels/NEABMWeightsReshapeKernel.h"

#include "arm_compute/core/NEON/kernels/NEIm2ColKernel.h"

#include "arm_compute/core/NEON/kernels/NEIm2ColKernel4S8.h"

#include "arm_compute/core/NEON/kernels/NEGEMMInterleave4x4Kernel.h"

#include "arm_compute/core/NEON/kernels/NEInterleave4S8Kernel.h"

#include "arm_compute/core/NEON/kernels/NEABMInputTransposeKernel.h"

#include "arm_compute/core/NEON/kernels/NECol2ImKernel.h"

#include "arm_compute/core/NEON/kernels/NECol2ImKernel4S8.h"

#include "arm_compute/runtime/NEON/functions/NEABMMatrixMultiply.h"
#include "arm_compute/core/Types.h"

#include "arm_compute/runtime/MemoryGroup.h"

#include "arm_compute/runtime/Tensor.h"#include <memory>

#include <vector>

#include <map>

using namespace std;

namespace arm_compute
{
class ITensor;

class NEABMReshapeWeights : public IFunction
{
public:

    /** Constructor */
    NEABMReshapeWeights();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEABMReshapeWeights(const NEABMReshapeWeights &) = delete;
    /** Default move constructor */
    NEABMReshapeWeights(NEABMReshapeWeights &&) = default;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEABMReshapeWeights &operator=(const NEABMReshapeWeights &) = delete;
    /** Default move assignment operator */
    NEABMReshapeWeights &operator=(NEABMReshapeWeights &&) = default;
    /** Set the input and output .
     *
     * @param[in]  weights Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: QASYMM8/QSYMM8_PER_CHANNEL/F16/F32.
     * @param[out] quantity  Record the values and occurrences of each patameters in weights tensor
     * @param[out] output  Rrcord the location of each parameters in weights tensor and the order of recording positions corresponds to the "quantity" table
     */
    void configure(const ITensor *weights, ITensor *value, ITensor *quantity, ITensor *location, unsigned int index[],int location_y_size=0);
    /** Static function to check if given info will lead to a valid configuration of @ref NEABMReshapeWeights
     *
     * @param[in]  weights Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: QASYMM8/QSYMM8_PER_CHANNEL/F16/F32.
     * @param[out] quantity  Record the values and occurrences of each patameters in weights tensor
     * @param[out] output  Rrcord the location of each parameters in weights tensor and the order of recording positions corresponds to the "quantity" table
     *
     * @return an error status
     */

    static Status validate(const ITensorInfo *weights, const ITensorInfo *value, const ITensorInfo *quantity, const ITensorInfo *location);
    // Inherited methods overridden:

    void run() override;
private:

    NEABMWeightsReshapeKernel _weights_reshape_kernel;

};
 
class NEABMConvolutionLayer : public IFunction
{

public:
    /** Constructor */
    NEABMConvolutionLayer(const std::shared_ptr<IMemoryManager> &memory_manager=nullptr);
    /** Prevent instances of this class from being copied (As this class contains pointers) */

    NEABMConvolutionLayer(const NEABMConvolutionLayer &) = delete;
    /** Default move constructor */

    NEABMConvolutionLayer(NEABMConvolutionLayer &&) = default;

    /** Prevent instances of this class from being copied (As this class contains pointers) */

    NEABMConvolutionLayer &operator=(const NEABMConvolutionLayer &) = delete;

    /** Default move assignment operator */

    NEABMConvolutionLayer &operator=(NEABMConvolutionLayer &&) = default;

    

    void configure_mm(const ITensor *input,ITensor *value,ITensor *quantity,ITensor *location, const ITensor *biases, ITensor *output, const ActivationLayerInfo &act_info, unsigned int precision[],unsigned int index[],int gemm_3d_depth,int kernel_width=0, int kernel_area=0,long long int total_number=0);

    /** Set the input and output tensors.
     *
     * @param[in]  input        Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *                          while every optional dimension from 4 and above represent a batch of inputs.
     *                          Data types supported: QASYMM8/F16/F32.
     * @param[in]  weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: QASYMM8/QSYMM8_PER_CHANNEL/F16/F32.
     * @param[in]  biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *                          Data type supported: Should match @p input data type, except for input of QASYMM8 type where biases should be of S32 type.
     * @param[out] output       Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     * Data types supported: Same as @p input.
     * @param[in]  conv_info    Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in]  weights_info Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     * tensor has also been transposed with NEGEMMTranspose1xWKernel. Data type supported: Same as @p input.
     * @param[in]  dilation     (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in]  act_info     (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     * @param[in]  num_groups   (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     */

    void configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info,  unsigned int precision[], unsigned int index[],const WeightsInfo &weights_info = WeightsInfo(),

                   const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), unsigned int num_groups = 1);

    /** Static function to check if given info will lead to a valid configuration of @ref NEABMConvolutionLayer
     *
     * @param[in] input        Source tensor. 3 lower dimensions represent a single input [width, height, IFM],
     *while every optional dimension from 4 and above represent a batch of inputs.
     *Data types supported: QASYMM8/F16/F32.
     * @param[in] weights      Weights tensor. Weights are 4D tensor with dimensions [kernel_x, kernel_y, IFM, OFM]. Data type supported: QASYMM8/QSYMM8_PER_CHANNEL/F16/F32.
     * @param[in] biases       Biases tensor. Shared biases supported. Biases are 1D tensor with dimensions [OFM].
     *Data type supported: Should match @p input data type, except for input of QASYMM8 type where biases should be of S32 type.
     * @param[in] output       Destination tensor. 3 lower dimensions represent a single output [width, height, OFM], while the rest represent batch of outputs.
     *Data types supported: Same as @p input.
     * @param[in] conv_info    Contains padding and stride information described in @ref PadStrideInfo.
     * @param[in] weights_info Specifies if the weights tensor has been reshaped with NEWeightsReshapeKernel. If this is not part of the fully connected layer the weights
     *tensor has also been transposed with NEGEMMTranspose1xWKernel. Data type supported: Same as @p input.
     * @param[in] dilation     (Optional) Dilation, in elements, across x and y. Defaults to (1, 1).
     * @param[in] act_info     (Optional) Activation layer information in case of a fused activation. Only RELU, BOUNDED_RELU and LU_BOUNDED_RELU supported.
     * @param[in] num_groups   (Optional) Number of groups when performing a grouped convolution. num_groups != 1 is not supported
     *
     * @return a status
     */

    static Status validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,

    const WeightsInfo &weights_info = WeightsInfo(), const Size2D &dilation = Size2D(1U, 1U), const ActivationLayerInfo &act_info = ActivationLayerInfo(), unsigned int num_groups = 1);
    // Inherited methods overridden:

    void run() override;

    void prepare() override;

private:

    MemoryGroup          _memory_group; //memory management
    NEABMReshapeWeights  _reshape_weights;    //runtime->core kernel
    //NEABMInputTransposeKernel   _input_transpose_kernel;
    NEIm2ColKernel       _im2col_kernel;       //core kernel
    NEIm2ColKernel4S8       _im2col_kernel_s8;
    NEInterleave4S8Kernel    _interleave_kernel;
    NEABMMatrixMultiply   _matrix_multiply;             //runtime->core kernel
    NECol2ImKernel4S8       _col2im_kernel;              //core kernel
    const ITensor *_original_weights;                    //keep a pointer to the original weights
    Tensor multiply_input;                    //a tensor to save the result of Im2colKernel output
    Tensor input_interleave;

    //Tensor input_transpose;
    Tensor value;
    Tensor quantity;     //describe the quantity of every value in all weights tensor
    Tensor location;
    Tensor multiply_output;                //a tensor to save the result of MatrixMultiply output and will be transfered to Col2ImKernel
    
    DataLayout _data_layout;   //NCHW or NHCW
    bool _skip_im2col;    //false
    bool _skip_col2im;     //false
    bool _is_quantized;     //
    bool _is_prepared;    
};

}//namespace arm_conpute

#endif//_ARM_CONPUTE_NEABMCONVOLUTIONLAYER_H_