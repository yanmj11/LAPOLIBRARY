#include "arm_compute/runtime/NEON/functions/NEABMConvolutionLayer.h"


#include "arm_compute/core/Size2D.h"
#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"

#include <set>
#include <tuple>

using namespace arm_compute;

NEABMReshapeWeights::NEABMReshapeWeights():_weights_reshape_kernel()
{}

void NEABMReshapeWeights::configure(const ITensor *weights, ITensor *value, ITensor *quantity, ITensor *location,unsigned int index[],int location_y_size)

{
    _weights_reshape_kernel.configure(weights,value,quantity,location,index,location_y_size);
}

Status NEABMReshapeWeights::validate(const ITensorInfo *weights, const ITensorInfo *value, const ITensorInfo *quantity, const ITensorInfo *location)
{
    // ARM_COMPUTE_RETURN_ERROR_ON_NULLPTR(weights);
    // ARM_COMPUTE_RETURN_ERROR_ON_DATA_TYPE_CHANNEL_NOT_IN(weights, 1, DataType::QASYMM8, DataType::QSYMM8_PER_CHANNEL, DataType::F16, DataType::F32);
    // ARM_COMPUTE_RETURN_ERROR_ON(weights->num_dimensions() > 4);
    // NEABMWeightsReshapeKernel::validate(weights,quantity,location);
    return Status{};

}
 
void NEABMReshapeWeights::run()
{
    NEScheduler::get().schedule(&_weights_reshape_kernel, 3);
    //Note in which dimension the scheduler execute on parallel!
}



NEABMConvolutionLayer::NEABMConvolutionLayer(const std::shared_ptr<IMemoryManager> &memory_manager )
    :_memory_group(memory_manager),_reshape_weights(),_im2col_kernel(),_im2col_kernel_s8(),_interleave_kernel(),_matrix_multiply(memory_manager),_col2im_kernel(),_original_weights(nullptr),
     multiply_input(),input_interleave(),value(),quantity(),location(),multiply_output(),_data_layout(DataLayout::NCHW),_skip_im2col(false),_skip_col2im(false),
     _is_quantized(false), _is_prepared(false)
{
}

/*configure_mm(&input_interleave, gemm_weights_value_use,gemm_weights_quantity_use,gemm_weights_location_use, biases, gemm_output_to_use, act_info, precision,index,gemm_3d_depth,kernel_width,kernel_area,total_number); */

void NEABMConvolutionLayer::configure_mm(const ITensor *input,ITensor *value, ITensor *quantity,ITensor *location, const ITensor *biases, ITensor *output, const ActivationLayerInfo &act_info, unsigned int precision[],unsigned int index[],int gemm_3d_depth,int kernel_width, int kernel_area,long long int total_number)
{
    // ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights);
    // ARM_COMPUTE_ERROR_THROW_ON(validate_mm(input->info(), weights->info(), biases == nullptr ? nullptr : biases->info(), output == nullptr ? nullptr : output->info(),
    //                                        act_info, gemm_3d_depth, _skip_im2col));
    // Create GEMMInfo structure
    const GEMMInfo &gemm_info = GEMMInfo(false, false, true /* Reshape weights only for the first run */,
                                         gemm_3d_depth, _skip_im2col /* Reinterpret the input as 3D if im2col is skipped */,
                                         false, GEMMLowpOutputStageInfo(), false, false, act_info);
    // Supported activations in GEMM
    const std::set<ActivationLayerInfo::ActivationFunction> supported_acts = { ActivationLayerInfo::ActivationFunction::RELU,
                                                                               ActivationLayerInfo::ActivationFunction::BOUNDED_RELU,
                                                                               ActivationLayerInfo::ActivationFunction::LU_BOUNDED_RELU
                                                                             };
    _matrix_multiply.configure(input, value, quantity,location, biases, output,precision,index, 1.0f, 0.0f, gemm_info, kernel_width,kernel_area,total_number);
}

void NEABMConvolutionLayer::configure(const ITensor *input, const ITensor *weights, const ITensor *biases, ITensor *output, const PadStrideInfo &conv_info, unsigned int precision[],unsigned int index[],const WeightsInfo &weights_info,
                                       const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(input, weights, output);
    ARM_COMPUTE_UNUSED(num_groups, weights_info);
    ARM_COMPUTE_ERROR_THROW_ON(NEGEMMConvolutionLayer::validate(input->info(),
                                                                weights->info(),
                                                                biases != nullptr ? biases->info() : nullptr,
                                                                output->info(),
                                                                conv_info,
                                                                weights_info,
                                                                dilation,
                                                                act_info,
                                                                num_groups));
 
    //const DataType   data_type   = input->info()->data_type();
    const DataLayout data_layout = input->info()->data_layout();
    const int        idx_width   = get_data_layout_dimension_index(data_layout, DataLayoutDimension::WIDTH);
    const int        idx_height  = get_data_layout_dimension_index(data_layout, DataLayoutDimension::HEIGHT);
    const int        idx_channel=get_data_layout_dimension_index(data_layout,DataLayoutDimension::CHANNEL);
    const int        idx_kernels = get_data_layout_dimension_index(data_layout, DataLayoutDimension::BATCHES);
    //std::cout<<"input_size:"<<std::endl;
    //std::cout<<input->info()->dimension(0)<<input->info()->dimension(1)<<input->info()->dimension(2)<<input->info()->dimension(3)<<std::endl;

    const unsigned int kernel_width  = weights->info()->dimension(idx_width);
    const unsigned int kernel_height = weights->info()->dimension(idx_height);

    _is_prepared      = weights_info.retain_internal_weights();
    _original_weights = weights;
    //_is_quantized     = is_data_type_quantized_asymmetric(input->info()->data_type());
    _data_layout      = data_layout;
    //_skip_im2col      = (data_layout == DataLayout::NHWC && kernel_width == 1 && kernel_height == 1 && conv_info.stride().first == 1 && conv_info.stride().second == 1);

    //const ITensor *gemm_input_to_use  = input;
    ITensor       *gemm_output_to_use = output;

    // Get convolved dimensions
    unsigned int conv_w = 0;
    unsigned int conv_h = 0;
    std::tie(conv_w, conv_h) = scaled_dimensions(input->info()->dimension(idx_width),
                                                 input->info()->dimension(idx_height),
                                                 kernel_width,
                                                 kernel_height,
                                                 conv_info,
                                                 dilation);
// std::cout<<"conv_size:"<<std::endl;
// std::cout<<conv_w<<conv_h<<std::endl;

    // Get parameters from conv_info

    unsigned int stride_x = 0;
    unsigned int stride_y = 0;
    std::tie(stride_x, stride_y) = conv_info.stride();

    unsigned int mat_weights_cols = weights->info()->dimension(idx_kernels);
    //init weights_value
    TensorShape value_shape=input->info()->tensor_shape();
    value_shape.set(0,1);value_shape.set(1,256);value_shape.set(2,mat_weights_cols);
    TensorInfo value_info(value_shape,1,DataType::S8);
    //set_quantization_info
    value.allocator()->init(value_info);
    value.allocator()->allocate();
    //memory_group_manage
    //init weights_quantity

    TensorShape quantity_shape=input->info()->tensor_shape();
    quantity_shape.set(0,1);
    quantity_shape.set(1,256);
    quantity_shape.set(2,mat_weights_cols);
    TensorInfo quantity_info(quantity_shape,1,DataType::U16);
    //set_quantization_info
    quantity.allocator()->init(quantity_info);
    quantity.allocator()->allocate();
    //memory_group_manage
    //init weights_location

    TensorShape location_shape=input->info()->tensor_shape();
    const long long int location_y_size=kernel_width*kernel_height*weights->info()->dimension(idx_channel);

    //std::cout<<location_y_size<<std::endl;
    location_shape.set(0,1);
    location_shape.set(1,location_y_size);
    location_shape.set(2,mat_weights_cols);//暂定为100

    TensorInfo location_info(location_shape,1,DataType::U16);
    //set_quantization_info
    location.allocator()->init(location_info);
    location.allocator()->allocate();
    //Reshape Weights
    ITensor *gemm_weights_value_use=&value;
    ITensor *gemm_weights_quantity_use=&quantity;
    ITensor *gemm_weights_location_use=&location;

    _reshape_weights.configure(weights,gemm_weights_value_use,gemm_weights_quantity_use,gemm_weights_location_use,index,location_y_size);

    // Create tensor to store im2col reshaped inputs

    //_skip_im2col= false
    if(!_skip_im2col)
    {
        _memory_group.manage(&multiply_input);
        _im2col_kernel_s8.configure(input, &multiply_input, Size2D(kernel_width, kernel_height), conv_info, false, dilation);
        multiply_input.allocator()->allocate();
        _memory_group.manage(&input_interleave);
        _interleave_kernel.configure(&multiply_input,&input_interleave);
        input_interleave.allocator()->allocate();
    }


    if(!_skip_col2im)
    {

        TensorShape shape_gemm;
        // Calculate GEMM output shape
        shape_gemm = multiply_input.info()->tensor_shape();
        shape_gemm.set(0, conv_w * conv_h);            //所有卷积操作
        shape_gemm.set(1, mat_weights_cols);            //所有卷积核

        // FIXME: input->clone() doesn't work with subtensors for grouped convolutions.

        TensorInfo info_gemm(shape_gemm, 1, DataType::S8);

        info_gemm.set_quantization_info(output->info()->quantization_info()).set_data_layout(input->info()->data_layout());

        multiply_output.allocator()->init(info_gemm);
        multiply_output.allocator()->allocate();
        _memory_group.manage(&multiply_output);

        // Update GEMM output

        gemm_output_to_use = &multiply_output;
    }

    // Configure GEMM
    // In case we need to skip col2im, GEMM3D (gemm_3d_depth != 0) must be called in order to avoid reshaping the output matrix

    const unsigned int kernel_area=kernel_width*kernel_height;
    const unsigned int gemm_3d_depth = _skip_col2im ? conv_h : 0;
    long long int total_number=conv_w*conv_h;
    configure_mm(&input_interleave, gemm_weights_value_use,gemm_weights_quantity_use,gemm_weights_location_use, biases, gemm_output_to_use, act_info, precision,index,gemm_3d_depth,kernel_width,kernel_area,total_number);

    if(!_skip_im2col)
    {
        //multiply_input.allocator()->allocate();
        // multiply_input.allocator()->allocate();
        // input_interleave.allocator()->allocate();
    }



    if(!_skip_col2im)
    {
        if(_data_layout == DataLayout::NCHW)
        {
            // Configure col2im
            _col2im_kernel.configure(gemm_output_to_use, output, Size2D(conv_w, conv_h));
        }
        else
        {
            // Configure reshape layer
           // _reshape_layer.configure(gemm_output_to_use, output);
        }

    }


    ARM_COMPUTE_ERROR_ON_MSG((output->info()->dimension(idx_width) != conv_w) || (output->info()->dimension(idx_height) != conv_h),
                             "Output shape does not match the expected one");

}

Status NEABMConvolutionLayer::validate(const ITensorInfo *input, const ITensorInfo *weights, const ITensorInfo *biases, const ITensorInfo *output, const PadStrideInfo &conv_info,

                                        const WeightsInfo &weights_info, const Size2D &dilation, const ActivationLayerInfo &act_info, unsigned int num_groups)

{

    return Status{};

}

void NEABMConvolutionLayer::run()

{

    prepare();                 

    if(!_skip_im2col)
    {
        unsigned int y_dim = get_data_layout_dimension_index(_data_layout, DataLayoutDimension::HEIGHT);
        NEScheduler::get().schedule(&_im2col_kernel_s8, y_dim);
        // arm_compute::utils::NPYLoader npy_sa2;
        // string file2="/media/sdcard/ComputeLibrary/data/neon_matrix/input_im2col.npy";
        // npy_sa2.save_to_npy2(multiply_input,file2,false);
        NEScheduler::get().schedule(&_interleave_kernel,Window::DimY);
        //  arm_compute::utils::NPYLoader npy_sa;
        //  string file="/media/sdcard/ComputeLibrary/data/neon_matrix/input_interleave.npy";
        //  npy_sa.save_to_npy4(input_interleave,file,false);

    }                       



    _matrix_multiply.run();
    arm_compute::utils::NPYLoader npy_sa2;
        string file2="/media/sdcard/ComputeLibrary/data/neon_matrix/multiply_output.npy";
        npy_sa2.save_to_npy2(multiply_output,file2,false);
  

         if(!_skip_col2im)

    {

        if(_data_layout == DataLayout::NCHW)

        {

           NEScheduler::get().schedule(&_col2im_kernel, Window::DimY);

        }

    }

}

void NEABMConvolutionLayer::prepare()

{

    if(!_is_prepared)

    {

            _reshape_weights.run();

            // string name1="/media/sdcard/ComputeLibrary/data/neon_matrix/value.npy";
            // string name2="/media/sdcard/ComputeLibrary/data/neon_matrix/quantity.npy";
            // string name3="/media/sdcard/ComputeLibrary/data/neon_matrix/location.npy";
            // arm_compute::utils::NPYLoader save;
            // save.save_to_npy2(value,name1,false);
            // save.save_to_npy3(quantity,name2,false);
            // save.save_to_npy3(location,name3,false);

            _original_weights->mark_as_unused();

            _matrix_multiply.prepare();

             _is_prepared = true;

    }

}


