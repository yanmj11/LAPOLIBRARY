#include "arm_compute/runtime/NEON/functions/NEABMMatrixMultiply.h"



#include "arm_compute/core/CPP/Validate.h"

#include "arm_compute/core/Error.h"

#include "arm_compute/core/Helpers.h"

#include "arm_compute/core/ITensor.h"

#include "arm_compute/core/TensorInfo.h"

#include "arm_compute/core/Types.h"

#include "arm_compute/core/Validate.h"

//#include "arm_compute/core/utils/misc/ShapeCalculator.h"

#include "arm_compute/runtime/NEON/NEScheduler.h"

//#include "arm_compute/runtime/NEON/functions/NEGEMMAssemblyDispatch.h"

#include "arm_compute/runtime/TensorAllocator.h"



#include <cmath>



namespace arm_compute

{

    NEABMMatrixMultiply::NEABMMatrixMultiply(std::shared_ptr<IMemoryManager> memory_manager)

            :_memory_group(memory_manager),_matrix_multiply_kernel(),_alpha_scale_func(), _add_bias_kernel(), _activation_func(), 

            _tmp_d(),_run_addition(false), _run_bias_addition(false), _run_activation(false), _is_prepared(false)

    {

    }



    void NEABMMatrixMultiply::configure(const ITensor *input,ITensor *value, ITensor *quantity, ITensor *location, const  ITensor *bias,  ITensor *output, unsigned int precision[],unsigned int index[],
                    float alpha, float beta, const GEMMInfo &gemm_info, int kernel_width, int kernel_area,long long int total_number)
    {
        ARM_COMPUTE_ERROR_THROW_ON(NEABMMatrixMultiply::validate(input->info(),value->info(),quantity->info(),location->info(),(bias != nullptr) ? bias->info() : nullptr,output->info(),alpha, beta, gemm_info));

        //_run_bias_addition                =bias != nullptr;
        // _run_addition                     =beta != 0 && bias != nullptr;
        // _run_activation                   =gemm_info.activation_info().enabled();
        _run_bias_addition                =false;
        _run_addition                     =false;
        _run_activation                   =false;
        // Pick output tensor in case bias addition should be performed
        ITensor *gemm_output_to_use = output;
        if(_run_bias_addition)
        {
            gemm_output_to_use = &_tmp_d;
            _memory_group.manage(&_tmp_d);
        }
        int m = (input->info()->dimension(1))*8; //m=卷积操作数
        int n = output->info()->dimension(0);       //n=卷积核个数
        int k = (input->info()->dimension(0))/8; //k=卷积操作操作数个数
        _matrix_multiply_kernel.configure(input,value,quantity,location,bias,gemm_output_to_use,precision,index,alpha,GEMMReshapeInfo(m, n, k),kernel_width,kernel_area,total_number);

        if(_run_bias_addition)
        {
            printf("bias_configure\n");
            _add_bias_kernel.configure(gemm_output_to_use, bias, output, ConvertPolicy::SATURATE);
            _tmp_d.allocator()->allocate();
        }

        if(_run_addition)
        {
            printf("addition_configure\n");
           // _ma_kernel.configure(bias, output, beta);
        }

         const ActivationLayerInfo &activation = gemm_info.activation_info();

         if(_run_activation)
        {
            printf("activation_configure\n");
        _activation_func.configure(output, nullptr, activation);
        }
    }



    Status NEABMMatrixMultiply::validate(const ITensorInfo *input, const ITensorInfo *value, const ITensorInfo *quantity, const ITensorInfo *location, const ITensorInfo *bias, const ITensorInfo *output, 
                    float alpha, float beta, const GEMMInfo &gemm_info)
    {
            return Status{};
    }


 
    void NEABMMatrixMultiply::run()
    {
        prepare();
        NEScheduler::get().schedule(&_matrix_multiply_kernel, Window::DimY);
        // Run bias addition kernel
        if(_run_bias_addition)
        {
            printf("bias_run\n");
            NEScheduler::get().schedule(&_add_bias_kernel, Window::DimY);
        }

    // Run matrix addition kernel
        if(_run_addition)
        {
            printf("addition_run\n");
            //NEScheduler::get().schedule(&_ma_kernel, Window::DimY);
        }
        // Run activation function
        if(_run_activation)
        {
            printf("activation_run\n");
            _activation_func.run();
        }
    }



    void NEABMMatrixMultiply::prepare()
    {
        //printf("NEABMMatrixMultiply prepare start!\n");
        //printf("NEABMMatrixMultiply prepare end!\n");
    }

}//end namespace arm_compute;
