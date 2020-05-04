#ifndef _ARM_COMPUTE_NEABMMATRIXMULTIPLY_H_

#define _ARM_COMPUTE_NEABMMATRIXMULTIPLY_H



#include "arm_compute/core/NEON/kernels/NEArithmeticAdditionKernel.h"

#include "arm_compute/core/NEON/kernels/NEABMMatrixMultiplyKernel.h"

#include "arm_compute/core/NEON/kernels/NEGEMMMatrixAdditionKernel.h"

#include "arm_compute/runtime/IFunction.h"

#include "arm_compute/runtime/IMemoryManager.h"

#include "arm_compute/runtime/MemoryGroup.h"

#include "arm_compute/runtime/NEON/functions/NEActivationLayer.h"

#include "arm_compute/runtime/Tensor.h"



#include<vector>

#include<map>

using namespace std;

namespace arm_compute

{



class NEABMMatrixMultiply : public IFunction

{

public:

    /** Constructor */

    NEABMMatrixMultiply(std::shared_ptr<IMemoryManager> memory_manager=nullptr );

    /** Prevent instances of this class from being copied (As this class contains pointers) */

    NEABMMatrixMultiply(const NEABMMatrixMultiply &) = delete;

    /** Default move constructor */

    NEABMMatrixMultiply(NEABMMatrixMultiply &&) = default;

    /** Prevent instances of this class from being copied (As this class contains pointers) */

    NEABMMatrixMultiply &operator=(const NEABMMatrixMultiply &) = delete;

    /** Default move assignment operator */

    NEABMMatrixMultiply &operator=(NEABMMatrixMultiply &&) = default;



    void configure(const ITensor *input, ITensor *value,ITensor *quantity, ITensor *location, const  ITensor *bias,  ITensor *output, unsigned int precision[],unsigned int index[],

                    float alpha, float beta, const GEMMInfo &gemm_info ,int kernel_width=0, int kernel_area=0,long long int total_number=0);



    static Status validate(const ITensorInfo *input, const ITensorInfo *value, const ITensorInfo *quantity, const ITensorInfo *location, const ITensorInfo *bias, const ITensorInfo *output, 

                    float alpha, float beta, const GEMMInfo &gemm_info );

    // Inherited methods overridden:

    void run() override;

    void prepare() override;

private:

    MemoryGroup                _memory_group;

    //IWeightsManager           *_weights_manager;

    //NEGEMMInterleave4x4Kernel  _interleave_kernel;

    //NEGEMMTranspose1xWKernel   _transpose_kernel;

    //NEGEMMMatrixMultiplyKernel _mm_kernel;

    //NEGEMMAssemblyDispatch     _asm_glue;

   // NEGEMMMatrixAdditionKernel _ma_kernel;

    NEABMMatrixMultiplyKernel _matrix_multiply_kernel;

    NEActivationLayer          _alpha_scale_func;

    NEArithmeticAdditionKernel _add_bias_kernel;

    NEActivationLayer          _activation_func;



    //Tensor         _tmp_a;

    //Tensor         _tmp_b;

    Tensor         _tmp_d;

    //const ITensor *_original_b;

    //bool           _run_vector_matrix_multiplication;

    //bool           _run_alpha_scale;

    bool           _run_addition;

    bool           _run_bias_addition;

    bool           _run_activation;

    //bool           _reshape_b_only_on_first_run;

    bool           _is_prepared;

};//end NEABMMatrixMultiply;

}//end namespace arm_compute;

#endif