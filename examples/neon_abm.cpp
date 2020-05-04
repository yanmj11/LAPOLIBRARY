//#ifndef ARM_COMPUTE_CL /* Needed by Utils.cpp to handle OpenCL exceptions properly */
//#error "This example needs to be built with -DARM_COMPUTE_CL"
//#endif /* ARM_COMPUTE_CL */

#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/CL/CLFunctions.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"
#include "arm_compute/runtime/Allocator.h"
#include "utils/ImageLoader.h"
#include "utils/Utils.h"
#include <ctime>
#include <cstdlib>

using namespace arm_compute;
using namespace utils;

class NEABMExample : public Example
{
public:
    bool do_setup(int argc, char **argv) override
    {
        //指定参数加载
        NPYLoader npy0;
        NPYLoader npy1;
        NPYLoader npy2;
        npy0.open(argv[1]);
        npy0.init_tensor2(src0, DataType::S8);

        npy1.open(argv[2]);
        npy1.init_tensor2(src1, DataType::S8);
        npy2.open(argv[3]);
        npy2.init_tensor2(src2, DataType::S8);

        unsigned int precision[4]={1,1,1,1};
        unsigned int index[3]={3,3,10};

        abm.configure(&src0, &src1, &src2, &dst, PadStrideInfo(1, 1, 0, 0),precision,index);


        src0.allocator()->allocate();
        src1.allocator()->allocate();
        src2.allocator()->allocate();
        dst.allocator()->allocate();
        if(npy0.is_open())
        {
            npy0.fill_tensor2(src0);
                npy1.fill_tensor2(src1);
            npy2.fill_tensor2(src2);

            is_fortran      = npy0.is_fortran();
        }
        //save the dst of abm_run()
        output_filename=argv[4];
        output_filename1=argv[5];

        return true;
    }
    void do_run() override
    {
       
        abm.run();
        if(!output_filename.empty()) /* Save to .npy file */
        {
            NPYLoader SSS;
            SSS.save_to_npy2(dst, output_filename, is_fortran);
        }
	}


private:
    Tensor      src0{}, src1{},src2{}, dst{};
 
    NEABMConvolutionLayer      abm{};
     
    bool        is_fortran{};
    std::string output_filename{};
    std::string output_filename1{};

};

/** Main program for ABM test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Matrix A, [optional] Matrix B, [optional] Matrix C, [optional] alpha, [optional] beta )
 */
int main(int argc, char **argv)
{
    return utils::run_example<NEABMExample>(argc, argv);
}
