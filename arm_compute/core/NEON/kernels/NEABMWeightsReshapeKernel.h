#ifndef _ARM_COMPUTE_ABMWEIGHTSRESHAPEKERNEL_H_
#define _ARM_COMPUTE_ABMWEIGHTSRESHAPEKERNEL_H_

#include "arm_compute/core/NEON/INEKernel.h"

#include<vector>
#include<map>
using namespace std;
namespace arm_compute
{
class ITensor;

class NEABMWeightsReshapeKernel : public INEKernel
{
public:
    const char *name() const override
    {
        return "NEABMWeightsReshapeKernel";
    }
    NEABMWeightsReshapeKernel();
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEABMWeightsReshapeKernel(const NEABMWeightsReshapeKernel &) = delete;
    /** Prevent instances of this class from being copied (As this class contains pointers) */
    NEABMWeightsReshapeKernel &operator=(const NEABMWeightsReshapeKernel &) = delete;
    /** Allow instances of this class to be moved */
    NEABMWeightsReshapeKernel(NEABMWeightsReshapeKernel &&) = default;
    /** Allow instances of this class to be moved */
    NEABMWeightsReshapeKernel &operator=(NEABMWeightsReshapeKernel &&) = default;
    /** Default destructor */
    ~NEABMWeightsReshapeKernel() = default;
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
    static Status validate(const ITensorInfo *weights,const ITensorInfo *value, const ITensorInfo *quantity, const ITensorInfo *location);
    // Inherited methods overridden:
    void run(const Window &window, const ThreadInfo &info) override;
private:
    const ITensor *_input;
    ITensor *_value;
    ITensor *_quantity;
    ITensor *_location;
    unsigned int _x_index;
    unsigned int _y_index;
    unsigned int _z_index;
    int _location_y_size;
};
}//namespace arm_compute
#endif
