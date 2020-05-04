#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/CPP/CPPScheduler.h"


#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/Allocator.h"
#include "utils/Utils.h"
#include <ctime>
#include <sched.h>

#include "support/ToolchainSupport.h"
#include <unistd.h>

#include <cstdlib>
#include <memory>

#include <iostream>

using namespace arm_compute;
using namespace utils;
using namespace std;

class NEONRESNETExample : public Example
{
public:
	bool do_setup(int argc, char **argv) override
	{
		constexpr unsigned int width_src_image  = 224;
        constexpr unsigned int height_src_image = 224;
        constexpr unsigned int ifm_src_img      = 3;

        const TensorShape src_shape(width_src_image, height_src_image, ifm_src_img);
        src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));
//first conv-batch-act-pooling
        constexpr unsigned int kern_x_con0 = 7;
		constexpr unsigned int kern_y_con0 =7 ;
		constexpr unsigned int ofm_con0 = 64;
		constexpr unsigned int out_x_con0 = 112;
		constexpr unsigned int out_y_con0 = 112;
		const TensorShape weights_shape_con0(kern_x_con0, kern_y_con0, src_shape.z(), ofm_con0);
		const TensorShape out_shape_con0(out_x_con0, out_y_con0, weights_shape_con0[3]);
		weights_con0.allocator()->init(TensorInfo(weights_shape_con0, 1, DataType::F32));
		out_con0.allocator()->init(TensorInfo(out_shape_con0, 1, DataType::F32));
		const TensorShape weights_mean_shape_batch0(out_shape_con0.z());
		const TensorShape weights_variance_shape_batch0(out_shape_con0.z());
		const TensorShape weights_gamma_shape_batch0(out_shape_con0.z());
		const TensorShape weights_beta_shape_batch0(out_shape_con0.z());
		weights_mean_batch0.allocator()->init(TensorInfo(weights_mean_shape_batch0, 1, DataType::F32));
		weights_variance_batch0.allocator()->init(TensorInfo(weights_variance_shape_batch0, 1, DataType::F32));
		weights_gamma_batch0.allocator()->init(TensorInfo(weights_gamma_shape_batch0, 1, DataType::F32));
		weights_beta_batch0.allocator()->init(TensorInfo(weights_beta_shape_batch0, 1, DataType::F32));
		out_batch0.allocator()->init(TensorInfo(out_shape_con0, 1, DataType::F32));
		out_act0.allocator()->init(TensorInfo(out_shape_con0,1,DataType::F32));
		TensorShape out_shape_pool0 = out_shape_con0;
		out_shape_pool0.set(0, out_shape_pool0.x() / 2); 
		out_shape_pool0.set(1, out_shape_pool0.y() / 2);
		out_pool0.allocator()->init(TensorInfo(out_shape_pool0, 1, DataType::F32));
//first  end
   
// block start          
// block1 
   //conv-batch-act
		constexpr unsigned int kern_x_block1r_con0 = 1;
		constexpr unsigned int kern_y_block1r_con0 = 1;
		constexpr unsigned int ofm_block1r_con0 = 64;
		constexpr unsigned int out_x_block1r_con0 = 56;
		constexpr unsigned int out_y_block1r_con0 = 56;
		const TensorShape weights_shape_block1r_con0(kern_x_block1r_con0, kern_y_block1r_con0, out_shape_pool0.z(), ofm_block1r_con0);
		const TensorShape out_shape_block1r_con0(out_x_block1r_con0, out_y_block1r_con0, weights_shape_block1r_con0[3]);
		weights_block1r_con0.allocator()->init(TensorInfo(weights_shape_block1r_con0, 1, DataType::F32));
		out_block1r_con0.allocator()->init(TensorInfo(out_shape_block1r_con0, 1, DataType::F32));
		const TensorShape weights_mean_shape_block1r_batch0(out_shape_block1r_con0.z());
		const TensorShape weights_variance_shape_block1r_batch0(out_shape_block1r_con0.z());
		const TensorShape weights_gamma_shape_block1r_batch0(out_shape_block1r_con0.z());
		const TensorShape weights_beta_shape_block1r_batch0(out_shape_block1r_con0.z());
		weights_mean_block1r_batch0.allocator()->init(TensorInfo(weights_mean_shape_block1r_batch0, 1, DataType::F32));
		weights_variance_block1r_batch0.allocator()->init(TensorInfo(weights_variance_shape_block1r_batch0, 1, DataType::F32));
		weights_gamma_block1r_batch0.allocator()->init(TensorInfo(weights_gamma_shape_block1r_batch0, 1, DataType::F32));
		weights_beta_block1r_batch0.allocator()->init(TensorInfo(weights_beta_shape_block1r_batch0, 1, DataType::F32));
		out_block1r_batch0.allocator()->init(TensorInfo(out_shape_block1r_con0, 1, DataType::F32));
		out_block1r_act0.allocator()->init(TensorInfo(out_shape_block1r_con0, 1, DataType::F32));
   //conv-batch-act	   
		constexpr unsigned int kern_x_block1r_con1 = 3;
		constexpr unsigned int kern_y_block1r_con1 = 3;
		constexpr unsigned int ofm_block1r_con1 = 64;
		constexpr unsigned int out_x_block1r_con1 = 56;
		constexpr unsigned int out_y_block1r_con1 = 56;
		const TensorShape weights_shape_block1r_con1(kern_x_block1r_con1, kern_y_block1r_con1, out_shape_block1r_con0.z(), ofm_block1r_con1);
		const TensorShape out_shape_block1r_con1(out_x_block1r_con1, out_y_block1r_con1, weights_shape_block1r_con1[3]);
		weights_block1r_con1.allocator()->init(TensorInfo(weights_shape_block1r_con1, 1, DataType::F32));
		out_block1r_con1.allocator()->init(TensorInfo(out_shape_block1r_con1, 1, DataType::F32));
		const TensorShape weights_mean_shape_block1r_batch1(out_shape_block1r_con1.z());
		const TensorShape weights_variance_shape_block1r_batch1(out_shape_block1r_con1.z());
		const TensorShape weights_gamma_shape_block1r_batch1(out_shape_block1r_con1.z());
		const TensorShape weights_beta_shape_block1r_batch1(out_shape_block1r_con1.z());
		weights_mean_block1r_batch1.allocator()->init(TensorInfo(weights_mean_shape_block1r_batch1, 1, DataType::F32));
		weights_variance_block1r_batch1.allocator()->init(TensorInfo(weights_variance_shape_block1r_batch1, 1, DataType::F32));
		weights_gamma_block1r_batch1.allocator()->init(TensorInfo(weights_gamma_shape_block1r_batch1, 1, DataType::F32));
		weights_beta_block1r_batch1.allocator()->init(TensorInfo(weights_beta_shape_block1r_batch1, 1, DataType::F32));
		out_block1r_batch1.allocator()->init(TensorInfo(out_shape_block1r_con1, 1, DataType::F32));
		out_block1r_act1.allocator()->init(TensorInfo(out_shape_block1r_con1, 1, DataType::F32));
   //conv-batch
		constexpr unsigned int kern_x_block1r_con2 = 1;
		constexpr unsigned int kern_y_block1r_con2 = 1;
		constexpr unsigned int ofm_block1r_con2 = 256;
		constexpr unsigned int out_x_block1r_con2 = 56;
		constexpr unsigned int out_y_block1r_con2 = 56;
		const TensorShape weights_shape_block1r_con2(kern_x_block1r_con2, kern_y_block1r_con2, out_shape_block1r_con1.z(), ofm_block1r_con2);
		const TensorShape out_shape_block1r_con2(out_x_block1r_con2, out_y_block1r_con2, weights_shape_block1r_con2[3]);
		weights_block1r_con2.allocator()->init(TensorInfo(weights_shape_block1r_con2, 1, DataType::F32));
		out_block1r_con2.allocator()->init(TensorInfo(out_shape_block1r_con2, 1, DataType::F32));
		const TensorShape weights_mean_shape_block1r_batch2(out_shape_block1r_con2.z());
		const TensorShape weights_variance_shape_block1r_batch2(out_shape_block1r_con2.z());
		const TensorShape weights_gamma_shape_block1r_batch2(out_shape_block1r_con2.z());
		const TensorShape weights_beta_shape_block1r_batch2(out_shape_block1r_con2.z());
		weights_mean_block1r_batch2.allocator()->init(TensorInfo(weights_mean_shape_block1r_batch2, 1, DataType::F32));
		weights_variance_block1r_batch2.allocator()->init(TensorInfo(weights_variance_shape_block1r_batch2, 1, DataType::F32));
		weights_gamma_block1r_batch2.allocator()->init(TensorInfo(weights_gamma_shape_block1r_batch2, 1, DataType::F32));
		weights_beta_block1r_batch2.allocator()->init(TensorInfo(weights_beta_shape_block1r_batch2, 1, DataType::F32));
		out_block1r_batch2.allocator()->init(TensorInfo(out_shape_block1r_con2, 1, DataType::F32));	
   //conv-batch
		constexpr unsigned int kern_x_block1l_con0 = 1;
		constexpr unsigned int kern_y_block1l_con0 = 1;
		constexpr unsigned int ofm_block1l_con0 = 256;
		constexpr unsigned int out_x_block1l_con0 = 56;
		constexpr unsigned int out_y_block1l_con0 = 56;
		const TensorShape weights_shape_block1l_con0(kern_x_block1l_con0, kern_y_block1l_con0, out_shape_pool0.z(), ofm_block1l_con0);
		const TensorShape out_shape_block1l_con0(out_x_block1l_con0, out_y_block1l_con0, weights_shape_block1l_con0[3]);
		weights_block1l_con0.allocator()->init(TensorInfo(weights_shape_block1l_con0, 1, DataType::F32));
		out_block1l_con0.allocator()->init(TensorInfo(out_shape_block1l_con0, 1, DataType::F32));
		const TensorShape weights_mean_shape_block1l_batch0(out_shape_block1l_con0.z());
		const TensorShape weights_variance_shape_block1l_batch0(out_shape_block1l_con0.z());
		const TensorShape weights_gamma_shape_block1l_batch0(out_shape_block1l_con0.z());
		const TensorShape weights_beta_shape_block1l_batch0(out_shape_block1l_con0.z());
		weights_mean_block1l_batch0.allocator()->init(TensorInfo(weights_mean_shape_block1l_batch0, 1, DataType::F32));
		weights_variance_block1l_batch0.allocator()->init(TensorInfo(weights_variance_shape_block1l_batch0, 1, DataType::F32));
		weights_gamma_block1l_batch0.allocator()->init(TensorInfo(weights_gamma_shape_block1l_batch0, 1, DataType::F32));
		weights_beta_block1l_batch0.allocator()->init(TensorInfo(weights_beta_shape_block1l_batch0, 1, DataType::F32));
		out_block1l_batch0.allocator()->init(TensorInfo(out_shape_block1l_con0, 1, DataType::F32));
   //add-act
		TensorShape out_shape_block1_0 = out_shape_block1r_con2;
		out_block1_add0.allocator()->init(TensorInfo(out_shape_block1_0, 1, DataType::F32));
		out_block1_act0.allocator()->init(TensorInfo(out_shape_block1_0, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block1r_con3 = 1;
		constexpr unsigned int kern_y_block1r_con3 = 1;
		constexpr unsigned int ofm_block1r_con3 = 64;
		constexpr unsigned int out_x_block1r_con3 = 56;
		constexpr unsigned int out_y_block1r_con3 = 56;
		const TensorShape weights_shape_block1r_con3(kern_x_block1r_con3, kern_y_block1r_con3, out_shape_block1_0.z(), ofm_block1r_con3);
		const TensorShape out_shape_block1r_con3(out_x_block1r_con3, out_y_block1r_con3, weights_shape_block1r_con3[3]);
		weights_block1r_con3.allocator()->init(TensorInfo(weights_shape_block1r_con3, 1, DataType::F32));
		out_block1r_con3.allocator()->init(TensorInfo(out_shape_block1r_con3, 1, DataType::F32));
		const TensorShape weights_mean_shape_block1r_batch3(out_shape_block1r_con3.z());
		const TensorShape weights_variance_shape_block1r_batch3(out_shape_block1r_con3.z());
		const TensorShape weights_gamma_shape_block1r_batch3(out_shape_block1r_con3.z());
		const TensorShape weights_beta_shape_block1r_batch3(out_shape_block1r_con3.z());
		weights_mean_block1r_batch3.allocator()->init(TensorInfo(weights_mean_shape_block1r_batch3, 1, DataType::F32));
		weights_variance_block1r_batch3.allocator()->init(TensorInfo(weights_variance_shape_block1r_batch3, 1, DataType::F32));
		weights_gamma_block1r_batch3.allocator()->init(TensorInfo(weights_gamma_shape_block1r_batch3, 1, DataType::F32));
		weights_beta_block1r_batch3.allocator()->init(TensorInfo(weights_beta_shape_block1r_batch3, 1, DataType::F32));
		out_block1r_batch3.allocator()->init(TensorInfo(out_shape_block1r_con3, 1, DataType::F32));
		out_block1r_act2.allocator()->init(TensorInfo(out_shape_block1r_con3, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block1r_con4 = 3;
		constexpr unsigned int kern_y_block1r_con4 = 3;
		constexpr unsigned int ofm_block1r_con4 = 64;
		constexpr unsigned int out_x_block1r_con4 = 56;
		constexpr unsigned int out_y_block1r_con4 = 56;
		const TensorShape weights_shape_block1r_con4(kern_x_block1r_con4, kern_y_block1r_con4, out_shape_block1r_con3.z(), ofm_block1r_con4);
		const TensorShape out_shape_block1r_con4(out_x_block1r_con4, out_y_block1r_con4, weights_shape_block1r_con4[3]);
		weights_block1r_con4.allocator()->init(TensorInfo(weights_shape_block1r_con4, 1, DataType::F32));
		out_block1r_con4.allocator()->init(TensorInfo(out_shape_block1r_con4, 1, DataType::F32));
		const TensorShape weights_mean_shape_block1r_batch4(out_shape_block1r_con4.z());
		const TensorShape weights_variance_shape_block1r_batch4(out_shape_block1r_con4.z());
		const TensorShape weights_gamma_shape_block1r_batch4(out_shape_block1r_con4.z());
		const TensorShape weights_beta_shape_block1r_batch4(out_shape_block1r_con4.z());
		weights_mean_block1r_batch4.allocator()->init(TensorInfo(weights_mean_shape_block1r_batch4, 1, DataType::F32));
		weights_variance_block1r_batch4.allocator()->init(TensorInfo(weights_variance_shape_block1r_batch4, 1, DataType::F32));
		weights_gamma_block1r_batch4.allocator()->init(TensorInfo(weights_gamma_shape_block1r_batch4, 1, DataType::F32));
		weights_beta_block1r_batch4.allocator()->init(TensorInfo(weights_beta_shape_block1r_batch4, 1, DataType::F32));
		out_block1r_batch4.allocator()->init(TensorInfo(out_shape_block1r_con4, 1, DataType::F32));
		out_block1r_act3.allocator()->init(TensorInfo(out_shape_block1r_con4, 1, DataType::F32));
   //conv-batch
		constexpr unsigned int kern_x_block1r_con5 = 1;
		constexpr unsigned int kern_y_block1r_con5 = 1;
		constexpr unsigned int ofm_block1r_con5 = 256;
		constexpr unsigned int out_x_block1r_con5 = 56;
		constexpr unsigned int out_y_block1r_con5 = 56;
		const TensorShape weights_shape_block1r_con5(kern_x_block1r_con5, kern_y_block1r_con5, out_shape_block1r_con4.z(), ofm_block1r_con5);
		const TensorShape out_shape_block1r_con5(out_x_block1r_con5, out_y_block1r_con5, weights_shape_block1r_con5[3]);
		weights_block1r_con5.allocator()->init(TensorInfo(weights_shape_block1r_con5, 1, DataType::F32));
		out_block1r_con5.allocator()->init(TensorInfo(out_shape_block1r_con5, 1, DataType::F32));
		const TensorShape weights_mean_shape_block1r_batch5(out_shape_block1r_con5.z());
		const TensorShape weights_variance_shape_block1r_batch5(out_shape_block1r_con5.z());
		const TensorShape weights_gamma_shape_block1r_batch5(out_shape_block1r_con5.z());
		const TensorShape weights_beta_shape_block1r_batch5(out_shape_block1r_con5.z());
		weights_mean_block1r_batch5.allocator()->init(TensorInfo(weights_mean_shape_block1r_batch5, 1, DataType::F32));
		weights_variance_block1r_batch5.allocator()->init(TensorInfo(weights_variance_shape_block1r_batch5, 1, DataType::F32));
		weights_gamma_block1r_batch5.allocator()->init(TensorInfo(weights_gamma_shape_block1r_batch5, 1, DataType::F32));
		weights_beta_block1r_batch5.allocator()->init(TensorInfo(weights_beta_shape_block1r_batch5, 1, DataType::F32));
		out_block1r_batch5.allocator()->init(TensorInfo(out_shape_block1r_con5, 1, DataType::F32));
   //add-act
		TensorShape out_shape_block1_1 = out_shape_block1r_con5;
		out_block1_add1.allocator()->init(TensorInfo(out_shape_block1_1, 1, DataType::F32));
		out_block1_act1.allocator()->init(TensorInfo(out_shape_block1_1, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block1r_con6 = 1;
		constexpr unsigned int kern_y_block1r_con6 = 1;
		constexpr unsigned int ofm_block1r_con6 = 64;
		constexpr unsigned int out_x_block1r_con6 = 56;
		constexpr unsigned int out_y_block1r_con6 = 56;
		const TensorShape weights_shape_block1r_con6(kern_x_block1r_con6, kern_y_block1r_con6, out_shape_block1_1.z(), ofm_block1r_con6);
		const TensorShape out_shape_block1r_con6(out_x_block1r_con6, out_y_block1r_con6, weights_shape_block1r_con6[3]);
		weights_block1r_con6.allocator()->init(TensorInfo(weights_shape_block1r_con6, 1, DataType::F32));
		out_block1r_con6.allocator()->init(TensorInfo(out_shape_block1r_con6, 1, DataType::F32));
		const TensorShape weights_mean_shape_block1r_batch6(out_shape_block1r_con6.z());
		const TensorShape weights_variance_shape_block1r_batch6(out_shape_block1r_con6.z());
		const TensorShape weights_gamma_shape_block1r_batch6(out_shape_block1r_con6.z());
		const TensorShape weights_beta_shape_block1r_batch6(out_shape_block1r_con6.z());
		weights_mean_block1r_batch6.allocator()->init(TensorInfo(weights_mean_shape_block1r_batch6, 1, DataType::F32));
		weights_variance_block1r_batch6.allocator()->init(TensorInfo(weights_variance_shape_block1r_batch6, 1, DataType::F32));
		weights_gamma_block1r_batch6.allocator()->init(TensorInfo(weights_gamma_shape_block1r_batch6, 1, DataType::F32));
		weights_beta_block1r_batch6.allocator()->init(TensorInfo(weights_beta_shape_block1r_batch6, 1, DataType::F32));
		out_block1r_batch6.allocator()->init(TensorInfo(out_shape_block1r_con6, 1, DataType::F32));
		out_block1r_act4.allocator()->init(TensorInfo(out_shape_block1r_con6, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block1r_con7= 3;
		constexpr unsigned int kern_y_block1r_con7= 3;
		constexpr unsigned int ofm_block1r_con7= 64;
		constexpr unsigned int out_x_block1r_con7= 28;
		constexpr unsigned int out_y_block1r_con7= 28;
		const TensorShape weights_shape_block1r_con7(kern_x_block1r_con7, kern_y_block1r_con7, out_shape_block1r_con6.z(), ofm_block1r_con7);
		const TensorShape out_shape_block1r_con7(out_x_block1r_con7, out_y_block1r_con7, weights_shape_block1r_con7[3]);
		weights_block1r_con7.allocator()->init(TensorInfo(weights_shape_block1r_con7, 1, DataType::F32));
		out_block1r_con7.allocator()->init(TensorInfo(out_shape_block1r_con7, 1, DataType::F32));
		const TensorShape weights_mean_shape_block1r_batch7(out_shape_block1r_con7.z());
		const TensorShape weights_variance_shape_block1r_batch7(out_shape_block1r_con7.z());
		const TensorShape weights_gamma_shape_block1r_batch7(out_shape_block1r_con7.z());
		const TensorShape weights_beta_shape_block1r_batch7(out_shape_block1r_con7.z());
		weights_mean_block1r_batch7.allocator()->init(TensorInfo(weights_mean_shape_block1r_batch7, 1, DataType::F32));
		weights_variance_block1r_batch7.allocator()->init(TensorInfo(weights_variance_shape_block1r_batch7, 1, DataType::F32));
		weights_gamma_block1r_batch7.allocator()->init(TensorInfo(weights_gamma_shape_block1r_batch7, 1, DataType::F32));
		weights_beta_block1r_batch7.allocator()->init(TensorInfo(weights_beta_shape_block1r_batch7, 1, DataType::F32));
		out_block1r_batch7.allocator()->init(TensorInfo(out_shape_block1r_con7, 1, DataType::F32));
		out_block1r_act5.allocator()->init(TensorInfo(out_shape_block1r_con7, 1, DataType::F32));
   //conv-batch
		constexpr unsigned int kern_x_block1r_con8 = 1;
		constexpr unsigned int kern_y_block1r_con8 = 1;
		constexpr unsigned int ofm_block1r_con8= 256;
		constexpr unsigned int out_x_block1r_con8= 28;
		constexpr unsigned int out_y_block1r_con8= 28;
		const TensorShape weights_shape_block1r_con8(kern_x_block1r_con8, kern_y_block1r_con8, out_shape_block1r_con7.z(), ofm_block1r_con8);
		const TensorShape out_shape_block1r_con8(out_x_block1r_con8, out_y_block1r_con8, weights_shape_block1r_con8[3]);
		weights_block1r_con8.allocator()->init(TensorInfo(weights_shape_block1r_con8, 1, DataType::F32));
		out_block1r_con8.allocator()->init(TensorInfo(out_shape_block1r_con8, 1, DataType::F32));
		const TensorShape weights_mean_shape_block1r_batch8(out_shape_block1r_con8.z());
		const TensorShape weights_variance_shape_block1r_batch8(out_shape_block1r_con8.z());
		const TensorShape weights_gamma_shape_block1r_batch8(out_shape_block1r_con8.z());
		const TensorShape weights_beta_shape_block1r_batch8(out_shape_block1r_con8.z());
		weights_mean_block1r_batch8.allocator()->init(TensorInfo(weights_mean_shape_block1r_batch8, 1, DataType::F32));
		weights_variance_block1r_batch8.allocator()->init(TensorInfo(weights_variance_shape_block1r_batch8, 1, DataType::F32));
		weights_gamma_block1r_batch8.allocator()->init(TensorInfo(weights_gamma_shape_block1r_batch8, 1, DataType::F32));
		weights_beta_block1r_batch8.allocator()->init(TensorInfo(weights_beta_shape_block1r_batch8, 1, DataType::F32));
		out_block1r_batch8.allocator()->init(TensorInfo(out_shape_block1r_con8, 1, DataType::F32));
   //pooling
		TensorShape out_shape_block1l_pool0 = out_shape_block1_1;
		out_shape_block1l_pool0.set(0, out_shape_block1l_pool0.x() / 2); 
		out_shape_block1l_pool0.set(1, out_shape_block1l_pool0.y() / 2);
		out_block1l_pool0.allocator()->init(TensorInfo(out_shape_block1l_pool0, 1, DataType::F32));
   //add-act
		TensorShape out_shape_block1_2 = out_shape_block1r_con8;
		out_block1_add2.allocator()->init(TensorInfo(out_shape_block1_2, 1, DataType::F32));
		out_block1_act2.allocator()->init(TensorInfo(out_shape_block1_2, 1, DataType::F32));

		
//block2
   //conv-batch-act
        constexpr unsigned int kern_x_block2r_con0 = 1;
		constexpr unsigned int kern_y_block2r_con0 = 1;
		constexpr unsigned int ofm_block2r_con0 = 128;
		constexpr unsigned int out_x_block2r_con0 = 28;
		constexpr unsigned int out_y_block2r_con0 = 28;
		const TensorShape weights_shape_block2r_con0(kern_x_block2r_con0, kern_y_block2r_con0, out_shape_block1_2.z(), ofm_block2r_con0);
		const TensorShape out_shape_block2r_con0(out_x_block2r_con0, out_y_block2r_con0, weights_shape_block2r_con0[3]);
		weights_block2r_con0.allocator()->init(TensorInfo(weights_shape_block2r_con0, 1, DataType::F32));
		out_block2r_con0.allocator()->init(TensorInfo(out_shape_block2r_con0, 1, DataType::F32));
		const TensorShape weights_mean_shape_block2r_batch0(out_shape_block2r_con0.z());
		const TensorShape weights_variance_shape_block2r_batch0(out_shape_block2r_con0.z());
		const TensorShape weights_gamma_shape_block2r_batch0(out_shape_block2r_con0.z());
		const TensorShape weights_beta_shape_block2r_batch0(out_shape_block2r_con0.z());
		weights_mean_block2r_batch0.allocator()->init(TensorInfo(weights_mean_shape_block2r_batch0, 1, DataType::F32));
		weights_variance_block2r_batch0.allocator()->init(TensorInfo(weights_variance_shape_block2r_batch0, 1, DataType::F32));
		weights_gamma_block2r_batch0.allocator()->init(TensorInfo(weights_gamma_shape_block2r_batch0, 1, DataType::F32));
		weights_beta_block2r_batch0.allocator()->init(TensorInfo(weights_beta_shape_block2r_batch0, 1, DataType::F32));
		out_block2r_batch0.allocator()->init(TensorInfo(out_shape_block2r_con0, 1, DataType::F32));
		out_block2r_act0.allocator()->init(TensorInfo(out_shape_block2r_con0, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block2r_con1 = 3;
		constexpr unsigned int kern_y_block2r_con1 = 3;
		constexpr unsigned int ofm_block2r_con1 = 128;
		constexpr unsigned int out_x_block2r_con1 = 28;
		constexpr unsigned int out_y_block2r_con1 = 28;
		const TensorShape weights_shape_block2r_con1(kern_x_block2r_con1, kern_y_block2r_con1, out_shape_block2r_con0.z(), ofm_block2r_con1);
		const TensorShape out_shape_block2r_con1(out_x_block2r_con1, out_y_block2r_con1, weights_shape_block2r_con1[3]);
		weights_block2r_con1.allocator()->init(TensorInfo(weights_shape_block2r_con1, 1, DataType::F32));
		out_block2r_con1.allocator()->init(TensorInfo(out_shape_block2r_con1, 1, DataType::F32));
		const TensorShape weights_mean_shape_block2r_batch1(out_shape_block2r_con1.z());
		const TensorShape weights_variance_shape_block2r_batch1(out_shape_block2r_con1.z());
		const TensorShape weights_gamma_shape_block2r_batch1(out_shape_block2r_con1.z());
		const TensorShape weights_beta_shape_block2r_batch1(out_shape_block2r_con1.z());
		weights_mean_block2r_batch1.allocator()->init(TensorInfo(weights_mean_shape_block2r_batch1, 1, DataType::F32));
		weights_variance_block2r_batch1.allocator()->init(TensorInfo(weights_variance_shape_block2r_batch1, 1, DataType::F32));
		weights_gamma_block2r_batch1.allocator()->init(TensorInfo(weights_gamma_shape_block2r_batch1, 1, DataType::F32));
		weights_beta_block2r_batch1.allocator()->init(TensorInfo(weights_beta_shape_block2r_batch1, 1, DataType::F32));
		out_block2r_batch1.allocator()->init(TensorInfo(out_shape_block2r_con1, 1, DataType::F32));
		out_block2r_act1.allocator()->init(TensorInfo(out_shape_block2r_con1, 1, DataType::F32));
   //conv-batch
		constexpr unsigned int kern_x_block2r_con2 = 1;
		constexpr unsigned int kern_y_block2r_con2 = 1;
		constexpr unsigned int ofm_block2r_con2 = 512;
		constexpr unsigned int out_x_block2r_con2 = 28;
		constexpr unsigned int out_y_block2r_con2 = 28;
		const TensorShape weights_shape_block2r_con2(kern_x_block2r_con2, kern_y_block2r_con2, out_shape_block2r_con1.z(), ofm_block2r_con2);
		const TensorShape out_shape_block2r_con2(out_x_block2r_con2, out_y_block2r_con2, weights_shape_block2r_con2[3]);
		weights_block2r_con2.allocator()->init(TensorInfo(weights_shape_block2r_con2, 1, DataType::F32));
		out_block2r_con2.allocator()->init(TensorInfo(out_shape_block2r_con2, 1, DataType::F32));
		const TensorShape weights_mean_shape_block2r_batch2(out_shape_block2r_con2.z());
		const TensorShape weights_variance_shape_block2r_batch2(out_shape_block2r_con2.z());
		const TensorShape weights_gamma_shape_block2r_batch2(out_shape_block2r_con2.z());
		const TensorShape weights_beta_shape_block2r_batch2(out_shape_block2r_con2.z());
		weights_mean_block2r_batch2.allocator()->init(TensorInfo(weights_mean_shape_block2r_batch2, 1, DataType::F32));
		weights_variance_block2r_batch2.allocator()->init(TensorInfo(weights_variance_shape_block2r_batch2, 1, DataType::F32));
		weights_gamma_block2r_batch2.allocator()->init(TensorInfo(weights_gamma_shape_block2r_batch2, 1, DataType::F32));
		weights_beta_block2r_batch2.allocator()->init(TensorInfo(weights_beta_shape_block2r_batch2, 1, DataType::F32));
		out_block2r_batch2.allocator()->init(TensorInfo(out_shape_block2r_con2, 1, DataType::F32));	
   //conv-batch
		constexpr unsigned int kern_x_block2l_con0 = 1;
		constexpr unsigned int kern_y_block2l_con0 = 1;
		constexpr unsigned int ofm_block2l_con0 = 512;
		constexpr unsigned int out_x_block2l_con0 = 28;
		constexpr unsigned int out_y_block2l_con0 = 28;
		const TensorShape weights_shape_block2l_con0(kern_x_block2l_con0, kern_y_block2l_con0, out_shape_block1_2.z(), ofm_block2l_con0);
		const TensorShape out_shape_block2l_con0(out_x_block2l_con0, out_y_block2l_con0, weights_shape_block2l_con0[3]);
		weights_block2l_con0.allocator()->init(TensorInfo(weights_shape_block2l_con0, 1, DataType::F32));
		out_block2l_con0.allocator()->init(TensorInfo(out_shape_block2l_con0, 1, DataType::F32));
		const TensorShape weights_mean_shape_block2l_batch0(out_shape_block2l_con0.z());
		const TensorShape weights_variance_shape_block2l_batch0(out_shape_block2l_con0.z());
		const TensorShape weights_gamma_shape_block2l_batch0(out_shape_block2l_con0.z());
		const TensorShape weights_beta_shape_block2l_batch0(out_shape_block2l_con0.z());
		weights_mean_block2l_batch0.allocator()->init(TensorInfo(weights_mean_shape_block2l_batch0, 1, DataType::F32));
		weights_variance_block2l_batch0.allocator()->init(TensorInfo(weights_variance_shape_block2l_batch0, 1, DataType::F32));
		weights_gamma_block2l_batch0.allocator()->init(TensorInfo(weights_gamma_shape_block2l_batch0, 1, DataType::F32));
		weights_beta_block2l_batch0.allocator()->init(TensorInfo(weights_beta_shape_block2l_batch0, 1, DataType::F32));
		out_block2l_batch0.allocator()->init(TensorInfo(out_shape_block2l_con0, 1, DataType::F32));
   //add-act
		TensorShape out_shape_block2_0 = out_shape_block2r_con2;
		out_block2_add0.allocator()->init(TensorInfo(out_shape_block2_0, 1, DataType::F32));
		out_block2_act0.allocator()->init(TensorInfo(out_shape_block2_0, 1, DataType::F32));
   //conv-batch-act
        constexpr unsigned int kern_x_block2r_con3 = 1;
		constexpr unsigned int kern_y_block2r_con3 = 1;
		constexpr unsigned int ofm_block2r_con3 = 128;
		constexpr unsigned int out_x_block2r_con3 = 28;
		constexpr unsigned int out_y_block2r_con3 = 28;
		const TensorShape weights_shape_block2r_con3(kern_x_block2r_con3, kern_y_block2r_con3, out_shape_block2_0.z(), ofm_block2r_con3);
		const TensorShape out_shape_block2r_con3(out_x_block2r_con3, out_y_block2r_con3, weights_shape_block2r_con3[3]);
		weights_block2r_con3.allocator()->init(TensorInfo(weights_shape_block2r_con3, 1, DataType::F32));
		out_block2r_con3.allocator()->init(TensorInfo(out_shape_block2r_con3, 1, DataType::F32));
		const TensorShape weights_mean_shape_block2r_batch3(out_shape_block2r_con3.z());
		const TensorShape weights_variance_shape_block2r_batch3(out_shape_block2r_con3.z());
		const TensorShape weights_gamma_shape_block2r_batch3(out_shape_block2r_con3.z());
		const TensorShape weights_beta_shape_block2r_batch3(out_shape_block2r_con3.z());
		weights_mean_block2r_batch3.allocator()->init(TensorInfo(weights_mean_shape_block2r_batch3, 1, DataType::F32));
		weights_variance_block2r_batch3.allocator()->init(TensorInfo(weights_variance_shape_block2r_batch3, 1, DataType::F32));
		weights_gamma_block2r_batch3.allocator()->init(TensorInfo(weights_gamma_shape_block2r_batch3, 1, DataType::F32));
		weights_beta_block2r_batch3.allocator()->init(TensorInfo(weights_beta_shape_block2r_batch3, 1, DataType::F32));
		out_block2r_batch3.allocator()->init(TensorInfo(out_shape_block2r_con3, 1, DataType::F32));
		out_block2r_act2.allocator()->init(TensorInfo(out_shape_block2r_con3, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block2r_con4 = 3;
		constexpr unsigned int kern_y_block2r_con4 = 3;
		constexpr unsigned int ofm_block2r_con4 = 128;
		constexpr unsigned int out_x_block2r_con4 = 28;
		constexpr unsigned int out_y_block2r_con4 = 28;
		const TensorShape weights_shape_block2r_con4(kern_x_block2r_con4, kern_y_block2r_con4, out_shape_block2r_con3.z(), ofm_block2r_con4);
		const TensorShape out_shape_block2r_con4(out_x_block2r_con4, out_y_block2r_con4, weights_shape_block2r_con4[3]);
		weights_block2r_con4.allocator()->init(TensorInfo(weights_shape_block2r_con4, 1, DataType::F32));
		out_block2r_con4.allocator()->init(TensorInfo(out_shape_block2r_con4, 1, DataType::F32));
		const TensorShape weights_mean_shape_block2r_batch4(out_shape_block2r_con4.z());
		const TensorShape weights_variance_shape_block2r_batch4(out_shape_block2r_con4.z());
		const TensorShape weights_gamma_shape_block2r_batch4(out_shape_block2r_con4.z());
		const TensorShape weights_beta_shape_block2r_batch4(out_shape_block2r_con4.z());
		weights_mean_block2r_batch4.allocator()->init(TensorInfo(weights_mean_shape_block2r_batch4, 1, DataType::F32));
		weights_variance_block2r_batch4.allocator()->init(TensorInfo(weights_variance_shape_block2r_batch4, 1, DataType::F32));
		weights_gamma_block2r_batch4.allocator()->init(TensorInfo(weights_gamma_shape_block2r_batch4, 1, DataType::F32));
		weights_beta_block2r_batch4.allocator()->init(TensorInfo(weights_beta_shape_block2r_batch4, 1, DataType::F32));
		out_block2r_batch4.allocator()->init(TensorInfo(out_shape_block2r_con4, 1, DataType::F32));
		out_block2r_act3.allocator()->init(TensorInfo(out_shape_block2r_con4, 1, DataType::F32));
   //conv-batch
		constexpr unsigned int kern_x_block2r_con5 = 1;
		constexpr unsigned int kern_y_block2r_con5 = 1;
		constexpr unsigned int ofm_block2r_con5 = 512;
		constexpr unsigned int out_x_block2r_con5 = 28;
		constexpr unsigned int out_y_block2r_con5 = 28;
		const TensorShape weights_shape_block2r_con5(kern_x_block2r_con5, kern_y_block2r_con5, out_shape_block2r_con4.z(), ofm_block2r_con5);
		const TensorShape out_shape_block2r_con5(out_x_block2r_con5, out_y_block2r_con5, weights_shape_block2r_con5[3]);
		weights_block2r_con5.allocator()->init(TensorInfo(weights_shape_block2r_con5, 1, DataType::F32));
		out_block2r_con5.allocator()->init(TensorInfo(out_shape_block2r_con5, 1, DataType::F32));
		const TensorShape weights_mean_shape_block2r_batch5(out_shape_block2r_con5.z());
		const TensorShape weights_variance_shape_block2r_batch5(out_shape_block2r_con5.z());
		const TensorShape weights_gamma_shape_block2r_batch5(out_shape_block2r_con5.z());
		const TensorShape weights_beta_shape_block2r_batch5(out_shape_block2r_con5.z());
		weights_mean_block2r_batch5.allocator()->init(TensorInfo(weights_mean_shape_block2r_batch5, 1, DataType::F32));
		weights_variance_block2r_batch5.allocator()->init(TensorInfo(weights_variance_shape_block2r_batch5, 1, DataType::F32));
		weights_gamma_block2r_batch5.allocator()->init(TensorInfo(weights_gamma_shape_block2r_batch5, 1, DataType::F32));
		weights_beta_block2r_batch5.allocator()->init(TensorInfo(weights_beta_shape_block2r_batch5, 1, DataType::F32));
		out_block2r_batch5.allocator()->init(TensorInfo(out_shape_block2r_con5, 1, DataType::F32));
   //add-act
		TensorShape out_shape_block2_1 = out_shape_block2r_con5;
		out_block2_add1.allocator()->init(TensorInfo(out_shape_block2_1, 1, DataType::F32));
		out_block2_act1.allocator()->init(TensorInfo(out_shape_block2_1, 1, DataType::F32));
   //conv-batch-act
        constexpr unsigned int kern_x_block2r_con6 = 1;
		constexpr unsigned int kern_y_block2r_con6 = 1;
		constexpr unsigned int ofm_block2r_con6 = 128;
		constexpr unsigned int out_x_block2r_con6 = 28;
		constexpr unsigned int out_y_block2r_con6 = 28;
		const TensorShape weights_shape_block2r_con6(kern_x_block2r_con6, kern_y_block2r_con6, out_shape_block2_1.z(), ofm_block2r_con6);
		const TensorShape out_shape_block2r_con6(out_x_block2r_con6, out_y_block2r_con6, weights_shape_block2r_con6[3]);
		weights_block2r_con6.allocator()->init(TensorInfo(weights_shape_block2r_con6, 1, DataType::F32));
		out_block2r_con6.allocator()->init(TensorInfo(out_shape_block2r_con6, 1, DataType::F32));
		const TensorShape weights_mean_shape_block2r_batch6(out_shape_block2r_con6.z());
		const TensorShape weights_variance_shape_block2r_batch6(out_shape_block2r_con6.z());
		const TensorShape weights_gamma_shape_block2r_batch6(out_shape_block2r_con6.z());
		const TensorShape weights_beta_shape_block2r_batch6(out_shape_block2r_con6.z());
		weights_mean_block2r_batch6.allocator()->init(TensorInfo(weights_mean_shape_block2r_batch6, 1, DataType::F32));
		weights_variance_block2r_batch6.allocator()->init(TensorInfo(weights_variance_shape_block2r_batch6, 1, DataType::F32));
		weights_gamma_block2r_batch6.allocator()->init(TensorInfo(weights_gamma_shape_block2r_batch6, 1, DataType::F32));
		weights_beta_block2r_batch6.allocator()->init(TensorInfo(weights_beta_shape_block2r_batch6, 1, DataType::F32));
		out_block2r_batch6.allocator()->init(TensorInfo(out_shape_block2r_con6, 1, DataType::F32));
		out_block2r_act4.allocator()->init(TensorInfo(out_shape_block2r_con6, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block2r_con7 = 3;
		constexpr unsigned int kern_y_block2r_con7 = 3;
		constexpr unsigned int ofm_block2r_con7 = 128;
		constexpr unsigned int out_x_block2r_con7 = 28;
		constexpr unsigned int out_y_block2r_con7 = 28;
		const TensorShape weights_shape_block2r_con7(kern_x_block2r_con7, kern_y_block2r_con7, out_shape_block2r_con6.z(), ofm_block2r_con7);
		const TensorShape out_shape_block2r_con7(out_x_block2r_con7, out_y_block2r_con7, weights_shape_block2r_con7[3]);
		weights_block2r_con7.allocator()->init(TensorInfo(weights_shape_block2r_con7, 1, DataType::F32));
		out_block2r_con7.allocator()->init(TensorInfo(out_shape_block2r_con7, 1, DataType::F32));
		const TensorShape weights_mean_shape_block2r_batch7(out_shape_block2r_con7.z());
		const TensorShape weights_variance_shape_block2r_batch7(out_shape_block2r_con7.z());
		const TensorShape weights_gamma_shape_block2r_batch7(out_shape_block2r_con7.z());
		const TensorShape weights_beta_shape_block2r_batch7(out_shape_block2r_con7.z());
		weights_mean_block2r_batch7.allocator()->init(TensorInfo(weights_mean_shape_block2r_batch7, 1, DataType::F32));
		weights_variance_block2r_batch7.allocator()->init(TensorInfo(weights_variance_shape_block2r_batch7, 1, DataType::F32));
		weights_gamma_block2r_batch7.allocator()->init(TensorInfo(weights_gamma_shape_block2r_batch7, 1, DataType::F32));
		weights_beta_block2r_batch7.allocator()->init(TensorInfo(weights_beta_shape_block2r_batch7, 1, DataType::F32));
		out_block2r_batch7.allocator()->init(TensorInfo(out_shape_block2r_con7, 1, DataType::F32));
		out_block2r_act5.allocator()->init(TensorInfo(out_shape_block2r_con7, 1, DataType::F32));
   //conv-batch
		constexpr unsigned int kern_x_block2r_con8 = 1;
		constexpr unsigned int kern_y_block2r_con8 = 1;
		constexpr unsigned int ofm_block2r_con8 = 512;
		constexpr unsigned int out_x_block2r_con8 = 28;
		constexpr unsigned int out_y_block2r_con8 = 28;
		const TensorShape weights_shape_block2r_con8(kern_x_block2r_con8, kern_y_block2r_con8, out_shape_block2r_con7.z(), ofm_block2r_con8);
		const TensorShape out_shape_block2r_con8(out_x_block2r_con8, out_y_block2r_con8, weights_shape_block2r_con8[3]);
		weights_block2r_con8.allocator()->init(TensorInfo(weights_shape_block2r_con8, 1, DataType::F32));
		out_block2r_con8.allocator()->init(TensorInfo(out_shape_block2r_con8, 1, DataType::F32));
		const TensorShape weights_mean_shape_block2r_batch8(out_shape_block2r_con8.z());
		const TensorShape weights_variance_shape_block2r_batch8(out_shape_block2r_con8.z());
		const TensorShape weights_gamma_shape_block2r_batch8(out_shape_block2r_con8.z());
		const TensorShape weights_beta_shape_block2r_batch8(out_shape_block2r_con8.z());
		weights_mean_block2r_batch8.allocator()->init(TensorInfo(weights_mean_shape_block2r_batch8, 1, DataType::F32));
		weights_variance_block2r_batch8.allocator()->init(TensorInfo(weights_variance_shape_block2r_batch8, 1, DataType::F32));
		weights_gamma_block2r_batch8.allocator()->init(TensorInfo(weights_gamma_shape_block2r_batch8, 1, DataType::F32));
		weights_beta_block2r_batch8.allocator()->init(TensorInfo(weights_beta_shape_block2r_batch8, 1, DataType::F32));
		out_block2r_batch8.allocator()->init(TensorInfo(out_shape_block2r_con8, 1, DataType::F32));
   //add-act
		TensorShape out_shape_block2_2 = out_shape_block2r_con8;
		out_block2_add2.allocator()->init(TensorInfo(out_shape_block2_2, 1, DataType::F32));
		out_block2_act2.allocator()->init(TensorInfo(out_shape_block2_2, 1, DataType::F32));
   //conv-batch-act
        constexpr unsigned int kern_x_block2r_con9 = 1;
		constexpr unsigned int kern_y_block2r_con9 = 1;
		constexpr unsigned int ofm_block2r_con9 = 128;
		constexpr unsigned int out_x_block2r_con9 = 28;
		constexpr unsigned int out_y_block2r_con9 = 28;
		const TensorShape weights_shape_block2r_con9(kern_x_block2r_con9, kern_y_block2r_con9, out_shape_block2_2.z(), ofm_block2r_con9);
		const TensorShape out_shape_block2r_con9(out_x_block2r_con9, out_y_block2r_con9, weights_shape_block2r_con9[3]);
		weights_block2r_con9.allocator()->init(TensorInfo(weights_shape_block2r_con9, 1, DataType::F32));
		out_block2r_con9.allocator()->init(TensorInfo(out_shape_block2r_con9, 1, DataType::F32));
		const TensorShape weights_mean_shape_block2r_batch9(out_shape_block2r_con9.z());
		const TensorShape weights_variance_shape_block2r_batch9(out_shape_block2r_con9.z());
		const TensorShape weights_gamma_shape_block2r_batch9(out_shape_block2r_con9.z());
		const TensorShape weights_beta_shape_block2r_batch9(out_shape_block2r_con9.z());
		weights_mean_block2r_batch9.allocator()->init(TensorInfo(weights_mean_shape_block2r_batch9, 1, DataType::F32));
		weights_variance_block2r_batch9.allocator()->init(TensorInfo(weights_variance_shape_block2r_batch9, 1, DataType::F32));
		weights_gamma_block2r_batch9.allocator()->init(TensorInfo(weights_gamma_shape_block2r_batch9, 1, DataType::F32));
		weights_beta_block2r_batch9.allocator()->init(TensorInfo(weights_beta_shape_block2r_batch9, 1, DataType::F32));
		out_block2r_batch9.allocator()->init(TensorInfo(out_shape_block2r_con9, 1, DataType::F32));
		out_block2r_act6.allocator()->init(TensorInfo(out_shape_block2r_con9, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block2r_con10 = 3;
		constexpr unsigned int kern_y_block2r_con10 = 3;
		constexpr unsigned int ofm_block2r_con10 = 128;
		constexpr unsigned int out_x_block2r_con10 = 14;
		constexpr unsigned int out_y_block2r_con10 = 14;
		const TensorShape weights_shape_block2r_con10(kern_x_block2r_con10, kern_y_block2r_con10, out_shape_block2r_con9.z(), ofm_block2r_con10);
		const TensorShape out_shape_block2r_con10(out_x_block2r_con10, out_y_block2r_con10, weights_shape_block2r_con10[3]);
		weights_block2r_con10.allocator()->init(TensorInfo(weights_shape_block2r_con10, 1, DataType::F32));
		out_block2r_con10.allocator()->init(TensorInfo(out_shape_block2r_con10, 1, DataType::F32));
		const TensorShape weights_mean_shape_block2r_batch10(out_shape_block2r_con10.z());
		const TensorShape weights_variance_shape_block2r_batch10(out_shape_block2r_con10.z());
		const TensorShape weights_gamma_shape_block2r_batch10(out_shape_block2r_con10.z());
		const TensorShape weights_beta_shape_block2r_batch10(out_shape_block2r_con10.z());
		weights_mean_block2r_batch10.allocator()->init(TensorInfo(weights_mean_shape_block2r_batch10, 1, DataType::F32));
		weights_variance_block2r_batch10.allocator()->init(TensorInfo(weights_variance_shape_block2r_batch10, 1, DataType::F32));
		weights_gamma_block2r_batch10.allocator()->init(TensorInfo(weights_gamma_shape_block2r_batch10, 1, DataType::F32));
		weights_beta_block2r_batch10.allocator()->init(TensorInfo(weights_beta_shape_block2r_batch10, 1, DataType::F32));
		out_block2r_batch10.allocator()->init(TensorInfo(out_shape_block2r_con10, 1, DataType::F32));
		out_block2r_act7.allocator()->init(TensorInfo(out_shape_block2r_con10, 1, DataType::F32));
   //conv-batch
		constexpr unsigned int kern_x_block2r_con11 = 1;
		constexpr unsigned int kern_y_block2r_con11= 1;
		constexpr unsigned int ofm_block2r_con11 = 512;
		constexpr unsigned int out_x_block2r_con11 = 14;
		constexpr unsigned int out_y_block2r_con11 = 14;
		const TensorShape weights_shape_block2r_con11(kern_x_block2r_con11, kern_y_block2r_con11, out_shape_block2r_con10.z(), ofm_block2r_con11);
		const TensorShape out_shape_block2r_con11(out_x_block2r_con11, out_y_block2r_con11, weights_shape_block2r_con11[3]);
		weights_block2r_con11.allocator()->init(TensorInfo(weights_shape_block2r_con11, 1, DataType::F32));
		out_block2r_con11.allocator()->init(TensorInfo(out_shape_block2r_con11, 1, DataType::F32));
		const TensorShape weights_mean_shape_block2r_batch11(out_shape_block2r_con11.z());
		const TensorShape weights_variance_shape_block2r_batch11(out_shape_block2r_con11.z());
		const TensorShape weights_gamma_shape_block2r_batch11(out_shape_block2r_con11.z());
		const TensorShape weights_beta_shape_block2r_batch11(out_shape_block2r_con11.z());
		weights_mean_block2r_batch11.allocator()->init(TensorInfo(weights_mean_shape_block2r_batch11, 1, DataType::F32));
		weights_variance_block2r_batch11.allocator()->init(TensorInfo(weights_variance_shape_block2r_batch11, 1, DataType::F32));
		weights_gamma_block2r_batch11.allocator()->init(TensorInfo(weights_gamma_shape_block2r_batch11, 1, DataType::F32));
		weights_beta_block2r_batch11.allocator()->init(TensorInfo(weights_beta_shape_block2r_batch11, 1, DataType::F32));
		out_block2r_batch11.allocator()->init(TensorInfo(out_shape_block2r_con11, 1, DataType::F32));
	//pooling
		TensorShape out_shape_block2l_pool0 = out_shape_block2_2;
		out_shape_block2l_pool0.set(0, out_shape_block2l_pool0.x() / 2); 
		out_shape_block2l_pool0.set(1, out_shape_block2l_pool0.y() / 2);
		out_block2l_pool0.allocator()->init(TensorInfo(out_shape_block2l_pool0, 1, DataType::F32));
   //add-act
		TensorShape out_shape_block2_3 = out_shape_block2r_con11;
		out_block2_add3.allocator()->init(TensorInfo(out_shape_block2_3, 1, DataType::F32));
		out_block2_act3.allocator()->init(TensorInfo(out_shape_block2_3, 1, DataType::F32));

//block3
   //conv-batch-act
        constexpr unsigned int kern_x_block3r_con0 = 1;
		constexpr unsigned int kern_y_block3r_con0 = 1;
		constexpr unsigned int ofm_block3r_con0 = 256;
		constexpr unsigned int out_x_block3r_con0 = 14;
		constexpr unsigned int out_y_block3r_con0 = 14;
		const TensorShape weights_shape_block3r_con0(kern_x_block3r_con0, kern_y_block3r_con0, out_shape_block2_3.z(), ofm_block3r_con0);
		const TensorShape out_shape_block3r_con0(out_x_block3r_con0, out_y_block3r_con0, weights_shape_block3r_con0[3]);
		weights_block3r_con0.allocator()->init(TensorInfo(weights_shape_block3r_con0, 1, DataType::F32));
		out_block3r_con0.allocator()->init(TensorInfo(out_shape_block3r_con0, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch0(out_shape_block3r_con0.z());
		const TensorShape weights_variance_shape_block3r_batch0(out_shape_block3r_con0.z());
		const TensorShape weights_gamma_shape_block3r_batch0(out_shape_block3r_con0.z());
		const TensorShape weights_beta_shape_block3r_batch0(out_shape_block3r_con0.z());
		weights_mean_block3r_batch0.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch0, 1, DataType::F32));
		weights_variance_block3r_batch0.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch0, 1, DataType::F32));
		weights_gamma_block3r_batch0.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch0, 1, DataType::F32));
		weights_beta_block3r_batch0.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch0, 1, DataType::F32));
		out_block3r_batch0.allocator()->init(TensorInfo(out_shape_block3r_con0, 1, DataType::F32));
		out_block3r_act0.allocator()->init(TensorInfo(out_shape_block3r_con0, 1, DataType::F32));
	//conv-batch-act
		constexpr unsigned int kern_x_block3r_con1 = 3;
		constexpr unsigned int kern_y_block3r_con1 = 3;
		constexpr unsigned int ofm_block3r_con1 = 256;
		constexpr unsigned int out_x_block3r_con1 = 14;
		constexpr unsigned int out_y_block3r_con1 = 14;
		const TensorShape weights_shape_block3r_con1(kern_x_block3r_con1, kern_y_block3r_con1, out_shape_block3r_con0.z(), ofm_block3r_con1);
		const TensorShape out_shape_block3r_con1(out_x_block3r_con1, out_y_block3r_con1, weights_shape_block3r_con1[3]);
		weights_block3r_con1.allocator()->init(TensorInfo(weights_shape_block3r_con1, 1, DataType::F32));
		out_block3r_con1.allocator()->init(TensorInfo(out_shape_block3r_con1, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch1(out_shape_block3r_con1.z());
		const TensorShape weights_variance_shape_block3r_batch1(out_shape_block3r_con1.z());
		const TensorShape weights_gamma_shape_block3r_batch1(out_shape_block3r_con1.z());
		const TensorShape weights_beta_shape_block3r_batch1(out_shape_block3r_con1.z());
		weights_mean_block3r_batch1.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch1, 1, DataType::F32));
		weights_variance_block3r_batch1.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch1, 1, DataType::F32));
		weights_gamma_block3r_batch1.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch1, 1, DataType::F32));
		weights_beta_block3r_batch1.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch1, 1, DataType::F32));
		out_block3r_batch1.allocator()->init(TensorInfo(out_shape_block3r_con1, 1, DataType::F32));
		out_block3r_act1.allocator()->init(TensorInfo(out_shape_block3r_con1, 1, DataType::F32));
   //conv-batch		
		constexpr unsigned int kern_x_block3r_con2 = 1;
		constexpr unsigned int kern_y_block3r_con2 = 1;
		constexpr unsigned int ofm_block3r_con2 = 1024;
		constexpr unsigned int out_x_block3r_con2 = 14;
		constexpr unsigned int out_y_block3r_con2 = 14;
		const TensorShape weights_shape_block3r_con2(kern_x_block3r_con2, kern_y_block3r_con2, out_shape_block3r_con1.z(), ofm_block3r_con2);
		const TensorShape out_shape_block3r_con2(out_x_block3r_con2, out_y_block3r_con2, weights_shape_block3r_con2[3]);
		weights_block3r_con2.allocator()->init(TensorInfo(weights_shape_block3r_con2, 1, DataType::F32));
		out_block3r_con2.allocator()->init(TensorInfo(out_shape_block3r_con2, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch2(out_shape_block3r_con2.z());
		const TensorShape weights_variance_shape_block3r_batch2(out_shape_block3r_con2.z());
		const TensorShape weights_gamma_shape_block3r_batch2(out_shape_block3r_con2.z());
		const TensorShape weights_beta_shape_block3r_batch2(out_shape_block3r_con2.z());
		weights_mean_block3r_batch2.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch2, 1, DataType::F32));
		weights_variance_block3r_batch2.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch2, 1, DataType::F32));
		weights_gamma_block3r_batch2.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch2, 1, DataType::F32));
		weights_beta_block3r_batch2.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch2, 1, DataType::F32));
		out_block3r_batch2.allocator()->init(TensorInfo(out_shape_block3r_con2, 1, DataType::F32));
   //conv-batch
		constexpr unsigned int kern_x_block3l_con0 = 1;
		constexpr unsigned int kern_y_block3l_con0 = 1;
		constexpr unsigned int ofm_block3l_con0 = 1024;
		constexpr unsigned int out_x_block3l_con0 = 14;
		constexpr unsigned int out_y_block3l_con0 = 14;
		const TensorShape weights_shape_block3l_con0(kern_x_block3l_con0, kern_y_block3l_con0, out_shape_block2_3.z(), ofm_block3l_con0);
		const TensorShape out_shape_block3l_con0(out_x_block3l_con0, out_y_block3l_con0, weights_shape_block3l_con0[3]);
		weights_block3l_con0.allocator()->init(TensorInfo(weights_shape_block3l_con0, 1, DataType::F32));
		out_block3l_con0.allocator()->init(TensorInfo(out_shape_block3l_con0, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3l_batch0(out_shape_block3l_con0.z());
		const TensorShape weights_variance_shape_block3l_batch0(out_shape_block3l_con0.z());
		const TensorShape weights_gamma_shape_block3l_batch0(out_shape_block3l_con0.z());
		const TensorShape weights_beta_shape_block3l_batch0(out_shape_block3l_con0.z());
		weights_mean_block3l_batch0.allocator()->init(TensorInfo(weights_mean_shape_block3l_batch0, 1, DataType::F32));
		weights_variance_block3l_batch0.allocator()->init(TensorInfo(weights_variance_shape_block3l_batch0, 1, DataType::F32));
		weights_gamma_block3l_batch0.allocator()->init(TensorInfo(weights_gamma_shape_block3l_batch0, 1, DataType::F32));
		weights_beta_block3l_batch0.allocator()->init(TensorInfo(weights_beta_shape_block3l_batch0, 1, DataType::F32));
		out_block3l_batch0.allocator()->init(TensorInfo(out_shape_block3l_con0, 1, DataType::F32));
   //add-act
		TensorShape out_shape_block3_0 = out_shape_block3r_con2;
		out_block3_add0.allocator()->init(TensorInfo(out_shape_block3_0, 1, DataType::F32));
		out_block3_act0.allocator()->init(TensorInfo(out_shape_block3_0, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block3r_con3 = 1;
		constexpr unsigned int kern_y_block3r_con3 = 1;
		constexpr unsigned int ofm_block3r_con3 = 256;
		constexpr unsigned int out_x_block3r_con3 = 14;
		constexpr unsigned int out_y_block3r_con3 = 14;
		const TensorShape weights_shape_block3r_con3(kern_x_block3r_con3, kern_y_block3r_con3, out_shape_block3_0.z(), ofm_block3r_con3);
		const TensorShape out_shape_block3r_con3(out_x_block3r_con3, out_y_block3r_con3, weights_shape_block3r_con3[3]);
		weights_block3r_con3.allocator()->init(TensorInfo(weights_shape_block3r_con3, 1, DataType::F32));
		out_block3r_con3.allocator()->init(TensorInfo(out_shape_block3r_con3, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch3(out_shape_block3r_con3.z());
		const TensorShape weights_variance_shape_block3r_batch3(out_shape_block3r_con3.z());
		const TensorShape weights_gamma_shape_block3r_batch3(out_shape_block3r_con3.z());
		const TensorShape weights_beta_shape_block3r_batch3(out_shape_block3r_con3.z());
		weights_mean_block3r_batch3.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch3, 1, DataType::F32));
		weights_variance_block3r_batch3.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch3, 1, DataType::F32));
		weights_gamma_block3r_batch3.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch3, 1, DataType::F32));
		weights_beta_block3r_batch3.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch3, 1, DataType::F32));
		out_block3r_batch3.allocator()->init(TensorInfo(out_shape_block3r_con3, 1, DataType::F32));
		out_block3r_act2.allocator()->init(TensorInfo(out_shape_block3r_con3, 1, DataType::F32));
   //conv-batch-act		
		constexpr unsigned int kern_x_block3r_con4 = 3;
		constexpr unsigned int kern_y_block3r_con4 = 3;
		constexpr unsigned int ofm_block3r_con4 = 256;
		constexpr unsigned int out_x_block3r_con4 = 14;
		constexpr unsigned int out_y_block3r_con4 = 14;
		const TensorShape weights_shape_block3r_con4(kern_x_block3r_con4, kern_y_block3r_con4, out_shape_block3r_con3.z(), ofm_block3r_con4);
		const TensorShape out_shape_block3r_con4(out_x_block3r_con4, out_y_block3r_con4, weights_shape_block3r_con4[3]);
		weights_block3r_con4.allocator()->init(TensorInfo(weights_shape_block3r_con4, 1, DataType::F32));
		out_block3r_con4.allocator()->init(TensorInfo(out_shape_block3r_con4, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch4(out_shape_block3r_con4.z());
		const TensorShape weights_variance_shape_block3r_batch4(out_shape_block3r_con4.z());
		const TensorShape weights_gamma_shape_block3r_batch4(out_shape_block3r_con4.z());
		const TensorShape weights_beta_shape_block3r_batch4(out_shape_block3r_con4.z());
		weights_mean_block3r_batch4.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch4, 1, DataType::F32));
		weights_variance_block3r_batch4.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch4, 1, DataType::F32));
		weights_gamma_block3r_batch4.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch4, 1, DataType::F32));
		weights_beta_block3r_batch4.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch4, 1, DataType::F32));
		out_block3r_batch4.allocator()->init(TensorInfo(out_shape_block3r_con4, 1, DataType::F32));
		out_block3r_act3.allocator()->init(TensorInfo(out_shape_block3r_con4, 1, DataType::F32));
   //conv-batch		
		constexpr unsigned int kern_x_block3r_con5 = 1;
		constexpr unsigned int kern_y_block3r_con5 = 1;
		constexpr unsigned int ofm_block3r_con5 = 1024;
		constexpr unsigned int out_x_block3r_con5 = 14;
		constexpr unsigned int out_y_block3r_con5 = 14;
		const TensorShape weights_shape_block3r_con5(kern_x_block3r_con5, kern_y_block3r_con5, out_shape_block3r_con4.z(), ofm_block3r_con5);
		const TensorShape out_shape_block3r_con5(out_x_block3r_con5, out_y_block3r_con5, weights_shape_block3r_con5[3]);
		weights_block3r_con5.allocator()->init(TensorInfo(weights_shape_block3r_con5, 1, DataType::F32));
		out_block3r_con5.allocator()->init(TensorInfo(out_shape_block3r_con5, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch5(out_shape_block3r_con5.z());
		const TensorShape weights_variance_shape_block3r_batch5(out_shape_block3r_con5.z());
		const TensorShape weights_gamma_shape_block3r_batch5(out_shape_block3r_con5.z());
		const TensorShape weights_beta_shape_block3r_batch5(out_shape_block3r_con5.z());
		weights_mean_block3r_batch5.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch5, 1, DataType::F32));
		weights_variance_block3r_batch5.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch5, 1, DataType::F32));
		weights_gamma_block3r_batch5.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch5, 1, DataType::F32));
		weights_beta_block3r_batch5.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch5, 1, DataType::F32));
		out_block3r_batch5.allocator()->init(TensorInfo(out_shape_block3r_con5, 1, DataType::F32));
   //add-act		
		TensorShape out_shape_block3_1 = out_shape_block3r_con5;
		out_block3_add1.allocator()->init(TensorInfo(out_shape_block3_1, 1, DataType::F32));
		out_block3_act1.allocator()->init(TensorInfo(out_shape_block3_1, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block3r_con6 = 1;
		constexpr unsigned int kern_y_block3r_con6 = 1;
		constexpr unsigned int ofm_block3r_con6 = 256;
		constexpr unsigned int out_x_block3r_con6 = 14;
		constexpr unsigned int out_y_block3r_con6 = 14;
		const TensorShape weights_shape_block3r_con6(kern_x_block3r_con6, kern_y_block3r_con6, out_shape_block3_1.z(), ofm_block3r_con6);
		const TensorShape out_shape_block3r_con6(out_x_block3r_con6, out_y_block3r_con6, weights_shape_block3r_con6[3]);
		weights_block3r_con6.allocator()->init(TensorInfo(weights_shape_block3r_con6, 1, DataType::F32));
		out_block3r_con6.allocator()->init(TensorInfo(out_shape_block3r_con6, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch6(out_shape_block3r_con6.z());
		const TensorShape weights_variance_shape_block3r_batch6(out_shape_block3r_con6.z());
		const TensorShape weights_gamma_shape_block3r_batch6(out_shape_block3r_con6.z());
		const TensorShape weights_beta_shape_block3r_batch6(out_shape_block3r_con6.z());
		weights_mean_block3r_batch6.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch6, 1, DataType::F32));
		weights_variance_block3r_batch6.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch6, 1, DataType::F32));
		weights_gamma_block3r_batch6.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch6, 1, DataType::F32));
		weights_beta_block3r_batch6.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch6, 1, DataType::F32));
		out_block3r_batch6.allocator()->init(TensorInfo(out_shape_block3r_con6, 1, DataType::F32));
		out_block3r_act4.allocator()->init(TensorInfo(out_shape_block3r_con6, 1, DataType::F32));
   //conv-batch-act		
		constexpr unsigned int kern_x_block3r_con7 = 3;
		constexpr unsigned int kern_y_block3r_con7 = 3;
		constexpr unsigned int ofm_block3r_con7 = 256;
		constexpr unsigned int out_x_block3r_con7 = 14;
		constexpr unsigned int out_y_block3r_con7 = 14;
		const TensorShape weights_shape_block3r_con7(kern_x_block3r_con7, kern_y_block3r_con7, out_shape_block3r_con6.z(), ofm_block3r_con7);
		const TensorShape out_shape_block3r_con7(out_x_block3r_con7, out_y_block3r_con7, weights_shape_block3r_con7[3]);
		weights_block3r_con7.allocator()->init(TensorInfo(weights_shape_block3r_con7, 1, DataType::F32));
		out_block3r_con7.allocator()->init(TensorInfo(out_shape_block3r_con7, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch7(out_shape_block3r_con7.z());
		const TensorShape weights_variance_shape_block3r_batch7(out_shape_block3r_con7.z());
		const TensorShape weights_gamma_shape_block3r_batch7(out_shape_block3r_con7.z());
		const TensorShape weights_beta_shape_block3r_batch7(out_shape_block3r_con7.z());
		weights_mean_block3r_batch7.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch7, 1, DataType::F32));
		weights_variance_block3r_batch7.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch7, 1, DataType::F32));
		weights_gamma_block3r_batch7.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch7, 1, DataType::F32));
		weights_beta_block3r_batch7.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch7, 1, DataType::F32));
		out_block3r_batch7.allocator()->init(TensorInfo(out_shape_block3r_con7, 1, DataType::F32));
		out_block3r_act5.allocator()->init(TensorInfo(out_shape_block3r_con7, 1, DataType::F32));
   //conv-batch		
		constexpr unsigned int kern_x_block3r_con8 = 1;
		constexpr unsigned int kern_y_block3r_con8 = 1;
		constexpr unsigned int ofm_block3r_con8 = 1024;
		constexpr unsigned int out_x_block3r_con8 = 14;
		constexpr unsigned int out_y_block3r_con8 = 14;
		const TensorShape weights_shape_block3r_con8(kern_x_block3r_con8, kern_y_block3r_con8, out_shape_block3r_con7.z(), ofm_block3r_con8);
		const TensorShape out_shape_block3r_con8(out_x_block3r_con8, out_y_block3r_con8, weights_shape_block3r_con8[3]);
		weights_block3r_con8.allocator()->init(TensorInfo(weights_shape_block3r_con8, 1, DataType::F32));
		out_block3r_con8.allocator()->init(TensorInfo(out_shape_block3r_con8, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch8(out_shape_block3r_con8.z());
		const TensorShape weights_variance_shape_block3r_batch8(out_shape_block3r_con8.z());
		const TensorShape weights_gamma_shape_block3r_batch8(out_shape_block3r_con8.z());
		const TensorShape weights_beta_shape_block3r_batch8(out_shape_block3r_con8.z());
		weights_mean_block3r_batch8.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch8, 1, DataType::F32));
		weights_variance_block3r_batch8.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch8, 1, DataType::F32));
		weights_gamma_block3r_batch8.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch8, 1, DataType::F32));
		weights_beta_block3r_batch8.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch8, 1, DataType::F32));
		out_block3r_batch8.allocator()->init(TensorInfo(out_shape_block3r_con8, 1, DataType::F32));
   //add-act		
		TensorShape out_shape_block3_2 = out_shape_block3r_con8;
		out_block3_add2.allocator()->init(TensorInfo(out_shape_block3_2, 1, DataType::F32));
		out_block3_act2.allocator()->init(TensorInfo(out_shape_block3_2, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block3r_con9 = 1;
		constexpr unsigned int kern_y_block3r_con9 = 1;
		constexpr unsigned int ofm_block3r_con9 = 256;
		constexpr unsigned int out_x_block3r_con9 = 14;
		constexpr unsigned int out_y_block3r_con9 = 14;
		const TensorShape weights_shape_block3r_con9(kern_x_block3r_con9, kern_y_block3r_con9, out_shape_block3_2.z(), ofm_block3r_con9);
		const TensorShape out_shape_block3r_con9(out_x_block3r_con9, out_y_block3r_con9, weights_shape_block3r_con9[3]);
		weights_block3r_con9.allocator()->init(TensorInfo(weights_shape_block3r_con9, 1, DataType::F32));
		out_block3r_con9.allocator()->init(TensorInfo(out_shape_block3r_con9, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch9(out_shape_block3r_con9.z());
		const TensorShape weights_variance_shape_block3r_batch9(out_shape_block3r_con9.z());
		const TensorShape weights_gamma_shape_block3r_batch9(out_shape_block3r_con9.z());
		const TensorShape weights_beta_shape_block3r_batch9(out_shape_block3r_con9.z());
		weights_mean_block3r_batch9.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch9, 1, DataType::F32));
		weights_variance_block3r_batch9.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch9, 1, DataType::F32));
		weights_gamma_block3r_batch9.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch9, 1, DataType::F32));
		weights_beta_block3r_batch9.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch9, 1, DataType::F32));
		out_block3r_batch9.allocator()->init(TensorInfo(out_shape_block3r_con9, 1, DataType::F32));
		out_block3r_act6.allocator()->init(TensorInfo(out_shape_block3r_con9, 1, DataType::F32));
   //conv-batch-act		
		constexpr unsigned int kern_x_block3r_con10 = 3;
		constexpr unsigned int kern_y_block3r_con10 = 3;
		constexpr unsigned int ofm_block3r_con10 = 256;
		constexpr unsigned int out_x_block3r_con10 = 14;
		constexpr unsigned int out_y_block3r_con10 = 14;
		const TensorShape weights_shape_block3r_con10(kern_x_block3r_con10, kern_y_block3r_con10, out_shape_block3r_con9.z(), ofm_block3r_con10);
		const TensorShape out_shape_block3r_con10(out_x_block3r_con10, out_y_block3r_con10, weights_shape_block3r_con10[3]);
		weights_block3r_con10.allocator()->init(TensorInfo(weights_shape_block3r_con10, 1, DataType::F32));
		out_block3r_con10.allocator()->init(TensorInfo(out_shape_block3r_con10, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch10(out_shape_block3r_con10.z());
		const TensorShape weights_variance_shape_block3r_batch10(out_shape_block3r_con10.z());
		const TensorShape weights_gamma_shape_block3r_batch10(out_shape_block3r_con10.z());
		const TensorShape weights_beta_shape_block3r_batch10(out_shape_block3r_con10.z());
		weights_mean_block3r_batch10.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch10, 1, DataType::F32));
		weights_variance_block3r_batch10.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch10, 1, DataType::F32));
		weights_gamma_block3r_batch10.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch10, 1, DataType::F32));
		weights_beta_block3r_batch10.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch10, 1, DataType::F32));
		out_block3r_batch10.allocator()->init(TensorInfo(out_shape_block3r_con10, 1, DataType::F32));
		out_block3r_act7.allocator()->init(TensorInfo(out_shape_block3r_con10, 1, DataType::F32));
   //conv-batch		
		constexpr unsigned int kern_x_block3r_con11 = 1;
		constexpr unsigned int kern_y_block3r_con11 = 1;
		constexpr unsigned int ofm_block3r_con11 = 1024;
		constexpr unsigned int out_x_block3r_con11 = 14;
		constexpr unsigned int out_y_block3r_con11 = 14;
		const TensorShape weights_shape_block3r_con11(kern_x_block3r_con11, kern_y_block3r_con11, out_shape_block3r_con10.z(), ofm_block3r_con11);
		const TensorShape out_shape_block3r_con11(out_x_block3r_con11, out_y_block3r_con11, weights_shape_block3r_con11[3]);
		weights_block3r_con11.allocator()->init(TensorInfo(weights_shape_block3r_con11, 1, DataType::F32));
		out_block3r_con11.allocator()->init(TensorInfo(out_shape_block3r_con11, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch11(out_shape_block3r_con11.z());
		const TensorShape weights_variance_shape_block3r_batch11(out_shape_block3r_con11.z());
		const TensorShape weights_gamma_shape_block3r_batch11(out_shape_block3r_con11.z());
		const TensorShape weights_beta_shape_block3r_batch11(out_shape_block3r_con11.z());
		weights_mean_block3r_batch11.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch11, 1, DataType::F32));
		weights_variance_block3r_batch11.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch11, 1, DataType::F32));
		weights_gamma_block3r_batch11.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch11, 1, DataType::F32));
		weights_beta_block3r_batch11.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch11, 1, DataType::F32));
		out_block3r_batch11.allocator()->init(TensorInfo(out_shape_block3r_con11, 1, DataType::F32));
   //add-act		
		TensorShape out_shape_block3_3 = out_shape_block3r_con11;
		out_block3_add3.allocator()->init(TensorInfo(out_shape_block3_3, 1, DataType::F32));
		out_block3_act3.allocator()->init(TensorInfo(out_shape_block3_3, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block3r_con12 = 1;
		constexpr unsigned int kern_y_block3r_con12 = 1;
		constexpr unsigned int ofm_block3r_con12 = 256;
		constexpr unsigned int out_x_block3r_con12 = 14;
		constexpr unsigned int out_y_block3r_con12 = 14;
		const TensorShape weights_shape_block3r_con12(kern_x_block3r_con12, kern_y_block3r_con12, out_shape_block3_3.z(), ofm_block3r_con12);
		const TensorShape out_shape_block3r_con12(out_x_block3r_con12, out_y_block3r_con12, weights_shape_block3r_con12[3]);
		weights_block3r_con12.allocator()->init(TensorInfo(weights_shape_block3r_con12, 1, DataType::F32));
		out_block3r_con12.allocator()->init(TensorInfo(out_shape_block3r_con12, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch12(out_shape_block3r_con12.z());
		const TensorShape weights_variance_shape_block3r_batch12(out_shape_block3r_con12.z());
		const TensorShape weights_gamma_shape_block3r_batch12(out_shape_block3r_con12.z());
		const TensorShape weights_beta_shape_block3r_batch12(out_shape_block3r_con12.z());
		weights_mean_block3r_batch12.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch12, 1, DataType::F32));
		weights_variance_block3r_batch12.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch12, 1, DataType::F32));
		weights_gamma_block3r_batch12.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch12, 1, DataType::F32));
		weights_beta_block3r_batch12.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch12, 1, DataType::F32));
		out_block3r_batch12.allocator()->init(TensorInfo(out_shape_block3r_con12, 1, DataType::F32));
		out_block3r_act8.allocator()->init(TensorInfo(out_shape_block3r_con12, 1, DataType::F32));
   //conv-batch-act		
		constexpr unsigned int kern_x_block3r_con13 = 3;
		constexpr unsigned int kern_y_block3r_con13 = 3;
		constexpr unsigned int ofm_block3r_con13 = 256;
		constexpr unsigned int out_x_block3r_con13 = 14;
		constexpr unsigned int out_y_block3r_con13 = 14;
		const TensorShape weights_shape_block3r_con13(kern_x_block3r_con13, kern_y_block3r_con13, out_shape_block3r_con12.z(), ofm_block3r_con13);
		const TensorShape out_shape_block3r_con13(out_x_block3r_con13, out_y_block3r_con13, weights_shape_block3r_con13[3]);
		weights_block3r_con13.allocator()->init(TensorInfo(weights_shape_block3r_con13, 1, DataType::F32));
		out_block3r_con13.allocator()->init(TensorInfo(out_shape_block3r_con13, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch13(out_shape_block3r_con13.z());
		const TensorShape weights_variance_shape_block3r_batch13(out_shape_block3r_con13.z());
		const TensorShape weights_gamma_shape_block3r_batch13(out_shape_block3r_con13.z());
		const TensorShape weights_beta_shape_block3r_batch13(out_shape_block3r_con13.z());
		weights_mean_block3r_batch13.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch13, 1, DataType::F32));
		weights_variance_block3r_batch13.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch13, 1, DataType::F32));
		weights_gamma_block3r_batch13.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch13, 1, DataType::F32));
		weights_beta_block3r_batch13.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch13, 1, DataType::F32));
		out_block3r_batch13.allocator()->init(TensorInfo(out_shape_block3r_con13, 1, DataType::F32));
		out_block3r_act9.allocator()->init(TensorInfo(out_shape_block3r_con13, 1, DataType::F32));
   //conv-batch		
		constexpr unsigned int kern_x_block3r_con14 = 1;
		constexpr unsigned int kern_y_block3r_con14 = 1;
		constexpr unsigned int ofm_block3r_con14 = 1024;
		constexpr unsigned int out_x_block3r_con14 = 14;
		constexpr unsigned int out_y_block3r_con14 = 14;
		const TensorShape weights_shape_block3r_con14(kern_x_block3r_con14, kern_y_block3r_con14, out_shape_block3r_con13.z(), ofm_block3r_con14);
		const TensorShape out_shape_block3r_con14(out_x_block3r_con14, out_y_block3r_con14, weights_shape_block3r_con14[3]);
		weights_block3r_con14.allocator()->init(TensorInfo(weights_shape_block3r_con14, 1, DataType::F32));
		out_block3r_con14.allocator()->init(TensorInfo(out_shape_block3r_con14, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch14(out_shape_block3r_con14.z());
		const TensorShape weights_variance_shape_block3r_batch14(out_shape_block3r_con14.z());
		const TensorShape weights_gamma_shape_block3r_batch14(out_shape_block3r_con14.z());
		const TensorShape weights_beta_shape_block3r_batch14(out_shape_block3r_con14.z());
		weights_mean_block3r_batch14.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch14, 1, DataType::F32));
		weights_variance_block3r_batch14.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch14, 1, DataType::F32));
		weights_gamma_block3r_batch14.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch14, 1, DataType::F32));
		weights_beta_block3r_batch14.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch14, 1, DataType::F32));
		out_block3r_batch14.allocator()->init(TensorInfo(out_shape_block3r_con14, 1, DataType::F32));
   //add-act		
		TensorShape out_shape_block3_4 = out_shape_block3r_con14;
		out_block3_add4.allocator()->init(TensorInfo(out_shape_block3_4, 1, DataType::F32));
		out_block3_act4.allocator()->init(TensorInfo(out_shape_block3_4, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block3r_con15 = 1;
		constexpr unsigned int kern_y_block3r_con15 = 1;
		constexpr unsigned int ofm_block3r_con15 = 256;
		constexpr unsigned int out_x_block3r_con15 = 14;
		constexpr unsigned int out_y_block3r_con15 = 14; 
		const TensorShape weights_shape_block3r_con15(kern_x_block3r_con15, kern_y_block3r_con15, out_shape_block3_4.z(), ofm_block3r_con15);
		const TensorShape out_shape_block3r_con15(out_x_block3r_con15, out_y_block3r_con15, weights_shape_block3r_con15[3]);
		weights_block3r_con15.allocator()->init(TensorInfo(weights_shape_block3r_con15, 1, DataType::F32));
		out_block3r_con15.allocator()->init(TensorInfo(out_shape_block3r_con15, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch15(out_shape_block3r_con15.z());
		const TensorShape weights_variance_shape_block3r_batch15(out_shape_block3r_con15.z());
		const TensorShape weights_gamma_shape_block3r_batch15(out_shape_block3r_con15.z());
		const TensorShape weights_beta_shape_block3r_batch15(out_shape_block3r_con15.z());
		weights_mean_block3r_batch15.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch15, 1, DataType::F32));
		weights_variance_block3r_batch15.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch15, 1, DataType::F32));
		weights_gamma_block3r_batch15.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch15, 1, DataType::F32));
		weights_beta_block3r_batch15.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch15, 1, DataType::F32));
		out_block3r_batch15.allocator()->init(TensorInfo(out_shape_block3r_con15, 1, DataType::F32));
		out_block3r_act10.allocator()->init(TensorInfo(out_shape_block3r_con15, 1, DataType::F32));
   //conv-batch-act		
		constexpr unsigned int kern_x_block3r_con16 = 3;
		constexpr unsigned int kern_y_block3r_con16 = 3;
		constexpr unsigned int ofm_block3r_con16 = 256;
		constexpr unsigned int out_x_block3r_con16 = 7;
		constexpr unsigned int out_y_block3r_con16 = 7;
		const TensorShape weights_shape_block3r_con16(kern_x_block3r_con16, kern_y_block3r_con16, out_shape_block3r_con15.z(), ofm_block3r_con16);
		const TensorShape out_shape_block3r_con16(out_x_block3r_con16, out_y_block3r_con16, weights_shape_block3r_con16[3]);
		weights_block3r_con16.allocator()->init(TensorInfo(weights_shape_block3r_con16, 1, DataType::F32));
		out_block3r_con16.allocator()->init(TensorInfo(out_shape_block3r_con16, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch16(out_shape_block3r_con16.z());
		const TensorShape weights_variance_shape_block3r_batch16(out_shape_block3r_con16.z());
		const TensorShape weights_gamma_shape_block3r_batch16(out_shape_block3r_con16.z());
		const TensorShape weights_beta_shape_block3r_batch16(out_shape_block3r_con16.z());
		weights_mean_block3r_batch16.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch16, 1, DataType::F32));
		weights_variance_block3r_batch16.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch16, 1, DataType::F32));
		weights_gamma_block3r_batch16.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch16, 1, DataType::F32));
		weights_beta_block3r_batch16.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch16, 1, DataType::F32));
		out_block3r_batch16.allocator()->init(TensorInfo(out_shape_block3r_con16, 1, DataType::F32));
		out_block3r_act11.allocator()->init(TensorInfo(out_shape_block3r_con16, 1, DataType::F32));
   //conv-batch	
		constexpr unsigned int kern_x_block3r_con17 = 1;
		constexpr unsigned int kern_y_block3r_con17 = 1;
		constexpr unsigned int ofm_block3r_con17 = 1024;
		constexpr unsigned int out_x_block3r_con17 = 7;
		constexpr unsigned int out_y_block3r_con17 = 7;
		const TensorShape weights_shape_block3r_con17(kern_x_block3r_con17, kern_y_block3r_con17, out_shape_block3r_con16.z(), ofm_block3r_con17);
		const TensorShape out_shape_block3r_con17(out_x_block3r_con17, out_y_block3r_con17, weights_shape_block3r_con17[3]);
		weights_block3r_con17.allocator()->init(TensorInfo(weights_shape_block3r_con17, 1, DataType::F32));
		out_block3r_con17.allocator()->init(TensorInfo(out_shape_block3r_con17, 1, DataType::F32));
		const TensorShape weights_mean_shape_block3r_batch17(out_shape_block3r_con17.z());
		const TensorShape weights_variance_shape_block3r_batch17(out_shape_block3r_con17.z());
		const TensorShape weights_gamma_shape_block3r_batch17(out_shape_block3r_con17.z());
		const TensorShape weights_beta_shape_block3r_batch17(out_shape_block3r_con17.z());
		weights_mean_block3r_batch17.allocator()->init(TensorInfo(weights_mean_shape_block3r_batch17, 1, DataType::F32));
		weights_variance_block3r_batch17.allocator()->init(TensorInfo(weights_variance_shape_block3r_batch17, 1, DataType::F32));
		weights_gamma_block3r_batch17.allocator()->init(TensorInfo(weights_gamma_shape_block3r_batch17, 1, DataType::F32));
		weights_beta_block3r_batch17.allocator()->init(TensorInfo(weights_beta_shape_block3r_batch17, 1, DataType::F32));
		out_block3r_batch17.allocator()->init(TensorInfo(out_shape_block3r_con17, 1, DataType::F32));
   //pooling		
		TensorShape out_shape_block3l_pool0 = out_shape_block3_4;
		out_shape_block3l_pool0.set(0, out_shape_block3l_pool0.x() / 2); 
		out_shape_block3l_pool0.set(1, out_shape_block3l_pool0.y() / 2);
		out_block3l_pool0.allocator()->init(TensorInfo(out_shape_block3l_pool0, 1, DataType::F32));

   //add-act		
		TensorShape out_shape_block3_5 = out_shape_block3r_con17;
		out_block3_add5.allocator()->init(TensorInfo(out_shape_block3_5, 1, DataType::F32));
		out_block3_act5.allocator()->init(TensorInfo(out_shape_block3_5, 1, DataType::F32));

//block4
   //conv-batch-act
		constexpr unsigned int kern_x_block4r_con0 = 1;
		constexpr unsigned int kern_y_block4r_con0 = 1;
		constexpr unsigned int ofm_block4r_con0 = 512;
		constexpr unsigned int out_x_block4r_con0 = 7;
		constexpr unsigned int out_y_block4r_con0 = 7;
		const TensorShape weights_shape_block4r_con0(kern_x_block4r_con0, kern_y_block4r_con0, out_shape_block3_5.z(), ofm_block4r_con0);
		const TensorShape out_shape_block4r_con0(out_x_block4r_con0, out_y_block4r_con0, weights_shape_block4r_con0[3]);
		weights_block4r_con0.allocator()->init(TensorInfo(weights_shape_block4r_con0, 1, DataType::F32));
		out_block4r_con0.allocator()->init(TensorInfo(out_shape_block4r_con0, 1, DataType::F32));
		const TensorShape weights_mean_shape_block4r_batch0(out_shape_block4r_con0.z());
		const TensorShape weights_variance_shape_block4r_batch0(out_shape_block4r_con0.z());
		const TensorShape weights_gamma_shape_block4r_batch0(out_shape_block4r_con0.z());
		const TensorShape weights_beta_shape_block4r_batch0(out_shape_block4r_con0.z());
		weights_mean_block4r_batch0.allocator()->init(TensorInfo(weights_mean_shape_block4r_batch0, 1, DataType::F32));
		weights_variance_block4r_batch0.allocator()->init(TensorInfo(weights_variance_shape_block4r_batch0, 1, DataType::F32));
		weights_gamma_block4r_batch0.allocator()->init(TensorInfo(weights_gamma_shape_block4r_batch0, 1, DataType::F32));
		weights_beta_block4r_batch0.allocator()->init(TensorInfo(weights_beta_shape_block4r_batch0, 1, DataType::F32));
		out_block4r_batch0.allocator()->init(TensorInfo(out_shape_block4r_con0, 1, DataType::F32));
		out_block4r_act0.allocator()->init(TensorInfo(out_shape_block4r_con0, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block4r_con1 = 3;
		constexpr unsigned int kern_y_block4r_con1 = 3;
		constexpr unsigned int ofm_block4r_con1 = 512;
		constexpr unsigned int out_x_block4r_con1 = 7;
		constexpr unsigned int out_y_block4r_con1 = 7;
		const TensorShape weights_shape_block4r_con1(kern_x_block4r_con1, kern_y_block4r_con1, out_shape_block4r_con0.z(), ofm_block4r_con1);
		const TensorShape out_shape_block4r_con1(out_x_block4r_con1, out_y_block4r_con1, weights_shape_block4r_con1[3]);
		weights_block4r_con1.allocator()->init(TensorInfo(weights_shape_block4r_con1, 1, DataType::F32));
		out_block4r_con1.allocator()->init(TensorInfo(out_shape_block4r_con1, 1, DataType::F32));
		const TensorShape weights_mean_shape_block4r_batch1(out_shape_block4r_con1.z());
		const TensorShape weights_variance_shape_block4r_batch1(out_shape_block4r_con1.z());
		const TensorShape weights_gamma_shape_block4r_batch1(out_shape_block4r_con1.z());
		const TensorShape weights_beta_shape_block4r_batch1(out_shape_block4r_con1.z());
		weights_mean_block4r_batch1.allocator()->init(TensorInfo(weights_mean_shape_block4r_batch1, 1, DataType::F32));
		weights_variance_block4r_batch1.allocator()->init(TensorInfo(weights_variance_shape_block4r_batch1, 1, DataType::F32));
		weights_gamma_block4r_batch1.allocator()->init(TensorInfo(weights_gamma_shape_block4r_batch1, 1, DataType::F32));
		weights_beta_block4r_batch1.allocator()->init(TensorInfo(weights_beta_shape_block4r_batch1, 1, DataType::F32));
		out_block4r_batch1.allocator()->init(TensorInfo(out_shape_block4r_con1, 1, DataType::F32));
		out_block4r_act1.allocator()->init(TensorInfo(out_shape_block4r_con1, 1, DataType::F32));
   //conv-batch
		constexpr unsigned int kern_x_block4r_con2 = 1;
		constexpr unsigned int kern_y_block4r_con2 = 1;
		constexpr unsigned int ofm_block4r_con2 = 2048;
		constexpr unsigned int out_x_block4r_con2 = 7;
		constexpr unsigned int out_y_block4r_con2 = 7;
		const TensorShape weights_shape_block4r_con2(kern_x_block4r_con2, kern_y_block4r_con2, out_shape_block4r_con1.z(), ofm_block4r_con2);
		const TensorShape out_shape_block4r_con2(out_x_block4r_con2, out_y_block4r_con2, weights_shape_block4r_con2[3]);
		weights_block4r_con2.allocator()->init(TensorInfo(weights_shape_block4r_con2, 1, DataType::F32));
		out_block4r_con2.allocator()->init(TensorInfo(out_shape_block4r_con2, 1, DataType::F32));
		const TensorShape weights_mean_shape_block4r_batch2(out_shape_block4r_con2.z());
		const TensorShape weights_variance_shape_block4r_batch2(out_shape_block4r_con2.z());
		const TensorShape weights_gamma_shape_block4r_batch2(out_shape_block4r_con2.z());
		const TensorShape weights_beta_shape_block4r_batch2(out_shape_block4r_con2.z());
		weights_mean_block4r_batch2.allocator()->init(TensorInfo(weights_mean_shape_block4r_batch2, 1, DataType::F32));
		weights_variance_block4r_batch2.allocator()->init(TensorInfo(weights_variance_shape_block4r_batch2, 1, DataType::F32));
		weights_gamma_block4r_batch2.allocator()->init(TensorInfo(weights_gamma_shape_block4r_batch2, 1, DataType::F32));
		weights_beta_block4r_batch2.allocator()->init(TensorInfo(weights_beta_shape_block4r_batch2, 1, DataType::F32));
		out_block4r_batch2.allocator()->init(TensorInfo(out_shape_block4r_con2, 1, DataType::F32));
  //conv-batch
		constexpr unsigned int kern_x_block4l_con0 = 1;
		constexpr unsigned int kern_y_block4l_con0 = 1;
		constexpr unsigned int ofm_block4l_con0 = 2048;
		constexpr unsigned int out_x_block4l_con0 = 7;
		constexpr unsigned int out_y_block4l_con0 = 7;
		const TensorShape weights_shape_block4l_con0(kern_x_block4l_con0, kern_y_block4l_con0, out_shape_block3_5.z(), ofm_block4l_con0);
		const TensorShape out_shape_block4l_con0(out_x_block4l_con0, out_y_block4l_con0, weights_shape_block4l_con0[3]);
		weights_block4l_con0.allocator()->init(TensorInfo(weights_shape_block4l_con0, 1, DataType::F32));
		out_block4l_con0.allocator()->init(TensorInfo(out_shape_block4l_con0, 1, DataType::F32));
		const TensorShape weights_mean_shape_block4l_batch0(out_shape_block4l_con0.z());
		const TensorShape weights_variance_shape_block4l_batch0(out_shape_block4l_con0.z());
		const TensorShape weights_gamma_shape_block4l_batch0(out_shape_block4l_con0.z());
		const TensorShape weights_beta_shape_block4l_batch0(out_shape_block4l_con0.z());
		weights_mean_block4l_batch0.allocator()->init(TensorInfo(weights_mean_shape_block4l_batch0, 1, DataType::F32));
		weights_variance_block4l_batch0.allocator()->init(TensorInfo(weights_variance_shape_block4l_batch0, 1, DataType::F32));
		weights_gamma_block4l_batch0.allocator()->init(TensorInfo(weights_gamma_shape_block4l_batch0, 1, DataType::F32));
		weights_beta_block4l_batch0.allocator()->init(TensorInfo(weights_beta_shape_block4l_batch0, 1, DataType::F32));
		out_block4l_batch0.allocator()->init(TensorInfo(out_shape_block4l_con0, 1, DataType::F32));
   //add-act
		TensorShape out_shape_block4_0 = out_shape_block4r_con2;
		out_block4_add0.allocator()->init(TensorInfo(out_shape_block4_0, 1, DataType::F32));
		out_block4_act0.allocator()->init(TensorInfo(out_shape_block4_0, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block4r_con3 = 1;
		constexpr unsigned int kern_y_block4r_con3 = 1;
		constexpr unsigned int ofm_block4r_con3 = 512;
		constexpr unsigned int out_x_block4r_con3 = 7;
		constexpr unsigned int out_y_block4r_con3 = 7;
		const TensorShape weights_shape_block4r_con3(kern_x_block4r_con3, kern_y_block4r_con3, out_shape_block4_0.z(), ofm_block4r_con3);
		const TensorShape out_shape_block4r_con3(out_x_block4r_con3, out_y_block4r_con3, weights_shape_block4r_con3[3]);
		weights_block4r_con3.allocator()->init(TensorInfo(weights_shape_block4r_con3, 1, DataType::F32));
		out_block4r_con3.allocator()->init(TensorInfo(out_shape_block4r_con3, 1, DataType::F32));
		const TensorShape weights_mean_shape_block4r_batch3(out_shape_block4r_con3.z());
		const TensorShape weights_variance_shape_block4r_batch3(out_shape_block4r_con3.z());
		const TensorShape weights_gamma_shape_block4r_batch3(out_shape_block4r_con3.z());
		const TensorShape weights_beta_shape_block4r_batch3(out_shape_block4r_con3.z());
		weights_mean_block4r_batch3.allocator()->init(TensorInfo(weights_mean_shape_block4r_batch3, 1, DataType::F32));
		weights_variance_block4r_batch3.allocator()->init(TensorInfo(weights_variance_shape_block4r_batch3, 1, DataType::F32));
		weights_gamma_block4r_batch3.allocator()->init(TensorInfo(weights_gamma_shape_block4r_batch3, 1, DataType::F32));
		weights_beta_block4r_batch3.allocator()->init(TensorInfo(weights_beta_shape_block4r_batch3, 1, DataType::F32));
		out_block4r_batch3.allocator()->init(TensorInfo(out_shape_block4r_con3, 1, DataType::F32));
		out_block4r_act2.allocator()->init(TensorInfo(out_shape_block4r_con3, 1, DataType::F32));
   //conv-batch-act		
		constexpr unsigned int kern_x_block4r_con4 = 3;
		constexpr unsigned int kern_y_block4r_con4 = 3;
		constexpr unsigned int ofm_block4r_con4 = 512;
		constexpr unsigned int out_x_block4r_con4 = 7;
		constexpr unsigned int out_y_block4r_con4 = 7;
		const TensorShape weights_shape_block4r_con4(kern_x_block4r_con4, kern_y_block4r_con4, out_shape_block4r_con3.z(), ofm_block4r_con4);
		const TensorShape out_shape_block4r_con4(out_x_block4r_con4, out_y_block4r_con4, weights_shape_block4r_con4[3]);
		weights_block4r_con4.allocator()->init(TensorInfo(weights_shape_block4r_con4, 1, DataType::F32));
		out_block4r_con4.allocator()->init(TensorInfo(out_shape_block4r_con4, 1, DataType::F32));
		const TensorShape weights_mean_shape_block4r_batch4(out_shape_block4r_con4.z());
		const TensorShape weights_variance_shape_block4r_batch4(out_shape_block4r_con4.z());
		const TensorShape weights_gamma_shape_block4r_batch4(out_shape_block4r_con4.z());
		const TensorShape weights_beta_shape_block4r_batch4(out_shape_block4r_con4.z());
		weights_mean_block4r_batch4.allocator()->init(TensorInfo(weights_mean_shape_block4r_batch4, 1, DataType::F32));
		weights_variance_block4r_batch4.allocator()->init(TensorInfo(weights_variance_shape_block4r_batch4, 1, DataType::F32));
		weights_gamma_block4r_batch4.allocator()->init(TensorInfo(weights_gamma_shape_block4r_batch4, 1, DataType::F32));
		weights_beta_block4r_batch4.allocator()->init(TensorInfo(weights_beta_shape_block4r_batch4, 1, DataType::F32));
		out_block4r_batch4.allocator()->init(TensorInfo(out_shape_block4r_con4, 1, DataType::F32));
		out_block4r_act3.allocator()->init(TensorInfo(out_shape_block4r_con4, 1, DataType::F32));
   //conv-batch		
		constexpr unsigned int kern_x_block4r_con5 = 1;
		constexpr unsigned int kern_y_block4r_con5 = 1;
		constexpr unsigned int ofm_block4r_con5 = 2048;
		constexpr unsigned int out_x_block4r_con5 = 7;
		constexpr unsigned int out_y_block4r_con5 = 7;
		const TensorShape weights_shape_block4r_con5(kern_x_block4r_con5, kern_y_block4r_con5, out_shape_block4r_con4.z(), ofm_block4r_con5);
		const TensorShape out_shape_block4r_con5(out_x_block4r_con5, out_y_block4r_con5, weights_shape_block4r_con5[3]);
		weights_block4r_con5.allocator()->init(TensorInfo(weights_shape_block4r_con5, 1, DataType::F32));
		out_block4r_con5.allocator()->init(TensorInfo(out_shape_block4r_con5, 1, DataType::F32));
		const TensorShape weights_mean_shape_block4r_batch5(out_shape_block4r_con5.z());
		const TensorShape weights_variance_shape_block4r_batch5(out_shape_block4r_con5.z());
		const TensorShape weights_gamma_shape_block4r_batch5(out_shape_block4r_con5.z());
		const TensorShape weights_beta_shape_block4r_batch5(out_shape_block4r_con5.z());
		weights_mean_block4r_batch5.allocator()->init(TensorInfo(weights_mean_shape_block4r_batch5, 1, DataType::F32));
		weights_variance_block4r_batch5.allocator()->init(TensorInfo(weights_variance_shape_block4r_batch5, 1, DataType::F32));
		weights_gamma_block4r_batch5.allocator()->init(TensorInfo(weights_gamma_shape_block4r_batch5, 1, DataType::F32));
		weights_beta_block4r_batch5.allocator()->init(TensorInfo(weights_beta_shape_block4r_batch5, 1, DataType::F32));
		out_block4r_batch5.allocator()->init(TensorInfo(out_shape_block4r_con5, 1, DataType::F32));
	//add-act
		TensorShape out_shape_block4_1 = out_shape_block4r_con5;
		out_block4_add1.allocator()->init(TensorInfo(out_shape_block4_1, 1, DataType::F32));
		out_block4_act1.allocator()->init(TensorInfo(out_shape_block4_1, 1, DataType::F32));
   //conv-batch-act
		constexpr unsigned int kern_x_block4r_con6 = 1;
		constexpr unsigned int kern_y_block4r_con6 = 1;
		constexpr unsigned int ofm_block4r_con6 = 512;
		constexpr unsigned int out_x_block4r_con6 = 7;
		constexpr unsigned int out_y_block4r_con6 = 7;
		const TensorShape weights_shape_block4r_con6(kern_x_block4r_con6, kern_y_block4r_con6, out_shape_block4_1.z(), ofm_block4r_con6);
		const TensorShape out_shape_block4r_con6(out_x_block4r_con6, out_y_block4r_con6, weights_shape_block4r_con6[3]);
		weights_block4r_con6.allocator()->init(TensorInfo(weights_shape_block4r_con6, 1, DataType::F32));
		out_block4r_con6.allocator()->init(TensorInfo(out_shape_block4r_con6, 1, DataType::F32));
		const TensorShape weights_mean_shape_block4r_batch6(out_shape_block4r_con6.z());
		const TensorShape weights_variance_shape_block4r_batch6(out_shape_block4r_con6.z());
		const TensorShape weights_gamma_shape_block4r_batch6(out_shape_block4r_con6.z());
		const TensorShape weights_beta_shape_block4r_batch6(out_shape_block4r_con6.z());
		weights_mean_block4r_batch6.allocator()->init(TensorInfo(weights_mean_shape_block4r_batch6, 1, DataType::F32));
		weights_variance_block4r_batch6.allocator()->init(TensorInfo(weights_variance_shape_block4r_batch6, 1, DataType::F32));
		weights_gamma_block4r_batch6.allocator()->init(TensorInfo(weights_gamma_shape_block4r_batch6, 1, DataType::F32));
		weights_beta_block4r_batch6.allocator()->init(TensorInfo(weights_beta_shape_block4r_batch6, 1, DataType::F32));
		out_block4r_batch6.allocator()->init(TensorInfo(out_shape_block4r_con6, 1, DataType::F32));
		out_block4r_act4.allocator()->init(TensorInfo(out_shape_block4r_con6, 1, DataType::F32));
   //conv-batch-act	
		constexpr unsigned int kern_x_block4r_con7 = 3;
		constexpr unsigned int kern_y_block4r_con7 = 3;
		constexpr unsigned int ofm_block4r_con7 = 512;
		constexpr unsigned int out_x_block4r_con7 = 7;
		constexpr unsigned int out_y_block4r_con7 = 7;
		const TensorShape weights_shape_block4r_con7(kern_x_block4r_con7, kern_y_block4r_con7, out_shape_block4r_con6.z(), ofm_block4r_con7);
		const TensorShape out_shape_block4r_con7(out_x_block4r_con7, out_y_block4r_con7, weights_shape_block4r_con7[3]);
		weights_block4r_con7.allocator()->init(TensorInfo(weights_shape_block4r_con7, 1, DataType::F32));
		out_block4r_con7.allocator()->init(TensorInfo(out_shape_block4r_con7, 1, DataType::F32));
		const TensorShape weights_mean_shape_block4r_batch7(out_shape_block4r_con7.z());
		const TensorShape weights_variance_shape_block4r_batch7(out_shape_block4r_con7.z());
		const TensorShape weights_gamma_shape_block4r_batch7(out_shape_block4r_con7.z());
		const TensorShape weights_beta_shape_block4r_batch7(out_shape_block4r_con7.z());
		weights_mean_block4r_batch7.allocator()->init(TensorInfo(weights_mean_shape_block4r_batch7, 1, DataType::F32));
		weights_variance_block4r_batch7.allocator()->init(TensorInfo(weights_variance_shape_block4r_batch7, 1, DataType::F32));
		weights_gamma_block4r_batch7.allocator()->init(TensorInfo(weights_gamma_shape_block4r_batch7, 1, DataType::F32));
		weights_beta_block4r_batch7.allocator()->init(TensorInfo(weights_beta_shape_block4r_batch7, 1, DataType::F32));
		out_block4r_batch7.allocator()->init(TensorInfo(out_shape_block4r_con7, 1, DataType::F32));
		out_block4r_act5.allocator()->init(TensorInfo(out_shape_block4r_con7, 1, DataType::F32));
   //conv-batch		
		constexpr unsigned int kern_x_block4r_con8 = 1;
		constexpr unsigned int kern_y_block4r_con8 = 1;
		constexpr unsigned int ofm_block4r_con8 = 2048;
		constexpr unsigned int out_x_block4r_con8 = 7;
		constexpr unsigned int out_y_block4r_con8 = 7;
		const TensorShape weights_shape_block4r_con8(kern_x_block4r_con8, kern_y_block4r_con8, out_shape_block4r_con7.z(), ofm_block4r_con8);
		const TensorShape out_shape_block4r_con8(out_x_block4r_con8, out_y_block4r_con8, weights_shape_block4r_con8[3]);
		weights_block4r_con8.allocator()->init(TensorInfo(weights_shape_block4r_con8, 1, DataType::F32));
		out_block4r_con8.allocator()->init(TensorInfo(out_shape_block4r_con8, 1, DataType::F32));
		const TensorShape weights_mean_shape_block4r_batch8(out_shape_block4r_con8.z());
		const TensorShape weights_variance_shape_block4r_batch8(out_shape_block4r_con8.z());
		const TensorShape weights_gamma_shape_block4r_batch8(out_shape_block4r_con8.z());
		const TensorShape weights_beta_shape_block4r_batch8(out_shape_block4r_con8.z());
		weights_mean_block4r_batch8.allocator()->init(TensorInfo(weights_mean_shape_block4r_batch8, 1, DataType::F32));
		weights_variance_block4r_batch8.allocator()->init(TensorInfo(weights_variance_shape_block4r_batch8, 1, DataType::F32));
		weights_gamma_block4r_batch8.allocator()->init(TensorInfo(weights_gamma_shape_block4r_batch8, 1, DataType::F32));
		weights_beta_block4r_batch8.allocator()->init(TensorInfo(weights_beta_shape_block4r_batch8, 1, DataType::F32));
		out_block4r_batch8.allocator()->init(TensorInfo(out_shape_block4r_con8, 1, DataType::F32));
		
	//add-act	
		TensorShape out_shape_block4_2 = out_shape_block4r_con8;
		out_block4_add2.allocator()->init(TensorInfo(out_shape_block4_2, 1, DataType::F32));
		out_block4_act2.allocator()->init(TensorInfo(out_shape_block4_2, 1, DataType::F32));
// block end  
       
//last pooling-conv-flatten-softmax
		const TensorShape out_shape_pool1 = out_shape_block4_2;
		out_shape_pool1.set(0, out_shape_pool1.x() / 7);
		out_shape_pool1.set(1, out_shape_pool1.y() / 7);
		out_pool1.allocator()->init(TensorInfo(out_shape_pool1, 1, DataType::F32));         
		constexpr unsigned int kern_x_con1 = 1;
		constexpr unsigned int kern_y_con1 = 1;
		constexpr unsigned int ofm_con1 = 1000;
		constexpr unsigned int out_x_con1 = 1;
		constexpr unsigned int out_y_con1 = 1;
		const TensorShape weights_shape_con1(kern_x_con1, kern_y_con1, out_shape_pool1.z(), ofm_con1);
		const TensorShape biases_shape_con1(weights_shape_con1[3]);
		const TensorShape out_shape_con1(out_x_con1, out_y_con1, weights_shape_con1[3]);
		weights_con1.allocator()->init(TensorInfo(weights_shape_con1, 1, DataType::F32));
		biases_con1.allocator()->init(TensorInfo(biases_shape_con1, 1, DataType::F32));
		out_con1.allocator()->init(TensorInfo(out_shape_con1, 1, DataType::F32));
		const TensorShape out_shape_flatten(out_shape_con1.x()*out_shape_con1.y()*out_shape_con1.z(),0);                     
		out_flatten.allocator()->init(TensorInfo(out_shape_flatten, 1, DataType::F32));
		const TensorShape out_shape_softmax(out_shape_flatten.x());
		out_softmax.allocator()->init(TensorInfo(out_shape_softmax, 1, DataType::F32));
//last end

//configure start
//first start
		con0.configure(&src, &weights_con0,nullptr, &out_con0, PadStrideInfo(2, 2, 3, 3));
		batch0.configure(&out_con0, &out_batch0, &weights_mean_batch0, &weights_variance_batch0, &weights_beta_batch0, &weights_gamma_batch0, 0.0000100099996416f);
		act0.configure(&out_batch0, &out_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		pool0.configure(&out_act0, &out_pool0, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR)));
//first end
//block start
// block1
		block1r_con0.configure(&out_pool0, &weights_block1r_con0, nullptr, &out_block1r_con0, PadStrideInfo(1, 1, 0, 0));
		block1r_batch0.configure(&out_block1r_con0, &out_block1r_batch0, &weights_mean_block1r_batch0, &weights_variance_block1r_batch0, &weights_beta_block1r_batch0, &weights_gamma_block1r_batch0, 0.0000100099996416f);
		block1r_act0.configure(&out_block1r_batch0, &out_block1r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block1r_con1.configure(&out_block1r_act0, &weights_block1r_con1, nullptr, &out_block1r_con1, PadStrideInfo(1, 1, 1, 1));
		block1r_batch1.configure(&out_block1r_con1, &out_block1r_batch1, &weights_mean_block1r_batch1, &weights_variance_block1r_batch1, &weights_beta_block1r_batch1, &weights_gamma_block1r_batch1, 0.0000100099996416f);
		block1r_act1.configure(&out_block1r_batch1, &out_block1r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block1r_con2.configure(&out_block1r_act1, &weights_block1r_con2, nullptr, &out_block1r_con2, PadStrideInfo(1, 1, 0, 0));
		block1r_batch2.configure(&out_block1r_con2, &out_block1r_batch2, &weights_mean_block1r_batch2, &weights_variance_block1r_batch2, &weights_beta_block1r_batch2, &weights_gamma_block1r_batch2, 0.0000100099996416f);
		block1l_con0.configure(&out_pool0, &weights_block1l_con0, nullptr, &out_block1l_con0, PadStrideInfo(1, 1, 0, 0));
		block1l_batch0.configure(&out_block1l_con0, &out_block1l_batch0, &weights_mean_block1l_batch0, &weights_variance_block1l_batch0, &weights_beta_block1l_batch0, &weights_gamma_block1l_batch0, 0.0000100099996416f);
		block1_add0.configure(&out_block1r_batch2, &out_block1l_batch0, &out_block1_add0,A);
		block1_act0.configure(&out_block1_add0, &out_block1_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		
		block1r_con3.configure(&out_block1_act0, &weights_block1r_con3, nullptr, &out_block1r_con3, PadStrideInfo(1, 1, 0, 0));
		block1r_batch3.configure(&out_block1r_con3, &out_block1r_batch3, &weights_mean_block1r_batch3, &weights_variance_block1r_batch3, &weights_beta_block1r_batch3, &weights_gamma_block1r_batch3, 0.0000100099996416f);
		block1r_act2.configure(&out_block1r_batch3, &out_block1r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block1r_con4.configure(&out_block1r_act2, &weights_block1r_con4, nullptr, &out_block1r_con4, PadStrideInfo(1, 1, 1, 1));
		block1r_batch4.configure(&out_block1r_con4, &out_block1r_batch4, &weights_mean_block1r_batch4, &weights_variance_block1r_batch4, &weights_beta_block1r_batch4, &weights_gamma_block1r_batch4, 0.0000100099996416f);
		block1r_act3.configure(&out_block1r_batch4, &out_block1r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block1r_con5.configure(&out_block1r_act3, &weights_block1r_con5, nullptr, &out_block1r_con5, PadStrideInfo(1, 1, 0, 0));
		block1r_batch5.configure(&out_block1r_con5, &out_block1r_batch5, &weights_mean_block1r_batch5, &weights_variance_block1r_batch5, &weights_beta_block1r_batch5, &weights_gamma_block1r_batch5, 0.0000100099996416f);
		block1_add1.configure(&out_block1r_batch5, &out_block1_act0, &out_block1_add1,A);
		block1_act1.configure(&out_block1_add1, &out_block1_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		
		block1r_con6.configure(&out_block1_act1, &weights_block1r_con6, nullptr, &out_block1r_con6, PadStrideInfo(1, 1, 0, 0));
		block1r_batch6.configure(&out_block1r_con6, &out_block1r_batch6, &weights_mean_block1r_batch6, &weights_variance_block1r_batch6, &weights_beta_block1r_batch6, &weights_gamma_block1r_batch6, 0.0000100099996416f);
		block1r_act4.configure(&out_block1r_batch6, &out_block1r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block1r_con7.configure(&out_block1r_act4, &weights_block1r_con7, nullptr, &out_block1r_con7, PadStrideInfo(2, 2, 1, 1));
		block1r_batch7.configure(&out_block1r_con7, &out_block1r_batch7, &weights_mean_block1r_batch7, &weights_variance_block1r_batch7, &weights_beta_block1r_batch7, &weights_gamma_block1r_batch7, 0.0000100099996416f);
		block1r_act5.configure(&out_block1r_batch7, &out_block1r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block1r_con8.configure(&out_block1r_act5, &weights_block1r_con8, nullptr, &out_block1r_con8, PadStrideInfo(1, 1, 0, 0));
		block1r_batch8.configure(&out_block1r_con8, &out_block1r_batch8, &weights_mean_block1r_batch8, &weights_variance_block1r_batch8, &weights_beta_block1r_batch8, &weights_gamma_block1r_batch8, 0.0000100099996416f);
		block1l_pool0.configure(&out_block1_act1, &out_block1l_pool0, PoolingLayerInfo(PoolingType::MAX, 1, PadStrideInfo(2,2, 0, 0), true));
		block1_add2.configure(&out_block1r_batch8, &out_block1l_pool0, &out_block1_add2,A);
		block1_act2.configure(&out_block1_add2, &out_block1_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
//end block1
//block2
		block2r_con0.configure(&out_block1_act2, &weights_block2r_con0, nullptr, &out_block2r_con0, PadStrideInfo(1, 1, 0, 0));
		block2r_batch0.configure(&out_block2r_con0, &out_block2r_batch0, &weights_mean_block2r_batch0, &weights_variance_block2r_batch0, &weights_beta_block2r_batch0, &weights_gamma_block2r_batch0, 0.0000100099996416f);
		block2r_act0.configure(&out_block2r_batch0, &out_block2r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block2r_con1.configure(&out_block2r_act0, &weights_block2r_con1, nullptr, &out_block2r_con1, PadStrideInfo(1, 1, 1, 1));
		block2r_batch1.configure(&out_block2r_con1, &out_block2r_batch1, &weights_mean_block2r_batch1, &weights_variance_block2r_batch1, &weights_beta_block2r_batch1, &weights_gamma_block2r_batch1, 0.0000100099996416f);
		block2r_act1.configure(&out_block2r_batch1, &out_block2r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block2r_con2.configure(&out_block2r_act1, &weights_block2r_con2, nullptr, &out_block2r_con2, PadStrideInfo(1, 1, 0, 0));
		block2r_batch2.configure(&out_block2r_con2, &out_block2r_batch2, &weights_mean_block2r_batch2, &weights_variance_block2r_batch2, &weights_beta_block2r_batch2, &weights_gamma_block2r_batch2, 0.0000100099996416f);
		block2l_con0.configure(&out_block1_act2, &weights_block2l_con0, nullptr, &out_block2l_con0, PadStrideInfo(1, 1, 0, 0));
		block2l_batch0.configure(&out_block2l_con0, &out_block2l_batch0, &weights_mean_block2l_batch0, &weights_variance_block2l_batch0, &weights_beta_block2l_batch0, &weights_gamma_block2l_batch0, 0.0000100099996416f);
		block2_add0.configure(&out_block2r_batch2, &out_block2l_batch0, &out_block2_add0, A);
		block2_act0.configure(&out_block2_add0, &out_block2_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		
		block2r_con3.configure(&out_block2_act0, &weights_block2r_con3, nullptr, &out_block2r_con3, PadStrideInfo(1, 1, 0, 0));
		block2r_batch3.configure(&out_block2r_con3, &out_block2r_batch3, &weights_mean_block2r_batch3, &weights_variance_block2r_batch3, &weights_beta_block2r_batch3, &weights_gamma_block2r_batch3, 0.0000100099996416f);
		block2r_act2.configure(&out_block2r_batch3, &out_block2r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block2r_con4.configure(&out_block2r_act2, &weights_block2r_con4, nullptr, &out_block2r_con4, PadStrideInfo(1, 1, 1, 1));
		block2r_batch4.configure(&out_block2r_con4, &out_block2r_batch4, &weights_mean_block2r_batch4, &weights_variance_block2r_batch4, &weights_beta_block2r_batch4, &weights_gamma_block2r_batch4, 0.0000100099996416f);
		block2r_act3.configure(&out_block2r_batch4, &out_block2r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block2r_con5.configure(&out_block2r_act3, &weights_block2r_con5, nullptr, &out_block2r_con5, PadStrideInfo(1, 1, 0, 0));
		block2r_batch5.configure(&out_block2r_con5, &out_block2r_batch5, &weights_mean_block2r_batch5, &weights_variance_block2r_batch5, &weights_beta_block2r_batch5, &weights_gamma_block2r_batch5, 0.0000100099996416f);
		block2_add1.configure(&out_block2r_batch5, &out_block2_act0, &out_block2_add1, A);
		block2_act1.configure(&out_block2_add1, &out_block2_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		
		block2r_con6.configure(&out_block2_act1, &weights_block2r_con6, nullptr, &out_block2r_con6, PadStrideInfo(1, 1, 0, 0));
		block2r_batch6.configure(&out_block2r_con6, &out_block2r_batch6, &weights_mean_block2r_batch6, &weights_variance_block2r_batch6, &weights_beta_block2r_batch6, &weights_gamma_block2r_batch6, 0.0000100099996416f);
		block2r_act4.configure(&out_block2r_batch6, &out_block2r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block2r_con7.configure(&out_block2r_act4, &weights_block2r_con7, nullptr, &out_block2r_con7, PadStrideInfo(1, 1, 1, 1));
		block2r_batch7.configure(&out_block2r_con7, &out_block2r_batch7, &weights_mean_block2r_batch7, &weights_variance_block2r_batch7, &weights_beta_block2r_batch7, &weights_gamma_block2r_batch7, 0.0000100099996416f);
		block2r_act5.configure(&out_block2r_batch7, &out_block2r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block2r_con8.configure(&out_block2r_act5, &weights_block2r_con8, nullptr, &out_block2r_con8, PadStrideInfo(1, 1, 0, 0));
		block2r_batch8.configure(&out_block2r_con8, &out_block2r_batch8, &weights_mean_block2r_batch8, &weights_variance_block2r_batch8, &weights_beta_block2r_batch8, &weights_gamma_block2r_batch8, 0.0000100099996416f);
		block2_add2.configure(&out_block2r_batch8, &out_block2_act1, &out_block2_add2, A);
		block2_act2.configure(&out_block2_add2, &out_block2_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		
		block2r_con9.configure(&out_block2_act2, &weights_block2r_con9, nullptr, &out_block2r_con9, PadStrideInfo(1, 1, 0, 0));
		block2r_batch9.configure(&out_block2r_con9, &out_block2r_batch9, &weights_mean_block2r_batch9, &weights_variance_block2r_batch9, &weights_beta_block2r_batch9, &weights_gamma_block2r_batch9, 0.0000100099996416f);
		block2r_act6.configure(&out_block2r_batch9, &out_block2r_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block2r_con10.configure(&out_block2r_act6, &weights_block2r_con10, nullptr, &out_block2r_con10, PadStrideInfo(2, 2, 1, 1));
		block2r_batch10.configure(&out_block2r_con10, &out_block2r_batch10, &weights_mean_block2r_batch10, &weights_variance_block2r_batch10, &weights_beta_block2r_batch10, &weights_gamma_block2r_batch10, 0.0000100099996416f);
		block2r_act7.configure(&out_block2r_batch10, &out_block2r_act7, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block2r_con11.configure(&out_block2r_act7, &weights_block2r_con11, nullptr, &out_block2r_con11, PadStrideInfo(1, 1, 0, 0));
		block2r_batch11.configure(&out_block2r_con11, &out_block2r_batch11, &weights_mean_block2r_batch11, &weights_variance_block2r_batch11, &weights_beta_block2r_batch11, &weights_gamma_block2r_batch11, 0.0000100099996416f);
		block2l_pool0.configure(&out_block2_act2, &out_block2l_pool0, PoolingLayerInfo(PoolingType::MAX, 1, PadStrideInfo(2,2, 0, 0), true));
		block2_add3.configure(&out_block2r_batch11, &out_block2l_pool0, &out_block2_add3, A);
		block2_act3.configure(&out_block2_add3, &out_block2_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		
//end block2
//block3
		block3r_con0.configure(&out_block2_act3, &weights_block3r_con0, nullptr, &out_block3r_con0, PadStrideInfo(1, 1, 0, 0));
		block3r_batch0.configure(&out_block3r_con0, &out_block3r_batch0, &weights_mean_block3r_batch0, &weights_variance_block3r_batch0, &weights_beta_block3r_batch0, &weights_gamma_block3r_batch0, 0.0000100099996416f);
		block3r_act0.configure(&out_block3r_batch0, &out_block3r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block3r_con1.configure(&out_block3r_act0, &weights_block3r_con1, nullptr, &out_block3r_con1, PadStrideInfo(1, 1, 1, 1));
		block3r_batch1.configure(&out_block3r_con1, &out_block3r_batch1, &weights_mean_block3r_batch1, &weights_variance_block3r_batch1, &weights_beta_block3r_batch1, &weights_gamma_block3r_batch1, 0.0000100099996416f);
		block3r_act1.configure(&out_block3r_batch1, &out_block3r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block3r_con2.configure(&out_block3r_act1, &weights_block3r_con2, nullptr, &out_block3r_con2, PadStrideInfo(1, 1, 0, 0));
		block3r_batch2.configure(&out_block3r_con2, &out_block3r_batch2, &weights_mean_block3r_batch2, &weights_variance_block3r_batch2, &weights_beta_block3r_batch2, &weights_gamma_block3r_batch2, 0.0000100099996416f);
		block3l_con0.configure(&out_block2_act3, &weights_block3l_con0, nullptr, &out_block3l_con0, PadStrideInfo(1, 1, 0, 0));
		block3l_batch0.configure(&out_block3l_con0, &out_block3l_batch0, &weights_mean_block3l_batch0, &weights_variance_block3l_batch0, &weights_beta_block3l_batch0, &weights_gamma_block3l_batch0, 0.0000100099996416f);
		block3_add0.configure(&out_block3r_batch2, &out_block3l_batch0, &out_block3_add0, A);
		block3_act0.configure(&out_block3_add0, &out_block3_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		block3r_con3.configure(&out_block3_act0, &weights_block3r_con3, nullptr, &out_block3r_con3, PadStrideInfo(1, 1, 0, 0));
		block3r_batch3.configure(&out_block3r_con3, &out_block3r_batch3, &weights_mean_block3r_batch3, &weights_variance_block3r_batch3, &weights_beta_block3r_batch3, &weights_gamma_block3r_batch3, 0.0000100099996416f);
		block3r_act2.configure(&out_block3r_batch3, &out_block3r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block3r_con4.configure(&out_block3r_act2, &weights_block3r_con4, nullptr, &out_block3r_con4, PadStrideInfo(1, 1, 1, 1));
		block3r_batch4.configure(&out_block3r_con4, &out_block3r_batch4, &weights_mean_block3r_batch4, &weights_variance_block3r_batch4, &weights_beta_block3r_batch4, &weights_gamma_block3r_batch4, 0.0000100099996416f);
		block3r_act3.configure(&out_block3r_batch4, &out_block3r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block3r_con5.configure(&out_block3r_act3, &weights_block3r_con5, nullptr, &out_block3r_con5, PadStrideInfo(1, 1, 0, 0));
		block3r_batch5.configure(&out_block3r_con5, &out_block3r_batch5, &weights_mean_block3r_batch5, &weights_variance_block3r_batch5, &weights_beta_block3r_batch5, &weights_gamma_block3r_batch5, 0.0000100099996416f);
		block3_add1.configure(&out_block3r_batch5, &out_block3_act0, &out_block3_add1, A);
		block3_act1.configure(&out_block3_add1, &out_block3_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        
		block3r_con6.configure(&out_block3_act1, &weights_block3r_con6, nullptr, &out_block3r_con6, PadStrideInfo(1, 1, 0, 0));
		block3r_batch6.configure(&out_block3r_con6, &out_block3r_batch6, &weights_mean_block3r_batch6, &weights_variance_block3r_batch6, &weights_beta_block3r_batch6, &weights_gamma_block3r_batch6, 0.0000100099996416f);
		block3r_act4.configure(&out_block3r_batch6, &out_block3r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block3r_con7.configure(&out_block3r_act4, &weights_block3r_con7, nullptr, &out_block3r_con7, PadStrideInfo(1, 1, 1, 1));
		block3r_batch7.configure(&out_block3r_con7, &out_block3r_batch7, &weights_mean_block3r_batch7, &weights_variance_block3r_batch7, &weights_beta_block3r_batch7, &weights_gamma_block3r_batch7, 0.0000100099996416f);
		block3r_act5.configure(&out_block3r_batch7, &out_block3r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block3r_con8.configure(&out_block3r_act5, &weights_block3r_con8, nullptr, &out_block3r_con8, PadStrideInfo(1, 1, 0, 0));
		block3r_batch8.configure(&out_block3r_con8, &out_block3r_batch8, &weights_mean_block3r_batch8, &weights_variance_block3r_batch8, &weights_beta_block3r_batch8, &weights_gamma_block3r_batch8, 0.0000100099996416f);
		block3_add2.configure(&out_block3r_batch8, &out_block3_act1, &out_block3_add2, A);
		block3_act2.configure(&out_block3_add2, &out_block3_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
      
		block3r_con9.configure(&out_block3_act2, &weights_block3r_con9, nullptr, &out_block3r_con9, PadStrideInfo(1, 1, 0, 0));
		block3r_batch9.configure(&out_block3r_con9, &out_block3r_batch9, &weights_mean_block3r_batch9, &weights_variance_block3r_batch9, &weights_beta_block3r_batch9, &weights_gamma_block3r_batch9, 0.0000100099996416f);
		block3r_act6.configure(&out_block3r_batch9, &out_block3r_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block3r_con10.configure(&out_block3r_act6, &weights_block3r_con10, nullptr, &out_block3r_con10, PadStrideInfo(1, 1, 1, 1));
		block3r_batch10.configure(&out_block3r_con10, &out_block3r_batch10, &weights_mean_block3r_batch10, &weights_variance_block3r_batch10, &weights_beta_block3r_batch10, &weights_gamma_block3r_batch10, 0.0000100099996416f);
		block3r_act7.configure(&out_block3r_batch10, &out_block3r_act7, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block3r_con11.configure(&out_block3r_act7, &weights_block3r_con11, nullptr, &out_block3r_con11, PadStrideInfo(1, 1, 0, 0));
		block3r_batch11.configure(&out_block3r_con11, &out_block3r_batch11, &weights_mean_block3r_batch11, &weights_variance_block3r_batch11, &weights_beta_block3r_batch11, &weights_gamma_block3r_batch11, 0.0000100099996416f);
		block3_add3.configure(&out_block3r_batch11, &out_block3_act2, &out_block3_add3, A);
		block3_act3.configure(&out_block3_add3, &out_block3_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        
		block3r_con12.configure(&out_block3_act3, &weights_block3r_con12, nullptr, &out_block3r_con12, PadStrideInfo(1, 1, 0, 0));
		block3r_batch12.configure(&out_block3r_con12, &out_block3r_batch12, &weights_mean_block3r_batch12, &weights_variance_block3r_batch12, &weights_beta_block3r_batch12, &weights_gamma_block3r_batch12, 0.0000100099996416f);
		block3r_act8.configure(&out_block3r_batch12, &out_block3r_act8, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block3r_con13.configure(&out_block3r_act8, &weights_block3r_con13, nullptr, &out_block3r_con13, PadStrideInfo(1, 1, 1, 1));
		block3r_batch13.configure(&out_block3r_con13, &out_block3r_batch13, &weights_mean_block3r_batch13, &weights_variance_block3r_batch13, &weights_beta_block3r_batch13, &weights_gamma_block3r_batch13, 0.0000100099996416f);
		block3r_act9.configure(&out_block3r_batch13, &out_block3r_act9, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block3r_con14.configure(&out_block3r_act9, &weights_block3r_con14, nullptr, &out_block3r_con14, PadStrideInfo(1, 1, 0, 0));
		block3r_batch14.configure(&out_block3r_con14, &out_block3r_batch14, &weights_mean_block3r_batch14, &weights_variance_block3r_batch14, &weights_beta_block3r_batch14, &weights_gamma_block3r_batch14, 0.0000100099996416f);
		block3_add4.configure(&out_block3r_batch14, &out_block3_act3, &out_block3_add4, A);
		block3_act4.configure(&out_block3_add4, &out_block3_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
      
		block3r_con15.configure(&out_block3_act4, &weights_block3r_con15, nullptr, &out_block3r_con15, PadStrideInfo(1, 1, 0, 0));
		block3r_batch15.configure(&out_block3r_con15, &out_block3r_batch15, &weights_mean_block3r_batch15, &weights_variance_block3r_batch15, &weights_beta_block3r_batch15, &weights_gamma_block3r_batch15, 0.0000100099996416f);
		block3r_act10.configure(&out_block3r_batch15, &out_block3r_act10, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block3r_con16.configure(&out_block3r_act10, &weights_block3r_con16, nullptr, &out_block3r_con16, PadStrideInfo(2, 2, 1, 1));
		block3r_batch16.configure(&out_block3r_con16, &out_block3r_batch16, &weights_mean_block3r_batch16, &weights_variance_block3r_batch16, &weights_beta_block3r_batch16, &weights_gamma_block3r_batch16, 0.0000100099996416f);
		block3r_act11.configure(&out_block3r_batch16, &out_block3r_act11, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block3r_con17.configure(&out_block3r_act11, &weights_block3r_con17, nullptr, &out_block3r_con17, PadStrideInfo(1, 1, 0, 0));
		block3r_batch17.configure(&out_block3r_con17, &out_block3r_batch17, &weights_mean_block3r_batch17, &weights_variance_block3r_batch17, &weights_beta_block3r_batch17, &weights_gamma_block3r_batch17, 0.0000100099996416f);
		block3l_pool0.configure(&out_block2_act3, &out_block3l_pool0, PoolingLayerInfo(PoolingType::MAX, 1, PadStrideInfo(2, 2, 0, 0), true));
		block3_add5.configure(&out_block3r_batch17, &out_block3l_pool0, &out_block3_add5, A);
		block3_act5.configure(&out_block3_add5, &out_block3_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		
		
//end block3
//block4
		block4r_con0.configure(&out_block3_act5, &weights_block4r_con0, nullptr, &out_block4r_con0, PadStrideInfo(1, 1, 0, 0));
		block4r_batch0.configure(&out_block4r_con0, &out_block4r_batch0, &weights_mean_block4r_batch0, &weights_variance_block4r_batch0, &weights_beta_block4r_batch0, &weights_gamma_block4r_batch0, 0.0000100099996416f);
		block4r_act0.configure(&out_block4r_batch0, &out_block4r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block4r_con1.configure(&out_block4r_act0, &weights_block4r_con1, nullptr, &out_block4r_con1, PadStrideInfo(1, 1, 1, 1));
		block4r_batch1.configure(&out_block4r_con1, &out_block4r_batch1, &weights_mean_block4r_batch1, &weights_variance_block4r_batch1, &weights_beta_block4r_batch1, &weights_gamma_block4r_batch1, 0.0000100099996416f);
		block4r_act1.configure(&out_block4r_batch1, &out_block4r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block4r_con2.configure(&out_block4r_act1, &weights_block4r_con2, nullptr, &out_block4r_con2, PadStrideInfo(1, 1, 0, 0));
		block4r_batch2.configure(&out_block4r_con2, &out_block4r_batch2, &weights_mean_block4r_batch2, &weights_variance_block4r_batch2, &weights_beta_block4r_batch2, &weights_gamma_block4r_batch2, 0.0000100099996416f);
		block4l_con0.configure(&out_block3_act5, &weights_block4l_con0, nullptr, &out_block4l_con0, PadStrideInfo(1, 1, 0, 0));
		block4l_batch0.configure(&out_block4l_con0, &out_block4l_batch0, &weights_mean_block4l_batch0, &weights_variance_block4l_batch0, &weights_beta_block4l_batch0, &weights_gamma_block4l_batch0, 0.0000100099996416f);
		block4_add0.configure(&out_block4r_batch2, &out_block4l_batch0, &out_block4_add0, A);
		block4_act0.configure(&out_block4_add0, &out_block4_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
     
		block4r_con3.configure(&out_block4_act0, &weights_block4r_con3, nullptr, &out_block4r_con3, PadStrideInfo(1, 1, 0, 0));
		block4r_batch3.configure(&out_block4r_con3, &out_block4r_batch3, &weights_mean_block4r_batch3, &weights_variance_block4r_batch3, &weights_beta_block4r_batch3, &weights_gamma_block4r_batch3, 0.0000100099996416f);
		block4r_act2.configure(&out_block4r_batch3, &out_block4r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block4r_con4.configure(&out_block4r_act2, &weights_block4r_con4, nullptr, &out_block4r_con4, PadStrideInfo(1, 1, 1, 1));
		block4r_batch4.configure(&out_block4r_con4, &out_block4r_batch4, &weights_mean_block4r_batch4, &weights_variance_block4r_batch4, &weights_beta_block4r_batch4, &weights_gamma_block4r_batch4, 0.0000100099996416f);
		block4r_act3.configure(&out_block4r_batch4, &out_block4r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block4r_con5.configure(&out_block4r_act3, &weights_block4r_con5, nullptr, &out_block4r_con5, PadStrideInfo(1, 1, 0, 0));
		block4r_batch5.configure(&out_block4r_con5, &out_block4r_batch5, &weights_mean_block4r_batch5, &weights_variance_block4r_batch5, &weights_beta_block4r_batch5, &weights_gamma_block4r_batch5, 0.0000100099996416f);
		block4_add1.configure(&out_block4r_batch5, &out_block4_act0, &out_block4_add1, A);
		block4_act1.configure(&out_block4_add1, &out_block4_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
      
		block4r_con6.configure(&out_block4_act1, &weights_block4r_con6, nullptr, &out_block4r_con6, PadStrideInfo(1, 1, 0, 0));
		block4r_batch6.configure(&out_block4r_con6, &out_block4r_batch6, &weights_mean_block4r_batch6, &weights_variance_block4r_batch6, &weights_beta_block4r_batch6, &weights_gamma_block4r_batch6, 0.0000100099996416f);
		block4r_act4.configure(&out_block4r_batch6, &out_block4r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block4r_con7.configure(&out_block4r_act4, &weights_block4r_con7, nullptr, &out_block4r_con7, PadStrideInfo(1, 1, 1, 1));
		block4r_batch7.configure(&out_block4r_con7, &out_block4r_batch7, &weights_mean_block4r_batch7, &weights_variance_block4r_batch7, &weights_beta_block4r_batch7, &weights_gamma_block4r_batch7, 0.0000100099996416f);
		block4r_act5.configure(&out_block4r_batch7, &out_block4r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		block4r_con8.configure(&out_block4r_act5, &weights_block4r_con8, nullptr, &out_block4r_con8, PadStrideInfo(1, 1, 0, 0));
		block4r_batch8.configure(&out_block4r_con8, &out_block4r_batch8, &weights_mean_block4r_batch8, &weights_variance_block4r_batch8, &weights_beta_block4r_batch8, &weights_gamma_block4r_batch8, 0.0000100099996416f);
		block4_add2.configure(&out_block4r_batch8, &out_block4_act1, &out_block4_add2,A);
		block4_act2.configure(&out_block4_add2, &out_block4_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
//end block4
//block end
//last start
	    pool1.configure(&out_block4_act2, &out_pool1, PoolingLayerInfo(PoolingType::AVG, 7, PadStrideInfo(1, 1, 0, 0, 0, 0, DimensionRoundingType::FLOOR)));
		con1.configure(&out_pool1, &weights_con1, &biases_con1, &out_con1, PadStrideInfo(1, 1, 0, 0));
		flatten.configure(&out_con1, &out_flatten);
		softmax.configure(&out_flatten, &out_softmax);
//last end
//configure end

//allocate start
//first allocate
	    out_con0.allocator()->allocate(); out_batch0.allocator()->allocate();
	    out_act0.allocator()->allocate(); out_pool0.allocator()->allocate();
//first allocate end
//block allocate
//block1
		out_block1r_con0.allocator()->allocate(); out_block1r_batch0.allocator()->allocate(); out_block1r_act0.allocator()->allocate();
		out_block1r_con1.allocator()->allocate(); out_block1r_batch1.allocator()->allocate(); out_block1r_act1.allocator()->allocate();
		out_block1r_con2.allocator()->allocate(); out_block1r_batch2.allocator()->allocate(); out_block1l_con0.allocator()->allocate(); out_block1l_batch0.allocator()->allocate();
		out_block1_add0.allocator()->allocate(); out_block1_act0.allocator()->allocate();

		out_block1r_con3.allocator()->allocate(); out_block1r_batch3.allocator()->allocate(); out_block1r_act2.allocator()->allocate();
		out_block1r_con4.allocator()->allocate(); out_block1r_batch4.allocator()->allocate(); out_block1r_act3.allocator()->allocate();
		out_block1r_con5.allocator()->allocate(); out_block1r_batch5.allocator()->allocate();
		out_block1_add1.allocator()->allocate(); out_block1_act1.allocator()->allocate();

		out_block1r_con6.allocator()->allocate(); out_block1r_batch6.allocator()->allocate(); out_block1r_act4.allocator()->allocate();
		out_block1r_con7.allocator()->allocate(); out_block1r_batch7.allocator()->allocate(); out_block1r_act5.allocator()->allocate();
		out_block1r_con8.allocator()->allocate(); out_block1r_batch8.allocator()->allocate(); out_block1l_pool0.allocator()->allocate();
		out_block1_add2.allocator()->allocate(); out_block1_act2.allocator()->allocate();
//end block1
//block2
		out_block2r_con0.allocator()->allocate(); out_block2r_batch0.allocator()->allocate(); out_block2r_act0.allocator()->allocate();
		out_block2r_con1.allocator()->allocate(); out_block2r_batch1.allocator()->allocate(); out_block2r_act1.allocator()->allocate();
		out_block2r_con2.allocator()->allocate(); out_block2r_batch2.allocator()->allocate(); out_block2l_con0.allocator()->allocate(); out_block2l_batch0.allocator()->allocate();
		out_block2_add0.allocator()->allocate(); out_block2_act0.allocator()->allocate();

		out_block2r_con3.allocator()->allocate(); out_block2r_batch3.allocator()->allocate(); out_block2r_act2.allocator()->allocate();
		out_block2r_con4.allocator()->allocate(); out_block2r_batch4.allocator()->allocate(); out_block2r_act3.allocator()->allocate();
		out_block2r_con5.allocator()->allocate(); out_block2r_batch5.allocator()->allocate();
		out_block2_add1.allocator()->allocate(); out_block2_act1.allocator()->allocate();

		out_block2r_con6.allocator()->allocate(); out_block2r_batch6.allocator()->allocate(); out_block2r_act4.allocator()->allocate();
		out_block2r_con7.allocator()->allocate(); out_block2r_batch7.allocator()->allocate(); out_block2r_act5.allocator()->allocate();
		out_block2r_con8.allocator()->allocate(); out_block2r_batch8.allocator()->allocate();
		out_block2_add2.allocator()->allocate(); out_block2_act2.allocator()->allocate();

		out_block2r_con9.allocator()->allocate(); out_block2r_batch9.allocator()->allocate(); out_block2r_act6.allocator()->allocate();
		out_block2r_con10.allocator()->allocate(); out_block2r_batch10.allocator()->allocate(); out_block2r_act7.allocator()->allocate();
		out_block2r_con11.allocator()->allocate(); out_block2r_batch11.allocator()->allocate(); out_block2l_pool0.allocator()->allocate();
		out_block2_add3.allocator()->allocate(); out_block2_act3.allocator()->allocate();
//end block2
//block3
		out_block3r_con0.allocator()->allocate(); out_block3r_batch0.allocator()->allocate(); out_block3r_act0.allocator()->allocate();
		out_block3r_con1.allocator()->allocate(); out_block3r_batch1.allocator()->allocate(); out_block3r_act1.allocator()->allocate();
		out_block3r_con2.allocator()->allocate(); out_block3r_batch2.allocator()->allocate(); out_block3l_con0.allocator()->allocate(); out_block3l_batch0.allocator()->allocate();
		out_block3_add0.allocator()->allocate(); out_block3_act0.allocator()->allocate();

		out_block3r_con3.allocator()->allocate(); out_block3r_batch3.allocator()->allocate(); out_block3r_act2.allocator()->allocate();
		out_block3r_con4.allocator()->allocate(); out_block3r_batch4.allocator()->allocate(); out_block3r_act3.allocator()->allocate();
		out_block3r_con5.allocator()->allocate(); out_block3r_batch5.allocator()->allocate();
		out_block3_add1.allocator()->allocate(); out_block3_act1.allocator()->allocate();

		out_block3r_con6.allocator()->allocate(); out_block3r_batch6.allocator()->allocate(); out_block3r_act4.allocator()->allocate();
		out_block3r_con7.allocator()->allocate(); out_block3r_batch7.allocator()->allocate(); out_block3r_act5.allocator()->allocate();
		out_block3r_con8.allocator()->allocate(); out_block3r_batch8.allocator()->allocate();
		out_block3_add2.allocator()->allocate(); out_block3_act2.allocator()->allocate();

		out_block3r_con9.allocator()->allocate(); out_block3r_batch9.allocator()->allocate(); out_block3r_act6.allocator()->allocate();
		out_block3r_con10.allocator()->allocate(); out_block3r_batch10.allocator()->allocate(); out_block3r_act7.allocator()->allocate();
		out_block3r_con11.allocator()->allocate(); out_block3r_batch11.allocator()->allocate();
		out_block3_add3.allocator()->allocate(); out_block3_act3.allocator()->allocate();

		out_block3r_con12.allocator()->allocate(); out_block3r_batch12.allocator()->allocate(); out_block3r_act8.allocator()->allocate();
		out_block3r_con13.allocator()->allocate(); out_block3r_batch13.allocator()->allocate(); out_block3r_act9.allocator()->allocate();
		out_block3r_con14.allocator()->allocate(); out_block3r_batch14.allocator()->allocate();
		out_block3_add4.allocator()->allocate(); out_block3_act4.allocator()->allocate();

		out_block3r_con15.allocator()->allocate(); out_block3r_batch15.allocator()->allocate(); out_block3r_act10.allocator()->allocate();
		out_block3r_con16.allocator()->allocate(); out_block3r_batch16.allocator()->allocate(); out_block3r_act11.allocator()->allocate();
		out_block3r_con17.allocator()->allocate(); out_block3r_batch17.allocator()->allocate(); out_block3l_pool0.allocator()->allocate();
		out_block3_add5.allocator()->allocate(); out_block3_act5.allocator()->allocate();

//end block3
//block4
		out_block4r_con0.allocator()->allocate(); out_block4r_batch0.allocator()->allocate(); out_block4r_act0.allocator()->allocate();
		out_block4r_con1.allocator()->allocate(); out_block4r_batch1.allocator()->allocate(); out_block4r_act1.allocator()->allocate();
		out_block4r_con2.allocator()->allocate(); out_block4r_batch2.allocator()->allocate(); out_block4l_con0.allocator()->allocate(); out_block4l_batch0.allocator()->allocate();
		out_block4_add0.allocator()->allocate(); out_block4_act0.allocator()->allocate();

		out_block4r_con3.allocator()->allocate(); out_block4r_batch3.allocator()->allocate(); out_block4r_act2.allocator()->allocate();
		out_block4r_con4.allocator()->allocate(); out_block4r_batch4.allocator()->allocate(); out_block4r_act3.allocator()->allocate();
		out_block4r_con5.allocator()->allocate(); out_block4r_batch5.allocator()->allocate();
		out_block4_add1.allocator()->allocate(); out_block4_act1.allocator()->allocate();

		out_block4r_con6.allocator()->allocate(); out_block4r_batch6.allocator()->allocate(); out_block4r_act4.allocator()->allocate();
		out_block4r_con7.allocator()->allocate(); out_block4r_batch7.allocator()->allocate(); out_block4r_act5.allocator()->allocate();
		out_block4r_con8.allocator()->allocate(); out_block4r_batch8.allocator()->allocate(); 
		out_block4_add2.allocator()->allocate(); out_block4_act2.allocator()->allocate();
//end block4
//block allocate end
//last allocate
        out_pool1.allocator()->allocate(); out_con1.allocator()->allocate(); out_flatten.allocator()->allocate(); out_softmax.allocator()->allocate();
//last allocate end
//
        src.allocator()->allocate(); weights_con0.allocator()->allocate();weights_mean_batch0.allocator()->allocate();  
		weights_variance_batch0.allocator()->allocate();weights_beta_batch0.allocator()->allocate();weights_gamma_batch0.allocator()->allocate();
		
		weights_block1r_con0.allocator()->allocate(); weights_block1r_con1.allocator()->allocate(); weights_block1r_con2.allocator()->allocate(); 
		weights_block1r_con3.allocator()->allocate(); weights_block1r_con4.allocator()->allocate(); weights_block1r_con5.allocator()->allocate(); 
		weights_block1r_con6.allocator()->allocate(); weights_block1r_con7.allocator()->allocate(); weights_block1r_con8.allocator()->allocate(); 
		weights_mean_block1r_batch0.allocator()->allocate(); weights_mean_block1r_batch1.allocator()->allocate(); weights_mean_block1r_batch2.allocator()->allocate();
		weights_mean_block1r_batch3.allocator()->allocate(); weights_mean_block1r_batch4.allocator()->allocate(); weights_mean_block1r_batch5.allocator()->allocate();
		weights_mean_block1r_batch6.allocator()->allocate(); weights_mean_block1r_batch7.allocator()->allocate(); weights_mean_block1r_batch8.allocator()->allocate();
		weights_variance_block1r_batch0.allocator()->allocate(); weights_variance_block1r_batch1.allocator()->allocate(); weights_variance_block1r_batch2.allocator()->allocate();
		weights_variance_block1r_batch3.allocator()->allocate(); weights_variance_block1r_batch4.allocator()->allocate(); weights_variance_block1r_batch5.allocator()->allocate();
		weights_variance_block1r_batch6.allocator()->allocate(); weights_variance_block1r_batch7.allocator()->allocate(); weights_variance_block1r_batch8.allocator()->allocate();
		weights_beta_block1r_batch0.allocator()->allocate(); weights_beta_block1r_batch1.allocator()->allocate(); weights_beta_block1r_batch2.allocator()->allocate();
		weights_beta_block1r_batch3.allocator()->allocate(); weights_beta_block1r_batch4.allocator()->allocate(); weights_beta_block1r_batch5.allocator()->allocate();
		weights_beta_block1r_batch6.allocator()->allocate(); weights_beta_block1r_batch7.allocator()->allocate(); weights_beta_block1r_batch8.allocator()->allocate();
		weights_gamma_block1r_batch0.allocator()->allocate(); weights_gamma_block1r_batch1.allocator()->allocate(); weights_gamma_block1r_batch2.allocator()->allocate();
		weights_gamma_block1r_batch3.allocator()->allocate(); weights_gamma_block1r_batch4.allocator()->allocate(); weights_gamma_block1r_batch5.allocator()->allocate();
		weights_gamma_block1r_batch6.allocator()->allocate(); weights_gamma_block1r_batch7.allocator()->allocate(); weights_gamma_block1r_batch8.allocator()->allocate();
		weights_block1l_con0.allocator()->allocate();weights_mean_block1l_batch0.allocator()->allocate();weights_variance_block1l_batch0.allocator()->allocate();
		weights_beta_block1l_batch0.allocator()->allocate();weights_gamma_block1l_batch0.allocator()->allocate();
		
		weights_block2r_con0.allocator()->allocate(); weights_block2r_con1.allocator()->allocate(); weights_block2r_con2.allocator()->allocate();
		weights_block2r_con3.allocator()->allocate(); weights_block2r_con4.allocator()->allocate(); weights_block2r_con5.allocator()->allocate();
		weights_block2r_con6.allocator()->allocate(); weights_block2r_con7.allocator()->allocate(); weights_block2r_con8.allocator()->allocate();
		weights_block2r_con9.allocator()->allocate(); weights_block2r_con10.allocator()->allocate(); weights_block2r_con11.allocator()->allocate();
		weights_mean_block2r_batch0.allocator()->allocate(); weights_mean_block2r_batch1.allocator()->allocate(); weights_mean_block2r_batch2.allocator()->allocate();
		weights_mean_block2r_batch3.allocator()->allocate(); weights_mean_block2r_batch4.allocator()->allocate(); weights_mean_block2r_batch5.allocator()->allocate();
		weights_mean_block2r_batch6.allocator()->allocate(); weights_mean_block2r_batch7.allocator()->allocate(); weights_mean_block2r_batch8.allocator()->allocate();
		weights_mean_block2r_batch9.allocator()->allocate(); weights_mean_block2r_batch10.allocator()->allocate(); weights_mean_block2r_batch11.allocator()->allocate();
		weights_variance_block2r_batch0.allocator()->allocate(); weights_variance_block2r_batch1.allocator()->allocate(); weights_variance_block2r_batch2.allocator()->allocate();
		weights_variance_block2r_batch3.allocator()->allocate(); weights_variance_block2r_batch4.allocator()->allocate(); weights_variance_block2r_batch5.allocator()->allocate();
		weights_variance_block2r_batch6.allocator()->allocate(); weights_variance_block2r_batch7.allocator()->allocate(); weights_variance_block2r_batch8.allocator()->allocate();
		weights_variance_block2r_batch9.allocator()->allocate(); weights_variance_block2r_batch10.allocator()->allocate(); weights_variance_block2r_batch11.allocator()->allocate();
		weights_beta_block2r_batch0.allocator()->allocate(); weights_beta_block2r_batch1.allocator()->allocate(); weights_beta_block2r_batch2.allocator()->allocate();
		weights_beta_block2r_batch3.allocator()->allocate(); weights_beta_block2r_batch4.allocator()->allocate(); weights_beta_block2r_batch5.allocator()->allocate();
		weights_beta_block2r_batch6.allocator()->allocate(); weights_beta_block2r_batch7.allocator()->allocate(); weights_beta_block2r_batch8.allocator()->allocate();
		weights_beta_block2r_batch9.allocator()->allocate(); weights_beta_block2r_batch10.allocator()->allocate(); weights_beta_block2r_batch11.allocator()->allocate();
		weights_gamma_block2r_batch0.allocator()->allocate(); weights_gamma_block2r_batch1.allocator()->allocate(); weights_gamma_block2r_batch2.allocator()->allocate();
		weights_gamma_block2r_batch3.allocator()->allocate(); weights_gamma_block2r_batch4.allocator()->allocate(); weights_gamma_block2r_batch5.allocator()->allocate();
		weights_gamma_block2r_batch6.allocator()->allocate(); weights_gamma_block2r_batch7.allocator()->allocate(); weights_gamma_block2r_batch8.allocator()->allocate();
		weights_gamma_block2r_batch9.allocator()->allocate(); weights_gamma_block2r_batch10.allocator()->allocate(); weights_gamma_block2r_batch11.allocator()->allocate();
		weights_block2l_con0.allocator()->allocate(); weights_mean_block2l_batch0.allocator()->allocate(); weights_variance_block2l_batch0.allocator()->allocate();
		weights_beta_block2l_batch0.allocator()->allocate(); weights_gamma_block2l_batch0.allocator()->allocate();
		
		weights_block3r_con0.allocator()->allocate(); weights_block3r_con1.allocator()->allocate(); weights_block3r_con2.allocator()->allocate();
		weights_block3r_con3.allocator()->allocate(); weights_block3r_con4.allocator()->allocate(); weights_block3r_con5.allocator()->allocate();
		weights_block3r_con6.allocator()->allocate(); weights_block3r_con7.allocator()->allocate(); weights_block3r_con8.allocator()->allocate();
		weights_block3r_con9.allocator()->allocate(); weights_block3r_con10.allocator()->allocate(); weights_block3r_con11.allocator()->allocate();
		weights_block3r_con12.allocator()->allocate(); weights_block3r_con13.allocator()->allocate(); weights_block3r_con14.allocator()->allocate();
		weights_block3r_con15.allocator()->allocate(); weights_block3r_con16.allocator()->allocate(); weights_block3r_con17.allocator()->allocate();
		weights_mean_block3r_batch0.allocator()->allocate(); weights_mean_block3r_batch1.allocator()->allocate(); weights_mean_block3r_batch2.allocator()->allocate();
		weights_mean_block3r_batch3.allocator()->allocate(); weights_mean_block3r_batch4.allocator()->allocate(); weights_mean_block3r_batch5.allocator()->allocate();
		weights_mean_block3r_batch6.allocator()->allocate(); weights_mean_block3r_batch7.allocator()->allocate(); weights_mean_block3r_batch8.allocator()->allocate();
		weights_mean_block3r_batch9.allocator()->allocate(); weights_mean_block3r_batch10.allocator()->allocate(); weights_mean_block3r_batch11.allocator()->allocate();
		weights_mean_block3r_batch12.allocator()->allocate(); weights_mean_block3r_batch13.allocator()->allocate(); weights_mean_block3r_batch14.allocator()->allocate();
		weights_mean_block3r_batch15.allocator()->allocate(); weights_mean_block3r_batch16.allocator()->allocate(); weights_mean_block3r_batch17.allocator()->allocate();
		weights_variance_block3r_batch0.allocator()->allocate(); weights_variance_block3r_batch1.allocator()->allocate(); weights_variance_block3r_batch2.allocator()->allocate();
		weights_variance_block3r_batch3.allocator()->allocate(); weights_variance_block3r_batch4.allocator()->allocate(); weights_variance_block3r_batch5.allocator()->allocate();
		weights_variance_block3r_batch6.allocator()->allocate(); weights_variance_block3r_batch7.allocator()->allocate(); weights_variance_block3r_batch8.allocator()->allocate();
		weights_variance_block3r_batch9.allocator()->allocate(); weights_variance_block3r_batch10.allocator()->allocate(); weights_variance_block3r_batch11.allocator()->allocate();
		weights_variance_block3r_batch12.allocator()->allocate(); weights_variance_block3r_batch13.allocator()->allocate(); weights_variance_block3r_batch14.allocator()->allocate();
		weights_variance_block3r_batch15.allocator()->allocate(); weights_variance_block3r_batch16.allocator()->allocate(); weights_variance_block3r_batch17.allocator()->allocate();
		weights_beta_block3r_batch0.allocator()->allocate(); weights_beta_block3r_batch1.allocator()->allocate(); weights_beta_block3r_batch2.allocator()->allocate();
		weights_beta_block3r_batch3.allocator()->allocate(); weights_beta_block3r_batch4.allocator()->allocate(); weights_beta_block3r_batch5.allocator()->allocate();
		weights_beta_block3r_batch6.allocator()->allocate(); weights_beta_block3r_batch7.allocator()->allocate(); weights_beta_block3r_batch8.allocator()->allocate();
		weights_beta_block3r_batch9.allocator()->allocate(); weights_beta_block3r_batch10.allocator()->allocate(); weights_beta_block3r_batch11.allocator()->allocate();
		weights_beta_block3r_batch12.allocator()->allocate(); weights_beta_block3r_batch13.allocator()->allocate(); weights_beta_block3r_batch14.allocator()->allocate();
		weights_beta_block3r_batch15.allocator()->allocate(); weights_beta_block3r_batch16.allocator()->allocate(); weights_beta_block3r_batch17.allocator()->allocate();
		weights_gamma_block3r_batch0.allocator()->allocate(); weights_gamma_block3r_batch1.allocator()->allocate(); weights_gamma_block3r_batch2.allocator()->allocate();
		weights_gamma_block3r_batch3.allocator()->allocate(); weights_gamma_block3r_batch4.allocator()->allocate(); weights_gamma_block3r_batch5.allocator()->allocate();
		weights_gamma_block3r_batch6.allocator()->allocate(); weights_gamma_block3r_batch7.allocator()->allocate(); weights_gamma_block3r_batch8.allocator()->allocate();
		weights_gamma_block3r_batch9.allocator()->allocate(); weights_gamma_block3r_batch10.allocator()->allocate(); weights_gamma_block3r_batch11.allocator()->allocate();
		weights_gamma_block3r_batch12.allocator()->allocate(); weights_gamma_block3r_batch13.allocator()->allocate(); weights_gamma_block3r_batch14.allocator()->allocate();
		weights_gamma_block3r_batch15.allocator()->allocate(); weights_gamma_block3r_batch16.allocator()->allocate(); weights_gamma_block3r_batch17.allocator()->allocate();
		weights_block3l_con0.allocator()->allocate(); weights_mean_block3l_batch0.allocator()->allocate(); weights_variance_block3l_batch0.allocator()->allocate();
		weights_beta_block3l_batch0.allocator()->allocate(); weights_gamma_block3l_batch0.allocator()->allocate();
		
		weights_block4r_con0.allocator()->allocate(); weights_block4r_con1.allocator()->allocate(); weights_block4r_con2.allocator()->allocate();
		weights_block4r_con3.allocator()->allocate(); weights_block4r_con4.allocator()->allocate(); weights_block4r_con5.allocator()->allocate();
		weights_block4r_con6.allocator()->allocate(); weights_block4r_con7.allocator()->allocate(); weights_block4r_con8.allocator()->allocate();
		weights_mean_block4r_batch0.allocator()->allocate(); weights_mean_block4r_batch1.allocator()->allocate(); weights_mean_block4r_batch2.allocator()->allocate();
		weights_mean_block4r_batch3.allocator()->allocate(); weights_mean_block4r_batch4.allocator()->allocate(); weights_mean_block4r_batch5.allocator()->allocate();
		weights_mean_block4r_batch6.allocator()->allocate(); weights_mean_block4r_batch7.allocator()->allocate(); weights_mean_block4r_batch8.allocator()->allocate();
		weights_variance_block4r_batch0.allocator()->allocate(); weights_variance_block4r_batch1.allocator()->allocate(); weights_variance_block4r_batch2.allocator()->allocate();
		weights_variance_block4r_batch3.allocator()->allocate(); weights_variance_block4r_batch4.allocator()->allocate(); weights_variance_block4r_batch5.allocator()->allocate();
		weights_variance_block4r_batch6.allocator()->allocate(); weights_variance_block4r_batch7.allocator()->allocate(); weights_variance_block4r_batch8.allocator()->allocate();
		weights_beta_block4r_batch0.allocator()->allocate(); weights_beta_block4r_batch1.allocator()->allocate(); weights_beta_block4r_batch2.allocator()->allocate();
		weights_beta_block4r_batch3.allocator()->allocate(); weights_beta_block4r_batch4.allocator()->allocate(); weights_beta_block4r_batch5.allocator()->allocate();
		weights_beta_block4r_batch6.allocator()->allocate(); weights_beta_block4r_batch7.allocator()->allocate(); weights_beta_block4r_batch8.allocator()->allocate();
		weights_gamma_block4r_batch0.allocator()->allocate(); weights_gamma_block4r_batch1.allocator()->allocate(); weights_gamma_block4r_batch2.allocator()->allocate();
		weights_gamma_block4r_batch3.allocator()->allocate(); weights_gamma_block4r_batch4.allocator()->allocate(); weights_gamma_block4r_batch5.allocator()->allocate();
		weights_gamma_block4r_batch6.allocator()->allocate(); weights_gamma_block4r_batch7.allocator()->allocate(); weights_gamma_block4r_batch8.allocator()->allocate();
		weights_block4l_con0.allocator()->allocate(); weights_mean_block4l_batch0.allocator()->allocate(); weights_variance_block4l_batch0.allocator()->allocate();
		weights_beta_block4l_batch0.allocator()->allocate(); weights_gamma_block4l_batch0.allocator()->allocate();
		
		weights_con1.allocator()->allocate(); biases_con1.allocator()->allocate();
		return true;
}//end of do_setup
void do_run()override
{
		con0.run(); batch0.run();
		act0.run(); pool0.run();
		

		block1r_con0.run(); block1r_batch0.run(); block1r_act0.run();
		block1r_con1.run(); block1r_batch1.run(); block1r_act1.run();
		block1r_con2.run(); block1r_batch2.run(); block1l_con0.run(); block1l_batch0.run();
		block1_add0.run(); block1_act0.run();
		
		
		block1r_con3.run(); block1r_batch3.run(); block1r_act2.run();
		block1r_con4.run(); block1r_batch4.run(); block1r_act3.run();
		block1r_con5.run(); block1r_batch5.run();
		block1_add1.run(); block1_act1.run();
		
		
		block1r_con6.run(); block1r_batch6.run(); block1r_act4.run();
		block1r_con7.run(); block1r_batch7.run(); block1r_act5.run();
		block1r_con8.run(); block1r_batch8.run(); block1l_pool0.run();
		block1_add2.run(); block1_act2.run();
	
	
		//end block1
		//block2 run
		block2r_con0.run(); block2r_batch0.run(); block2r_act0.run();
		block2r_con1.run(); block2r_batch1.run(); block2r_act1.run();
		block2r_con2.run(); block2r_batch2.run(); block2l_con0.run(); block2l_batch0.run();
		block2_add0.run(); block2_act0.run();
		
		
		block2r_con3.run(); block2r_batch3.run(); block2r_act2.run();
		block2r_con4.run(); block2r_batch4.run(); block2r_act3.run();
		block2r_con5.run(); block2r_batch5.run();
		block2_add1.run(); block2_act1.run();
		

		block2r_con6.run(); block2r_batch6.run(); block2r_act4.run();
		block2r_con7.run(); block2r_batch7.run(); block2r_act5.run();
		block2r_con8.run(); block2r_batch8.run();
		block2_add2.run(); block2_act2.run();
		
		
		block2r_con9.run(); block2r_batch9.run(); block2r_act6.run();
		block2r_con10.run(); block2r_batch10.run(); block2r_act7.run();
		block2r_con11.run(); block2r_batch11.run(); block2l_pool0.run();
		block2_add3.run(); block2_act3.run();
		
		
		//end block2
		//block3 run
		block3r_con0.run(); block3r_batch0.run(); block3r_act0.run();
		block3r_con1.run(); block3r_batch1.run(); block3r_act1.run();
		block3r_con2.run(); block3r_batch2.run(); block3l_con0.run(); block3l_batch0.run();
		block3_add0.run(); block3_act0.run();
		
		

		block3r_con3.run(); block3r_batch3.run(); block3r_act2.run();
		block3r_con4.run(); block3r_batch4.run(); block3r_act3.run();
		block3r_con5.run(); block3r_batch5.run();
		block3_add1.run(); block3_act1.run();
		
		

		block3r_con6.run(); block3r_batch6.run(); block3r_act4.run();
		block3r_con7.run(); block3r_batch7.run(); block3r_act5.run();
		block3r_con8.run(); block3r_batch8.run();
		block3_add2.run(); block3_act2.run();
		
		

		block3r_con9.run(); block3r_batch9.run(); block3r_act6.run();
		block3r_con10.run(); block3r_batch10.run(); block3r_act7.run();
		block3r_con11.run(); block3r_batch11.run();
		block3_add3.run(); block3_act3.run();
		
		

		block3r_con12.run(); block3r_batch12.run(); block3r_act8.run();
		block3r_con13.run(); block3r_batch13.run(); block3r_act9.run();
		block3r_con14.run(); block3r_batch14.run();
		block3_add4.run(); block3_act4.run();
		
		

		block3r_con15.run(); block3r_batch15.run(); block3r_act10.run();
		block3r_con16.run(); block3r_batch16.run(); block3r_act11.run();
		block3r_con17.run(); block3r_batch17.run(); block3l_pool0.run();
		block3_add5.run(); block3_act5.run();
		
		

		//end block3
		//block4 run
		block4r_con0.run(); block4r_batch0.run(); block4r_act0.run();
		block4r_con1.run(); block4r_batch1.run(); block4r_act1.run();
		block4r_con2.run(); block4r_batch2.run(); block4l_con0.run(); block4l_batch0.run();
		block4_add0.run(); block4_act0.run();
		
		

		block4r_con3.run(); block4r_batch3.run(); block4r_act2.run();
		block4r_con4.run(); block4r_batch4.run(); block4r_act3.run();
		block4r_con5.run(); block4r_batch5.run();
		block4_add1.run(); block4_act1.run();
		
		

		block4r_con6.run(); block4r_batch6.run(); block4r_act4.run();
		block4r_con7.run(); block4r_batch7.run(); block4r_act5.run();
		block4r_con8.run(); block4r_batch8.run(); 
		block4_add2.run(); block4_act2.run();
	
	
		//end block4

		//last run
		pool1.run(); con1.run(); flatten.run(); softmax.run();
		//end last
	
	}
}//end do_run()
private:
	//Tensor
	Tensor src{}; Tensor weights_con0{}; Tensor weights_mean_batch0{};
	Tensor weights_variance_batch0{}; Tensor weights_beta_batch0{}; Tensor weights_gamma_batch0{};

	Tensor weights_block1r_con0{}; Tensor weights_block1r_con1{}; Tensor weights_block1r_con2{};
	Tensor weights_block1r_con3{}; Tensor weights_block1r_con4{}; Tensor weights_block1r_con5{};
	Tensor weights_block1r_con6{}; Tensor weights_block1r_con7{}; Tensor weights_block1r_con8{};
	Tensor weights_mean_block1r_batch0{}; Tensor weights_mean_block1r_batch1{}; Tensor weights_mean_block1r_batch2{};
	Tensor weights_mean_block1r_batch3{}; Tensor weights_mean_block1r_batch4{}; Tensor weights_mean_block1r_batch5{};
	Tensor weights_mean_block1r_batch6{}; Tensor weights_mean_block1r_batch7{}; Tensor weights_mean_block1r_batch8{};
	Tensor weights_variance_block1r_batch0{}; Tensor weights_variance_block1r_batch1{}; Tensor weights_variance_block1r_batch2{};
	Tensor weights_variance_block1r_batch3{}; Tensor weights_variance_block1r_batch4{}; Tensor weights_variance_block1r_batch5{};
	Tensor weights_variance_block1r_batch6{}; Tensor weights_variance_block1r_batch7{}; Tensor weights_variance_block1r_batch8{};
	Tensor weights_beta_block1r_batch0{}; Tensor weights_beta_block1r_batch1{}; Tensor weights_beta_block1r_batch2{};
	Tensor weights_beta_block1r_batch3{}; Tensor weights_beta_block1r_batch4{}; Tensor weights_beta_block1r_batch5{};
	Tensor weights_beta_block1r_batch6{}; Tensor weights_beta_block1r_batch7{}; Tensor weights_beta_block1r_batch8{};
	Tensor weights_gamma_block1r_batch0{}; Tensor weights_gamma_block1r_batch1{}; Tensor weights_gamma_block1r_batch2{};
	Tensor weights_gamma_block1r_batch3{}; Tensor weights_gamma_block1r_batch4{}; Tensor weights_gamma_block1r_batch5{};
	Tensor weights_gamma_block1r_batch6{}; Tensor weights_gamma_block1r_batch7{}; Tensor weights_gamma_block1r_batch8{};
	Tensor weights_block1l_con0{}; Tensor weights_mean_block1l_batch0{}; Tensor weights_variance_block1l_batch0{};
	Tensor weights_beta_block1l_batch0{}; Tensor weights_gamma_block1l_batch0{};

	Tensor weights_block2r_con0{}; Tensor weights_block2r_con1{}; Tensor weights_block2r_con2{};
	Tensor weights_block2r_con3{}; Tensor weights_block2r_con4{}; Tensor weights_block2r_con5{};
	Tensor weights_block2r_con6{}; Tensor weights_block2r_con7{}; Tensor weights_block2r_con8{};
	Tensor weights_block2r_con9{}; Tensor weights_block2r_con10{}; Tensor weights_block2r_con11{};
	Tensor weights_mean_block2r_batch0{}; Tensor weights_mean_block2r_batch1{}; Tensor weights_mean_block2r_batch2{};
	Tensor weights_mean_block2r_batch3{}; Tensor weights_mean_block2r_batch4{}; Tensor weights_mean_block2r_batch5{};
	Tensor weights_mean_block2r_batch6{}; Tensor weights_mean_block2r_batch7{}; Tensor weights_mean_block2r_batch8{};
	Tensor weights_mean_block2r_batch9{}; Tensor weights_mean_block2r_batch10{}; Tensor weights_mean_block2r_batch11{};
	Tensor weights_variance_block2r_batch0{}; Tensor weights_variance_block2r_batch1{}; Tensor weights_variance_block2r_batch2{};
	Tensor weights_variance_block2r_batch3{}; Tensor weights_variance_block2r_batch4{}; Tensor weights_variance_block2r_batch5{};
	Tensor weights_variance_block2r_batch6{}; Tensor weights_variance_block2r_batch7{}; Tensor weights_variance_block2r_batch8{};
	Tensor weights_variance_block2r_batch9{}; Tensor weights_variance_block2r_batch10{}; Tensor weights_variance_block2r_batch11{};
	Tensor weights_beta_block2r_batch0{}; Tensor weights_beta_block2r_batch1{}; Tensor weights_beta_block2r_batch2{};
	Tensor weights_beta_block2r_batch3{}; Tensor weights_beta_block2r_batch4{}; Tensor weights_beta_block2r_batch5{};
	Tensor weights_beta_block2r_batch6{}; Tensor weights_beta_block2r_batch7{}; Tensor weights_beta_block2r_batch8{};
	Tensor weights_beta_block2r_batch9{}; Tensor weights_beta_block2r_batch10{}; Tensor weights_beta_block2r_batch11{};
	Tensor weights_gamma_block2r_batch0{}; Tensor weights_gamma_block2r_batch1{}; Tensor weights_gamma_block2r_batch2{};
	Tensor weights_gamma_block2r_batch3{}; Tensor weights_gamma_block2r_batch4{}; Tensor weights_gamma_block2r_batch5{};
	Tensor weights_gamma_block2r_batch6{}; Tensor weights_gamma_block2r_batch7{}; Tensor weights_gamma_block2r_batch8{};
	Tensor weights_gamma_block2r_batch9{}; Tensor weights_gamma_block2r_batch10{}; Tensor weights_gamma_block2r_batch11{};
	Tensor weights_block2l_con0{}; Tensor weights_mean_block2l_batch0{}; Tensor weights_variance_block2l_batch0{};
	Tensor weights_beta_block2l_batch0{}; Tensor weights_gamma_block2l_batch0{};

	Tensor weights_block3r_con0{}; Tensor weights_block3r_con1{}; Tensor weights_block3r_con2{};
	Tensor weights_block3r_con3{}; Tensor weights_block3r_con4{}; Tensor weights_block3r_con5{};
	Tensor weights_block3r_con6{}; Tensor weights_block3r_con7{}; Tensor weights_block3r_con8{};
	Tensor weights_block3r_con9{}; Tensor weights_block3r_con10{}; Tensor weights_block3r_con11{};
	Tensor weights_block3r_con12{}; Tensor weights_block3r_con13{}; Tensor weights_block3r_con14{};
	Tensor weights_block3r_con15{}; Tensor weights_block3r_con16{}; Tensor weights_block3r_con17{};
	Tensor weights_mean_block3r_batch0{}; Tensor weights_mean_block3r_batch1{}; Tensor weights_mean_block3r_batch2{};
	Tensor weights_mean_block3r_batch3{}; Tensor weights_mean_block3r_batch4{}; Tensor weights_mean_block3r_batch5{};
	Tensor weights_mean_block3r_batch6{}; Tensor weights_mean_block3r_batch7{}; Tensor weights_mean_block3r_batch8{};
	Tensor weights_mean_block3r_batch9{}; Tensor weights_mean_block3r_batch10{}; Tensor weights_mean_block3r_batch11{};
	Tensor weights_mean_block3r_batch12{}; Tensor weights_mean_block3r_batch13{}; Tensor weights_mean_block3r_batch14{};
	Tensor weights_mean_block3r_batch15{}; Tensor weights_mean_block3r_batch16{}; Tensor weights_mean_block3r_batch17{};
	Tensor weights_variance_block3r_batch0{}; Tensor weights_variance_block3r_batch1{}; Tensor weights_variance_block3r_batch2{};
	Tensor weights_variance_block3r_batch3{}; Tensor weights_variance_block3r_batch4{}; Tensor weights_variance_block3r_batch5{};
	Tensor weights_variance_block3r_batch6{}; Tensor weights_variance_block3r_batch7{}; Tensor weights_variance_block3r_batch8{};
	Tensor weights_variance_block3r_batch9{}; Tensor weights_variance_block3r_batch10{}; Tensor weights_variance_block3r_batch11{};
	Tensor weights_variance_block3r_batch12{}; Tensor weights_variance_block3r_batch13{}; Tensor weights_variance_block3r_batch14{};
	Tensor weights_variance_block3r_batch15{}; Tensor weights_variance_block3r_batch16{}; Tensor weights_variance_block3r_batch17{};
	Tensor weights_beta_block3r_batch0{}; Tensor weights_beta_block3r_batch1{}; Tensor weights_beta_block3r_batch2{};
	Tensor weights_beta_block3r_batch3{}; Tensor weights_beta_block3r_batch4{}; Tensor weights_beta_block3r_batch5{};
	Tensor weights_beta_block3r_batch6{}; Tensor weights_beta_block3r_batch7{}; Tensor weights_beta_block3r_batch8{};
	Tensor weights_beta_block3r_batch9{}; Tensor weights_beta_block3r_batch10{}; Tensor weights_beta_block3r_batch11{};
	Tensor weights_beta_block3r_batch12{}; Tensor weights_beta_block3r_batch13{}; Tensor weights_beta_block3r_batch14{};
	Tensor weights_beta_block3r_batch15{}; Tensor weights_beta_block3r_batch16{}; Tensor weights_beta_block3r_batch17{};
	Tensor weights_gamma_block3r_batch0{}; Tensor weights_gamma_block3r_batch1{}; Tensor weights_gamma_block3r_batch2{};
	Tensor weights_gamma_block3r_batch3{}; Tensor weights_gamma_block3r_batch4{}; Tensor weights_gamma_block3r_batch5{};
	Tensor weights_gamma_block3r_batch6{}; Tensor weights_gamma_block3r_batch7{}; Tensor weights_gamma_block3r_batch8{};
	Tensor weights_gamma_block3r_batch9{}; Tensor weights_gamma_block3r_batch10{}; Tensor weights_gamma_block3r_batch11{};
	Tensor weights_gamma_block3r_batch12{}; Tensor weights_gamma_block3r_batch13{}; Tensor weights_gamma_block3r_batch14{};
	Tensor weights_gamma_block3r_batch15{}; Tensor weights_gamma_block3r_batch16{}; Tensor weights_gamma_block3r_batch17{};
	Tensor weights_block3l_con0{}; Tensor weights_mean_block3l_batch0{}; Tensor weights_variance_block3l_batch0{};
	Tensor weights_beta_block3l_batch0{}; Tensor weights_gamma_block3l_batch0{};

	Tensor weights_block4r_con0{}; Tensor weights_block4r_con1{}; Tensor weights_block4r_con2{};
	Tensor weights_block4r_con3{}; Tensor weights_block4r_con4{}; Tensor weights_block4r_con5{};
	Tensor weights_block4r_con6{}; Tensor weights_block4r_con7{}; Tensor weights_block4r_con8{};
	Tensor weights_mean_block4r_batch0{}; Tensor weights_mean_block4r_batch1{}; Tensor weights_mean_block4r_batch2{};
	Tensor weights_mean_block4r_batch3{}; Tensor weights_mean_block4r_batch4{}; Tensor weights_mean_block4r_batch5{};
	Tensor weights_mean_block4r_batch6{}; Tensor weights_mean_block4r_batch7{}; Tensor weights_mean_block4r_batch8{};
	Tensor weights_variance_block4r_batch0{}; Tensor weights_variance_block4r_batch1{}; Tensor weights_variance_block4r_batch2{};
	Tensor weights_variance_block4r_batch3{}; Tensor weights_variance_block4r_batch4{}; Tensor weights_variance_block4r_batch5{};
	Tensor weights_variance_block4r_batch6{}; Tensor weights_variance_block4r_batch7{}; Tensor weights_variance_block4r_batch8{};
	Tensor weights_beta_block4r_batch0{}; Tensor weights_beta_block4r_batch1{}; Tensor weights_beta_block4r_batch2{};
	Tensor weights_beta_block4r_batch3{}; Tensor weights_beta_block4r_batch4{}; Tensor weights_beta_block4r_batch5{};
	Tensor weights_beta_block4r_batch6{}; Tensor weights_beta_block4r_batch7{}; Tensor weights_beta_block4r_batch8{};
	Tensor weights_gamma_block4r_batch0{}; Tensor weights_gamma_block4r_batch1{}; Tensor weights_gamma_block4r_batch2{};
	Tensor weights_gamma_block4r_batch3{}; Tensor weights_gamma_block4r_batch4{}; Tensor weights_gamma_block4r_batch5{};
	Tensor weights_gamma_block4r_batch6{}; Tensor weights_gamma_block4r_batch7{}; Tensor weights_gamma_block4r_batch8{};
	Tensor weights_block4l_con0{}; Tensor weights_mean_block4l_batch0{}; Tensor weights_variance_block4l_batch0{};
	Tensor weights_beta_block4l_batch0{}; Tensor weights_gamma_block4l_batch0{};

	Tensor weights_con1{}, biases_con1{};


	Tensor out_con0{}; Tensor out_batch0{};
	Tensor out_act0{}; Tensor out_pool0{};
	//block1
	Tensor out_block1r_con0{}; Tensor out_block1r_batch0{}; Tensor out_block1r_act0{};
	Tensor out_block1r_con1{}; Tensor out_block1r_batch1{}; Tensor out_block1r_act1{};
	Tensor out_block1r_con2{}; Tensor out_block1r_batch2{}; Tensor out_block1l_con0{}; Tensor out_block1l_batch0{};
	Tensor out_block1_add0{}; Tensor out_block1_act0{};

	Tensor out_block1r_con3{}; Tensor out_block1r_batch3{}; Tensor out_block1r_act2{};
	Tensor out_block1r_con4{}; Tensor out_block1r_batch4{}; Tensor out_block1r_act3{};
	Tensor out_block1r_con5{}; Tensor out_block1r_batch5{};
	Tensor out_block1_add1{}; Tensor out_block1_act1{};

	Tensor out_block1r_con6{}; Tensor out_block1r_batch6{}; Tensor out_block1r_act4{};
	Tensor out_block1r_con7{}; Tensor out_block1r_batch7{}; Tensor out_block1r_act5{};
	Tensor out_block1r_con8{}; Tensor out_block1r_batch8{}; Tensor out_block1l_pool0{};
	Tensor out_block1_add2{}; Tensor out_block1_act2{};
	//block2
	Tensor out_block2r_con0{}; Tensor out_block2r_batch0{}; Tensor out_block2r_act0{};
	Tensor out_block2r_con1{}; Tensor out_block2r_batch1{}; Tensor out_block2r_act1{};
	Tensor out_block2r_con2{}; Tensor out_block2r_batch2{}; Tensor out_block2l_con0{}; Tensor out_block2l_batch0{};
	Tensor out_block2_add0{}; Tensor out_block2_act0{};

	Tensor out_block2r_con3{}; Tensor out_block2r_batch3{}; Tensor out_block2r_act2{};
	Tensor out_block2r_con4{}; Tensor out_block2r_batch4{}; Tensor out_block2r_act3{};
	Tensor out_block2r_con5{}; Tensor out_block2r_batch5{};
	Tensor out_block2_add1{}; Tensor out_block2_act1{};

	Tensor out_block2r_con6{}; Tensor out_block2r_batch6{}; Tensor out_block2r_act4{};
	Tensor out_block2r_con7{}; Tensor out_block2r_batch7{}; Tensor out_block2r_act5{};
	Tensor out_block2r_con8{}; Tensor out_block2r_batch8{};
	Tensor out_block2_add2{}; Tensor out_block2_act2{};

	Tensor out_block2r_con9{}; Tensor out_block2r_batch9{}; Tensor out_block2r_act6{};
	Tensor out_block2r_con10{}; Tensor out_block2r_batch10{}; Tensor out_block2r_act7{};
	Tensor out_block2r_con11{}; Tensor out_block2r_batch11{}; Tensor out_block2l_pool0{};
	Tensor out_block2_add3{}; Tensor out_block2_act3{};
	//block3
	Tensor out_block3r_con0{}; Tensor out_block3r_batch0{}; Tensor out_block3r_act0{};
	Tensor out_block3r_con1{}; Tensor out_block3r_batch1{}; Tensor out_block3r_act1{};
	Tensor out_block3r_con2{}; Tensor out_block3r_batch2{}; Tensor out_block3l_con0{}; Tensor out_block3l_batch0{};
	Tensor out_block3_add0{}; Tensor out_block3_act0{};

	Tensor out_block3r_con3{}; Tensor out_block3r_batch3{}; Tensor out_block3r_act2{};
	Tensor out_block3r_con4{}; Tensor out_block3r_batch4{}; Tensor out_block3r_act3{};
	Tensor out_block3r_con5{}; Tensor out_block3r_batch5{};
	Tensor out_block3_add1{}; Tensor out_block3_act1{};

	Tensor out_block3r_con6{}; Tensor out_block3r_batch6{}; Tensor out_block3r_act4{};
	Tensor out_block3r_con7{}; Tensor out_block3r_batch7{}; Tensor out_block3r_act5{};
	Tensor out_block3r_con8{}; Tensor out_block3r_batch8{};
	Tensor out_block3_add2{}; Tensor out_block3_act2{};

	Tensor out_block3r_con9{}; Tensor out_block3r_batch9{}; Tensor out_block3r_act6{};
	Tensor out_block3r_con10{}; Tensor out_block3r_batch10{}; Tensor out_block3r_act7{};
	Tensor out_block3r_con11{}; Tensor out_block3r_batch11{};
	Tensor out_block3_add3{}; Tensor out_block3_act3{};

	Tensor out_block3r_con12{}; Tensor out_block3r_batch12{}; Tensor out_block3r_act8{};
	Tensor out_block3r_con13{}; Tensor out_block3r_batch13{}; Tensor out_block3r_act9{};
	Tensor out_block3r_con14{}; Tensor out_block3r_batch14{};
	Tensor out_block3_add4{}; Tensor out_block3_act4{};

	Tensor out_block3r_con15{}; Tensor out_block3r_batch15{}; Tensor out_block3r_act10{};
	Tensor out_block3r_con16{}; Tensor out_block3r_batch16{}; Tensor out_block3r_act11{};
	Tensor out_block3r_con17{}; Tensor out_block3r_batch17{}; Tensor out_block3l_pool0{};
	Tensor out_block3_add5{}; Tensor out_block3_act5{};

	//block4
	Tensor out_block4r_con0{}; Tensor out_block4r_batch0{}; Tensor out_block4r_act0{};
	Tensor out_block4r_con1{}; Tensor out_block4r_batch1{}; Tensor out_block4r_act1{};
	Tensor out_block4r_con2{}; Tensor out_block4r_batch2{}; Tensor out_block4l_con0{}; Tensor out_block4l_batch0{};
	Tensor out_block4_add0{}; Tensor out_block4_act0{};

	Tensor out_block4r_con3{}; Tensor out_block4r_batch3{}; Tensor out_block4r_act2{};
	Tensor out_block4r_con4{}; Tensor out_block4r_batch4{}; Tensor out_block4r_act3{};
	Tensor out_block4r_con5{}; Tensor out_block4r_batch5{};
	Tensor out_block4_add1{}; Tensor out_block4_act1{};

	Tensor out_block4r_con6{}; Tensor out_block4r_batch6{}; Tensor out_block4r_act4{};
	Tensor out_block4r_con7{}; Tensor out_block4r_batch7{}; Tensor out_block4r_act5{};
	Tensor out_block4r_con8{}; Tensor out_block4r_batch8{}; 
	Tensor out_block4_add2{}; Tensor out_block4_act2{};

	Tensor out_pool1{}; Tensor out_con1{}; Tensor out_flatten{}; Tensor out_softmax{};

	//Layer
	NEConvolutionLayer con0; NEBatchNormalizationLayer batch0; NEActivationLayer act0; NEPoolingLayer pool0;
	//block1
	NEConvolutionLayer  block1r_con0{}; NEBatchNormalizationLayer  block1r_batch0{}; NEActivationLayer  block1r_act0{};
	NEConvolutionLayer  block1r_con1{}; NEBatchNormalizationLayer  block1r_batch1{}; NEActivationLayer  block1r_act1{};
	NEConvolutionLayer  block1r_con2{}; NEBatchNormalizationLayer  block1r_batch2{}; NEConvolutionLayer block1l_con0{}; NEBatchNormalizationLayer block1l_batch0{};
	NEArithmeticAddition  block1_add0{}; NEActivationLayer  block1_act0{};

	NEConvolutionLayer  block1r_con3{}; NEBatchNormalizationLayer  block1r_batch3{}; NEActivationLayer  block1r_act2{};
	NEConvolutionLayer  block1r_con4{}; NEBatchNormalizationLayer  block1r_batch4{}; NEActivationLayer  block1r_act3{};
	NEConvolutionLayer  block1r_con5{}; NEBatchNormalizationLayer  block1r_batch5{};
	NEArithmeticAddition  block1_add1{}; NEActivationLayer  block1_act1{};

	NEConvolutionLayer  block1r_con6{}; NEBatchNormalizationLayer  block1r_batch6{}; NEActivationLayer  block1r_act4{};
	NEConvolutionLayer  block1r_con7{}; NEBatchNormalizationLayer  block1r_batch7{}; NEActivationLayer  block1r_act5{};
	NEConvolutionLayer  block1r_con8{}; NEBatchNormalizationLayer  block1r_batch8{}; NEPoolingLayer block1l_pool0{};
	NEArithmeticAddition  block1_add2{}; NEActivationLayer  block1_act2{};
	//block2
	NEConvolutionLayer  block2r_con0{}; NEBatchNormalizationLayer  block2r_batch0{}; NEActivationLayer  block2r_act0{};
	NEConvolutionLayer  block2r_con1{}; NEBatchNormalizationLayer  block2r_batch1{}; NEActivationLayer  block2r_act1{};
	NEConvolutionLayer  block2r_con2{}; NEBatchNormalizationLayer  block2r_batch2{}; NEConvolutionLayer block2l_con0{}; NEBatchNormalizationLayer block2l_batch0{};
	NEArithmeticAddition  block2_add0{}; NEActivationLayer  block2_act0{};

	NEConvolutionLayer  block2r_con3{}; NEBatchNormalizationLayer  block2r_batch3{}; NEActivationLayer  block2r_act2{};
	NEConvolutionLayer  block2r_con4{}; NEBatchNormalizationLayer  block2r_batch4{}; NEActivationLayer  block2r_act3{};
	NEConvolutionLayer  block2r_con5{}; NEBatchNormalizationLayer  block2r_batch5{};
	NEArithmeticAddition  block2_add1{}; NEActivationLayer  block2_act1{};

	NEConvolutionLayer  block2r_con6{}; NEBatchNormalizationLayer  block2r_batch6{}; NEActivationLayer  block2r_act4{};
	NEConvolutionLayer  block2r_con7{}; NEBatchNormalizationLayer  block2r_batch7{}; NEActivationLayer  block2r_act5{};
	NEConvolutionLayer  block2r_con8{}; NEBatchNormalizationLayer  block2r_batch8{};
	NEArithmeticAddition  block2_add2{}; NEActivationLayer  block2_act2{};

	NEConvolutionLayer  block2r_con9{}; NEBatchNormalizationLayer  block2r_batch9{}; NEActivationLayer  block2r_act6{};
	NEConvolutionLayer  block2r_con10{}; NEBatchNormalizationLayer  block2r_batch10{}; NEActivationLayer  block2r_act7{};
	NEConvolutionLayer  block2r_con11{}; NEBatchNormalizationLayer  block2r_batch11{}; NEPoolingLayer block2l_pool0{};
	NEArithmeticAddition  block2_add3{}; NEActivationLayer  block2_act3{};
	//block3
	NEConvolutionLayer  block3r_con0{}; NEBatchNormalizationLayer  block3r_batch0{}; NEActivationLayer  block3r_act0{};
	NEConvolutionLayer  block3r_con1{}; NEBatchNormalizationLayer  block3r_batch1{}; NEActivationLayer  block3r_act1{};
	NEConvolutionLayer  block3r_con2{}; NEBatchNormalizationLayer  block3r_batch2{}; NEConvolutionLayer block3l_con0{}; NEBatchNormalizationLayer block3l_batch0{};
	NEArithmeticAddition  block3_add0{}; NEActivationLayer  block3_act0{};

	NEConvolutionLayer  block3r_con3{}; NEBatchNormalizationLayer  block3r_batch3{}; NEActivationLayer  block3r_act2{};
	NEConvolutionLayer  block3r_con4{}; NEBatchNormalizationLayer  block3r_batch4{}; NEActivationLayer  block3r_act3{};
	NEConvolutionLayer  block3r_con5{}; NEBatchNormalizationLayer  block3r_batch5{};
	NEArithmeticAddition  block3_add1{}; NEActivationLayer  block3_act1{};

	NEConvolutionLayer  block3r_con6{}; NEBatchNormalizationLayer  block3r_batch6{}; NEActivationLayer  block3r_act4{};
	NEConvolutionLayer  block3r_con7{}; NEBatchNormalizationLayer  block3r_batch7{}; NEActivationLayer  block3r_act5{};
	NEConvolutionLayer  block3r_con8{}; NEBatchNormalizationLayer  block3r_batch8{};
	NEArithmeticAddition  block3_add2{}; NEActivationLayer  block3_act2{};

	NEConvolutionLayer  block3r_con9{}; NEBatchNormalizationLayer  block3r_batch9{}; NEActivationLayer  block3r_act6{};
	NEConvolutionLayer  block3r_con10{}; NEBatchNormalizationLayer  block3r_batch10{}; NEActivationLayer  block3r_act7{};
	NEConvolutionLayer  block3r_con11{}; NEBatchNormalizationLayer  block3r_batch11{};
	NEArithmeticAddition  block3_add3{}; NEActivationLayer  block3_act3{};

	NEConvolutionLayer  block3r_con12{}; NEBatchNormalizationLayer  block3r_batch12{}; NEActivationLayer  block3r_act8{};
	NEConvolutionLayer  block3r_con13{}; NEBatchNormalizationLayer  block3r_batch13{}; NEActivationLayer  block3r_act9{};
	NEConvolutionLayer  block3r_con14{}; NEBatchNormalizationLayer  block3r_batch14{};
	NEArithmeticAddition  block3_add4{}; NEActivationLayer  block3_act4{};

	NEConvolutionLayer  block3r_con15{}; NEBatchNormalizationLayer  block3r_batch15{}; NEActivationLayer  block3r_act10{};
	NEConvolutionLayer  block3r_con16{}; NEBatchNormalizationLayer  block3r_batch16{}; NEActivationLayer  block3r_act11{};
	NEConvolutionLayer  block3r_con17{}; NEBatchNormalizationLayer  block3r_batch17{}; NEPoolingLayer block3l_pool0{};
	NEArithmeticAddition  block3_add5{}; NEActivationLayer  block3_act5{};
	//block4
	NEConvolutionLayer  block4r_con0{}; NEBatchNormalizationLayer  block4r_batch0{}; NEActivationLayer  block4r_act0{};
	NEConvolutionLayer  block4r_con1{}; NEBatchNormalizationLayer  block4r_batch1{}; NEActivationLayer  block4r_act1{};
	NEConvolutionLayer  block4r_con2{}; NEBatchNormalizationLayer  block4r_batch2{}; NEConvolutionLayer block4l_con0{}; NEBatchNormalizationLayer block4l_batch0{};
	NEArithmeticAddition  block4_add0{}; NEActivationLayer  block4_act0{};

	NEConvolutionLayer  block4r_con3{}; NEBatchNormalizationLayer  block4r_batch3{}; NEActivationLayer  block4r_act2{};
	NEConvolutionLayer  block4r_con4{}; NEBatchNormalizationLayer  block4r_batch4{}; NEActivationLayer  block4r_act3{};
	NEConvolutionLayer  block4r_con5{}; NEBatchNormalizationLayer  block4r_batch5{};
	NEArithmeticAddition  block4_add1{}; NEActivationLayer  block4_act1{};

	NEConvolutionLayer  block4r_con6{}; NEBatchNormalizationLayer  block4r_batch6{}; NEActivationLayer  block4r_act4{};
	NEConvolutionLayer  block4r_con7{}; NEBatchNormalizationLayer  block4r_batch7{}; NEActivationLayer  block4r_act5{};
	NEConvolutionLayer  block4r_con8{}; NEBatchNormalizationLayer  block4r_batch8{};
	NEArithmeticAddition  block4_add2{}; NEActivationLayer  block4_act2{};



	NEPoolingLayer pool1; NEConvolutionLayer con1; NEFlattenLayer flatten; NESoftmaxLayer softmax;
	
	ConvertPolicy A;

};//end of class
int main(int argc, char **argv)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    //1s-4s
    for(int little=0;little<=3;little++){
        CPU_SET(little, &cpuset);
        int e = sched_setaffinity(getpid(), sizeof(cpuset), &cpuset);
        if(e !=0) {
            std::cout << "Error in setting sched_setaffinity \n";
        }

        int threadnum=0;
        for (int j = 0; j < 8; j++) {
            if (CPU_ISSET(j, &cpuset)) {
                threadnum+=1;
            }
        }

        CPPScheduler::get().set_num_threads(threadnum);
        std::cout<<"thread="<<CPPScheduler::get().num_threads()<<std::endl<<std::endl;
        utils::run_example<NEONRESNETExample>(argc, argv);
    }


    //1b-4b
    CPU_ZERO(&cpuset);
    for(int big=4;big<=7;big++){
        CPU_SET(big, &cpuset);
        int e = sched_setaffinity(getpid(), sizeof(cpuset), &cpuset);
        if(e !=0) {
            std::cout << "Error in setting sched_setaffinity \n";
        }

        int threadnum=0;
        for (int j = 0; j < 8; j++) {
            if (CPU_ISSET(j, &cpuset)) {
                threadnum+=1;
            }
        }

        CPPScheduler::get().set_num_threads(threadnum);
        std::cout<<"thread="<<CPPScheduler::get().num_threads()<<std::endl<<std::endl;
        utils::run_example<NEONRESNETExample>(argc, argv);
    }

    //mixed
    CPU_ZERO(&cpuset);
    for(int little=0;little<=3;little++){  
        CPU_SET(little, &cpuset);
        for(int big=4;big<8;big++){
            CPU_CLR(big,&cpuset);
        }
        for(int big=4;big<8;big++){
            CPU_SET(big, &cpuset);
            int e = sched_setaffinity(getpid(), sizeof(cpuset), &cpuset);
            if(e !=0) {
                std::cout << "Error in setting sched_setaffinity \n";
            }

            int threadnum=0;
            for (int j = 0; j < 8; j++) {
                if (CPU_ISSET(j, &cpuset)) {
                    threadnum+=1;
                }
            }

            CPPScheduler::get().set_num_threads(threadnum);
            std::cout<<"thread="<<CPPScheduler::get().num_threads()<<std::endl<<std::endl;
            utils::run_example<NEONRESNETExample>(argc, argv);
        }
        
    }
	//return utils::run_example<NEONRESNETExample>(argc, argv);
}