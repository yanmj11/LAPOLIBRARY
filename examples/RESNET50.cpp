/*#ifndef ARM_COMPUTE_CL  Needed by Utils.cpp to handle OpenCL exceptions properly */
/*#error "This example needs to be built with -DARM_COMPUTE_CL"*/
/*#endif ARM_COMPUTE_CL */

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

class NEONRESNETExample : public Example
{
public:
	bool do_setup(int argc, char **argv) override
	{
		string data_path="/media/sdcard/ComputeLibrary/data/neon_resnet50/";
		NPYLoader npy0;npy0.open(data_path+"input.npy");npy0.init_tensor2(src,DataType:: S8);
		/*first conv-batch-act-pooling*/
		NPYLoader npy1;npy1.open(data_path+"1.conv1.weight.npy");npy1.init_tensor2(weights_con0,DataType:: S8);
		NPYLoader npyb1;npyb1.open(data_path+"1.conv1.bias.npy");npyb1.init_tensor2(bias_con0,DataType:: S8);
		const TensorShape out_shape_con0(112, 112, 64);
		out_con0.allocator()->init(TensorInfo(out_shape_con0, 1, DataType:: S8));
		/* NPYLoader npy1_1;npy1_1.open(data_path+"bn1.running_mean.npy");npy1_1.init_tensor2(weights_mean_batch0,DataType:: S8);*/
		/* NPYLoader npy1_2;npy1_2.open(data_path+"bn1.running_var.npy");npy1_2.init_tensor2(weights_variance_batch0,DataType:: S8);*/
		/*NPYLoader npy1_3;npy1_3.open(data_path+"bn1.weight.npy");npy1_3.init_tensor2(weights_gamma_batch0,DataType:: S8);*/
		/* NPYLoader npy1_4;npy1_4.open(data_path+"bn1.bias.npy");npy1_4.init_tensor2(weights_beta_batch0,DataType:: S8);*/
		/* out_batch0.allocator()->init(TensorInfo(out_shape_con0, 1, DataType:: S8));*/
		out_act0.allocator()->init(TensorInfo(out_shape_con0,1,DataType:: F32));
		TensorShape out_shape_pool0 = out_shape_con0;
		out_shape_pool0.set(0, out_shape_pool0.x() / 2);
		out_shape_pool0.set(1, out_shape_pool0.y() / 2);
		out_pool0.allocator()->init(TensorInfo(out_shape_pool0, 1, DataType:: F32));

		/*pool0fs.allocator()->init(TensorInfo(out_shape_pool0, 1, DataType:: S8));*/
		/*first  end*/
	
		/* block start          */
		/* block1 */
		/* conv-batch-act*/
		NPYLoader npy2;npy2.open(data_path+"1.layer1.0.conv1.weight.npy");npy2.init_tensor2(weights_block1r_con0,DataType:: S8);
		NPYLoader npyb2;npyb2.open(data_path+"1.layer1.0.conv1.bias.npy");npyb2.init_tensor2(bias_block1r_con0,DataType:: S8);
		
		const TensorShape out_shape_block1r_con0(56, 56, 64);
		out_block1r_con0.allocator()->init(TensorInfo(out_shape_block1r_con0, 1, DataType:: S8));
		/* NPYLoader npy2_1;npy2_1.open(data_path+"layer1.0.bn1.running_mean.npy");npy2_1.init_tensor2(weights_mean_block1r_batch0,DataType:: S8);*/
		/* NPYLoader npy2_2;npy2_2.open(data_path+"layer1.0.bn1.running_var.npy");npy2_2.init_tensor2(weights_variance_block1r_batch0,DataType:: S8);*/
		/* NPYLoader npy2_3;npy2_3.open(data_path+"layer1.0.bn1.weight.npy");npy2_3.init_tensor2(weights_gamma_block1r_batch0,DataType:: S8);*/
		/* NPYLoader npy2_4;npy2_4.open(data_path+"layer1.0.bn1.bias.npy");npy2_4.init_tensor2(weights_beta_block1r_batch0,DataType:: S8);*/
		/*out_block1r_batch0.allocator()->init(TensorInfo(out_shape_block1r_con0, 1, DataType:: S8));*/
		out_block1r_act0.allocator()->init(TensorInfo(out_shape_block1r_con0, 1, DataType:: F32));
   		/*conv-batch-act	  */
		NPYLoader npy3;npy3.open(data_path+"1.layer1.0.conv2.weight.npy");npy3.init_tensor2(weights_block1r_con1,DataType:: S8);
		NPYLoader npyb3;npyb3.open(data_path+"1.layer1.0.conv2.bias.npy");npyb3.init_tensor2(bias_block1r_con1,DataType:: S8);
		const TensorShape out_shape_block1r_con1(56, 56, 64);
		out_block1r_con1.allocator()->init(TensorInfo(out_shape_block1r_con1, 1, DataType:: S8));
		/* NPYLoader npy3_1;npy3_1.open(data_path+"layer1.0.bn2.running_mean.npy");npy3_1.init_tensor2(weights_mean_block1r_batch1,DataType:: S8);*/
		/* NPYLoader npy3_2;npy3_2.open(data_path+"layer1.0.bn2.running_var.npy");npy3_2.init_tensor2(weights_variance_block1r_batch1,DataType:: S8);*/
		/* NPYLoader npy3_3;npy3_3.open(data_path+"layer1.0.bn2.weight.npy");npy3_3.init_tensor2(weights_gamma_block1r_batch1,DataType:: S8);*/
		 /* NPYLoader npy3_4;npy3_4.open(data_path+"layer1.0.bn2.bias.npy");npy3_4.init_tensor2(weights_beta_block1r_batch1,DataType:: S8);*/
		/* out_block1r_batch1.allocator()->init(TensorInfo(out_shape_block1r_con1, 1, DataType:: S8));*/
		out_block1r_act1.allocator()->init(TensorInfo(out_shape_block1r_con1, 1, DataType:: F32));
  		 /*conv-batch     */
		NPYLoader npy4;npy4.open(data_path+"1.layer1.0.conv3.weight.npy");npy4.init_tensor2(weights_block1r_con2,DataType:: S8);
		NPYLoader npyb4;npyb4.open(data_path+"1.layer1.0.conv3.bias.npy");npyb4.init_tensor2(bias_block1r_con2,DataType:: S8);
		const TensorShape out_shape_block1r_con2(56, 56,256);
		out_block1r_con2.allocator()->init(TensorInfo(out_shape_block1r_con2, 1, DataType:: S8));
		/* NPYLoader npy4_1;npy4_1.open(data_path+"layer1.0.bn3.running_mean.npy");npy4_1.init_tensor2(weights_mean_block1r_batch2,DataType:: S8);*/
		/* NPYLoader npy4_2;npy4_2.open(data_path+"layer1.0.bn3.running_var.npy");npy4_2.init_tensor2(weights_variance_block1r_batch2,DataType:: S8);*/
		/* NPYLoader npy4_3;npy4_3.open(data_path+"layer1.0.bn3.weight.npy");npy4_3.init_tensor2(weights_gamma_block1r_batch2,DataType:: S8);*/
		/* NPYLoader npy4_4;npy4_4.open(data_path+"layer1.0.bn3.bias.npy");npy4_4.init_tensor2(weights_beta_block1r_batch2,DataType:: S8);*/
		/* out_block1r_batch2.allocator()->init(TensorInfo(out_shape_block1r_con2, 1, DataType:: S8));	*/
   		/*conv-batch*/
		NPYLoader npy5;npy5.open(data_path+"1.layer1.0.downsample.0.weight.npy");npy5.init_tensor2(weights_block1l_con0,DataType:: S8);
		NPYLoader npyb5;npyb5.open(data_path+"1.layer1.0.downsample.0.bias.npy");npyb5.init_tensor2(bias_block1l_con0,DataType:: S8);
		const TensorShape out_shape_block1l_con0(56, 56, 256);
		out_block1l_con0.allocator()->init(TensorInfo(out_shape_block1l_con0, 1, DataType:: S8));
		/* 		NPYLoader npy5_1;npy5_1.open(data_path+"layer1.0.downsample.1.running_mean.npy");npy5_1.init_tensor2(weights_mean_block1l_batch0,DataType:: S8);*/
		/* 		NPYLoader npy5_2;npy5_2.open(data_path+"layer1.0.downsample.1.running_var.npy");npy5_2.init_tensor2(weights_variance_block1l_batch0,DataType:: S8);*/
		/* 		NPYLoader npy5_3;npy5_3.open(data_path+"layer1.0.downsample.1.weight.npy");npy5_3.init_tensor2(weights_gamma_block1l_batch0,DataType:: S8);*/
		/* 		NPYLoader npy5_4;npy5_4.open(data_path+"layer1.0.downsample.1.bias.npy");npy5_4.init_tensor2(weights_beta_block1l_batch0,DataType:: S8);*/
		/* 		out_block1l_batch0.allocator()->init(TensorInfo(out_shape_block1l_con0, 1, DataType:: S8));*/
		/*    //add-act*/
		TensorShape out_shape_block1_0 = out_shape_block1r_con2;
		out_block1_add0.allocator()->init(TensorInfo(out_shape_block1_0, 1, DataType:: S8));
		out_block1_act0.allocator()->init(TensorInfo(out_shape_block1_0, 1, DataType:: F32));
   		/*conv-batch-act*/
		NPYLoader npy6;npy6.open(data_path+"1.layer1.1.conv1.weight.npy");npy6.init_tensor2(weights_block1r_con3,DataType:: S8);
		NPYLoader npyb6;npyb6.open(data_path+"1.layer1.1.conv1.bias.npy");npyb6.init_tensor2(bias_block1r_con3,DataType:: S8);
		const TensorShape out_shape_block1r_con3(56, 56,64);
		out_block1r_con3.allocator()->init(TensorInfo(out_shape_block1r_con3, 1, DataType:: S8));
		/* NPYLoader npy6_1;npy6_1.open(data_path+"layer1.1.bn1.running_mean.npy");npy6_1.init_tensor2(weights_mean_block1r_batch3,DataType:: S8);*/
		/* NPYLoader npy6_2;npy6_2.open(data_path+"layer1.1.bn1.running_var.npy");npy6_2.init_tensor2(weights_variance_block1r_batch3,DataType:: S8);*/
		/* NPYLoader npy6_3;npy6_3.open(data_path+"layer1.1.bn1.weight.npy");npy6_3.init_tensor2(weights_gamma_block1r_batch3,DataType:: S8);*/
		/* NPYLoader npy6_4;npy6_4.open(data_path+"layer1.1.bn1.bias.npy");npy6_4.init_tensor2(weights_beta_block1r_batch3,DataType:: S8);*/
		/* out_block1r_batch3.allocator()->init(TensorInfo(out_shape_block1r_con3, 1, DataType:: S8));*/
		out_block1r_act2.allocator()->init(TensorInfo(out_shape_block1r_con3, 1, DataType:: F32));
   		/*conv-batch-act*/
		NPYLoader npy7;npy7.open(data_path+"1.layer1.1.conv2.weight.npy");npy7.init_tensor2(weights_block1r_con4,DataType:: S8);
		NPYLoader npyb7;npyb7.open(data_path+"1.layer1.1.conv2.bias.npy");npyb7.init_tensor2(bias_block1r_con4,DataType:: S8);
		const TensorShape out_shape_block1r_con4(56, 56, 64);
		out_block1r_con4.allocator()->init(TensorInfo(out_shape_block1r_con4, 1, DataType:: S8));
		/* NPYLoader npy7_1;npy7_1.open(data_path+"layer1.1.bn2.running_mean.npy");npy7_1.init_tensor2(weights_mean_block1r_batch4,DataType:: S8);*/
		/* NPYLoader npy7_2;npy7_2.open(data_path+"layer1.1.bn2.running_var.npy");npy7_2.init_tensor2(weights_variance_block1r_batch4,DataType:: S8);*/
		/* NPYLoader npy7_3;npy7_3.open(data_path+"layer1.1.bn2.weight.npy");npy7_3.init_tensor2(weights_gamma_block1r_batch4,DataType:: S8);*/
		/* NPYLoader npy7_4;npy7_4.open(data_path+"layer1.1.bn2.bias.npy");npy7_4.init_tensor2(weights_beta_block1r_batch4,DataType:: S8);*/
		/* out_block1r_batch4.allocator()->init(TensorInfo(out_shape_block1r_con4, 1, DataType:: S8));*/
		out_block1r_act3.allocator()->init(TensorInfo(out_shape_block1r_con4, 1, DataType:: F32));
   		/*conv-batch*/
		NPYLoader npy8;npy8.open(data_path+"1.layer1.1.conv3.weight.npy");npy8.init_tensor2(weights_block1r_con5,DataType:: S8);
		NPYLoader npyb8;npyb8.open(data_path+"1.layer1.1.conv3.bias.npy");npyb8.init_tensor2(bias_block1r_con5,DataType:: S8);
		const TensorShape out_shape_block1r_con5(56, 56,256);
		out_block1r_con5.allocator()->init(TensorInfo(out_shape_block1r_con5, 1, DataType:: S8));
		/* NPYLoader npy8_1;npy8_1.open(data_path+"layer1.1.bn3.running_mean.npy");npy8_1.init_tensor2(weights_mean_block1r_batch5,DataType:: S8);*/
		/* NPYLoader npy8_2;npy8_2.open(data_path+"layer1.1.bn3.running_var.npy");npy8_2.init_tensor2(weights_variance_block1r_batch5,DataType:: S8);*/
		/* NPYLoader npy8_3;npy8_3.open(data_path+"layer1.1.bn3.weight.npy");npy8_3.init_tensor2(weights_gamma_block1r_batch5,DataType:: S8);*/
		/* NPYLoader npy8_4;npy8_4.open(data_path+"layer1.1.bn3.bias.npy");npy8_4.init_tensor2(weights_beta_block1r_batch5,DataType:: S8);*/
		/* out_block1r_batch5.allocator()->init(TensorInfo(out_shape_block1r_con5, 1, DataType:: S8));*/
   		/*add-act*/
		TensorShape out_shape_block1_1 = out_shape_block1r_con5;
		out_block1_add1.allocator()->init(TensorInfo(out_shape_block1_1, 1, DataType:: S8));
		out_block1_act1.allocator()->init(TensorInfo(out_shape_block1_1, 1, DataType:: F32));
  		 /*conv-batch-act*/
		NPYLoader npy9;npy9.open(data_path+"1.layer1.2.conv1.weight.npy");npy9.init_tensor2(weights_block1r_con6,DataType:: S8);
		NPYLoader npyb9;npyb9.open(data_path+"1.layer1.2.conv1.bias.npy");npyb9.init_tensor2(bias_block1r_con6,DataType:: S8);
		const TensorShape out_shape_block1r_con6(56, 56, 64);
		out_block1r_con6.allocator()->init(TensorInfo(out_shape_block1r_con6, 1, DataType:: S8));
		/* NPYLoader npy9_1;npy9_1.open(data_path+"layer1.2.bn1.running_mean.npy");npy9_1.init_tensor2(weights_mean_block1r_batch6,DataType:: S8);*/
		/* NPYLoader npy9_2;npy9_2.open(data_path+"layer1.2.bn1.running_var.npy");npy9_2.init_tensor2(weights_variance_block1r_batch6,DataType:: S8);*/
		/* NPYLoader npy9_3;npy9_3.open(data_path+"layer1.2.bn1.weight.npy");npy9_3.init_tensor2(weights_gamma_block1r_batch6,DataType:: S8);*/
		/* NPYLoader npy9_4;npy9_4.open(data_path+"layer1.2.bn1.bias.npy");npy9_4.init_tensor2(weights_beta_block1r_batch6,DataType:: S8);*/
		/* out_block1r_batch6.allocator()->init(TensorInfo(out_shape_block1r_con6, 1, DataType:: S8));*/
		out_block1r_act4.allocator()->init(TensorInfo(out_shape_block1r_con6, 1, DataType:: F32));
  		 /*conv-batch-act*/
		NPYLoader npy10;npy10.open(data_path+"1.layer1.2.conv2.weight.npy");npy10.init_tensor2(weights_block1r_con7,DataType:: S8);
		NPYLoader npyb10;npyb10.open(data_path+"1.layer1.2.conv2.bias.npy");npyb10.init_tensor2(bias_block1r_con7,DataType:: S8);
		const TensorShape out_shape_block1r_con7(56, 56,64);
		out_block1r_con7.allocator()->init(TensorInfo(out_shape_block1r_con7, 1, DataType:: S8));
		/* NPYLoader npy10_1;npy10_1.open(data_path+"layer1.2.bn2.running_mean.npy");npy10_1.init_tensor2(weights_mean_block1r_batch7,DataType:: S8);*/
		/* NPYLoader npy10_2;npy10_2.open(data_path+"layer1.2.bn2.running_var.npy");npy10_2.init_tensor2(weights_variance_block1r_batch7,DataType:: S8);*/
		/* NPYLoader npy10_3;npy10_3.open(data_path+"layer1.2.bn2.weight.npy");npy10_3.init_tensor2(weights_gamma_block1r_batch7,DataType:: S8);*/
		/* NPYLoader npy10_4;npy10_4.open(data_path+"layer1.2.bn2.bias.npy");npy10_4.init_tensor2(weights_beta_block1r_batch7,DataType:: S8);*/
		/* out_block1r_batch7.allocator()->init(TensorInfo(out_shape_block1r_con7, 1, DataType:: S8));*/
		out_block1r_act5.allocator()->init(TensorInfo(out_shape_block1r_con7, 1, DataType:: F32));
   		/*conv-batch*/
		NPYLoader npy11;npy11.open(data_path+"1.layer1.2.conv3.weight.npy");npy11.init_tensor2(weights_block1r_con8,DataType:: S8);
		NPYLoader npyb11;npyb11.open(data_path+"1.layer1.2.conv3.bias.npy");npyb11.init_tensor2(bias_block1r_con8,DataType:: S8);

		const TensorShape out_shape_block1r_con8(56, 56,256);
		out_block1r_con8.allocator()->init(TensorInfo(out_shape_block1r_con8, 1, DataType:: S8));
		/* NPYLoader npy11_1;npy11_1.open(data_path+"layer1.2.bn3.running_mean.npy");npy11_1.init_tensor2(weights_mean_block1r_batch8,DataType:: S8);*/
		/* NPYLoader npy11_2;npy11_2.open(data_path+"layer1.2.bn3.running_var.npy");npy11_2.init_tensor2(weights_variance_block1r_batch8,DataType:: S8);*/
		/* NPYLoader npy11_3;npy11_3.open(data_path+"layer1.2.bn3.weight.npy");npy11_3.init_tensor2(weights_gamma_block1r_batch8,DataType:: S8);*/
		/* NPYLoader npy11_4;npy11_4.open(data_path+"layer1.2.bn3.bias.npy");npy11_4.init_tensor2(weights_beta_block1r_batch8,DataType:: S8);*/
		/* out_block1r_batch8.allocator()->init(TensorInfo(out_shape_block1r_con8, 1, DataType:: S8));*/
		/*pooling*/
		/*add-act*/
		TensorShape out_shape_block1_2 = out_shape_block1r_con8;
		out_block1_add2.allocator()->init(TensorInfo(out_shape_block1_2, 1, DataType:: S8));
		out_block1_act2.allocator()->init(TensorInfo(out_shape_block1_2, 1, DataType:: F32));

		
		/*block2*/
   		/*conv-batch-act*/
        NPYLoader npy12;npy12.open(data_path+"1.layer2.0.conv1.weight.npy");npy12.init_tensor2(weights_block2r_con0,DataType:: S8);
		NPYLoader npyb12;npyb12.open(data_path+"1.layer2.0.conv1.bias.npy");npyb12.init_tensor2(bias_block2r_con0,DataType:: S8);
		const TensorShape out_shape_block2r_con0(28, 28, 128);
		out_block2r_con0.allocator()->init(TensorInfo(out_shape_block2r_con0, 1, DataType:: S8));
		/* NPYLoader npy12_1;npy12_1.open(data_path+"layer2.0.bn1.running_mean.npy");npy12_1.init_tensor2(weights_mean_block2r_batch0,DataType:: S8);*/
		/* NPYLoader npy12_2;npy12_2.open(data_path+"layer2.0.bn1.running_var.npy");npy12_2.init_tensor2(weights_variance_block2r_batch0,DataType:: S8);*/
		/* NPYLoader npy12_3;npy12_3.open(data_path+"layer2.0.bn1.weight.npy");npy12_3.init_tensor2(weights_gamma_block2r_batch0,DataType:: S8);*/
		/* NPYLoader npy12_4;npy12_4.open(data_path+"layer2.0.bn1.bias.npy");npy12_4.init_tensor2(weights_beta_block2r_batch0,DataType:: S8);*/
		/* out_block2r_batch0.allocator()->init(TensorInfo(out_shape_block2r_con0, 1, DataType:: S8));*/
		out_block2r_act0.allocator()->init(TensorInfo(out_shape_block2r_con0, 1, DataType:: F32));
   		/*conv-batch-act*/
		NPYLoader npy13;npy13.open(data_path+"1.layer2.0.conv2.weight.npy");npy13.init_tensor2(weights_block2r_con1,DataType:: S8);
		NPYLoader npyb13;npyb13.open(data_path+"1.layer2.0.conv2.bias.npy");npyb13.init_tensor2(bias_block2r_con1,DataType:: S8);
		const TensorShape out_shape_block2r_con1(28, 28, 128);
		out_block2r_con1.allocator()->init(TensorInfo(out_shape_block2r_con1, 1, DataType:: S8));
		/* NPYLoader npy13_1;npy13_1.open(data_path+"layer2.0.bn2.running_mean.npy");npy13_1.init_tensor2(weights_mean_block2r_batch1,DataType:: S8);*/
		/* NPYLoader npy13_2;npy13_2.open(data_path+"layer2.0.bn2.running_var.npy");npy13_2.init_tensor2(weights_variance_block2r_batch1,DataType:: S8);*/
		/* NPYLoader npy13_3;npy13_3.open(data_path+"layer2.0.bn2.weight.npy");npy13_3.init_tensor2(weights_gamma_block2r_batch1,DataType:: S8);*/
		/* NPYLoader npy13_4;npy13_4.open(data_path+"layer2.0.bn2.bias.npy");npy13_4.init_tensor2(weights_beta_block2r_batch1,DataType:: S8);*/
		/* out_block2r_batch1.allocator()->init(TensorInfo(out_shape_block2r_con1, 1, DataType:: S8));*/
		out_block2r_act1.allocator()->init(TensorInfo(out_shape_block2r_con1, 1, DataType:: F32));
   		/*conv-batch*/
		NPYLoader npy14;npy14.open(data_path+"1.layer2.0.conv3.weight.npy");npy14.init_tensor2(weights_block2r_con2,DataType:: S8);
		NPYLoader npyb14;npyb14.open(data_path+"1.layer2.0.conv3.bias.npy");npyb14.init_tensor2(bias_block2r_con2,DataType:: S8);
		const TensorShape out_shape_block2r_con2(28, 28, 512);
		out_block2r_con2.allocator()->init(TensorInfo(out_shape_block2r_con2, 1, DataType:: S8));
		/* NPYLoader npy14_1;npy14_1.open(data_path+"layer2.0.bn3.running_mean.npy");npy14_1.init_tensor2(weights_mean_block2r_batch2,DataType:: S8);*/
		/* NPYLoader npy14_2;npy14_2.open(data_path+"layer2.0.bn3.running_var.npy");npy14_2.init_tensor2(weights_variance_block2r_batch2,DataType:: S8);*/
		/* NPYLoader npy14_3;npy14_3.open(data_path+"layer2.0.bn3.weight.npy");npy14_3.init_tensor2(weights_gamma_block2r_batch2,DataType:: S8);*/
		/* NPYLoader npy14_4;npy14_4.open(data_path+"layer2.0.bn3.bias.npy");npy14_4.init_tensor2(weights_beta_block2r_batch2,DataType:: S8);*/
		/* out_block2r_batch2.allocator()->init(TensorInfo(out_shape_block2r_con2, 1, DataType:: S8));	*/
   		/*conv-batch*/
		NPYLoader npy15;npy15.open(data_path+"1.layer2.0.downsample.0.weight.npy");npy15.init_tensor2(weights_block2l_con0,DataType:: S8);
		NPYLoader npyb15;npyb15.open(data_path+"1.layer2.0.downsample.0.bias.npy");npyb15.init_tensor2(bias_block2l_con0,DataType:: S8);
		const TensorShape out_shape_block2l_con0(28, 28, 512);
		out_block2l_con0.allocator()->init(TensorInfo(out_shape_block2l_con0, 1, DataType:: S8));
		/* NPYLoader npy15_1;npy15_1.open(data_path+"layer2.0.downsample.1.running_mean.npy");npy15_1.init_tensor2(weights_mean_block2l_batch0,DataType:: S8);*/
		/* NPYLoader npy15_2;npy15_2.open(data_path+"layer2.0.downsample.1.running_var.npy");npy15_2.init_tensor2(weights_variance_block2l_batch0,DataType:: S8);*/
		/* NPYLoader npy15_3;npy15_3.open(data_path+"layer2.0.downsample.1.weight.npy");npy15_3.init_tensor2(weights_gamma_block2l_batch0,DataType:: S8);*/
		/* NPYLoader npy15_4;npy15_4.open(data_path+"layer2.0.downsample.1.bias.npy");npy15_4.init_tensor2(weights_beta_block2l_batch0,DataType:: S8);*/
		/* out_block2l_batch0.allocator()->init(TensorInfo(out_shape_block2l_con0, 1, DataType:: S8));*/
   		/*add-act*/
		TensorShape out_shape_block2_0 = out_shape_block2r_con2;
		out_block2_add0.allocator()->init(TensorInfo(out_shape_block2_0, 1, DataType:: S8));
		out_block2_act0.allocator()->init(TensorInfo(out_shape_block2_0, 1, DataType:: F32));
   		/*conv-batch-act*/
        NPYLoader npy16;npy16.open(data_path+"1.layer2.1.conv1.weight.npy");npy16.init_tensor2(weights_block2r_con3,DataType:: S8);
		NPYLoader npyb16;npyb16.open(data_path+"1.layer2.1.conv1.bias.npy");npyb16.init_tensor2(bias_block2r_con3,DataType:: S8);
		const TensorShape out_shape_block2r_con3(28, 28, 128);
		out_block2r_con3.allocator()->init(TensorInfo(out_shape_block2r_con3, 1, DataType:: S8));
		/* NPYLoader npy16_1;npy16_1.open(data_path+"layer2.1.bn1.running_mean.npy");npy16_1.init_tensor2(weights_mean_block2r_batch3,DataType:: S8);*/
		/* NPYLoader npy16_2;npy16_2.open(data_path+"layer2.1.bn1.running_var.npy");npy16_2.init_tensor2(weights_variance_block2r_batch3,DataType:: S8);*/
		/* NPYLoader npy16_3;npy16_3.open(data_path+"layer2.1.bn1.weight.npy");npy16_3.init_tensor2(weights_gamma_block2r_batch3,DataType:: S8);*/
		/* NPYLoader npy16_4;npy16_4.open(data_path+"layer2.1.bn1.bias.npy");npy16_4.init_tensor2(weights_beta_block2r_batch3,DataType:: S8);*/
		/* out_block2r_batch3.allocator()->init(TensorInfo(out_shape_block2r_con3, 1, DataType:: S8));*/
		out_block2r_act2.allocator()->init(TensorInfo(out_shape_block2r_con3, 1, DataType:: F32));
  		 /*conv-batch-act*/
		NPYLoader npy17;npy17.open(data_path+"1.layer2.1.conv2.weight.npy");npy17.init_tensor2(weights_block2r_con4,DataType:: S8);
		NPYLoader npyb17;npyb17.open(data_path+"1.layer2.1.conv2.bias.npy");npyb17.init_tensor2(bias_block2r_con4,DataType:: S8);
		const TensorShape out_shape_block2r_con4(28, 28,128);
		out_block2r_con4.allocator()->init(TensorInfo(out_shape_block2r_con4, 1, DataType:: S8));
		/* NPYLoader npy17_1;npy17_1.open(data_path+"layer2.1.bn2.running_mean.npy");npy17_1.init_tensor2(weights_mean_block2r_batch4,DataType:: S8);*/
		/* NPYLoader npy17_2;npy17_2.open(data_path+"layer2.1.bn2.running_var.npy");npy17_2.init_tensor2(weights_variance_block2r_batch4,DataType:: S8);*/
		/* NPYLoader npy17_3;npy17_3.open(data_path+"layer2.1.bn2.weight.npy");npy17_3.init_tensor2(weights_gamma_block2r_batch4,DataType:: S8);*/
		/* NPYLoader npy17_4;npy17_4.open(data_path+"layer2.1.bn2.bias.npy");npy17_4.init_tensor2(weights_beta_block2r_batch4,DataType:: S8);*/
		/* out_block2r_batch4.allocator()->init(TensorInfo(out_shape_block2r_con4, 1, DataType:: S8));*/
		out_block2r_act3.allocator()->init(TensorInfo(out_shape_block2r_con4, 1, DataType:: F32));
 		/*conv-batch*/
		NPYLoader npy18;npy18.open(data_path+"1.layer2.1.conv3.weight.npy");npy18.init_tensor2(weights_block2r_con5,DataType:: S8);
		NPYLoader npyb18;npyb18.open(data_path+"1.layer2.1.conv3.bias.npy");npyb18.init_tensor2(bias_block2r_con5,DataType:: S8);
		const TensorShape out_shape_block2r_con5(28, 28,512);
		out_block2r_con5.allocator()->init(TensorInfo(out_shape_block2r_con5, 1, DataType:: S8));
		/* NPYLoader npy18_1;npy18_1.open(data_path+"layer2.1.bn3.running_mean.npy");npy18_1.init_tensor2(weights_mean_block2r_batch5,DataType:: S8);*/
		/* NPYLoader npy18_2;npy18_2.open(data_path+"layer2.1.bn3.running_var.npy");npy18_2.init_tensor2(weights_variance_block2r_batch5,DataType:: S8);*/
		/* NPYLoader npy18_3;npy18_3.open(data_path+"layer2.1.bn3.weight.npy");npy18_3.init_tensor2(weights_gamma_block2r_batch5,DataType:: S8);*/
		/* NPYLoader npy18_4;npy18_4.open(data_path+"layer2.1.bn3.bias.npy");npy18_4.init_tensor2(weights_beta_block2r_batch5,DataType:: S8);*/
		/* out_block2r_batch5.allocator()->init(TensorInfo(out_shape_block2r_con5, 1, DataType:: S8));*/
  		 /*add-act*/
		TensorShape out_shape_block2_1 = out_shape_block2r_con5;
		out_block2_add1.allocator()->init(TensorInfo(out_shape_block2_1, 1, DataType:: S8));
		out_block2_act1.allocator()->init(TensorInfo(out_shape_block2_1, 1, DataType:: F32));
  		 /*conv-batch-act*/
        NPYLoader npy19;npy19.open(data_path+"1.layer2.2.conv1.weight.npy");npy19.init_tensor2(weights_block2r_con6,DataType:: S8);
		 NPYLoader npyb19;npyb19.open(data_path+"1.layer2.2.conv1.bias.npy");npyb19.init_tensor2(bias_block2r_con6,DataType:: S8);
		const TensorShape out_shape_block2r_con6(28, 28, 128);
		out_block2r_con6.allocator()->init(TensorInfo(out_shape_block2r_con6, 1, DataType:: S8));
		/* NPYLoader npy19_1;npy19_1.open(data_path+"layer2.2.bn1.running_mean.npy");npy19_1.init_tensor2(weights_mean_block2r_batch6,DataType:: S8);*/
		/* NPYLoader npy19_2;npy19_2.open(data_path+"layer2.2.bn1.running_var.npy");npy19_2.init_tensor2(weights_variance_block2r_batch6,DataType:: S8);*/
		/* NPYLoader npy19_3;npy19_3.open(data_path+"layer2.2.bn1.weight.npy");npy19_3.init_tensor2(weights_gamma_block2r_batch6,DataType:: S8);*/
		/* NPYLoader npy19_4;npy19_4.open(data_path+"layer2.2.bn1.bias.npy");npy19_4.init_tensor2(weights_beta_block2r_batch6,DataType:: S8);*/
		/* out_block2r_batch6.allocator()->init(TensorInfo(out_shape_block2r_con6, 1, DataType:: S8));*/
		out_block2r_act4.allocator()->init(TensorInfo(out_shape_block2r_con6, 1, DataType:: F32));
  		 /*conv-batch-act*/
		NPYLoader npy20;npy20.open(data_path+"1.layer2.2.conv2.weight.npy");npy20.init_tensor2(weights_block2r_con7,DataType:: S8);
		NPYLoader npyb20;npyb20.open(data_path+"1.layer2.2.conv2.bias.npy");npyb20.init_tensor2(bias_block2r_con7,DataType:: S8);
		const TensorShape out_shape_block2r_con7(28, 28, 128);
		out_block2r_con7.allocator()->init(TensorInfo(out_shape_block2r_con7, 1, DataType:: S8));
		/* NPYLoader npy20_1;npy20_1.open(data_path+"layer2.2.bn2.running_mean.npy");npy20_1.init_tensor2(weights_mean_block2r_batch7,DataType:: S8);*/
		/* NPYLoader npy20_2;npy20_2.open(data_path+"layer2.2.bn2.running_var.npy");npy20_2.init_tensor2(weights_variance_block2r_batch7,DataType:: S8);*/
		/* NPYLoader npy20_3;npy20_3.open(data_path+"layer2.2.bn2.weight.npy");npy20_3.init_tensor2(weights_gamma_block2r_batch7,DataType:: S8);*/
		/* NPYLoader npy20_4;npy20_4.open(data_path+"layer2.2.bn2.bias.npy");npy20_4.init_tensor2(weights_beta_block2r_batch7,DataType:: S8);*/
		/* out_block2r_batch7.allocator()->init(TensorInfo(out_shape_block2r_con7, 1, DataType:: S8));*/
		out_block2r_act5.allocator()->init(TensorInfo(out_shape_block2r_con7, 1, DataType:: F32));
   		/*conv-batch*/
		NPYLoader npy21;npy21.open(data_path+"1.layer2.2.conv3.weight.npy");npy21.init_tensor2(weights_block2r_con8,DataType:: S8);
		NPYLoader npyb21;npyb21.open(data_path+"1.layer2.2.conv3.bias.npy");npyb21.init_tensor2(bias_block2r_con8,DataType:: S8);
		const TensorShape out_shape_block2r_con8(28, 28, 512);
		out_block2r_con8.allocator()->init(TensorInfo(out_shape_block2r_con8, 1, DataType:: S8));
		/* NPYLoader npy21_1;npy21_1.open(data_path+"layer2.2.bn3.running_mean.npy");npy21_1.init_tensor2(weights_mean_block2r_batch8,DataType:: S8);*/
		/* NPYLoader npy21_2;npy21_2.open(data_path+"layer2.2.bn3.running_var.npy");npy21_2.init_tensor2(weights_variance_block2r_batch8,DataType:: S8);*/
		/* NPYLoader npy21_3;npy21_3.open(data_path+"layer2.2.bn3.weight.npy");npy21_3.init_tensor2(weights_gamma_block2r_batch8,DataType:: S8);*/
		/* NPYLoader npy21_4;npy21_4.open(data_path+"layer2.2.bn3.bias.npy");npy21_4.init_tensor2(weights_beta_block2r_batch8,DataType:: S8);*/
		/* out_block2r_batch8.allocator()->init(TensorInfo(out_shape_block2r_con8, 1, DataType:: S8));*/
   		/*add-act*/
		TensorShape out_shape_block2_2 = out_shape_block2r_con8;
		out_block2_add2.allocator()->init(TensorInfo(out_shape_block2_2, 1, DataType:: S8));
		out_block2_act2.allocator()->init(TensorInfo(out_shape_block2_2, 1, DataType:: F32));
   		/*conv-batch-act*/
        NPYLoader npy22;npy22.open(data_path+"1.layer2.3.conv1.weight.npy");npy22.init_tensor2(weights_block2r_con9,DataType:: S8);
		NPYLoader npyb22;npyb22.open(data_path+"1.layer2.3.conv1.bias.npy");npyb22.init_tensor2(bias_block2r_con9,DataType:: S8);
		const TensorShape out_shape_block2r_con9(28, 28,128);
		out_block2r_con9.allocator()->init(TensorInfo(out_shape_block2r_con9, 1, DataType:: S8));
		/* NPYLoader npy22_1;npy22_1.open(data_path+"layer2.3.bn1.running_mean.npy");npy22_1.init_tensor2(weights_mean_block2r_batch9,DataType:: S8);*/
		/* NPYLoader npy22_2;npy22_2.open(data_path+"layer2.3.bn1.running_var.npy");npy22_2.init_tensor2(weights_variance_block2r_batch9,DataType:: S8);*/
		/* NPYLoader npy22_3;npy22_3.open(data_path+"layer2.3.bn1.weight.npy");npy22_3.init_tensor2(weights_gamma_block2r_batch9,DataType:: S8);*/
		/* NPYLoader npy22_4;npy22_4.open(data_path+"layer2.3.bn1.bias.npy");npy22_4.init_tensor2(weights_beta_block2r_batch9,DataType:: S8);*/
		/* out_block2r_batch9.allocator()->init(TensorInfo(out_shape_block2r_con9, 1, DataType:: S8));*/
		out_block2r_act6.allocator()->init(TensorInfo(out_shape_block2r_con9, 1, DataType:: F32));
   		/*conv-batch-act*/
		NPYLoader npy23;npy23.open(data_path+"1.layer2.3.conv2.weight.npy");npy23.init_tensor2(weights_block2r_con10,DataType:: S8);
		NPYLoader npyb23;npyb23.open(data_path+"1.layer2.3.conv2.bias.npy");npyb23.init_tensor2(bias_block2r_con10,DataType:: S8);
		const TensorShape out_shape_block2r_con10(28, 28, 128);
		out_block2r_con10.allocator()->init(TensorInfo(out_shape_block2r_con10, 1, DataType:: S8));
		/* NPYLoader npy23_1;npy23_1.open(data_path+"layer2.3.bn2.running_mean.npy");npy23_1.init_tensor2(weights_mean_block2r_batch10,DataType:: S8);*/
		/* NPYLoader npy23_2;npy23_2.open(data_path+"layer2.3.bn2.running_var.npy");npy23_2.init_tensor2(weights_variance_block2r_batch10,DataType:: S8);*/
		/* NPYLoader npy23_3;npy23_3.open(data_path+"layer2.3.bn2.weight.npy");npy23_3.init_tensor2(weights_gamma_block2r_batch10,DataType:: S8);*/
		/* NPYLoader npy23_4;npy23_4.open(data_path+"layer2.3.bn2.bias.npy");npy23_4.init_tensor2(weights_beta_block2r_batch10,DataType:: S8);*/
		/* out_block2r_batch10.allocator()->init(TensorInfo(out_shape_block2r_con10, 1, DataType:: S8));*/
		out_block2r_act7.allocator()->init(TensorInfo(out_shape_block2r_con10, 1, DataType:: F32));
   		/*conv-batch*/
		NPYLoader npy24;npy24.open(data_path+"1.layer2.3.conv3.weight.npy");npy24.init_tensor2(weights_block2r_con11,DataType:: S8);
		NPYLoader npyb24;npyb24.open(data_path+"1.layer2.3.conv3.bias.npy");npyb24.init_tensor2(bias_block2r_con11,DataType:: S8);
		const TensorShape out_shape_block2r_con11(28, 28, 512);
		out_block2r_con11.allocator()->init(TensorInfo(out_shape_block2r_con11, 1, DataType:: S8));
		/* NPYLoader npy24_1;npy24_1.open(data_path+"layer2.3.bn3.running_mean.npy");npy24_1.init_tensor2(weights_mean_block2r_batch11,DataType:: S8);*/
		/* NPYLoader npy24_2;npy24_2.open(data_path+"layer2.3.bn3.running_var.npy");npy24_2.init_tensor2(weights_variance_block2r_batch11,DataType:: S8);*/
		/* NPYLoader npy24_3;npy24_3.open(data_path+"layer2.3.bn3.weight.npy");npy24_3.init_tensor2(weights_gamma_block2r_batch11,DataType:: S8);*/
		/* NPYLoader npy24_4;npy24_4.open(data_path+"layer2.3.bn3.bias.npy");npy24_4.init_tensor2(weights_beta_block2r_batch11,DataType:: S8);*/
		/* out_block2r_batch11.allocator()->init(TensorInfo(out_shape_block2r_con11, 1, DataType:: S8));*/
		/*pooling*/
  		 /*add-act*/
		TensorShape out_shape_block2_3 = out_shape_block2r_con11;
		out_block2_add3.allocator()->init(TensorInfo(out_shape_block2_3, 1, DataType:: S8));
		out_block2_act3.allocator()->init(TensorInfo(out_shape_block2_3, 1, DataType:: F32));

		/*block3*/
  		 /*conv-batch-act*/
        NPYLoader npy25;npy25.open(data_path+"1.layer3.0.conv1.weight.npy");npy25.init_tensor2(weights_block3r_con0,DataType:: S8);
		NPYLoader npyb25;npyb25.open(data_path+"1.layer3.0.conv1.bias.npy");npyb25.init_tensor2(bias_block3r_con0,DataType:: S8);
		const TensorShape out_shape_block3r_con0(14, 14, 256);
		out_block3r_con0.allocator()->init(TensorInfo(out_shape_block3r_con0, 1, DataType:: S8));
		/* NPYLoader npy25_1;npy25_1.open(data_path+"layer3.0.bn1.running_mean.npy");npy25_1.init_tensor2(weights_mean_block3r_batch0,DataType:: S8);*/
		/* NPYLoader npy25_2;npy25_2.open(data_path+"layer3.0.bn1.running_var.npy");npy25_2.init_tensor2(weights_variance_block3r_batch0,DataType:: S8);*/
		/* NPYLoader npy25_3;npy25_3.open(data_path+"layer3.0.bn1.weight.npy");npy25_3.init_tensor2(weights_gamma_block3r_batch0,DataType:: S8);*/
		/* NPYLoader npy25_4;npy25_4.open(data_path+"layer3.0.bn1.bias.npy");npy25_4.init_tensor2(weights_beta_block3r_batch0,DataType:: S8);*/
		/* out_block3r_batch0.allocator()->init(TensorInfo(out_shape_block3r_con0, 1, DataType:: S8));*/
		out_block3r_act0.allocator()->init(TensorInfo(out_shape_block3r_con0, 1, DataType:: F32));
		/*conv-batch-act*/
		NPYLoader npy26;npy26.open(data_path+"1.layer3.0.conv2.weight.npy");npy26.init_tensor2(weights_block3r_con1,DataType:: S8);
		NPYLoader npyb26;npyb26.open(data_path+"1.layer3.0.conv2.bias.npy");npyb26.init_tensor2(bias_block3r_con1,DataType:: S8);
		const TensorShape out_shape_block3r_con1(14, 14, 256);
		out_block3r_con1.allocator()->init(TensorInfo(out_shape_block3r_con1, 1, DataType:: S8));
		/* NPYLoader npy26_1;npy26_1.open(data_path+"layer3.0.bn2.running_mean.npy");npy26_1.init_tensor2(weights_mean_block3r_batch1,DataType:: S8);*/
		/* NPYLoader npy26_2;npy26_2.open(data_path+"layer3.0.bn2.running_var.npy");npy26_2.init_tensor2(weights_variance_block3r_batch1,DataType:: S8);*/
		/* NPYLoader npy26_3;npy26_3.open(data_path+"layer3.0.bn2.weight.npy");npy26_3.init_tensor2(weights_gamma_block3r_batch1,DataType:: S8);*/
		/* NPYLoader npy26_4;npy26_4.open(data_path+"layer3.0.bn2.bias.npy");npy26_4.init_tensor2(weights_beta_block3r_batch1,DataType:: S8);*/
		/* out_block3r_batch1.allocator()->init(TensorInfo(out_shape_block3r_con1, 1, DataType:: S8));*/
		out_block3r_act1.allocator()->init(TensorInfo(out_shape_block3r_con1, 1, DataType:: F32));
   		/*conv-batch		*/
		NPYLoader npy27;npy27.open(data_path+"1.layer3.0.conv3.weight.npy");npy27.init_tensor2(weights_block3r_con2,DataType:: S8);
		NPYLoader npyb27;npyb27.open(data_path+"1.layer3.0.conv3.bias.npy");npyb27.init_tensor2(bias_block3r_con2,DataType:: S8);
		const TensorShape out_shape_block3r_con2(14, 14, 1024);
		out_block3r_con2.allocator()->init(TensorInfo(out_shape_block3r_con2, 1, DataType:: S8));
		/* NPYLoader npy27_1;npy27_1.open(data_path+"layer3.0.bn3.running_mean.npy");npy27_1.init_tensor2(weights_mean_block3r_batch2,DataType:: S8);*/
		/* NPYLoader npy27_2;npy27_2.open(data_path+"layer3.0.bn3.running_var.npy");npy27_2.init_tensor2(weights_variance_block3r_batch2,DataType:: S8);*/
		/* NPYLoader npy27_3;npy27_3.open(data_path+"layer3.0.bn3.weight.npy");npy27_3.init_tensor2(weights_gamma_block3r_batch2,DataType:: S8);*/
		/* NPYLoader npy27_4;npy27_4.open(data_path+"layer3.0.bn3.bias.npy");npy27_4.init_tensor2(weights_beta_block3r_batch2,DataType:: S8);*/
		/* out_block3r_batch2.allocator()->init(TensorInfo(out_shape_block3r_con2, 1, DataType:: S8));*/
  		 /*conv-batch*/
		NPYLoader npy28;npy28.open(data_path+"1.layer3.0.downsample.0.weight.npy");npy28.init_tensor2(weights_block3l_con0,DataType:: S8);
		NPYLoader npyb28;npyb28.open(data_path+"1.layer3.0.downsample.0.bias.npy");npyb28.init_tensor2(bias_block3l_con0,DataType:: S8);
		const TensorShape out_shape_block3l_con0(14, 14, 1024);
		out_block3l_con0.allocator()->init(TensorInfo(out_shape_block3l_con0, 1, DataType:: S8));
		/* NPYLoader npy28_1;npy28_1.open(data_path+"layer3.0.downsample.1.running_mean.npy");npy28_1.init_tensor2(weights_mean_block3l_batch0,DataType:: S8);*/
		/* NPYLoader npy28_2;npy28_2.open(data_path+"layer3.0.downsample.1.running_var.npy");npy28_2.init_tensor2(weights_variance_block3l_batch0,DataType:: S8);*/
		/* NPYLoader npy28_3;npy28_3.open(data_path+"layer3.0.downsample.1.weight.npy");npy28_3.init_tensor2(weights_gamma_block3l_batch0,DataType:: S8);*/
		/* NPYLoader npy28_4;npy28_4.open(data_path+"layer3.0.downsample.1.bias.npy");npy28_4.init_tensor2(weights_beta_block3l_batch0,DataType:: S8);*/
		/* out_block3l_batch0.allocator()->init(TensorInfo(out_shape_block3l_con0, 1, DataType:: S8));*/
   		/*add-act*/
		TensorShape out_shape_block3_0 = out_shape_block3r_con2;
		out_block3_add0.allocator()->init(TensorInfo(out_shape_block3_0, 1, DataType:: S8));
		out_block3_act0.allocator()->init(TensorInfo(out_shape_block3_0, 1, DataType:: F32));
   		/*conv-batch-act*/
		NPYLoader npy29;npy29.open(data_path+"1.layer3.1.conv1.weight.npy");npy29.init_tensor2(weights_block3r_con3,DataType:: S8);
		NPYLoader npyb29;npyb29.open(data_path+"1.layer3.1.conv1.bias.npy");npyb29.init_tensor2(bias_block3r_con3,DataType:: S8);
		const TensorShape out_shape_block3r_con3(14, 14,256);
		out_block3r_con3.allocator()->init(TensorInfo(out_shape_block3r_con3, 1, DataType:: S8));
		/* NPYLoader npy29_1;npy29_1.open(data_path+"layer3.1.bn1.running_mean.npy");npy29_1.init_tensor2(weights_mean_block3r_batch3,DataType:: S8);*/
		/* NPYLoader npy29_2;npy29_2.open(data_path+"layer3.1.bn1.running_var.npy");npy29_2.init_tensor2(weights_variance_block3r_batch3,DataType:: S8);*/
		/* NPYLoader npy29_3;npy29_3.open(data_path+"layer3.1.bn1.weight.npy");npy29_3.init_tensor2(weights_gamma_block3r_batch3,DataType:: S8);*/
		/* NPYLoader npy29_4;npy29_4.open(data_path+"layer3.1.bn1.bias.npy");npy29_4.init_tensor2(weights_beta_block3r_batch3,DataType:: S8);*/
		/* out_block3r_batch3.allocator()->init(TensorInfo(out_shape_block3r_con3, 1, DataType:: S8));*/
		out_block3r_act2.allocator()->init(TensorInfo(out_shape_block3r_con3, 1, DataType:: F32));
  		 /*conv-batch-act		*/
		NPYLoader npy30;npy30.open(data_path+"1.layer3.1.conv2.weight.npy");npy30.init_tensor2(weights_block3r_con4,DataType:: S8);
		NPYLoader npyb30;npyb30.open(data_path+"1.layer3.1.conv2.bias.npy");npyb30.init_tensor2(bias_block3r_con4,DataType:: S8);
		const TensorShape out_shape_block3r_con4(14, 14,256);
		out_block3r_con4.allocator()->init(TensorInfo(out_shape_block3r_con4, 1, DataType:: S8));
		/* NPYLoader npy30_1;npy30_1.open(data_path+"layer3.1.bn2.running_mean.npy");npy30_1.init_tensor2(weights_mean_block3r_batch4,DataType:: S8);*/
		/* NPYLoader npy30_2;npy30_2.open(data_path+"layer3.1.bn2.running_var.npy");npy30_2.init_tensor2(weights_variance_block3r_batch4,DataType:: S8);*/
		/* NPYLoader npy30_3;npy30_3.open(data_path+"layer3.1.bn2.weight.npy");npy30_3.init_tensor2(weights_gamma_block3r_batch4,DataType:: S8);*/
		/* NPYLoader npy30_4;npy30_4.open(data_path+"layer3.1.bn2.bias.npy");npy30_4.init_tensor2(weights_beta_block3r_batch4,DataType:: S8);*/
		/* out_block3r_batch4.allocator()->init(TensorInfo(out_shape_block3r_con4, 1, DataType:: S8));*/
		out_block3r_act3.allocator()->init(TensorInfo(out_shape_block3r_con4, 1, DataType:: F32));
   		/*conv-batch		*/
		NPYLoader npy31;npy31.open(data_path+"1.layer3.1.conv3.weight.npy");npy31.init_tensor2(weights_block3r_con5,DataType:: S8);
		NPYLoader npyb31;npyb31.open(data_path+"1.layer3.1.conv3.bias.npy");npyb31.init_tensor2(bias_block3r_con5,DataType:: S8);
		const TensorShape out_shape_block3r_con5(14, 14, 1024);
		out_block3r_con5.allocator()->init(TensorInfo(out_shape_block3r_con5, 1, DataType:: S8));
		/* NPYLoader npy31_1;npy31_1.open(data_path+"layer3.1.bn3.running_mean.npy");npy31_1.init_tensor2(weights_mean_block3r_batch5,DataType:: S8);*/
		/* NPYLoader npy31_2;npy31_2.open(data_path+"layer3.1.bn3.running_var.npy");npy31_2.init_tensor2(weights_variance_block3r_batch5,DataType:: S8);*/
		/* NPYLoader npy31_3;npy31_3.open(data_path+"layer3.1.bn3.weight.npy");npy31_3.init_tensor2(weights_gamma_block3r_batch5,DataType:: S8);*/
		/* NPYLoader npy31_4;npy31_4.open(data_path+"layer3.1.bn3.bias.npy");npy31_4.init_tensor2(weights_beta_block3r_batch5,DataType:: S8);*/
		/* out_block3r_batch5.allocator()->init(TensorInfo(out_shape_block3r_con5, 1, DataType:: S8));*/
   		/*add-act		*/
		TensorShape out_shape_block3_1 = out_shape_block3r_con5;
		out_block3_add1.allocator()->init(TensorInfo(out_shape_block3_1, 1, DataType:: S8));
		out_block3_act1.allocator()->init(TensorInfo(out_shape_block3_1, 1, DataType:: F32));
   		/*conv-batch-act*/
		NPYLoader npy32;npy32.open(data_path+"1.layer3.2.conv1.weight.npy");npy32.init_tensor2(weights_block3r_con6,DataType:: S8);
		NPYLoader npyb32;npyb32.open(data_path+"1.layer3.2.conv1.bias.npy");npyb32.init_tensor2(bias_block3r_con6,DataType:: S8);
		const TensorShape out_shape_block3r_con6(14, 14, 256);
		out_block3r_con6.allocator()->init(TensorInfo(out_shape_block3r_con6, 1, DataType:: S8));
		/* NPYLoader npy32_1;npy32_1.open(data_path+"layer3.2.bn1.running_mean.npy");npy32_1.init_tensor2(weights_mean_block3r_batch6,DataType:: S8);*/
		/* NPYLoader npy32_2;npy32_2.open(data_path+"layer3.2.bn1.running_var.npy");npy32_2.init_tensor2(weights_variance_block3r_batch6,DataType:: S8);*/
		/* NPYLoader npy32_3;npy32_3.open(data_path+"layer3.2.bn1.weight.npy");npy32_3.init_tensor2(weights_gamma_block3r_batch6,DataType:: S8);*/
		/* NPYLoader npy32_4;npy32_4.open(data_path+"layer3.2.bn1.bias.npy");npy32_4.init_tensor2(weights_beta_block3r_batch6,DataType:: S8);*/
		/* out_block3r_batch6.allocator()->init(TensorInfo(out_shape_block3r_con6, 1, DataType:: S8));*/
		out_block3r_act4.allocator()->init(TensorInfo(out_shape_block3r_con6, 1, DataType:: F32));
  		 /*conv-batch-act		*/
		NPYLoader npy33;npy33.open(data_path+"1.layer3.2.conv2.weight.npy");npy33.init_tensor2(weights_block3r_con7,DataType:: S8);
		NPYLoader npyb33;npyb33.open(data_path+"1.layer3.2.conv2.bias.npy");npyb33.init_tensor2(bias_block3r_con7,DataType:: S8);
		const TensorShape out_shape_block3r_con7(14, 14, 256);
		out_block3r_con7.allocator()->init(TensorInfo(out_shape_block3r_con7, 1, DataType:: S8));
		/* NPYLoader npy33_1;npy33_1.open(data_path+"layer3.2.bn2.running_mean.npy");npy33_1.init_tensor2(weights_mean_block3r_batch7,DataType:: S8);*/
		/* NPYLoader npy33_2;npy33_2.open(data_path+"layer3.2.bn2.running_var.npy");npy33_2.init_tensor2(weights_variance_block3r_batch7,DataType:: S8);*/
		/* NPYLoader npy33_3;npy33_3.open(data_path+"layer3.2.bn2.weight.npy");npy33_3.init_tensor2(weights_gamma_block3r_batch7,DataType:: S8);*/
		/* NPYLoader npy33_4;npy33_4.open(data_path+"layer3.2.bn2.bias.npy");npy33_4.init_tensor2(weights_beta_block3r_batch7,DataType:: S8);*/
		/* out_block3r_batch7.allocator()->init(TensorInfo(out_shape_block3r_con7, 1, DataType:: S8));*/
		out_block3r_act5.allocator()->init(TensorInfo(out_shape_block3r_con7, 1, DataType:: F32));
   		/*conv-batch		*/
		NPYLoader npy34;npy34.open(data_path+"1.layer3.2.conv3.weight.npy");npy34.init_tensor2(weights_block3r_con8,DataType:: S8);
		NPYLoader npyb34;npyb34.open(data_path+"1.layer3.2.conv3.bias.npy");npyb34.init_tensor2(bias_block3r_con8,DataType:: S8);
		const TensorShape out_shape_block3r_con8(14, 14, 1024);
		out_block3r_con8.allocator()->init(TensorInfo(out_shape_block3r_con8, 1, DataType:: S8));
		/* NPYLoader npy34_1;npy34_1.open(data_path+"layer3.2.bn3.running_mean.npy");npy34_1.init_tensor2(weights_mean_block3r_batch8,DataType:: S8);*/
		/* NPYLoader npy34_2;npy34_2.open(data_path+"layer3.2.bn3.running_var.npy");npy34_2.init_tensor2(weights_variance_block3r_batch8,DataType:: S8);*/
		/* NPYLoader npy34_3;npy34_3.open(data_path+"layer3.2.bn3.weight.npy");npy34_3.init_tensor2(weights_gamma_block3r_batch8,DataType:: S8);*/
		/* NPYLoader npy34_4;npy34_4.open(data_path+"layer3.2.bn3.bias.npy");npy34_4.init_tensor2(weights_beta_block3r_batch8,DataType:: S8);*/
		/* out_block3r_batch8.allocator()->init(TensorInfo(out_shape_block3r_con8, 1, DataType:: S8));*/
   		/*add-act		*/
		TensorShape out_shape_block3_2 = out_shape_block3r_con8;
		out_block3_add2.allocator()->init(TensorInfo(out_shape_block3_2, 1, DataType:: S8));
		out_block3_act2.allocator()->init(TensorInfo(out_shape_block3_2, 1, DataType:: F32));
  		 /*conv-batch-act*/
		NPYLoader npy35;npy35.open(data_path+"1.layer3.3.conv1.weight.npy");npy35.init_tensor2(weights_block3r_con9,DataType:: S8);
		NPYLoader npyb35;npyb35.open(data_path+"1.layer3.3.conv1.bias.npy");npyb35.init_tensor2(bias_block3r_con9,DataType:: S8);
		const TensorShape out_shape_block3r_con9(14, 14, 256);
		out_block3r_con9.allocator()->init(TensorInfo(out_shape_block3r_con9, 1, DataType:: S8));
		/* NPYLoader npy35_1;npy35_1.open(data_path+"layer3.3.bn1.running_mean.npy");npy35_1.init_tensor2(weights_mean_block3r_batch9,DataType:: S8);*/
		/* NPYLoader npy35_2;npy35_2.open(data_path+"layer3.3.bn1.running_var.npy");npy35_2.init_tensor2(weights_variance_block3r_batch9,DataType:: S8);*/
		/* NPYLoader npy35_3;npy35_3.open(data_path+"layer3.3.bn1.weight.npy");npy35_3.init_tensor2(weights_gamma_block3r_batch9,DataType:: S8);*/
		/* NPYLoader npy35_4;npy35_4.open(data_path+"layer3.3.bn1.bias.npy");npy35_4.init_tensor2(weights_beta_block3r_batch9,DataType:: S8);*/
		/* out_block3r_batch9.allocator()->init(TensorInfo(out_shape_block3r_con9, 1, DataType:: S8));*/
		out_block3r_act6.allocator()->init(TensorInfo(out_shape_block3r_con9, 1, DataType:: F32));
  		 /*conv-batch-act		*/
		NPYLoader npy36;npy36.open(data_path+"1.layer3.3.conv2.weight.npy");npy36.init_tensor2(weights_block3r_con10,DataType:: S8);
		NPYLoader npyb36;npyb36.open(data_path+"1.layer3.3.conv2.bias.npy");npyb36.init_tensor2(bias_block3r_con10,DataType:: S8);
		const TensorShape out_shape_block3r_con10(14, 14, 256);
		out_block3r_con10.allocator()->init(TensorInfo(out_shape_block3r_con10, 1, DataType:: S8));
		/* NPYLoader npy36_1;npy36_1.open(data_path+"layer3.3.bn2.running_mean.npy");npy36_1.init_tensor2(weights_mean_block3r_batch10,DataType:: S8);*/
		/* NPYLoader npy36_2;npy36_2.open(data_path+"layer3.3.bn2.running_var.npy");npy36_2.init_tensor2(weights_variance_block3r_batch10,DataType:: S8);*/
		/* NPYLoader npy36_3;npy36_3.open(data_path+"layer3.3.bn2.weight.npy");npy36_3.init_tensor2(weights_gamma_block3r_batch10,DataType:: S8);*/
		/* NPYLoader npy36_4;npy36_4.open(data_path+"layer3.3.bn2.bias.npy");npy36_4.init_tensor2(weights_beta_block3r_batch10,DataType:: S8);*/
		/* out_block3r_batch10.allocator()->init(TensorInfo(out_shape_block3r_con10, 1, DataType:: S8));*/
		out_block3r_act7.allocator()->init(TensorInfo(out_shape_block3r_con10, 1, DataType:: F32));
   		/*conv-batch		*/
		NPYLoader npy37;npy37.open(data_path+"1.layer3.3.conv3.weight.npy");npy37.init_tensor2(weights_block3r_con11,DataType:: S8);
		NPYLoader npyb37;npyb37.open(data_path+"1.layer3.3.conv3.bias.npy");npyb37.init_tensor2(bias_block3r_con11,DataType:: S8);
		const TensorShape out_shape_block3r_con11(14, 14, 1024);
		out_block3r_con11.allocator()->init(TensorInfo(out_shape_block3r_con11, 1, DataType:: S8));
		/* NPYLoader npy37_1;npy37_1.open(data_path+"layer3.3.bn3.running_mean.npy");npy37_1.init_tensor2(weights_mean_block3r_batch11,DataType:: S8);*/
		/* NPYLoader npy37_2;npy37_2.open(data_path+"layer3.3.bn3.running_var.npy");npy37_2.init_tensor2(weights_variance_block3r_batch11,DataType:: S8);*/
		/* NPYLoader npy37_3;npy37_3.open(data_path+"layer3.3.bn3.weight.npy");npy37_3.init_tensor2(weights_gamma_block3r_batch11,DataType:: S8);*/
		/* NPYLoader npy37_4;npy37_4.open(data_path+"layer3.3.bn3.bias.npy");npy37_4.init_tensor2(weights_beta_block3r_batch11,DataType:: S8);*/
		/*out_block3r_batch11.allocator()->init(TensorInfo(out_shape_block3r_con11, 1, DataType:: S8));*/
   		/*add-act		*/
		TensorShape out_shape_block3_3 = out_shape_block3r_con11;
		out_block3_add3.allocator()->init(TensorInfo(out_shape_block3_3, 1, DataType:: S8));
		out_block3_act3.allocator()->init(TensorInfo(out_shape_block3_3, 1, DataType:: F32));
   		/*conv-batch-act*/
		NPYLoader npy38;npy38.open(data_path+"1.layer3.4.conv1.weight.npy");npy38.init_tensor2(weights_block3r_con12,DataType:: S8);
		NPYLoader npyb38;npyb38.open(data_path+"1.layer3.4.conv1.bias.npy");npyb38.init_tensor2(bias_block3r_con12,DataType:: S8);
		const TensorShape out_shape_block3r_con12(14, 14, 256);
		out_block3r_con12.allocator()->init(TensorInfo(out_shape_block3r_con12, 1, DataType:: S8));
		/* NPYLoader npy38_1;npy38_1.open(data_path+"layer3.4.bn1.running_mean.npy");npy38_1.init_tensor2(weights_mean_block3r_batch12,DataType:: S8);*/
		/* NPYLoader npy38_2;npy38_2.open(data_path+"layer3.4.bn1.running_var.npy");npy38_2.init_tensor2(weights_variance_block3r_batch12,DataType:: S8);*/
		/* NPYLoader npy38_3;npy38_3.open(data_path+"layer3.4.bn1.weight.npy");npy38_3.init_tensor2(weights_gamma_block3r_batch12,DataType:: S8);*/
		/* NPYLoader npy38_4;npy38_4.open(data_path+"layer3.4.bn1.bias.npy");npy38_4.init_tensor2(weights_beta_block3r_batch12,DataType:: S8);*/
		/* out_block3r_batch12.allocator()->init(TensorInfo(out_shape_block3r_con12, 1, DataType:: S8));*/
		out_block3r_act8.allocator()->init(TensorInfo(out_shape_block3r_con12, 1, DataType:: F32));
   		/*conv-batch-act		*/
		NPYLoader npy39;npy39.open(data_path+"1.layer3.4.conv2.weight.npy");npy39.init_tensor2(weights_block3r_con13,DataType:: S8);
		NPYLoader npyb39;npyb39.open(data_path+"1.layer3.4.conv2.bias.npy");npyb39.init_tensor2(bias_block3r_con13,DataType:: S8);
		const TensorShape out_shape_block3r_con13(14, 14, 256);
		out_block3r_con13.allocator()->init(TensorInfo(out_shape_block3r_con13, 1, DataType:: S8));
		/* NPYLoader npy39_1;npy39_1.open(data_path+"layer3.4.bn2.running_mean.npy");npy39_1.init_tensor2(weights_mean_block3r_batch13,DataType:: S8);*/
		/* NPYLoader npy39_2;npy39_2.open(data_path+"layer3.4.bn2.running_var.npy");npy39_2.init_tensor2(weights_variance_block3r_batch13,DataType:: S8);*/
		/* NPYLoader npy39_3;npy39_3.open(data_path+"layer3.4.bn2.weight.npy");npy39_3.init_tensor2(weights_gamma_block3r_batch13,DataType:: S8);*/
		/* NPYLoader npy39_4;npy39_4.open(data_path+"layer3.4.bn2.bias.npy");npy39_4.init_tensor2(weights_beta_block3r_batch13,DataType:: S8);*/
		/* out_block3r_batch13.allocator()->init(TensorInfo(out_shape_block3r_con13, 1, DataType:: S8));*/
		out_block3r_act9.allocator()->init(TensorInfo(out_shape_block3r_con13, 1, DataType:: F32));
   		/*conv-batch		*/
		NPYLoader npy40;npy40.open(data_path+"1.layer3.4.conv3.weight.npy");npy40.init_tensor2(weights_block3r_con14,DataType:: S8);
		NPYLoader npyb40;npyb40.open(data_path+"1.layer3.4.conv3.bias.npy");npyb40.init_tensor2(bias_block3r_con14,DataType:: S8);
		const TensorShape out_shape_block3r_con14(14, 14, 1024);
		out_block3r_con14.allocator()->init(TensorInfo(out_shape_block3r_con14, 1, DataType:: S8));
		/* NPYLoader npy40_1;npy40_1.open(data_path+"layer3.4.bn3.running_mean.npy");npy40_1.init_tensor2(weights_mean_block3r_batch14,DataType:: S8);*/
		/* NPYLoader npy40_2;npy40_2.open(data_path+"layer3.4.bn3.running_var.npy");npy40_2.init_tensor2(weights_variance_block3r_batch14,DataType:: S8);*/
		/* NPYLoader npy40_3;npy40_3.open(data_path+"layer3.4.bn3.weight.npy");npy40_3.init_tensor2(weights_gamma_block3r_batch14,DataType:: S8);*/
		/* NPYLoader npy40_4;npy40_4.open(data_path+"layer3.4.bn3.bias.npy");npy40_4.init_tensor2(weights_beta_block3r_batch14,DataType:: S8);*/
		/* out_block3r_batch14.allocator()->init(TensorInfo(out_shape_block3r_con14, 1, DataType:: S8));*/
  		 /*add-act		*/
		TensorShape out_shape_block3_4 = out_shape_block3r_con14;
		out_block3_add4.allocator()->init(TensorInfo(out_shape_block3_4, 1, DataType:: S8));
		out_block3_act4.allocator()->init(TensorInfo(out_shape_block3_4, 1, DataType:: F32));
   		/*conv-batch-act*/
		NPYLoader npy41;npy41.open(data_path+"1.layer3.5.conv1.weight.npy");npy41.init_tensor2(weights_block3r_con15,DataType:: S8);
		NPYLoader npyb41;npyb41.open(data_path+"1.layer3.5.conv1.bias.npy");npyb41.init_tensor2(bias_block3r_con15,DataType:: S8);
		const TensorShape out_shape_block3r_con15(14, 14,256);
		out_block3r_con15.allocator()->init(TensorInfo(out_shape_block3r_con15, 1, DataType:: S8));
		/* NPYLoader npy41_1;npy41_1.open(data_path+"layer3.5.bn1.running_mean.npy");npy41_1.init_tensor2(weights_mean_block3r_batch15,DataType:: S8);*/
		/* NPYLoader npy41_2;npy41_2.open(data_path+"layer3.5.bn1.running_var.npy");npy41_2.init_tensor2(weights_variance_block3r_batch15,DataType:: S8);*/
		/* NPYLoader npy41_3;npy41_3.open(data_path+"layer3.5.bn1.weight.npy");npy41_3.init_tensor2(weights_gamma_block3r_batch15,DataType:: S8);*/
		/* NPYLoader npy41_4;npy41_4.open(data_path+"layer3.5.bn1.bias.npy");npy41_4.init_tensor2(weights_beta_block3r_batch15,DataType:: S8);*/
		/* out_block3r_batch15.allocator()->init(TensorInfo(out_shape_block3r_con15, 1, DataType:: S8));*/
		out_block3r_act10.allocator()->init(TensorInfo(out_shape_block3r_con15, 1, DataType:: F32));
   		/*conv-batch-act		*/
		NPYLoader npy42;npy42.open(data_path+"1.layer3.5.conv2.weight.npy");npy42.init_tensor2(weights_block3r_con16,DataType:: S8);
		NPYLoader npyb42;npyb42.open(data_path+"1.layer3.5.conv2.bias.npy");npyb42.init_tensor2(bias_block3r_con16,DataType:: S8);
		const TensorShape out_shape_block3r_con16(14, 14, 256);
		out_block3r_con16.allocator()->init(TensorInfo(out_shape_block3r_con16, 1, DataType:: S8));
		/* NPYLoader npy42_1;npy42_1.open(data_path+"layer3.5.bn2.running_mean.npy");npy42_1.init_tensor2(weights_mean_block3r_batch16,DataType:: S8);*/
		/* NPYLoader npy42_2;npy42_2.open(data_path+"layer3.5.bn2.running_var.npy");npy42_2.init_tensor2(weights_variance_block3r_batch16,DataType:: S8);*/
		/* NPYLoader npy42_3;npy42_3.open(data_path+"layer3.5.bn2.weight.npy");npy42_3.init_tensor2(weights_gamma_block3r_batch16,DataType:: S8);*/
		/* NPYLoader npy42_4;npy42_4.open(data_path+"layer3.5.bn2.bias.npy");npy42_4.init_tensor2(weights_beta_block3r_batch16,DataType:: S8);*/
		/* out_block3r_batch16.allocator()->init(TensorInfo(out_shape_block3r_con16, 1, DataType:: S8));*/
		out_block3r_act11.allocator()->init(TensorInfo(out_shape_block3r_con16, 1, DataType:: F32));
   		/*conv-batch	*/
		NPYLoader npy43;npy43.open(data_path+"1.layer3.5.conv3.weight.npy");npy43.init_tensor2(weights_block3r_con17,DataType:: S8);
		NPYLoader npyb43;npyb43.open(data_path+"1.layer3.5.conv3.bias.npy");npyb43.init_tensor2(bias_block3r_con17,DataType:: S8);
		const TensorShape out_shape_block3r_con17(14, 14, 1024);
		out_block3r_con17.allocator()->init(TensorInfo(out_shape_block3r_con17, 1, DataType:: S8));
		/* NPYLoader npy43_1;npy43_1.open(data_path+"layer3.5.bn3.running_mean.npy");npy43_1.init_tensor2(weights_mean_block3r_batch17,DataType:: S8);*/
		/* NPYLoader npy43_2;npy43_2.open(data_path+"layer3.5.bn3.running_var.npy");npy43_2.init_tensor2(weights_variance_block3r_batch17,DataType:: S8);*/
		/* NPYLoader npy43_3;npy43_3.open(data_path+"layer3.5.bn3.weight.npy");npy43_3.init_tensor2(weights_gamma_block3r_batch17,DataType:: S8);*/
		/* NPYLoader npy43_4;npy43_4.open(data_path+"layer3.5.bn3.bias.npy");npy43_4.init_tensor2(weights_beta_block3r_batch17,DataType:: S8);*/
		/* out_block3r_batch17.allocator()->init(TensorInfo(out_shape_block3r_con17, 1, DataType:: S8));*/
		/*pooling		*/
		/*add-act		*/
		TensorShape out_shape_block3_5 = out_shape_block3r_con17;
		out_block3_add5.allocator()->init(TensorInfo(out_shape_block3_5, 1, DataType:: S8));
		out_block3_act5.allocator()->init(TensorInfo(out_shape_block3_5, 1, DataType:: F32));

		/*block4*/
   		/*conv-batch-act*/
		NPYLoader npy44;npy44.open(data_path+"1.layer4.0.conv1.weight.npy");npy44.init_tensor2(weights_block4r_con0,DataType:: S8);
		NPYLoader npyb44;npyb44.open(data_path+"1.layer4.0.conv1.bias.npy");npyb44.init_tensor2(bias_block4r_con0,DataType:: S8);
		const TensorShape out_shape_block4r_con0(7, 7, 512);
		out_block4r_con0.allocator()->init(TensorInfo(out_shape_block4r_con0, 1, DataType:: S8));
		/* NPYLoader npy44_1;npy44_1.open(data_path+"layer4.0.bn1.running_mean.npy");npy44_1.init_tensor2(weights_mean_block4r_batch0,DataType:: S8);*/
		/* NPYLoader npy44_2;npy44_2.open(data_path+"layer4.0.bn1.running_var.npy");npy44_2.init_tensor2(weights_variance_block4r_batch0,DataType:: S8);*/
		/* NPYLoader npy44_3;npy44_3.open(data_path+"layer4.0.bn1.weight.npy");npy44_3.init_tensor2(weights_gamma_block4r_batch0,DataType:: S8);*/
		/* NPYLoader npy44_4;npy44_4.open(data_path+"layer4.0.bn1.bias.npy");npy44_4.init_tensor2(weights_beta_block4r_batch0,DataType:: S8);*/
		/* out_block4r_batch0.allocator()->init(TensorInfo(out_shape_block4r_con0, 1, DataType:: S8));*/
		out_block4r_act0.allocator()->init(TensorInfo(out_shape_block4r_con0, 1, DataType:: F32));
  		 /*conv-batch-act*/
		NPYLoader npy45;npy45.open(data_path+"1.layer4.0.conv2.weight.npy");npy45.init_tensor2(weights_block4r_con1,DataType:: S8);
		NPYLoader npyb45;npyb45.open(data_path+"1.layer4.0.conv2.bias.npy");npyb45.init_tensor2(bias_block4r_con1,DataType:: S8);
		const TensorShape out_shape_block4r_con1(7, 7,512);
		out_block4r_con1.allocator()->init(TensorInfo(out_shape_block4r_con1, 1, DataType:: S8));
		/* NPYLoader npy45_1;npy45_1.open(data_path+"layer4.0.bn2.running_mean.npy");npy45_1.init_tensor2(weights_mean_block4r_batch1,DataType:: S8);*/
		/* NPYLoader npy45_2;npy45_2.open(data_path+"layer4.0.bn2.running_var.npy");npy45_2.init_tensor2(weights_variance_block4r_batch1,DataType:: S8);*/
		/* NPYLoader npy45_3;npy45_3.open(data_path+"layer4.0.bn2.weight.npy");npy45_3.init_tensor2(weights_gamma_block4r_batch1,DataType:: S8);*/
		/* NPYLoader npy45_4;npy45_4.open(data_path+"layer4.0.bn2.bias.npy");npy45_4.init_tensor2(weights_beta_block4r_batch1,DataType:: S8);*/
		/* out_block4r_batch1.allocator()->init(TensorInfo(out_shape_block4r_con1, 1, DataType:: S8));*/
		out_block4r_act1.allocator()->init(TensorInfo(out_shape_block4r_con1, 1, DataType:: F32));
  		 /*conv-batch*/
		NPYLoader npy46;npy46.open(data_path+"1.layer4.0.conv3.weight.npy");npy46.init_tensor2(weights_block4r_con2,DataType:: S8);
		NPYLoader npyb46;npyb46.open(data_path+"1.layer4.0.conv3.bias.npy");npyb46.init_tensor2(bias_block4r_con2,DataType:: S8);
		const TensorShape out_shape_block4r_con2(7, 7, 2048);
		out_block4r_con2.allocator()->init(TensorInfo(out_shape_block4r_con2, 1, DataType:: S8));
		/* NPYLoader npy46_1;npy46_1.open(data_path+"layer4.0.bn3.running_mean.npy");npy46_1.init_tensor2(weights_mean_block4r_batch2,DataType:: S8);*/
		/* NPYLoader npy46_2;npy46_2.open(data_path+"layer4.0.bn3.running_var.npy");npy46_2.init_tensor2(weights_variance_block4r_batch2,DataType:: S8);*/
		/* NPYLoader npy46_3;npy46_3.open(data_path+"layer4.0.bn3.weight.npy");npy46_3.init_tensor2(weights_gamma_block4r_batch2,DataType:: S8);*/
		/* NPYLoader npy46_4;npy46_4.open(data_path+"layer4.0.bn3.bias.npy");npy46_4.init_tensor2(weights_beta_block4r_batch2,DataType:: S8);*/
		/* out_block4r_batch2.allocator()->init(TensorInfo(out_shape_block4r_con2, 1, DataType:: S8));*/
  		/*conv-batch*/
		NPYLoader npy47;npy47.open(data_path+"1.layer4.0.downsample.0.weight.npy");npy47.init_tensor2(weights_block4l_con0,DataType:: S8);
		NPYLoader npyb47;npyb47.open(data_path+"1.layer4.0.downsample.0.bias.npy");npyb47.init_tensor2(bias_block4l_con0,DataType:: S8);
		const TensorShape out_shape_block4l_con0(7, 7, 2048);
		out_block4l_con0.allocator()->init(TensorInfo(out_shape_block4l_con0, 1, DataType:: S8));
		/* NPYLoader npy47_1;npy47_1.open(data_path+"layer4.0.downsample.1.running_mean.npy");npy47_1.init_tensor2(weights_mean_block4l_batch0,DataType:: S8);*/
		/* NPYLoader npy47_2;npy47_2.open(data_path+"layer4.0.downsample.1.running_var.npy");npy47_2.init_tensor2(weights_variance_block4l_batch0,DataType:: S8);*/
		/* NPYLoader npy47_3;npy47_3.open(data_path+"layer4.0.downsample.1.weight.npy");npy47_3.init_tensor2(weights_gamma_block4l_batch0,DataType:: S8);*/
		/* NPYLoader npy47_4;npy47_4.open(data_path+"layer4.0.downsample.1.bias.npy");npy47_4.init_tensor2(weights_beta_block4l_batch0,DataType:: S8);*/
		/* out_block4l_batch0.allocator()->init(TensorInfo(out_shape_block4l_con0, 1, DataType:: S8));*/
   		/*add-act*/
		TensorShape out_shape_block4_0 = out_shape_block4r_con2;
		out_block4_add0.allocator()->init(TensorInfo(out_shape_block4_0, 1, DataType:: S8));
		out_block4_act0.allocator()->init(TensorInfo(out_shape_block4_0, 1, DataType:: F32));
   		/*conv-batch-act*/
		NPYLoader npy48;npy48.open(data_path+"1.layer4.1.conv1.weight.npy");npy48.init_tensor2(weights_block4r_con3,DataType:: S8);
		NPYLoader npyb48;npyb48.open(data_path+"1.layer4.1.conv1.bias.npy");npyb48.init_tensor2(bias_block4r_con3,DataType:: S8);
		const TensorShape out_shape_block4r_con3(7, 7, 512);
		out_block4r_con3.allocator()->init(TensorInfo(out_shape_block4r_con3, 1, DataType:: S8));
		/* NPYLoader npy48_1;npy48_1.open(data_path+"layer4.1.bn1.running_mean.npy");npy48_1.init_tensor2(weights_mean_block4r_batch3,DataType:: S8);*/
		/* NPYLoader npy48_2;npy48_2.open(data_path+"layer4.1.bn1.running_var.npy");npy48_2.init_tensor2(weights_variance_block4r_batch3,DataType:: S8);*/
		/* NPYLoader npy48_3;npy48_3.open(data_path+"layer4.1.bn1.weight.npy");npy48_3.init_tensor2(weights_gamma_block4r_batch3,DataType:: S8);*/
		/* NPYLoader npy48_4;npy48_4.open(data_path+"layer4.1.bn1.bias.npy");npy48_4.init_tensor2(weights_beta_block4r_batch3,DataType:: S8);*/
		/* out_block4r_batch3.allocator()->init(TensorInfo(out_shape_block4r_con3, 1, DataType:: S8));*/
		out_block4r_act2.allocator()->init(TensorInfo(out_shape_block4r_con3, 1, DataType:: F32));
   		/*conv-batch-act		*/
		NPYLoader npy49;npy49.open(data_path+"1.layer4.1.conv2.weight.npy");npy49.init_tensor2(weights_block4r_con4,DataType:: S8);
		NPYLoader npyb49;npyb49.open(data_path+"1.layer4.1.conv2.bias.npy");npyb49.init_tensor2(bias_block4r_con4,DataType:: S8);
		const TensorShape out_shape_block4r_con4(7, 7, 512);
		out_block4r_con4.allocator()->init(TensorInfo(out_shape_block4r_con4, 1, DataType:: S8));
		/* NPYLoader npy49_1;npy49_1.open(data_path+"layer4.1.bn2.running_mean.npy");npy49_1.init_tensor2(weights_mean_block4r_batch4,DataType:: S8);*/
		/* NPYLoader npy49_2;npy49_2.open(data_path+"layer4.1.bn2.running_var.npy");npy49_2.init_tensor2(weights_variance_block4r_batch4,DataType:: S8);*/
		/* NPYLoader npy49_3;npy49_3.open(data_path+"layer4.1.bn2.weight.npy");npy49_3.init_tensor2(weights_gamma_block4r_batch4,DataType:: S8);*/
		/* NPYLoader npy49_4;npy49_4.open(data_path+"layer4.1.bn2.bias.npy");npy49_4.init_tensor2(weights_beta_block4r_batch4,DataType:: S8);*/
		/* out_block4r_batch4.allocator()->init(TensorInfo(out_shape_block4r_con4, 1, DataType:: S8));*/
		out_block4r_act3.allocator()->init(TensorInfo(out_shape_block4r_con4, 1, DataType:: F32));
   		/*conv-batch		*/
		NPYLoader npy50;npy50.open(data_path+"1.layer4.1.conv3.weight.npy");npy50.init_tensor2(weights_block4r_con5,DataType:: S8);
		NPYLoader npyb50;npyb50.open(data_path+"1.layer4.1.conv3.bias.npy");npyb50.init_tensor2(bias_block4r_con5,DataType:: S8);
		const TensorShape out_shape_block4r_con5(7, 7, 2048);
		out_block4r_con5.allocator()->init(TensorInfo(out_shape_block4r_con5, 1, DataType:: S8));
		/* NPYLoader npy50_1;npy50_1.open(data_path+"layer4.1.bn3.running_mean.npy");npy50_1.init_tensor2(weights_mean_block4r_batch5,DataType:: S8);*/
		/* NPYLoader npy50_2;npy50_2.open(data_path+"layer4.1.bn3.running_var.npy");npy50_2.init_tensor2(weights_variance_block4r_batch5,DataType:: S8);*/
		/* NPYLoader npy50_3;npy50_3.open(data_path+"layer4.1.bn3.weight.npy");npy50_3.init_tensor2(weights_gamma_block4r_batch5,DataType:: S8);*/
		/* NPYLoader npy50_4;npy50_4.open(data_path+"layer4.1.bn3.bias.npy");npy50_4.init_tensor2(weights_beta_block4r_batch5,DataType:: S8);*/
		/* out_block4r_batch5.allocator()->init(TensorInfo(out_shape_block4r_con5, 1, DataType:: S8));*/
		/*add-act*/
		TensorShape out_shape_block4_1 = out_shape_block4r_con5;
		out_block4_add1.allocator()->init(TensorInfo(out_shape_block4_1, 1, DataType:: S8));
		out_block4_act1.allocator()->init(TensorInfo(out_shape_block4_1, 1, DataType:: F32));
  		 /*conv-batch-act*/
		NPYLoader npy51;npy51.open(data_path+"1.layer4.2.conv1.weight.npy");npy51.init_tensor2(weights_block4r_con6,DataType:: S8);
		NPYLoader npyb51;npyb51.open(data_path+"1.layer4.2.conv1.bias.npy");npyb51.init_tensor2(bias_block4r_con6,DataType:: S8);
		const TensorShape out_shape_block4r_con6(7, 7, 512);
		out_block4r_con6.allocator()->init(TensorInfo(out_shape_block4r_con6, 1, DataType:: S8));
		/* NPYLoader npy51_1;npy51_1.open(data_path+"layer4.2.bn1.running_mean.npy");npy51_1.init_tensor2(weights_mean_block4r_batch6,DataType:: S8);*/
		/* NPYLoader npy51_2;npy51_2.open(data_path+"layer4.2.bn1.running_var.npy");npy51_2.init_tensor2(weights_variance_block4r_batch6,DataType:: S8);*/
		/* NPYLoader npy51_3;npy51_3.open(data_path+"layer4.2.bn1.weight.npy");npy51_3.init_tensor2(weights_gamma_block4r_batch6,DataType:: S8);*/
		/* NPYLoader npy51_4;npy51_4.open(data_path+"layer4.2.bn1.bias.npy");npy51_4.init_tensor2(weights_beta_block4r_batch6,DataType:: S8);*/
		/* out_block4r_batch6.allocator()->init(TensorInfo(out_shape_block4r_con6, 1, DataType:: S8));*/
		out_block4r_act4.allocator()->init(TensorInfo(out_shape_block4r_con6, 1, DataType:: F32));
   		/*conv-batch-act	*/
		NPYLoader npy52;npy52.open(data_path+"1.layer4.2.conv2.weight.npy");npy52.init_tensor2(weights_block4r_con7,DataType:: S8);
		NPYLoader npyb52;npyb52.open(data_path+"1.layer4.2.conv2.bias.npy");npyb52.init_tensor2(bias_block4r_con7,DataType:: S8);
		const TensorShape out_shape_block4r_con7(7, 7, 512);
		out_block4r_con7.allocator()->init(TensorInfo(out_shape_block4r_con7, 1, DataType:: S8));
		/* NPYLoader npy52_1;npy52_1.open(data_path+"layer4.2.bn2.running_mean.npy");npy52_1.init_tensor2(weights_mean_block4r_batch7,DataType:: S8);*/
		/* NPYLoader npy52_2;npy52_2.open(data_path+"layer4.2.bn2.running_var.npy");npy52_2.init_tensor2(weights_variance_block4r_batch7,DataType:: S8);*/
		/* NPYLoader npy52_3;npy52_3.open(data_path+"layer4.2.bn2.weight.npy");npy52_3.init_tensor2(weights_gamma_block4r_batch7,DataType:: S8);*/
		/* NPYLoader npy52_4;npy52_4.open(data_path+"layer4.2.bn2.bias.npy");npy52_4.init_tensor2(weights_beta_block4r_batch7,DataType:: S8);*/
		/* out_block4r_batch7.allocator()->init(TensorInfo(out_shape_block4r_con7, 1, DataType:: S8));*/
		out_block4r_act5.allocator()->init(TensorInfo(out_shape_block4r_con7, 1, DataType:: F32));
   		/*conv-batch		*/
		NPYLoader npy53;npy53.open(data_path+"1.layer4.2.conv3.weight.npy");npy53.init_tensor2(weights_block4r_con8,DataType:: S8);
		NPYLoader npyb53;npyb53.open(data_path+"1.layer4.2.conv3.bias.npy");npyb53.init_tensor2(bias_block4r_con8,DataType:: S8);
		const TensorShape out_shape_block4r_con8(7, 7, 2048);
		out_block4r_con8.allocator()->init(TensorInfo(out_shape_block4r_con8, 1, DataType:: S8));
		/* NPYLoader npy53_1;npy53_1.open(data_path+"layer4.2.bn3.running_mean.npy");npy53_1.init_tensor2(weights_mean_block4r_batch8,DataType:: S8);*/
		/* NPYLoader npy53_2;npy53_2.open(data_path+"layer4.2.bn3.running_var.npy");npy53_2.init_tensor2(weights_variance_block4r_batch8,DataType:: S8);*/
		/* NPYLoader npy53_3;npy53_3.open(data_path+"layer4.2.bn3.weight.npy");npy53_3.init_tensor2(weights_gamma_block4r_batch8,DataType:: S8);*/
		/* NPYLoader npy53_4;npy53_4.open(data_path+"layer4.2.bn3.bias.npy");npy53_4.init_tensor2(weights_beta_block4r_batch8,DataType:: S8);*/
		/* out_block4r_batch8.allocator()->init(TensorInfo(out_shape_block4r_con8, 1, DataType:: S8));*/
		
		/*add-act	*/
		TensorShape out_shape_block4_2 = out_shape_block4r_con8;
		out_block4_add2.allocator()->init(TensorInfo(out_shape_block4_2, 1, DataType:: S8));
		out_block4_act2.allocator()->init(TensorInfo(out_shape_block4_2, 1, DataType:: F32));
		/* block end  */
       
		/*last pooling-conv-flatten-softmax*/
		TensorShape out_shape_pool1 = out_shape_block4_2;
		out_shape_pool1.set(0, out_shape_pool1.x() / 7);
		out_shape_pool1.set(1, out_shape_pool1.y() / 7);
		out_pool1.allocator()->init(TensorInfo(out_shape_pool1, 1, DataType:: F32));      
		NPYLoader npy100;npy100.open(data_path+"input_fc.npy");npy100.init_tensor2(input_fc,DataType:: S8);
		NPYLoader npy54;npy54.open(data_path+"1.fc.weight.reshape.npy");npy54.init_tensor2(weights_con1,DataType:: S8);
		NPYLoader npyb54;npyb54.open(data_path+"1.fc.bias.npy");npyb54.init_tensor2(bias_con1,DataType:: S8);
		const TensorShape out_shape_con1(1, 1, 1000);
		out_con1.allocator()->init(TensorInfo(out_shape_con1, 1, DataType:: S8));
		const TensorShape out_shape_flatten(out_shape_con1.x()*out_shape_con1.y()*out_shape_con1.z(),0);                     
		out_flatten.allocator()->init(TensorInfo(out_shape_flatten, 1, DataType:: F32));
		const TensorShape out_shape_softmax(out_shape_flatten.x());
		out_softmax.allocator()->init(TensorInfo(out_shape_softmax, 1, DataType:: F32));
		/*last end*/

		/*configure start*/
		/*first start*/
		con0.configure(&src, &weights_con0,&bias_con0, &out_con0, PadStrideInfo(2, 2, 3, 3),precision[0],index[0]);
		lconv0sf.configure(&out_con0,&conv0sf);
		/*batch0.configure(&out_con0, &out_batch0, &weights_mean_batch0, &weights_variance_batch0, &weights_beta_batch0, &weights_gamma_batch0, 0.0000100099996416f);*/
		act0.configure(&conv0sf, &out_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		pool0.configure(&out_act0, &out_pool0, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 1, 0, 1, DimensionRoundingType::FLOOR)));
		lpool0fs.configure(&out_pool0,&pool0fs);
		/*first end*/
		/*block start*/
		/* block1*/
		block1r_con0.configure(&pool0fs, &weights_block1r_con0, &bias_block1r_con0, &out_block1r_con0, PadStrideInfo(1, 1, 0, 0),precision[1],index[1]);
		lb1rconv0sf.configure(&out_block1r_con0,&b1rconv0sf);
		block1r_act0.configure(&b1rconv0sf, &out_block1r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb1ract0fs.configure(&out_block1r_act0,&b1ract0fs);
		block1r_con1.configure(&b1ract0fs, &weights_block1r_con1, &bias_block1r_con1, &out_block1r_con1, PadStrideInfo(1, 1, 1, 1),precision[2],index[2]);
		lb1rconv1sf.configure(&out_block1r_con1,&b1rconv1sf);
		block1r_act1.configure(&b1rconv1sf, &out_block1r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb1ract1fs.configure(&out_block1r_act1,&b1ract1fs);
		block1r_con2.configure(&b1ract1fs, &weights_block1r_con2, &bias_block1r_con2, &out_block1r_con2, PadStrideInfo(1, 1, 0, 0),precision[3],index[3]);
		block1l_con0.configure(&pool0fs, &weights_block1l_con0, &bias_block1l_con0, &out_block1l_con0, PadStrideInfo(1, 1, 0, 0),precision[4],index[4]);
		block1_add0.configure(&out_block1r_con2, &out_block1l_con0, &out_block1_add0,fp[0]);
	
	
	out_block1_add02.allocator()->init(TensorInfo(out_shape_block1_0, 1, DataType:: F32));
	/*testsf1.configure(&out_block1r_con2,&test1);
	testsf2.configure(&out_block1l_con0,&test2);
	block1_add02.configure(&test1, &test2, &out_block1_add02,fp[0]);
	testfs.configure(&out_block1_add02,&test);*/


		lb1add0sf.configure(&out_block1_add0,&b1add0sf);
		block1_act0.configure(&b1add0sf, &out_block1_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
	 
		lb1act0fs.configure(&out_block1_act0, &b1act0fs);
		block1r_con3.configure(&b1act0fs, &weights_block1r_con3, &bias_block1r_con3, &out_block1r_con3, PadStrideInfo(1, 1, 0, 0),precision[5],index[5]);
		lb1rconv3sf.configure(&out_block1r_con3,&b1rconv3sf);
		block1r_act2.configure(&b1rconv3sf, &out_block1r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb1ract2fs.configure(&out_block1r_act2,&b1ract2fs);
		block1r_con4.configure(&b1ract2fs, &weights_block1r_con4, &bias_block1r_con4, &out_block1r_con4, PadStrideInfo(1, 1, 1, 1),precision[6],index[6]);
		lb1rconv4sf.configure(&out_block1r_con4,&b1rconv4sf);
		block1r_act3.configure(&b1rconv4sf, &out_block1r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb1ract3fs.configure(&out_block1r_act3,&b1ract3fs);
		block1r_con5.configure(&b1ract3fs, &weights_block1r_con5, &bias_block1r_con5, &out_block1r_con5, PadStrideInfo(1, 1, 0, 0),precision[7],index[7]);
		block1_add1.configure(&out_block1r_con5, &b1act0fs, &out_block1_add1,fp[1]);
		lb1add1sf.configure(&out_block1_add1,&b1add1sf);
		block1_act1.configure(&b1add1sf, &out_block1_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		
		lb1act1fs.configure(&out_block1_act1,&b1act1fs);
		block1r_con6.configure(&b1act1fs, &weights_block1r_con6, &bias_block1r_con6, &out_block1r_con6, PadStrideInfo(1, 1, 0, 0),precision[8],index[8]);
		lb1rconv6sf.configure(&out_block1r_con6,&b1rconv6sf);
		block1r_act4.configure(&b1rconv6sf, &out_block1r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb1ract4fs.configure(&out_block1r_act4,&b1ract4fs);
		block1r_con7.configure(&b1ract4fs, &weights_block1r_con7, &bias_block1r_con7, &out_block1r_con7, PadStrideInfo(1, 1, 1, 1),precision[9],index[9]);
		lb1rconv7sf.configure(&out_block1r_con7,&b1rconv7sf);
		block1r_act5.configure(&b1rconv7sf, &out_block1r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb1ract5fs.configure(&out_block1r_act5,&b1ract5fs);
		block1r_con8.configure(&b1ract5fs, &weights_block1r_con8, &bias_block1r_con8, &out_block1r_con8, PadStrideInfo(1, 1, 0, 0),precision[10],index[10]);
		block1_add2.configure(&out_block1r_con8, &b1act1fs, &out_block1_add2,fp[2]);
		lb1add2sf.configure(&out_block1_add2,&b1add2sf);
		block1_act2.configure(&b1add2sf, &out_block1_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		/*end block1*/
		/*block2*/
		lb1act2fs.configure(&out_block1_act2,&b1act2fs);
		block2r_con0.configure(&b1act2fs, &weights_block2r_con0, &bias_block2r_con0, &out_block2r_con0, PadStrideInfo(2, 2, 0, 0),precision[11],index[11]);
		lb2rconv0sf.configure(&out_block2r_con0,&b2rconv0sf);
		block2r_act0.configure(&b2rconv0sf, &out_block2r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb2ract0fs.configure(&out_block2r_act0,&b2ract0fs);
		block2r_con1.configure(&b2ract0fs, &weights_block2r_con1, &bias_block2r_con1, &out_block2r_con1, PadStrideInfo(1, 1, 1, 1),precision[12],index[12]);
		lb2rconv1sf.configure(&out_block2r_con1,&b2rconv1sf);
		block2r_act1.configure(&b2rconv1sf, &out_block2r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb2ract1fs.configure(&out_block2r_act1, &b2ract1fs);
		block2r_con2.configure(&b2ract1fs, &weights_block2r_con2, &bias_block2r_con2, &out_block2r_con2, PadStrideInfo(1, 1, 0, 0),precision[13],index[13]);
		block2l_con0.configure(&b1act2fs, &weights_block2l_con0, &bias_block2l_con0, &out_block2l_con0, PadStrideInfo(2, 2, 0, 0),precision[14],index[14]);
		block2_add0.configure(&out_block2r_con2, &out_block2l_con0, &out_block2_add0, fp[3]);
		lb2add0sf.configure(&out_block2_add0,&b2add0sf);
		block2_act0.configure(&b2add0sf, &out_block2_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		
		lb2act0fs.configure(&out_block2_act0,&b2act0fs);
		block2r_con3.configure(&b2act0fs, &weights_block2r_con3, &bias_block2r_con3, &out_block2r_con3, PadStrideInfo(1, 1, 0, 0),precision[15],index[15]);
		lb2rconv3sf.configure(&out_block2r_con3,&b2rconv3sf);
		block2r_act2.configure(&b2rconv3sf, &out_block2r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb2ract2fs.configure(&out_block2r_act2,&b2ract2fs);
		block2r_con4.configure(&b2ract2fs, &weights_block2r_con4, &bias_block2r_con4, &out_block2r_con4, PadStrideInfo(1, 1, 1, 1),precision[16],index[16]);
		lb2rconv4sf.configure(&out_block2r_con4,&b2rconv4sf);
		block2r_act3.configure(&b2rconv4sf, &out_block2r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb2ract3fs.configure(&out_block2r_act3,&b2ract3fs);
		block2r_con5.configure(&b2ract3fs, &weights_block2r_con5, &bias_block2r_con5, &out_block2r_con5, PadStrideInfo(1, 1, 0, 0),precision[17],index[17]);
		block2_add1.configure(&out_block2r_con5, &b2act0fs, &out_block2_add1, fp[4]);
		lb2add1sf.configure(&out_block2_add1,&b2add1sf);
		block2_act1.configure(&b2add1sf, &out_block2_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		
		lb2act1fs.configure(&out_block2_act1,&b2act1fs);
		block2r_con6.configure(&b2act1fs, &weights_block2r_con6, &bias_block2r_con6, &out_block2r_con6, PadStrideInfo(1, 1, 0, 0),precision[18],index[18]);
		lb2rconv6sf.configure(&out_block2r_con6,&b2rconv6sf);
		block2r_act4.configure(&b2rconv6sf, &out_block2r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb2ract4fs.configure(&out_block2r_act4,&b2ract4fs);
		block2r_con7.configure(&b2ract4fs, &weights_block2r_con7, &bias_block2r_con7, &out_block2r_con7, PadStrideInfo(1, 1, 1, 1),precision[19],index[19]);
		lb2rconv7sf.configure(&out_block2r_con7,&b2rconv7sf);
		block2r_act5.configure(&b2rconv7sf, &out_block2r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb2ract5fs.configure(&out_block2r_act5,&b2ract5fs);
		block2r_con8.configure(&b2ract5fs, &weights_block2r_con8, &bias_block2r_con8, &out_block2r_con8, PadStrideInfo(1, 1, 0, 0),precision[20],index[20]);
		block2_add2.configure(&out_block2r_con8, &b2act1fs, &out_block2_add2, fp[5]);
		lb2add2sf.configure(&out_block2_add2,&b2add2sf);
		block2_act2.configure(&b2add2sf, &out_block2_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		
		lb2act2fs.configure(&out_block2_act2,&b2act2fs);
		block2r_con9.configure(&b2act2fs, &weights_block2r_con9, &bias_block2r_con9, &out_block2r_con9, PadStrideInfo(1, 1, 0, 0),precision[21],index[21]);
		lb2rconv9sf.configure(&out_block2r_con9,&b2rconv9sf);
		block2r_act6.configure(&b2rconv9sf, &out_block2r_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb2ract6fs.configure(&out_block2r_act6,&b2ract6fs);
		block2r_con10.configure(&b2ract6fs, &weights_block2r_con10, &bias_block2r_con10, &out_block2r_con10, PadStrideInfo(1, 1, 1, 1),precision[22],index[22]);
		lb2rconv10sf.configure(&out_block2r_con10,&b2rconv10sf);
		block2r_act7.configure(&b2rconv10sf, &out_block2r_act7, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb2ract7fs.configure(&out_block2r_act7,&b2ract7fs);
		block2r_con11.configure(&b2ract7fs, &weights_block2r_con11, &bias_block2r_con11, &out_block2r_con11, PadStrideInfo(1, 1, 0, 0),precision[23],index[23]);
		block2_add3.configure(&out_block2r_con11, &b2act2fs, &out_block2_add3, fp[6]);
		lb2add3sf.configure(&out_block2_add3,&b2add3sf);
		block2_act3.configure(&b2add3sf, &out_block2_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		
		/*end block2*/
		/*block3*/
		lb2act3fs.configure(&out_block2_act3,&b2act3fs);
		block3r_con0.configure(&b2act3fs, &weights_block3r_con0, &bias_block3r_con0, &out_block3r_con0, PadStrideInfo(2, 2, 0, 0),precision[24],index[24]);
		lb3rconv0sf.configure(&out_block3r_con0,&b3rconv0sf);
		block3r_act0.configure(&b3rconv0sf, &out_block3r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb3ract0fs.configure(&out_block3r_act0,&b3ract0fs);
		block3r_con1.configure(&b3ract0fs, &weights_block3r_con1,  &bias_block3r_con1, &out_block3r_con1, PadStrideInfo(1, 1, 1, 1),precision[25],index[25]);
		lb3rconv1sf.configure(&out_block3r_con1,&b3rconv1sf);
		block3r_act1.configure(&b3rconv1sf, &out_block3r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb3ract1fs.configure(&out_block3r_act1,&b3ract1fs);
		block3r_con2.configure(&b3ract1fs, &weights_block3r_con2,  &bias_block3r_con2, &out_block3r_con2, PadStrideInfo(1, 1, 0, 0),precision[26],index[26]);
		block3l_con0.configure(&b2act3fs, &weights_block3l_con0,  &bias_block3l_con0, &out_block3l_con0, PadStrideInfo(2, 2, 0, 0),precision[27],index[27]);
		block3_add0.configure(&out_block3r_con2, &out_block3l_con0, &out_block3_add0, fp[7]);
		lb3add0sf.configure(&out_block3_add0,&b3add0sf);
		block3_act0.configure(&b3add0sf, &out_block3_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

		lb3act0fs.configure(&out_block3_act0,&b3act0fs);
		block3r_con3.configure(&b3act0fs, &weights_block3r_con3,  &bias_block3r_con3, &out_block3r_con3, PadStrideInfo(1, 1, 0, 0),precision[28],index[28]);
		lb3rconv3sf.configure(&out_block3r_con3,&b3rconv3sf);
		block3r_act2.configure(&b3rconv3sf, &out_block3r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb3ract2fs.configure(&out_block3r_act2,&b3ract2fs);
		block3r_con4.configure(&b3ract2fs, &weights_block3r_con4,  &bias_block3r_con4, &out_block3r_con4, PadStrideInfo(1, 1, 1, 1),precision[29],index[29]);
		lb3rconv4sf.configure(&out_block3r_con4,&b3rconv4sf);
		block3r_act3.configure(&b3rconv4sf, &out_block3r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb3ract3fs.configure(&out_block3r_act3,&b3ract3fs);
		block3r_con5.configure(&b3ract3fs, &weights_block3r_con5,  &bias_block3r_con5, &out_block3r_con5, PadStrideInfo(1, 1, 0, 0),precision[30],index[30]);
		block3_add1.configure(&out_block3r_con5, &b3act0fs, &out_block3_add1, fp[8]);
		lb3add1sf.configure(&out_block3_add1,&b3add1sf);
		block3_act1.configure(&b3add1sf, &out_block3_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        
		lb3act1fs.configure(&out_block3_act1,&b3act1fs);
		block3r_con6.configure(&b3act1fs, &weights_block3r_con6,  &bias_block3r_con6, &out_block3r_con6, PadStrideInfo(1, 1, 0, 0),precision[31],index[31]);
		lb3rconv6sf.configure(&out_block3r_con6,&b3rconv6sf);
		block3r_act4.configure(&b3rconv6sf, &out_block3r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb3ract4fs.configure(&out_block3r_act4,&b3ract4fs);
		block3r_con7.configure(&b3ract4fs, &weights_block3r_con7,  &bias_block3r_con7, &out_block3r_con7, PadStrideInfo(1, 1, 1, 1),precision[32],index[32]);
		lb3rconv7sf.configure(&out_block3r_con7,&b3rconv7sf);
		block3r_act5.configure(&b3rconv7sf, &out_block3r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb3ract5fs.configure(&out_block3r_act5,&b3ract5fs);
		block3r_con8.configure(&b3ract5fs, &weights_block3r_con8,  &bias_block3r_con8, &out_block3r_con8, PadStrideInfo(1, 1, 0, 0),precision[33],index[33]);
		block3_add2.configure(&out_block3r_con8, &b3act1fs, &out_block3_add2, fp[9]);
		lb3add2sf.configure(&out_block3_add2,&b3add2sf);
		block3_act2.configure(&b3add2sf, &out_block3_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
      
	  	lb3act2fs.configure(&out_block3_act2,&b3act2fs);
		block3r_con9.configure(&b3act2fs, &weights_block3r_con9,  &bias_block3r_con9, &out_block3r_con9, PadStrideInfo(1, 1, 0, 0),precision[34],index[34]);
		lb3rconv9sf.configure(&out_block3r_con9,&b3rconv9sf);
		block3r_act6.configure(&b3rconv9sf, &out_block3r_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb3ract6fs.configure(&out_block3r_act6,&b3ract6fs);
		block3r_con10.configure(&b3ract6fs, &weights_block3r_con10,  &bias_block3r_con10, &out_block3r_con10, PadStrideInfo(1, 1, 1, 1),precision[35],index[35]);
		lb3rconv10sf.configure(&out_block3r_con10,&b3rconv10sf);
		block3r_act7.configure(&b3rconv10sf, &out_block3r_act7, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb3ract7fs.configure(&out_block3r_act7,&b3ract7fs);
		block3r_con11.configure(&b3ract7fs, &weights_block3r_con11,  &bias_block3r_con11, &out_block3r_con11, PadStrideInfo(1, 1, 0, 0),precision[36],index[36]);
		block3_add3.configure(&out_block3r_con11, &b3act2fs, &out_block3_add3, fp[10]);
		lb3add3sf.configure(&out_block3_add3,&b3add3sf);
		block3_act3.configure(&b3add3sf, &out_block3_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        
		lb3act3fs.configure(&out_block3_act3,&b3act3fs);
		block3r_con12.configure(&b3act3fs, &weights_block3r_con12,  &bias_block3r_con12, &out_block3r_con12, PadStrideInfo(1, 1, 0, 0),precision[37],index[37]);
		lb3rconv12sf.configure(&out_block3r_con12,&b3rconv12sf);
		block3r_act8.configure(&b3rconv12sf, &out_block3r_act8, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb3ract8fs.configure(&out_block3r_act8,&b3ract8fs);
		block3r_con13.configure(&b3ract8fs, &weights_block3r_con13,  &bias_block3r_con13, &out_block3r_con13, PadStrideInfo(1, 1, 1, 1),precision[38],index[38]);
		lb3rconv13sf.configure(&out_block3r_con13,&b3rconv13sf);
		block3r_act9.configure(&b3rconv13sf, &out_block3r_act9, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb3ract9fs.configure(&out_block3r_act9,&b3ract9fs);
		block3r_con14.configure(&b3ract9fs, &weights_block3r_con14,  &bias_block3r_con14, &out_block3r_con14, PadStrideInfo(1, 1, 0, 0),precision[39],index[39]);
		block3_add4.configure(&out_block3r_con14, &b3act3fs, &out_block3_add4, fp[11]);
		lb3add4sf.configure(&out_block3_add4,&b3add4sf);
		block3_act4.configure(&b3add4sf, &out_block3_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
      
	  	lb3act4fs.configure(&out_block3_act4,&b3act4fs);
		block3r_con15.configure(&b3act4fs, &weights_block3r_con15,  &bias_block3r_con15, &out_block3r_con15, PadStrideInfo(1, 1, 0, 0),precision[40],index[40]);
		lb3rconv15sf.configure(&out_block3r_con15,&b3rconv15sf);
		block3r_act10.configure(&b3rconv15sf, &out_block3r_act10, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb3ract10fs.configure(&out_block3r_act10,&b3ract10fs);
		block3r_con16.configure(&b3ract10fs, &weights_block3r_con16,  &bias_block3r_con16, &out_block3r_con16, PadStrideInfo(1, 1, 1, 1),precision[41],index[41]);
		lb3rconv16sf.configure(&out_block3r_con16,&b3rconv16sf);
		block3r_act11.configure(&b3rconv16sf, &out_block3r_act11, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb3ract11fs.configure(&out_block3r_act11,&b3ract11fs);
		block3r_con17.configure(&b3ract11fs, &weights_block3r_con17,  &bias_block3r_con17, &out_block3r_con17, PadStrideInfo(1, 1, 0, 0),precision[42],index[42]);
		block3_add5.configure(&out_block3r_con17, &b3act4fs, &out_block3_add5, fp[12]);
		lb3add5sf.configure(&out_block3_add5,&b3add5sf);
		block3_act5.configure(&b3add5sf, &out_block3_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		
				
		/*end block3*/
		/*block4*/
		lb3act5fs.configure(&out_block3_act5,&b3act5fs);
		block4r_con0.configure(&b3act5fs, &weights_block4r_con0, &bias_block4r_con0, &out_block4r_con0, PadStrideInfo(2, 2, 0, 0),precision[43],index[43]);
		lb4rconv0sf.configure(&out_block4r_con0,&b4rconv0sf);
		block4r_act0.configure(&b4rconv0sf, &out_block4r_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb4ract0fs.configure(&out_block4r_act0,&b4ract0fs);
		block4r_con1.configure(&b4ract0fs, &weights_block4r_con1, &bias_block4r_con1, &out_block4r_con1, PadStrideInfo(1, 1, 1, 1),precision[44],index[44]);
		lb4rconv1sf.configure(&out_block4r_con1,&b4rconv1sf);
		block4r_act1.configure(&b4rconv1sf, &out_block4r_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb4ract1fs.configure(&out_block4r_act1,&b4ract1fs);
		block4r_con2.configure(&b4ract1fs, &weights_block4r_con2, &bias_block4r_con2, &out_block4r_con2, PadStrideInfo(1, 1, 0, 0),precision[45],index[45]);
		block4l_con0.configure(&b3act5fs, &weights_block4l_con0, &bias_block4l_con0, &out_block4l_con0, PadStrideInfo(2, 2, 0, 0),precision[46],index[46]);
		block4_add0.configure(&out_block4r_con2, &out_block4l_con0, &out_block4_add0, fp[13]);
		lb4add0sf.configure(&out_block4_add0,&b4add0sf);
		block4_act0.configure(&b4add0sf, &out_block4_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
     
	 	lb4act0fs.configure(&out_block4_act0,&b4act0fs);
		block4r_con3.configure(&b4act0fs, &weights_block4r_con3, &bias_block4r_con3, &out_block4r_con3, PadStrideInfo(1, 1, 0, 0),precision[47],index[47]);
		lb4rconv3sf.configure(&out_block4r_con3,&b4rconv3sf);
		block4r_act2.configure(&b4rconv3sf, &out_block4r_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb4ract2fs.configure(&out_block4r_act2,&b4ract2fs);
		block4r_con4.configure(&b4ract2fs, &weights_block4r_con4, &bias_block4r_con4, &out_block4r_con4, PadStrideInfo(1, 1, 1, 1),precision[48],index[48]);
		lb4rconv4sf.configure(&out_block4r_con4,&b4rconv4sf);
		block4r_act3.configure(&b4rconv4sf, &out_block4r_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb4ract3fs.configure(&out_block4r_act3,&b4ract3fs);
		block4r_con5.configure(&b4ract3fs, &weights_block4r_con5, &bias_block4r_con5, &out_block4r_con5, PadStrideInfo(1, 1, 0, 0),precision[49],index[49]);
		block4_add1.configure(&out_block4r_con5, &b4act0fs, &out_block4_add1, fp[14]);
		lb4add1sf.configure(&out_block4_add1,&b4add1sf);
		block4_act1.configure(&b4add1sf, &out_block4_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
      
	  	lb4act1fs.configure(&out_block4_act1,&b4act1fs);
		block4r_con6.configure(&b4act1fs, &weights_block4r_con6, &bias_block4r_con6, &out_block4r_con6, PadStrideInfo(1, 1, 0, 0),precision[50],index[50]);
		lb4rconv6sf.configure(&out_block4r_con6,&b4rconv6sf);
		block4r_act4.configure(&b4rconv6sf, &out_block4r_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb4ract4fs.configure(&out_block4r_act4,&b4ract4fs);
		block4r_con7.configure(&b4ract4fs, &weights_block4r_con7, &bias_block4r_con7, &out_block4r_con7, PadStrideInfo(1, 1, 1, 1),precision[51],index[51]);
		lb4rconv7sf.configure(&out_block4r_con7,&b4rconv7sf);
		block4r_act5.configure(&b4rconv7sf, &out_block4r_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		lb4ract5fs.configure(&out_block4r_act5,&b4ract5fs);
		block4r_con8.configure(&b4ract5fs, &weights_block4r_con8, &bias_block4r_con8, &out_block4r_con8, PadStrideInfo(1, 1, 0, 0),precision[52],index[52]);
		block4_add2.configure(&out_block4r_con8, &b4act1fs, &out_block4_add2,fp[15]);
		lb4add2sf.configure(&out_block4_add2,&b4add2sf);
		block4_act2.configure(&b4add2sf, &out_block4_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
		/*end block4*/
		/*block end*/
		/*last start*/
	    pool1.configure(&out_block4_act2, &out_pool1, PoolingLayerInfo(PoolingType::AVG, 7, PadStrideInfo(1, 1, 0, 0, 0, 0, DimensionRoundingType::FLOOR)));
		lpool1fs.configure(&out_pool1,&pool1fs);
		con1.configure(&input_fc, &weights_con1, &bias_con1, &out_con1, PadStrideInfo(1, 1, 0, 0),precision[53],index[53]);
		lconv1sf.configure(&out_con1,&conv1sf);
		flatten.configure(&conv1sf, &out_flatten);
		softmax.configure(&out_flatten, &out_softmax);
		/*last end*/
		/*configure end*/


		/*allocate start*/
		/*first allocate*/
	    out_con0.allocator()->allocate(); 
	    out_act0.allocator()->allocate(); out_pool0.allocator()->allocate();
		/*first allocate end*/
		/*block allocate*/
		/*block1*/
		out_block1r_con0.allocator()->allocate();  out_block1r_act0.allocator()->allocate();
		out_block1r_con1.allocator()->allocate(); out_block1r_act1.allocator()->allocate();
		out_block1r_con2.allocator()->allocate(); out_block1l_con0.allocator()->allocate(); 
		out_block1_add0.allocator()->allocate(); out_block1_act0.allocator()->allocate();

		out_block1r_con3.allocator()->allocate(); out_block1r_act2.allocator()->allocate();
		out_block1r_con4.allocator()->allocate(); out_block1r_act3.allocator()->allocate();
		out_block1r_con5.allocator()->allocate();
		out_block1_add1.allocator()->allocate(); out_block1_act1.allocator()->allocate();

		out_block1r_con6.allocator()->allocate(); out_block1r_act4.allocator()->allocate();
		out_block1r_con7.allocator()->allocate(); out_block1r_act5.allocator()->allocate();
		out_block1r_con8.allocator()->allocate();  
		out_block1_add2.allocator()->allocate(); out_block1_act2.allocator()->allocate();
		/*end block1*/
		/*block2*/
		out_block2r_con0.allocator()->allocate(); out_block2r_act0.allocator()->allocate();
		out_block2r_con1.allocator()->allocate(); out_block2r_act1.allocator()->allocate();
		out_block2r_con2.allocator()->allocate(); out_block2l_con0.allocator()->allocate(); 
		out_block2_add0.allocator()->allocate(); out_block2_act0.allocator()->allocate();

		out_block2r_con3.allocator()->allocate(); out_block2r_act2.allocator()->allocate();
		out_block2r_con4.allocator()->allocate();  out_block2r_act3.allocator()->allocate();
		out_block2r_con5.allocator()->allocate(); 
		out_block2_add1.allocator()->allocate(); out_block2_act1.allocator()->allocate();

		out_block2r_con6.allocator()->allocate(); out_block2r_act4.allocator()->allocate();
		out_block2r_con7.allocator()->allocate(); out_block2r_act5.allocator()->allocate();
		out_block2r_con8.allocator()->allocate();
		out_block2_add2.allocator()->allocate(); out_block2_act2.allocator()->allocate();

		out_block2r_con9.allocator()->allocate(); out_block2r_act6.allocator()->allocate();
		out_block2r_con10.allocator()->allocate(); out_block2r_act7.allocator()->allocate();
		out_block2r_con11.allocator()->allocate();
		out_block2_add3.allocator()->allocate(); out_block2_act3.allocator()->allocate();
		/*end block2*/
		/*block3*/
		out_block3r_con0.allocator()->allocate(); out_block3r_act0.allocator()->allocate();
		out_block3r_con1.allocator()->allocate();  out_block3r_act1.allocator()->allocate();
		out_block3r_con2.allocator()->allocate(); out_block3l_con0.allocator()->allocate(); 
		out_block3_add0.allocator()->allocate(); out_block3_act0.allocator()->allocate();

		out_block3r_con3.allocator()->allocate();  out_block3r_act2.allocator()->allocate();
		out_block3r_con4.allocator()->allocate();out_block3r_act3.allocator()->allocate();
		out_block3r_con5.allocator()->allocate(); 
		out_block3_add1.allocator()->allocate(); out_block3_act1.allocator()->allocate();

		out_block3r_con6.allocator()->allocate();out_block3r_act4.allocator()->allocate();
		out_block3r_con7.allocator()->allocate();  out_block3r_act5.allocator()->allocate();
		out_block3r_con8.allocator()->allocate(); 
		out_block3_add2.allocator()->allocate(); out_block3_act2.allocator()->allocate();

		out_block3r_con9.allocator()->allocate(); out_block3r_act6.allocator()->allocate();
		out_block3r_con10.allocator()->allocate(); out_block3r_act7.allocator()->allocate();
		out_block3r_con11.allocator()->allocate();
		out_block3_add3.allocator()->allocate(); out_block3_act3.allocator()->allocate();

		out_block3r_con12.allocator()->allocate();  out_block3r_act8.allocator()->allocate();
		out_block3r_con13.allocator()->allocate(); out_block3r_act9.allocator()->allocate();
		out_block3r_con14.allocator()->allocate(); 
		out_block3_add4.allocator()->allocate(); out_block3_act4.allocator()->allocate();

		out_block3r_con15.allocator()->allocate(); ; out_block3r_act10.allocator()->allocate();
		out_block3r_con16.allocator()->allocate(); out_block3r_act11.allocator()->allocate();
		out_block3r_con17.allocator()->allocate();  
		out_block3_add5.allocator()->allocate(); out_block3_act5.allocator()->allocate();

		/*end block3*/
		/*block4*/
		out_block4r_con0.allocator()->allocate();  out_block4r_act0.allocator()->allocate();
		out_block4r_con1.allocator()->allocate(); out_block4r_act1.allocator()->allocate();
		out_block4r_con2.allocator()->allocate(); out_block4l_con0.allocator()->allocate(); 
		out_block4_add0.allocator()->allocate(); out_block4_act0.allocator()->allocate();

		out_block4r_con3.allocator()->allocate(); out_block4r_act2.allocator()->allocate();
		out_block4r_con4.allocator()->allocate();  out_block4r_act3.allocator()->allocate();
		out_block4r_con5.allocator()->allocate(); 
		out_block4_add1.allocator()->allocate(); out_block4_act1.allocator()->allocate();

		out_block4r_con6.allocator()->allocate();  out_block4r_act4.allocator()->allocate();
		out_block4r_con7.allocator()->allocate(); out_block4r_act5.allocator()->allocate();
		out_block4r_con8.allocator()->allocate(); 
		out_block4_add2.allocator()->allocate(); out_block4_act2.allocator()->allocate();
		/*end block4*/
		/*block allocate end*/
		/*last allocate*/
        out_pool1.allocator()->allocate(); out_con1.allocator()->allocate(); out_flatten.allocator()->allocate(); out_softmax.allocator()->allocate();
		/*last allocate end*/
        src.allocator()->allocate(); weights_con0.allocator()->allocate();bias_con0.allocator()->allocate();
		
		weights_block1r_con0.allocator()->allocate(); weights_block1r_con1.allocator()->allocate(); weights_block1r_con2.allocator()->allocate(); 
		weights_block1r_con3.allocator()->allocate(); weights_block1r_con4.allocator()->allocate(); weights_block1r_con5.allocator()->allocate(); 
		weights_block1r_con6.allocator()->allocate(); weights_block1r_con7.allocator()->allocate(); weights_block1r_con8.allocator()->allocate(); 
		weights_block1l_con0.allocator()->allocate();
		
		weights_block2r_con0.allocator()->allocate(); weights_block2r_con1.allocator()->allocate(); weights_block2r_con2.allocator()->allocate();
		weights_block2r_con3.allocator()->allocate(); weights_block2r_con4.allocator()->allocate(); weights_block2r_con5.allocator()->allocate();
		weights_block2r_con6.allocator()->allocate(); weights_block2r_con7.allocator()->allocate(); weights_block2r_con8.allocator()->allocate();
		weights_block2r_con9.allocator()->allocate(); weights_block2r_con10.allocator()->allocate(); weights_block2r_con11.allocator()->allocate();
		weights_block2l_con0.allocator()->allocate(); 
		
		weights_block3r_con0.allocator()->allocate(); weights_block3r_con1.allocator()->allocate(); weights_block3r_con2.allocator()->allocate();
		weights_block3r_con3.allocator()->allocate(); weights_block3r_con4.allocator()->allocate(); weights_block3r_con5.allocator()->allocate();
		weights_block3r_con6.allocator()->allocate(); weights_block3r_con7.allocator()->allocate(); weights_block3r_con8.allocator()->allocate();
		weights_block3r_con9.allocator()->allocate(); weights_block3r_con10.allocator()->allocate(); weights_block3r_con11.allocator()->allocate();
		weights_block3r_con12.allocator()->allocate(); weights_block3r_con13.allocator()->allocate(); weights_block3r_con14.allocator()->allocate();
		weights_block3r_con15.allocator()->allocate(); weights_block3r_con16.allocator()->allocate(); weights_block3r_con17.allocator()->allocate();
		weights_block3l_con0.allocator()->allocate(); 
		
		weights_block4r_con0.allocator()->allocate(); weights_block4r_con1.allocator()->allocate(); weights_block4r_con2.allocator()->allocate();
		weights_block4r_con3.allocator()->allocate(); weights_block4r_con4.allocator()->allocate(); weights_block4r_con5.allocator()->allocate();
		weights_block4r_con6.allocator()->allocate(); weights_block4r_con7.allocator()->allocate(); weights_block4r_con8.allocator()->allocate();
		weights_block4l_con0.allocator()->allocate(); 
		
		bias_block1r_con0.allocator()->allocate(); bias_block1r_con1.allocator()->allocate(); bias_block1r_con2.allocator()->allocate(); 
		bias_block1r_con3.allocator()->allocate(); bias_block1r_con4.allocator()->allocate(); bias_block1r_con5.allocator()->allocate(); 
		bias_block1r_con6.allocator()->allocate(); bias_block1r_con7.allocator()->allocate(); bias_block1r_con8.allocator()->allocate(); 
		bias_block1l_con0.allocator()->allocate();
		
		bias_block2r_con0.allocator()->allocate(); bias_block2r_con1.allocator()->allocate(); bias_block2r_con2.allocator()->allocate();
		bias_block2r_con3.allocator()->allocate(); bias_block2r_con4.allocator()->allocate(); bias_block2r_con5.allocator()->allocate();
		bias_block2r_con6.allocator()->allocate(); bias_block2r_con7.allocator()->allocate(); bias_block2r_con8.allocator()->allocate();
		bias_block2r_con9.allocator()->allocate(); bias_block2r_con10.allocator()->allocate(); bias_block2r_con11.allocator()->allocate();
		bias_block2l_con0.allocator()->allocate(); 
		
		bias_block3r_con0.allocator()->allocate(); bias_block3r_con1.allocator()->allocate(); bias_block3r_con2.allocator()->allocate();
		bias_block3r_con3.allocator()->allocate(); bias_block3r_con4.allocator()->allocate(); bias_block3r_con5.allocator()->allocate();
		bias_block3r_con6.allocator()->allocate(); bias_block3r_con7.allocator()->allocate(); bias_block3r_con8.allocator()->allocate();
		bias_block3r_con9.allocator()->allocate(); bias_block3r_con10.allocator()->allocate(); bias_block3r_con11.allocator()->allocate();
		bias_block3r_con12.allocator()->allocate(); bias_block3r_con13.allocator()->allocate(); bias_block3r_con14.allocator()->allocate();
		bias_block3r_con15.allocator()->allocate(); bias_block3r_con16.allocator()->allocate(); bias_block3r_con17.allocator()->allocate();
		bias_block3l_con0.allocator()->allocate(); 
		
		bias_block4r_con0.allocator()->allocate(); bias_block4r_con1.allocator()->allocate(); bias_block4r_con2.allocator()->allocate();
		bias_block4r_con3.allocator()->allocate(); bias_block4r_con4.allocator()->allocate(); bias_block4r_con5.allocator()->allocate();
		bias_block4r_con6.allocator()->allocate(); bias_block4r_con7.allocator()->allocate(); bias_block4r_con8.allocator()->allocate();
		bias_block4l_con0.allocator()->allocate(); 

		input_fc.allocator()->allocate();
		weights_con1.allocator()->allocate(); bias_con1.allocator()->allocate();

		/*type change tensor allocate*/
		 conv0sf.allocator()->allocate();  pool0fs.allocator()->allocate();

		 b1rconv0sf.allocator()->allocate();  b1ract0fs.allocator()->allocate();  b1rconv1sf.allocator()->allocate();  b1ract1fs.allocator()->allocate(); b1add0sf.allocator()->allocate();
		 b1act0fs.allocator()->allocate();  b1rconv3sf.allocator()->allocate();  b1ract2fs.allocator()->allocate();  b1rconv4sf.allocator()->allocate();  b1ract3fs.allocator()->allocate();  b1add1sf.allocator()->allocate();
		 b1act1fs.allocator()->allocate();  b1rconv6sf.allocator()->allocate();  b1ract4fs.allocator()->allocate();  b1rconv7sf.allocator()->allocate();  b1ract5fs.allocator()->allocate(); b1add2sf.allocator()->allocate();

		 b1act2fs.allocator()->allocate(); b2rconv0sf.allocator()->allocate();  b2ract0fs.allocator()->allocate();  b2rconv1sf.allocator()->allocate();  b2ract1fs.allocator()->allocate();  b2add0sf.allocator()->allocate();
		 b2act0fs.allocator()->allocate();  b2rconv3sf.allocator()->allocate();  b2ract2fs.allocator()->allocate();  b2rconv4sf.allocator()->allocate();  b2ract3fs.allocator()->allocate();  b2add1sf.allocator()->allocate();
		 b2act1fs.allocator()->allocate();  b2rconv6sf.allocator()->allocate();  b2ract4fs.allocator()->allocate();  b2rconv7sf.allocator()->allocate();  b2ract5fs.allocator()->allocate();  b2add2sf.allocator()->allocate();
		 b2act2fs.allocator()->allocate();  b2rconv9sf.allocator()->allocate();  b2ract6fs.allocator()->allocate();  b2rconv10sf.allocator()->allocate();  b2ract7fs.allocator()->allocate(); b2add3sf.allocator()->allocate();

		 b2act3fs.allocator()->allocate(); b3rconv0sf.allocator()->allocate();  b3ract0fs.allocator()->allocate();  b3rconv1sf.allocator()->allocate();  b3ract1fs.allocator()->allocate();  b3add0sf.allocator()->allocate();
		 b3act0fs.allocator()->allocate();  b3rconv3sf.allocator()->allocate();  b3ract2fs.allocator()->allocate();  b3rconv4sf.allocator()->allocate();  b3ract3fs.allocator()->allocate();  b3add1sf.allocator()->allocate();
		 b3act1fs.allocator()->allocate();  b3rconv6sf.allocator()->allocate();  b3ract4fs.allocator()->allocate();  b3rconv7sf.allocator()->allocate();  b3ract5fs.allocator()->allocate(); b3add2sf.allocator()->allocate();
		 b3act2fs.allocator()->allocate();  b3rconv9sf.allocator()->allocate();  b3ract6fs.allocator()->allocate();  b3rconv10sf.allocator()->allocate();  b3ract7fs.allocator()->allocate();  b3add3sf.allocator()->allocate();
		 b3act3fs.allocator()->allocate();  b3rconv12sf.allocator()->allocate();  b3ract8fs.allocator()->allocate();  b3rconv13sf.allocator()->allocate();  b3ract9fs.allocator()->allocate();  b3add4sf.allocator()->allocate();
		 b3act4fs.allocator()->allocate();  b3rconv15sf.allocator()->allocate();  b3ract10fs.allocator()->allocate();  b3rconv16sf.allocator()->allocate();  b3ract11fs.allocator()->allocate();  b3add5sf.allocator()->allocate();

		 b3act5fs.allocator()->allocate(); b4rconv0sf.allocator()->allocate();  b4ract0fs.allocator()->allocate();  b4rconv1sf.allocator()->allocate();  b4ract1fs.allocator()->allocate();  b4add0sf.allocator()->allocate();
		 b4act0fs.allocator()->allocate();  b4rconv3sf.allocator()->allocate();  b4ract2fs.allocator()->allocate();  b4rconv4sf.allocator()->allocate();  b4ract3fs.allocator()->allocate();  b4add1sf.allocator()->allocate();
		 b4act1fs.allocator()->allocate();  b4rconv6sf.allocator()->allocate();  b4ract4fs.allocator()->allocate();  b4rconv7sf.allocator()->allocate();  b4ract5fs.allocator()->allocate(); b4add2sf.allocator()->allocate();

		 pool1fs.allocator()->allocate();  conv1sf.allocator()->allocate();

		 test1.allocator()->allocate(); test2.allocator()->allocate(); test.allocator()->allocate(); out_block1_add02.allocator()->allocate();


		/*Fill tensor*/
		npy0.fill_tensor2(src);
		npy1.fill_tensor2(weights_con0);
		npy2.fill_tensor2(weights_block1r_con0);
		npy3.fill_tensor2(weights_block1r_con1);
		npy4.fill_tensor2(weights_block1r_con2);
		npy5.fill_tensor2(weights_block1l_con0);
		npy6.fill_tensor2(weights_block1r_con3);
		npy7.fill_tensor2(weights_block1r_con4);
		npy8.fill_tensor2(weights_block1r_con5);
		npy9.fill_tensor2(weights_block1r_con6);
		npy10.fill_tensor2(weights_block1r_con7);
		npy11.fill_tensor2(weights_block1r_con8);
		npy12.fill_tensor2(weights_block2r_con0);
		npy13.fill_tensor2(weights_block2r_con1);
		npy14.fill_tensor2(weights_block2r_con2);
		npy15.fill_tensor2(weights_block2l_con0);
		npy16.fill_tensor2(weights_block2r_con3);
		npy17.fill_tensor2(weights_block2r_con4);
		npy18.fill_tensor2(weights_block2r_con5);
		npy19.fill_tensor2(weights_block2r_con6);
		npy20.fill_tensor2(weights_block2r_con7);
		npy21.fill_tensor2(weights_block2r_con8);
		npy22.fill_tensor2(weights_block2r_con9);
		npy23.fill_tensor2(weights_block2r_con10);
		npy24.fill_tensor2(weights_block2r_con11);
		npy25.fill_tensor2(weights_block3r_con0);
		npy26.fill_tensor2(weights_block3r_con1);
		npy27.fill_tensor2(weights_block3r_con2);
		npy28.fill_tensor2(weights_block3l_con0);
		npy29.fill_tensor2(weights_block3r_con3);
		npy30.fill_tensor2(weights_block3r_con4);
		npy31.fill_tensor2(weights_block3r_con5);
		npy32.fill_tensor2(weights_block3r_con6);
		npy33.fill_tensor2(weights_block3r_con7);
		npy34.fill_tensor2(weights_block3r_con8);
		npy35.fill_tensor2(weights_block3r_con9);
		npy36.fill_tensor2(weights_block3r_con10);
		npy37.fill_tensor2(weights_block3r_con11);
		npy38.fill_tensor2(weights_block3r_con12);
		npy39.fill_tensor2(weights_block3r_con13);
		npy40.fill_tensor2(weights_block3r_con14);
		npy41.fill_tensor2(weights_block3r_con15);
		npy42.fill_tensor2(weights_block3r_con16);
		npy43.fill_tensor2(weights_block3r_con17);
		npy44.fill_tensor2(weights_block4r_con0);
		npy45.fill_tensor2(weights_block4r_con1);
		npy46.fill_tensor2(weights_block4r_con2);
		npy47.fill_tensor2(weights_block4l_con0);
		npy48.fill_tensor2(weights_block4r_con3);
		npy49.fill_tensor2(weights_block4r_con4);
		npy50.fill_tensor2(weights_block4r_con5);
		npy51.fill_tensor2(weights_block4r_con6);
		npy52.fill_tensor2(weights_block4r_con7);
		npy53.fill_tensor2(weights_block4r_con8);
		npy54.fill_tensor2(weights_con1);
		npyb1.fill_tensor2(bias_con0);
		npyb2.fill_tensor2(bias_block1r_con0);
		npyb3.fill_tensor2(bias_block1r_con1);
		npyb4.fill_tensor2(bias_block1r_con2);
		npyb5.fill_tensor2(bias_block1l_con0);
		npyb6.fill_tensor2(bias_block1r_con3);
		npyb7.fill_tensor2(bias_block1r_con4);
		npyb8.fill_tensor2(bias_block1r_con5);
		npyb9.fill_tensor2(bias_block1r_con6);
		npyb10.fill_tensor2(bias_block1r_con7);
		npyb11.fill_tensor2(bias_block1r_con8);
		npyb12.fill_tensor2(bias_block2r_con0);
		npyb13.fill_tensor2(bias_block2r_con1);
		npyb14.fill_tensor2(bias_block2r_con2);
		npyb15.fill_tensor2(bias_block2l_con0);
		npyb16.fill_tensor2(bias_block2r_con3);
		npyb17.fill_tensor2(bias_block2r_con4);
		npyb18.fill_tensor2(bias_block2r_con5);
		npyb19.fill_tensor2(bias_block2r_con6);
		npyb20.fill_tensor2(bias_block2r_con7);
		npyb21.fill_tensor2(bias_block2r_con8);
		npyb22.fill_tensor2(bias_block2r_con9);
		npyb23.fill_tensor2(bias_block2r_con10);
		npyb24.fill_tensor2(bias_block2r_con11);
		npyb25.fill_tensor2(bias_block3r_con0);
		npyb26.fill_tensor2(bias_block3r_con1);
		npyb27.fill_tensor2(bias_block3r_con2);
		npyb28.fill_tensor2(bias_block3l_con0);
		npyb29.fill_tensor2(bias_block3r_con3);
		npyb30.fill_tensor2(bias_block3r_con4);
		npyb31.fill_tensor2(bias_block3r_con5);
		npyb32.fill_tensor2(bias_block3r_con6);
		npyb33.fill_tensor2(bias_block3r_con7);
		npyb34.fill_tensor2(bias_block3r_con8);
		npyb35.fill_tensor2(bias_block3r_con9);
		npyb36.fill_tensor2(bias_block3r_con10);
		npyb37.fill_tensor2(bias_block3r_con11);
		npyb38.fill_tensor2(bias_block3r_con12);
		npyb39.fill_tensor2(bias_block3r_con13);
		npyb40.fill_tensor2(bias_block3r_con14);
		npyb41.fill_tensor2(bias_block3r_con15);
		npyb42.fill_tensor2(bias_block3r_con16);
		npyb43.fill_tensor2(bias_block3r_con17);
		npyb44.fill_tensor2(bias_block4r_con0);
		npyb45.fill_tensor2(bias_block4r_con1);
		npyb46.fill_tensor2(bias_block4r_con2);
		npyb47.fill_tensor2(bias_block4l_con0);
		npyb48.fill_tensor2(bias_block4r_con3);
		npyb49.fill_tensor2(bias_block4r_con4);
		npyb50.fill_tensor2(bias_block4r_con5);
		npyb51.fill_tensor2(bias_block4r_con6);
		npyb52.fill_tensor2(bias_block4r_con7);
		npyb53.fill_tensor2(bias_block4r_con8);
		npyb54.fill_tensor2(bias_con1);
		npy100.fill_tensor2(input_fc);
		is_fortran      = npy0.is_fortran();

		/*allocate end*/
		return true;
}/*end of do_setup*/
void do_run()override
{
	std::cout<<"ABMConvolutionLayer changed!"<<std::endl;
	clock_t start;
	/*clock_t end;*/
	clock_t end01,end02,end03,end04,end05;

	clock_t end111,end112,end113,end114,end115,end116,end117,end118,end119,end1110,end1111,end1112,end1113;
	clock_t end121,end122,end123,end124,end125,end126,end127,end128,end129,end1210,end1211,end1212,end1213;
	clock_t end131,end132,end133,end134,end135,end136,end137,end138,end139,end1310,end1311,end1312,end1313;

	clock_t end211,end212,end213,end214,end215,end216,end217,end218,end219,end2110,end2111,end2112,end2113,end2114;
	clock_t end221,end222,end223,end224,end225,end226,end227,end228,end229,end2210,end2211,end2212,end2213;
	clock_t end231,end232,end233,end234,end235,end236,end237,end238,end239,end2310,end2311,end2312,end2313;
	clock_t end241,end242,end243,end244,end245,end246,end247,end248,end249,end2410,end2411,end2412,end2413;

	clock_t end311,end312,end313,end314,end315,end316,end317,end318,end319,end3110,end3111,end3112,end3113,end3114;
	clock_t end321,end322,end323,end324,end325,end326,end327,end328,end329,end3210,end3211,end3212,end3213;
	clock_t end331,end332,end333,end334,end335,end336,end337,end338,end339,end3310,end3311,end3312,end3313;
	clock_t end341,end342,end343,end344,end345,end346,end347,end348,end349,end3410,end3411,end3412,end3413;
	clock_t end351,end352,end353,end354,end355,end356,end357,end358,end359,end3510,end3511,end3512,end3513;
	clock_t end361,end362,end363,end364,end365,end366,end367,end368,end369,end3610,end3611,end3612,end3613;

	clock_t end411,end412,end413,end414,end415,end416,end417,end418,end419,end4110,end4111,end4112,end4113,end4114;
	clock_t end421,end422,end423,end424,end425,end426,end427,end428,end429,end4210,end4211,end4212,end4213;
	clock_t end431,end432,end433,end434,end435,end436,end437,end438,end439,end4310,end4311,end4312,end4313;

    clock_t end11, end12, end13, end14,end15,end16;


	double lend01=0,lend02=0,lend03=0;

	double lend111=0,lend112=0,lend113=0,lend114=0,lend115=0,lend116=0,lend117=0,lend118=0;
	double lend121=0,lend122=0,lend123=0,lend124=0,lend125=0,lend126=0,lend127=0;
	double lend131=0,lend132=0,lend133=0,lend134=0,lend135=0,lend136=0,lend137=0;

	double lend211=0,lend212=0,lend213=0,lend214=0,lend215=0,lend216=0,lend217=0,lend218=0;
	double lend221=0,lend222=0,lend223=0,lend224=0,lend225=0,lend226=0,lend227=0;
	double lend231=0,lend232=0,lend233=0,lend234=0,lend235=0,lend236=0,lend237=0;
	double lend241=0,lend242=0,lend243=0,lend244=0,lend245=0,lend246=0,lend247=0;

	double lend311=0,lend312=0,lend313=0,lend314=0,lend315=0,lend316=0,lend317=0,lend318=0;
	double lend321=0,lend322=0,lend323=0,lend324=0,lend325=0,lend326=0,lend327=0;
	double lend331=0,lend332=0,lend333=0,lend334=0,lend335=0,lend336=0,lend337=0;
	double lend341=0,lend342=0,lend343=0,lend344=0,lend345=0,lend346=0,lend347=0;
	double lend351=0,lend352=0,lend353=0,lend354=0,lend355=0,lend356=0,lend357=0;
	double lend361=0,lend362=0,lend363=0,lend364=0,lend365=0,lend366=0,lend367=0;

	double lend411=0,lend412=0,lend413=0,lend414=0,lend415=0,lend416=0,lend417=0,lend418=0;
	double lend421=0,lend422=0,lend423=0,lend424=0,lend425=0,lend426=0,lend427=0;
	double lend431=0,lend432=0,lend433=0,lend434=0,lend435=0,lend436=0,lend437=0;

    double lend11=0, lend12=0, lend13=0, lend14=0; 

	/*double endtime=0;*/
	double total_time=0;
	double time=0;
	for (int i = 0; i < 51; i++)
	{
		/*std::cout<<"Resnet50 run start!"<<std::endl;*/
		start = clock();
		/*first run*/
		 con0.run(); end01=clock();
		 lconv0sf.run();end02=clock();
		/*batch0.run();end02=clock();*/
		act0.run();end03=clock(); 
		pool0.run();end04=clock();
		 lpool0fs.run();end05=clock();
		/* end first*/
		/*block1 run*/
		block1r_con0.run(); end111=clock();
		lb1rconv0sf.run();end112=clock();
		/*0.run(); end112=clock();*/
		block1r_act0.run();end113=clock();
		lb1ract0fs.run();end114=clock();
		block1r_con1.run(); end115=clock();
		lb1rconv1sf.run();end116=clock();
		/*1.run(); end115=clock();*/
		block1r_act1.run();end117=clock();
		lb1ract1fs.run();end118=clock();
		block1r_con2.run(); end119=clock();
		/*2.run(); end118=clock();*/
		block1l_con0.run(); end1110=clock();
		/*0.run();end1110=clock();*/
		block1_add0.run(); end1111=clock();
	/*testsf1.run();testsf2.run();block1_add02.run();testfs.run();end1111=clock();*/
		lb1add0sf.run();end1112=clock();
		block1_act0.run();end1113=clock();

		lb1act0fs.run();end121=clock();
		block1r_con3.run();end122=clock();
		lb1rconv3sf.run();end123=clock();
		/*3.run();end122=clock();*/
		block1r_act2.run();end124=clock();
		lb1ract2fs.run();end125=clock();
		block1r_con4.run(); end126=clock();
		lb1rconv4sf.run();end127=clock();
		/*4.run();end125=clock();*/
		block1r_act3.run();end128=clock();
		lb1ract3fs.run();end129=clock();
		block1r_con5.run(); end1210=clock();
		/*5.run();end128=clock();*/
		block1_add1.run(); end1211=clock();
		lb1add1sf.run();end1212=clock();
		block1_act1.run();end1213=clock();

		lb1act1fs.run();end131=clock();
		block1r_con6.run(); end132=clock();
		lb1rconv6sf.run();end133=clock();
		/*6.run();  end132=clock();*/
		block1r_act4.run(); end134=clock();
		lb1ract4fs.run();end135=clock();
		block1r_con7.run();  end136=clock();
		lb1rconv7sf.run();end137=clock();
		/*7.run();  end135=clock();*/
		block1r_act5.run(); end138=clock();
		lb1ract5fs.run();end139=clock();
		block1r_con8.run();  end1310=clock();
		/*8.run();  end138=clock(); */
		block1_add2.run();  end1311=clock();
		lb1add2sf.run();end1312=clock();
		block1_act2.run(); end1313=clock();
		/*end block1*/
		/*block2 run*/
		lb1act2fs.run();end211=clock();
		block2r_con0.run(); end212=clock();
		lb2rconv0sf.run();end213=clock();
		/*0.run(); end212=clock();*/
		block2r_act0.run();end214=clock();
		lb2ract0fs.run();end215=clock();
		block2r_con1.run(); end216=clock();
		lb2rconv1sf.run();end217=clock();
		/*1.run(); end215=clock();*/
		block2r_act1.run();end218=clock();
		lb2ract1fs.run();end219=clock();
		block2r_con2.run(); end2110=clock();
		/*2.run(); end218=clock();*/
		block2l_con0.run(); end2111=clock();
		/*0.run();end2110=clock();*/
		block2_add0.run(); end2112=clock();
		lb2add0sf.run();end2113=clock();
		block2_act0.run();end2114=clock();

		lb2act0fs.run();end221=clock();
		block2r_con3.run(); end222=clock();
		lb2rconv3sf.run();end223=clock();
		/*3.run(); end222=clock();*/
		block2r_act2.run();end224=clock();
		lb2ract2fs.run();end225=clock();
		block2r_con4.run();end226=clock();
		lb2rconv4sf.run();end227=clock();
		/*4.run(); end225=clock();*/
		block2r_act3.run();end228=clock();
		lb2ract3fs.run();end229=clock();
		block2r_con5.run(); end2210=clock();
		/*5.run();end228=clock();*/
		block2_add1.run(); end2211=clock();
		lb2add1sf.run();end2212=clock();
		block2_act1.run();end2213=clock();

		lb2act1fs.run();end231=clock();
		block2r_con6.run(); end232=clock();
		lb2rconv6sf.run();end233=clock();
		/*6.run(); end232=clock();*/
		block2r_act4.run();end234=clock();
		lb2ract4fs.run();end235=clock();
		block2r_con7.run();end236=clock();
		lb2rconv7sf.run();end237=clock();
		 /*7.run(); end235=clock();*/
		block2r_act5.run();end238=clock();
		lb2ract5fs.run();end239=clock();
		block2r_con8.run(); end2310=clock();
		/*8.run();end238=clock();*/
		block2_add2.run();end2311=clock();
		lb2add2sf.run();end2312=clock();
		block2_act2.run();end2313=clock();

		lb2act2fs.run();end241=clock();
		block2r_con9.run(); end242=clock();
		lb2rconv9sf.run();end243=clock();
		/*9.run(); end242=clock();*/
		block2r_act6.run();end244=clock();
		lb2ract6fs.run();end245=clock();
		block2r_con10.run();end246=clock();
		lb2rconv10sf.run();end247=clock();
		 /*10.run(); end245=clock();*/
		block2r_act7.run();end248=clock();
		lb2ract7fs.run();end249=clock();
		block2r_con11.run(); end2410=clock();
		/*11.run(); end248=clock();*/
		block2_add3.run(); end2411=clock();
		lb2add3sf.run();end2412=clock();
		block2_act3.run();end2413=clock();
		/*end block2*/
		/*block3 run*/

		lb2act3fs.run();end311=clock();
		block3r_con0.run();end312=clock();
		lb3rconv0sf.run();end313=clock();
		/*0.run(); end312=clock();*/
		block3r_act0.run();end314=clock();
		lb3ract0fs.run();end315=clock();
		block3r_con1.run(); end316=clock();
		lb3rconv1sf.run();end317=clock();
		/*1.run(); end315=clock();*/
		block3r_act1.run();end318=clock();
		lb3ract1fs.run();end319=clock();
		block3r_con2.run(); end3110=clock();
		/*2.run();end318=clock();*/
		 block3l_con0.run(); end3111=clock();
		 /*0.run();end3110=clock();*/
		block3_add0.run(); end3112=clock();
		lb3add0sf.run();end3113=clock();
		block3_act0.run();end3114=clock();

		lb3act0fs.run();end321=clock();
		block3r_con3.run(); end322=clock();
		lb3rconv3sf.run();end323=clock();
		/*3.run(); end322=clock();*/
		block3r_act2.run();end324=clock();
		lb3ract2fs.run();end325=clock();
		block3r_con4.run(); end326=clock();
		lb3rconv4sf.run();end327=clock();
		/*4.run(); end325=clock();*/
		block3r_act3.run();end328=clock();
		lb3ract3fs.run();end329=clock();
		block3r_con5.run(); end3210=clock();
		/*5.run();end328=clock();*/
		block3_add1.run(); end3211=clock();
		lb3add1sf.run();end3212=clock();
		 block3_act1.run();end3213=clock();

		lb3act1fs.run();end331=clock();
		block3r_con6.run(); end332=clock();
		lb3rconv6sf.run();end333=clock();
		/*6.run(); end332=clock();*/
		block3r_act4.run();end334=clock();
		lb3ract4fs.run();end335=clock();
		block3r_con7.run(); end336=clock();
		lb3rconv7sf.run();end337=clock();
		/*7.run(); end335=clock();*/
		block3r_act5.run();end338=clock();
		lb3ract5fs.run();end339=clock();
		block3r_con8.run(); end3310=clock();
		/*8.run();end338=clock();*/
		block3_add2.run();end3311=clock();
		lb3add2sf.run();end3312=clock();
		block3_act2.run();end3313=clock();

		lb3act2fs.run();end341=clock();
		block3r_con9.run(); end342=clock();
		lb3rconv9sf.run();end343=clock();
		/*9.run(); end342=clock();*/
		block3r_act6.run();end344=clock();
		lb3ract6fs.run();end345=clock();
		block3r_con10.run(); end346=clock();
		lb3rconv10sf.run();end347=clock();
		/*10.run(); end345=clock();*/
		block3r_act7.run();end348=clock();
		lb3ract7fs.run();end349=clock();
		block3r_con11.run(); end3410=clock();
		/*11.run();end348=clock();*/
		block3_add3.run(); end3411=clock();
		lb3add3sf.run();end3412=clock();
		block3_act3.run();end3413=clock();

		lb3act3fs.run();end351=clock();
		block3r_con12.run(); end352=clock();
		lb3rconv12sf.run();end353=clock();
		/*12.run(); end352=clock();*/
		block3r_act8.run();end354=clock();
		lb3ract8fs.run();end355=clock();
		block3r_con13.run(); end356=clock();
		lb3rconv13sf.run();end357=clock();
		/*13.run(); end355=clock();*/
		block3r_act9.run();end358=clock();
		lb3ract9fs.run();end359=clock();
		block3r_con14.run(); end3510=clock();
		/*14.run();end358=clock();*/
		block3_add4.run();end3511=clock();
		lb3add4sf.run();end3512=clock();
		block3_act4.run();end3513=clock();
		
		lb3act4fs.run();end361=clock();
		block3r_con15.run();end362=clock();
		lb3rconv15sf.run();end363=clock();
		/*15.run(); end362=clock();*/
		block3r_act10.run();end364=clock();
		lb3ract10fs.run();end365=clock();
		block3r_con16.run(); end366=clock();
		lb3rconv16sf.run();end367=clock();
		/*16.run(); end365=clock();*/
		block3r_act11.run();end368=clock();
		lb3ract11fs.run();end369=clock();
		block3r_con17.run(); end3610=clock();
		/*17.run();end368=clock();*/
		block3_add5.run();end3611=clock();
		lb3add5sf.run();end3612=clock();
		block3_act5.run();end3613=clock();

		/*end block3*/
		/*block4 run*/
		lb3act5fs.run();end411=clock();
		block4r_con0.run(); end412=clock();
		lb4rconv0sf.run();end413=clock();
		/*0.run();  end412=clock();*/
		block4r_act0.run(); end414=clock();
		lb4ract0fs.run();end415=clock();
		block4r_con1.run();  end416=clock();
		lb4rconv1sf.run();end417=clock();
		/*1.run();  end415=clock();*/
		block4r_act1.run(); end418=clock();
		lb4ract1fs.run();end419=clock();
		block4r_con2.run();  end4110=clock();
		/*2.run();  end418=clock();*/
		block4l_con0.run();  end4111=clock();
		/*0.run(); end4110=clock();*/
		block4_add0.run();  end4112=clock();
		lb4add0sf.run();end4113=clock();
		block4_act0.run(); end4114=clock();

		lb4act0fs.run();end421=clock();
		block4r_con3.run(); end422=clock();
		lb4rconv3sf.run();end423=clock();
		/*3.run(); end422=clock();*/
		block4r_act2.run();end424=clock();
		lb4ract2fs.run();end425=clock();
		block4r_con4.run(); end426=clock();
		lb4rconv4sf.run();end427=clock();
		/*4.run(); end425=clock();*/
		block4r_act3.run();end428=clock();
		lb4ract3fs.run();end429=clock();
		block4r_con5.run(); end4210=clock();
		/*5.run();end428=clock();*/
		block4_add1.run(); end4211=clock();
		lb4add1sf.run();end4212=clock();
		block4_act1.run();end4213=clock();

		lb4act1fs.run();end431=clock();
		block4r_con6.run(); end432=clock();
		lb4rconv6sf.run();end433=clock();
		/*6.run(); end432=clock();*/
		block4r_act4.run();end434=clock();
		lb4ract4fs.run();end435=clock();
		block4r_con7.run(); end436=clock();
		lb4rconv7sf.run();end437=clock();
		/*7.run(); end435=clock();*/
		block4r_act5.run();end438=clock();
		lb4ract5fs.run();end439=clock();
		block4r_con8.run(); end4310=clock();
		/*8.run(); end438=clock();*/
		block4_add2.run(); end4311=clock();
		lb4add2sf.run();end4312=clock();
		block4_act2.run();end4313=clock();
		/*end block4*/

		/* last run*/
		pool1.run();end11=clock();
		lpool1fs.run();end12=clock();
		 con1.run(); end13=clock();
		 lconv1sf.run();end14=clock();
		flatten.run();end15=clock();
		softmax.run();end16=clock();
		/*end = clock();*/
			if(i>0){
			double one_runtime=0;
			time = (double)(end01 - start) / CLOCKS_PER_SEC;lend01+=time; one_runtime+=time;
			time = (double)(end03 - end02) / CLOCKS_PER_SEC;lend02+=time; one_runtime+=time;
			time = (double)(end04 - end03) / CLOCKS_PER_SEC;lend03+=time; one_runtime+=time;
			/*  end first*/
			/*block1 run*/
			time = (double)(end111 - end05) / CLOCKS_PER_SEC;lend111+=time; one_runtime+=time;
			time = (double)(end113 - end112) / CLOCKS_PER_SEC;lend112+=time; one_runtime+=time;
			time = (double)(end115 - end114) / CLOCKS_PER_SEC;lend113+=time; one_runtime+=time;
			time = (double)(end117 - end116) / CLOCKS_PER_SEC;lend114+=time; one_runtime+=time;
			time = (double)(end119 - end118) / CLOCKS_PER_SEC;lend115+=time; one_runtime+=time;
			time = (double)(end1110 - end119) / CLOCKS_PER_SEC;lend116+=time; one_runtime+=time;
			time = (double)(end1111 - end1110) / CLOCKS_PER_SEC;lend117+=time; one_runtime+=time;
			time = (double)(end1113 - end1112) / CLOCKS_PER_SEC;lend118+=time; one_runtime+=time;


			time = (double)(end122 - end121) / CLOCKS_PER_SEC;lend121+=time; one_runtime+=time;
			time = (double)(end124 - end123) / CLOCKS_PER_SEC;lend122+=time; one_runtime+=time;
			time = (double)(end126 - end125) / CLOCKS_PER_SEC;lend123+=time; one_runtime+=time;
			time = (double)(end128 - end127) / CLOCKS_PER_SEC;lend124+=time; one_runtime+=time;
			time = (double)(end1210 - end129) / CLOCKS_PER_SEC;lend125+=time; one_runtime+=time;
			time = (double)(end1211 - end1210) / CLOCKS_PER_SEC;lend126+=time; one_runtime+=time;
			time = (double)(end1213 - end1212) / CLOCKS_PER_SEC;lend127+=time; one_runtime+=time;


			time = (double)(end132 - end131) / CLOCKS_PER_SEC;lend131+=time; one_runtime+=time;
			time = (double)(end134 - end133) / CLOCKS_PER_SEC;lend132+=time; one_runtime+=time;
			time = (double)(end136 - end135) / CLOCKS_PER_SEC;lend133+=time; one_runtime+=time;
			time = (double)(end138 - end137) / CLOCKS_PER_SEC;lend134+=time; one_runtime+=time;
			time = (double)(end1310 - end139) / CLOCKS_PER_SEC;lend135+=time; one_runtime+=time;
			time = (double)(end1311 - end1310) / CLOCKS_PER_SEC;lend136+=time; one_runtime+=time;
			time = (double)(end1313 - end1312) / CLOCKS_PER_SEC;lend137+=time; one_runtime+=time;

			/* end block1*/
			/* block2 run*/
			time = (double)(end212 - end211) / CLOCKS_PER_SEC;lend211+=time; one_runtime+=time;
			time = (double)(end214 - end213) / CLOCKS_PER_SEC;lend212+=time; one_runtime+=time;
			time = (double)(end216 - end215) / CLOCKS_PER_SEC;lend213+=time; one_runtime+=time;
			time = (double)(end218 - end217) / CLOCKS_PER_SEC;lend214+=time; one_runtime+=time;
			time = (double)(end2110 - end219) / CLOCKS_PER_SEC;lend215+=time; one_runtime+=time;
			time = (double)(end2111 - end2110) / CLOCKS_PER_SEC;lend216+=time; one_runtime+=time;
			time = (double)(end2112 - end2111) / CLOCKS_PER_SEC;lend217+=time; one_runtime+=time;
			time = (double)(end2114 - end2113) / CLOCKS_PER_SEC;lend218+=time; one_runtime+=time;


			time = (double)(end222 - end221) / CLOCKS_PER_SEC;lend221+=time; one_runtime+=time;
			time = (double)(end224 - end223) / CLOCKS_PER_SEC;lend222+=time; one_runtime+=time;
			time = (double)(end226 - end225) / CLOCKS_PER_SEC;lend223+=time; one_runtime+=time;
			time = (double)(end228 - end227) / CLOCKS_PER_SEC;lend224+=time; one_runtime+=time;
			time = (double)(end2210 - end229) / CLOCKS_PER_SEC;lend225+=time; one_runtime+=time;
			time = (double)(end2211 - end2210) / CLOCKS_PER_SEC;lend226+=time; one_runtime+=time;
			time = (double)(end2213 - end2212) / CLOCKS_PER_SEC;lend227+=time; one_runtime+=time;


			time = (double)(end232 - end231) / CLOCKS_PER_SEC;lend231+=time; one_runtime+=time;
			time = (double)(end234 - end233) / CLOCKS_PER_SEC;lend232+=time; one_runtime+=time;
			time = (double)(end236 - end235) / CLOCKS_PER_SEC;lend233+=time; one_runtime+=time;
			time = (double)(end238 - end237) / CLOCKS_PER_SEC;lend234+=time; one_runtime+=time;
			time = (double)(end2310 - end239) / CLOCKS_PER_SEC;lend235+=time; one_runtime+=time;
			time = (double)(end2311 - end2310) / CLOCKS_PER_SEC;lend236+=time; one_runtime+=time;
			time = (double)(end2313 - end2312) / CLOCKS_PER_SEC;lend237+=time; one_runtime+=time;


			time = (double)(end242 - end241) / CLOCKS_PER_SEC;lend241+=time; one_runtime+=time;
			time = (double)(end244 - end243) / CLOCKS_PER_SEC;lend242+=time; one_runtime+=time;
			time = (double)(end246 - end245) / CLOCKS_PER_SEC;lend243+=time; one_runtime+=time;
			time = (double)(end248 - end247) / CLOCKS_PER_SEC;lend244+=time; one_runtime+=time;
			time = (double)(end2410 - end249) / CLOCKS_PER_SEC;lend245+=time; one_runtime+=time;
			time = (double)(end2411 - end2410) / CLOCKS_PER_SEC;lend246+=time; one_runtime+=time;
			time = (double)(end2413 - end2412) / CLOCKS_PER_SEC;lend247+=time; one_runtime+=time;

			/*end block2*/
			/*block3 run*/
			time = (double)(end312 - end311) / CLOCKS_PER_SEC;lend311+=time; one_runtime+=time;
			time = (double)(end314 - end313) / CLOCKS_PER_SEC;lend312+=time; one_runtime+=time;
			time = (double)(end316 - end315) / CLOCKS_PER_SEC;lend313+=time; one_runtime+=time;
			time = (double)(end318 - end317) / CLOCKS_PER_SEC;lend314+=time; one_runtime+=time;
			time = (double)(end3110 - end319) / CLOCKS_PER_SEC;lend315+=time; one_runtime+=time;
			time = (double)(end3111 - end3110) / CLOCKS_PER_SEC;lend316+=time; one_runtime+=time;
			time = (double)(end3112 - end3111) / CLOCKS_PER_SEC;lend317+=time; one_runtime+=time;
			time = (double)(end3114 - end3113) / CLOCKS_PER_SEC;lend318+=time; one_runtime+=time;


			time = (double)(end322 - end321) / CLOCKS_PER_SEC;lend321+=time; one_runtime+=time;
			time = (double)(end324 - end323) / CLOCKS_PER_SEC;lend322+=time; one_runtime+=time;
			time = (double)(end326 - end325) / CLOCKS_PER_SEC;lend323+=time; one_runtime+=time;
			time = (double)(end328 - end327) / CLOCKS_PER_SEC;lend324+=time; one_runtime+=time;
			time = (double)(end3210 - end329) / CLOCKS_PER_SEC;lend325+=time; one_runtime+=time;
			time = (double)(end3211 - end3210) / CLOCKS_PER_SEC;lend326+=time; one_runtime+=time;
			time = (double)(end3213 - end3212) / CLOCKS_PER_SEC;lend327+=time; one_runtime+=time;


			time = (double)(end332 - end331) / CLOCKS_PER_SEC;lend331+=time; one_runtime+=time;
			time = (double)(end334 - end333) / CLOCKS_PER_SEC;lend332+=time; one_runtime+=time;
			time = (double)(end336 - end335) / CLOCKS_PER_SEC;lend333+=time; one_runtime+=time;
			time = (double)(end338 - end337) / CLOCKS_PER_SEC;lend334+=time; one_runtime+=time;
			time = (double)(end3310 - end339) / CLOCKS_PER_SEC;lend335+=time; one_runtime+=time;
			time = (double)(end3311 - end3310) / CLOCKS_PER_SEC;lend336+=time; one_runtime+=time;
			time = (double)(end3313 - end3312) / CLOCKS_PER_SEC;lend337+=time; one_runtime+=time;


			time = (double)(end342 - end341) / CLOCKS_PER_SEC;lend341+=time; one_runtime+=time;
			time = (double)(end344 - end343) / CLOCKS_PER_SEC;lend342+=time; one_runtime+=time;
			time = (double)(end346 - end345) / CLOCKS_PER_SEC;lend343+=time; one_runtime+=time;
			time = (double)(end348 - end347) / CLOCKS_PER_SEC;lend344+=time; one_runtime+=time;
			time = (double)(end3410 - end349) / CLOCKS_PER_SEC;lend345+=time; one_runtime+=time;
			time = (double)(end3411 - end3410) / CLOCKS_PER_SEC;lend346+=time; one_runtime+=time;
			time = (double)(end3413 - end3412) / CLOCKS_PER_SEC;lend347+=time; one_runtime+=time;


			time = (double)(end352 - end351) / CLOCKS_PER_SEC;lend351+=time; one_runtime+=time;
			time = (double)(end354 - end353) / CLOCKS_PER_SEC;lend352+=time; one_runtime+=time;
			time = (double)(end356 - end355) / CLOCKS_PER_SEC;lend353+=time; one_runtime+=time;
			time = (double)(end358 - end357) / CLOCKS_PER_SEC;lend354+=time; one_runtime+=time;
			time = (double)(end3510 - end359) / CLOCKS_PER_SEC;lend355+=time; one_runtime+=time;
			time = (double)(end3511 - end3510) / CLOCKS_PER_SEC;lend356+=time; one_runtime+=time;
			time = (double)(end3513 - end3512) / CLOCKS_PER_SEC;lend357+=time; one_runtime+=time;

			time = (double)(end362 - end361) / CLOCKS_PER_SEC;lend361+=time; one_runtime+=time;
			time = (double)(end364 - end363) / CLOCKS_PER_SEC;lend362+=time; one_runtime+=time;
			time = (double)(end366 - end365) / CLOCKS_PER_SEC;lend363+=time; one_runtime+=time;
			time = (double)(end368 - end367) / CLOCKS_PER_SEC;lend364+=time; one_runtime+=time;
			time = (double)(end3610 - end369) / CLOCKS_PER_SEC;lend365+=time; one_runtime+=time;
			time = (double)(end3611 - end3610) / CLOCKS_PER_SEC;lend366+=time; one_runtime+=time;
			time = (double)(end3613 - end3612) / CLOCKS_PER_SEC;lend367+=time; one_runtime+=time;


			/*end block3*/
			/*block4 run*/
			time = (double)(end412 - end411) / CLOCKS_PER_SEC;lend411+=time; one_runtime+=time;
			time = (double)(end414 - end413) / CLOCKS_PER_SEC;lend412+=time; one_runtime+=time;
			time = (double)(end416 - end415) / CLOCKS_PER_SEC;lend413+=time; one_runtime+=time;
			time = (double)(end418 - end417) / CLOCKS_PER_SEC;lend414+=time; one_runtime+=time;
			time = (double)(end4110 - end419) / CLOCKS_PER_SEC;lend415+=time; one_runtime+=time;
			time = (double)(end4111 - end4110) / CLOCKS_PER_SEC;lend416+=time; one_runtime+=time;
			time = (double)(end4112 - end4111) / CLOCKS_PER_SEC;lend417+=time; one_runtime+=time;
			time = (double)(end4114 - end4113) / CLOCKS_PER_SEC;lend418+=time; one_runtime+=time;


			time = (double)(end422 - end421) / CLOCKS_PER_SEC;lend421+=time; one_runtime+=time;
			time = (double)(end424 - end423) / CLOCKS_PER_SEC;lend422+=time; one_runtime+=time;
			time = (double)(end426 - end425) / CLOCKS_PER_SEC;lend423+=time; one_runtime+=time;
			time = (double)(end428 - end427) / CLOCKS_PER_SEC;lend424+=time; one_runtime+=time;
			time = (double)(end4210 - end429) / CLOCKS_PER_SEC;lend425+=time; one_runtime+=time;
			time = (double)(end4211 - end4210) / CLOCKS_PER_SEC;lend426+=time; one_runtime+=time;
			time = (double)(end4213 - end4212) / CLOCKS_PER_SEC;lend427+=time; one_runtime+=time;


			time = (double)(end432 - end431) / CLOCKS_PER_SEC;lend431+=time; one_runtime+=time;
			time = (double)(end434 - end433) / CLOCKS_PER_SEC;lend432+=time; one_runtime+=time;
			time = (double)(end436 - end435) / CLOCKS_PER_SEC;lend433+=time; one_runtime+=time;
			time = (double)(end438 - end437) / CLOCKS_PER_SEC;lend434+=time; one_runtime+=time;
			time = (double)(end4310 - end439) / CLOCKS_PER_SEC;lend435+=time; one_runtime+=time;
			time = (double)(end4311 - end4310) / CLOCKS_PER_SEC;lend436+=time; one_runtime+=time;
			time = (double)(end4313 - end4312) / CLOCKS_PER_SEC;lend437+=time; one_runtime+=time;

			/*end block4*/

			/* last run*/
			time = (double)(end11 - end4313) / CLOCKS_PER_SEC;lend11+=time; one_runtime+=time;
			time = (double)(end13 - end12) / CLOCKS_PER_SEC;lend12+=time; one_runtime+=time;
			time = (double)(end15 - end14) / CLOCKS_PER_SEC;lend13+=time; one_runtime+=time;
			time = (double)(end16 - end15) / CLOCKS_PER_SEC;lend14+=time; one_runtime+=time;
			std::cout<<i<<"---run:"<<std::endl;
			std::cout<<"time="<<one_runtime*1000<<"ms"<<std::endl;/*不包含datatype change层的时间*/
			/*endtime = (double)(end - start) / CLOCKS_PER_SEC;   包含datatype change层的时间*/
			total_time+=one_runtime;    }
	}
		arm_compute::utils::NPYLoader save;
		save.save_to_npy2(pool1fs,output_filename,false);
		save.save_to_npy2(out_con1,output_filename1,false);
		save.save_to_npy2(out_block4_add2,output_filename2,false);
		save.save_to_npy2(pool1fs,output_filename3,false);
		
		/*print resnet50 layer time*/
			std::cout<<"Resnet50"<<std::endl;
			std::cout << "---conv1       " << "		"<< lend01 * 1000/50 << "ms" << std::endl;
			/*std::cout << "---bn1         " << "		"<<lend02 * 1000/50 << "ms" << std::endl;*/
			std::cout << "---relu1       " <<"		"<< lend02* 1000/50 << "ms" << std::endl;
			std::cout << "---pooling1    " <<"		"<< lend03 * 1000/50 << "ms" << std::endl;

			std::cout<<"---layer1      "<<std::endl;
			std::cout<<"  ---0         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend111 * 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend112 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend112 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend113 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend115 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend114 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend115 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend118 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---conv    " << "		"<< lend116 * 1000/50 << "ms" << std::endl;
			/*       " << "		"<< lend1110 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend117 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend118 * 1000/50 << "ms" << std::endl;
			
			std::cout<<"  ---1         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend121 * 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend122 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend122 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend123 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend125 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend124 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend125 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend128 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend126 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend127 * 1000/50 << "ms" << std::endl;

			std::cout<<"  ---2         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend131 * 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend132 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend132 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend133 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend135 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend134 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend135 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend138 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend136 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend137 * 1000/50 << "ms" << std::endl;

			std::cout<<"---layer2      "<<std::endl;
			std::cout<<"  ---0         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend211 * 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend212 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend212 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend213 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend215 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend214 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend215 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend218 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---conv    " << "		"<< lend216 * 1000/50 << "ms" << std::endl;
			/*       " << "		"<< lend2110 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend217 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend218 * 1000/50 << "ms" << std::endl;
			
			std::cout<<"  ---1         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend221 * 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend222 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend222 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend223 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend225 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend224 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend225 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend228 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend226 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend227 * 1000/50 << "ms" << std::endl;

			std::cout<<"  ---2         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend231* 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend232 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend232 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend233 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend235 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend234 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend235 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend238 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend236 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend237 * 1000/50 << "ms" << std::endl;

			std::cout<<"  ---3         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend241 * 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend242 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend242 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend243 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend245 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend244 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend245 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend248 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend246 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend247 * 1000/50 << "ms" << std::endl;

			std::cout<<"---layer3      "<<std::endl;
			std::cout<<"  ---0         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend311 * 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend312 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend312 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend313 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend315 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend314 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend315 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend318 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---conv    " << "		"<< lend316 * 1000/50 << "ms" << std::endl;
			/*       " << "		"<< lend3110 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend317 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend318 * 1000/50 << "ms" << std::endl;
			
			std::cout<<"  ---1         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend321 * 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend322 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend322 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend323 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend325 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend324 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend325 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend328 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend326 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend327 * 1000/50 << "ms" << std::endl;

			std::cout<<"  ---2         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend331 * 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend332 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend332 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend333 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend335 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend334 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend335 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend338 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend336 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend337 * 1000/50 << "ms" << std::endl;

			std::cout<<"  ---3         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend341 * 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend342 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend342 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend343 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend345 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend344 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend345 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend348 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend346 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend347 * 1000/50 << "ms" << std::endl;

			std::cout<<"  ---4         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend351 * 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend352 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend352 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend353 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend355 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend354 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend355 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend358 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend356 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend357 * 1000/50 << "ms" << std::endl;

			std::cout<<"  ---5         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend361 * 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend362 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend362 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend363 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend365 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend364 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend365 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend368 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend366 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend367 * 1000/50 << "ms" << std::endl;

			std::cout<<"---layer4      "<<std::endl;
			std::cout<<"  ---0         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend411 * 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend412 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend412 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend413 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend415 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend414 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend415 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend418 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---conv    " << "		"<< lend416 * 1000/50 << "ms" << std::endl;
			/*       " << "		"<< lend4110 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend417 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend418 * 1000/50 << "ms" << std::endl;
			
			std::cout<<"  ---1         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend421 * 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend422 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend422 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend423 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend425 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend424 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend425 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend428 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend426 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend427 * 1000/50 << "ms" << std::endl;

			std::cout<<"  ---2         "<<std::endl;
			std::cout << "   ---conv1    " << "		"<< lend431 * 1000/50 << "ms" << std::endl;
			/*1      " << "		"<< lend432 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu1    " << "		"<< lend432 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv2    " << "		"<< lend433 * 1000/50 << "ms" << std::endl;
			/*2      " << "		"<< lend435 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---relu2    " << "		"<< lend434 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---conv3    " << "		"<< lend435 * 1000/50 << "ms" << std::endl;
			/*3      " << "		"<< lend438 * 1000/50 << "ms" << std::endl;*/
			std::cout << "   ---add      " << "		"<< lend436 * 1000/50 << "ms" << std::endl;
			std::cout << "   ---relu     " << "		"<< lend437 * 1000/50 << "ms" << std::endl;	

			std::cout << "---pooling     " << "		"<< lend11 * 1000/50 << "ms" << std::endl;
			std::cout << "---conv1       " << "		"<<lend12 * 1000/50 << "ms" << std::endl;
			std::cout << "---flatten     "<<"		"<< lend13 * 1000/50 << "ms" << std::endl;
			std::cout << "---softmax   " <<"		"<< lend14 * 1000/50 << "ms" << std::endl;
		
			std::cout<<"avg time="<<total_time*1000/50<<"ms"<<std::endl;
		/* end print resnet50  layer time*/
		
}/*end do_run()*/
private:
	/*precision table*/
	unsigned int  precision[54][4] ={
                        {8, 7, 4, 6                      
                         },/*layer-1*/
                        {7, 7, 6,  5
                         },/*layer-2*/
                        {7, 7, 5, 6
                         },/*layer-3*/
                        {5, 6, 6, 7
                         },/*layer-4*/
                        {7, 7, 6, 5
                        },/*layer-5*/
                        {8, 6, 6, 6
                         },/*layer-6*/
                        {7, 7, 6, 6
                         },/*layer-7*/
                        {7, 6, 6, 6
                        },/*layer-8*/
                        {6, 8, 6, 7
                         },/*layer-9*/
                        {7, 8, 7, 4
                         },/*layer-10*/
                        {4, 7, 4, 4
                       },/*layer-11*/
                        {7, 7, 7, 7
                         },/*layer-12*/
                        {9, 6, 7, 7
                         },/*layer-13*/
                        {7, 6, 7, 6
                         },/*layer-14*/
                        {8, 6, 7, 6
                        },/*layer-15*/
                        {6, 8, 4, 4
                         },/*layer-16*/
                        {7, 9, 4, 5
                         },/*layer-17*/
                        {5, 9, 5, 5
                        },/*layer-18*/
                        {7, 7, 7, 7
                         },/*layer-19*/
                        {6, 8, 7, 4
                         },/*layer-20*/
                        {5, 8, 4, 8
                        },/*layer-21*/
                        {8, 7, 6, 3
                         },/*layer-22*/
                        {6, 8, 3, 6
                         },/*layer-23*/
                        {7, 6, 6, 6
                        },/*layer-24*/
                        {8, 7, 7, 5
                         },/*layer-25*/
                        {10, 6, 5, 5
                         },/*layer-26*/
                        {6, 7, 5, 7
                         },/*layer-27*/
                        {8, 7, 7, 7
                        },/*layer-28*/
                        {9, 8, 5, 4
                         },/*layer-29*/
                        {8, 8, 4, 4
                         },/*layer-30*/
                        {8, 7, 4, 8
                        },/*layer-31*/
                        {8, 9, 6, 6
                         },/*layer-32*/
                        {8, 7, 6, 7
                         },/*layer-33*/
                        {7, 8, 7, 7
                        },/*layer-34*/
                        {7, 9, 5, 5
                         },/*layer-35*/
                        {8, 9, 5, 6
                         },/*layer-36*/
                        {7, 8, 6, 6
                        },/*layer-37*/
                        {7, 9, 5, 5
                         },/*layer-38*/
                        {7, 8, 5, 5
                         },/*layer-39*/
                        {6, 6, 5, 7
                        },/*layer-40*/
                        {9, 9, 6, 6
                         },/*layer-41*/
                        {9, 7, 6, 7
                         },/*layer-42*/
                        {6, 6, 7, 7
                        },/*layer-43*/
                        {8, 9, 6, 6
                         },/*layer-44*/
                        {9, 7, 6, 6
                         },/*layer-45*/
                        {7, 9, 6, 6
                         },/*layer-46*/
                        {8, 8, 6, 6
                        },/*layer-47*/
                        {7, 7, 7, 5
                         },/*layer-48*/
                        {9, 7, 5, 7
                         },/*layer-49*/
                        {8, 6, 7, 5
                     	  },/*layer-50*/
                        {7, 7, 5, 5
                         },/*layer-51*/
                        {9, 8, 5, 7
                         },/*layer-52*/
                        {3, 7, 7, 0
                        },/*layer-53*/
                        {8, 9, 2, 1
                         }};/*layer-54*/

	/*index table*/
	unsigned int  index[54][3] ={
							{3, 3, 10}, /* layer-1 max weight kernel size is 2^3, 2^3, 2^10*/
                            {3, 3, 10}, /* layer-2*/
                            {3, 3, 10}, /* layer-3*/
                            {3, 3, 10}, /* layer-4*/
                            {3, 3, 10}, /* layer-5*/
                            {3, 3, 10}, /* layer-6*/
                            {3, 3, 10}, /* layer-7*/
                            {3, 3, 10}, /* layer-8*/
                            {3, 3, 10}, /* layer-9*/
                            {3, 3, 10}, /* layer-10*/
                            {3, 3, 10}, /* layer-11*/
                            {3, 3, 10}, /* layer-12*/
                            {3, 3, 10}, /* layer-13*/
                            {3, 3, 10}, /* layer-14*/
                            {3, 3, 10}, /* layer-15*/
                            {3, 3, 10}, /* layer-16*/
                            {3, 3, 10}, /* layer-17*/
                            {3, 3, 10}, /* layer-18*/
                            {3, 3, 10}, /* layer-19*/
                            {3, 3, 10}, /* layer-20*/
                            {3, 3, 10}, /* layer-21*/
                            {3, 3, 10}, /* layer-22*/
                            {3, 3, 10}, /* layer-23*/
                            {3, 3, 10}, /* layer-24*/
                            {3, 3, 10}, /* layer-25*/
                            {3, 3, 10}, /* layer-26*/
                            {3, 3, 10}, /* layer-27*/
                            {3, 3, 10}, /* layer-28*/
                            {3, 3, 10}, /* layer-29*/
                            {3, 3, 10}, /* layer-30*/
                            {3, 3, 10}, /* layer-31*/
                            {3, 3, 10}, /* layer-32*/
                            {3, 3, 10}, /* layer-33*/
                            {3, 3, 10}, /* layer-34*/
                            {3, 3, 10}, /* layer-35*/
                            {3, 3, 10}, /* layer-36*/
                            {3, 3, 10}, /* layer-37*/
                            {3, 3, 10}, /* layer-38*/
                            {3, 3, 10}, /* layer-39*/
                            {3, 3, 10}, /* layer-40*/
                            {3, 3, 10}, /* layer-41*/
                            {3, 3, 10}, /* layer-42*/
                            {3, 3, 10}, /* layer-43*/
                            {3, 3, 10}, /* layer-44*/
                            {3, 3, 10}, /* layer-45*/
                            {3, 3, 10}, /* layer-46*/
                            {3, 3, 10}, /* layer-47*/
                            {2, 2, 12}, /* layer-48*/
                            {3, 3, 10}, /* layer-49*/
                            {3, 3, 10}, /* layer-50*/
                            {2, 2, 12}, /* layer-51*/
                            {3, 3, 10}, /* layer-52*/
                            {3, 3, 10}, /* layer-53*/
                            {7, 4, 5}   /* layer-54 fc is resized to 32,4,16 */
                            };
	
	
	int fp[16][3]={
		{7,5,6},
		{6,6,6},
		{4,6,7},
		{6,6,4},
		{5,4,7},
		{8,7,6},
		{6,6,7},
		{7,7,5},
		{8,5,6},
		{7,6,5},
		{6,5,5},
		{7,5,6},
		{7,6,6},
		{6,6,7},
		{5,7,5},
		{0,5,2}
	};
	/*Tensor*/
	bool is_fortran{};
	string output_filename="/media/sdcard/ComputeLibrary/data/resnet50_resnet50_float_output/output.npy";
	string output_filename1="/media/sdcard/ComputeLibrary/data/resnet50_resnet50_float_output/output1.npy";
	string output_filename2="/media/sdcard/ComputeLibrary/data/resnet50_resnet50_float_output/output2.npy";
	string output_filename3="/media/sdcard/ComputeLibrary/data/resnet50_resnet50_float_output/output3.npy";
	Tensor src{}; Tensor weights_con0{};Tensor bias_con0{};

	Tensor weights_block1r_con0{}; Tensor weights_block1r_con1{}; Tensor weights_block1r_con2{};
	Tensor weights_block1r_con3{}; Tensor weights_block1r_con4{}; Tensor weights_block1r_con5{};
	Tensor weights_block1r_con6{}; Tensor weights_block1r_con7{}; Tensor weights_block1r_con8{};
	Tensor weights_block1l_con0{}; 

	Tensor weights_block2r_con0{}; Tensor weights_block2r_con1{}; Tensor weights_block2r_con2{};
	Tensor weights_block2r_con3{}; Tensor weights_block2r_con4{}; Tensor weights_block2r_con5{};
	Tensor weights_block2r_con6{}; Tensor weights_block2r_con7{}; Tensor weights_block2r_con8{};
	Tensor weights_block2r_con9{}; Tensor weights_block2r_con10{}; Tensor weights_block2r_con11{};
	Tensor weights_block2l_con0{}; 

	Tensor weights_block3r_con0{}; Tensor weights_block3r_con1{}; Tensor weights_block3r_con2{};
	Tensor weights_block3r_con3{}; Tensor weights_block3r_con4{}; Tensor weights_block3r_con5{};
	Tensor weights_block3r_con6{}; Tensor weights_block3r_con7{}; Tensor weights_block3r_con8{};
	Tensor weights_block3r_con9{}; Tensor weights_block3r_con10{}; Tensor weights_block3r_con11{};
	Tensor weights_block3r_con12{}; Tensor weights_block3r_con13{}; Tensor weights_block3r_con14{};
	Tensor weights_block3r_con15{}; Tensor weights_block3r_con16{}; Tensor weights_block3r_con17{};
	Tensor weights_block3l_con0{}; 

	Tensor weights_block4r_con0{}; Tensor weights_block4r_con1{}; Tensor weights_block4r_con2{};
	Tensor weights_block4r_con3{}; Tensor weights_block4r_con4{}; Tensor weights_block4r_con5{};
	Tensor weights_block4r_con6{}; Tensor weights_block4r_con7{}; Tensor weights_block4r_con8{};
	Tensor weights_block4l_con0{}; 

	Tensor bias_block1r_con0{}; Tensor bias_block1r_con1{}; Tensor bias_block1r_con2{};
	Tensor bias_block1r_con3{}; Tensor bias_block1r_con4{}; Tensor bias_block1r_con5{};
	Tensor bias_block1r_con6{}; Tensor bias_block1r_con7{}; Tensor bias_block1r_con8{};
	Tensor bias_block1l_con0{}; 

	Tensor bias_block2r_con0{}; Tensor bias_block2r_con1{}; Tensor bias_block2r_con2{};
	Tensor bias_block2r_con3{}; Tensor bias_block2r_con4{}; Tensor bias_block2r_con5{};
	Tensor bias_block2r_con6{}; Tensor bias_block2r_con7{}; Tensor bias_block2r_con8{};
	Tensor bias_block2r_con9{}; Tensor bias_block2r_con10{}; Tensor bias_block2r_con11{};
	Tensor bias_block2l_con0{}; 

	Tensor bias_block3r_con0{}; Tensor bias_block3r_con1{}; Tensor bias_block3r_con2{};
	Tensor bias_block3r_con3{}; Tensor bias_block3r_con4{}; Tensor bias_block3r_con5{};
	Tensor bias_block3r_con6{}; Tensor bias_block3r_con7{}; Tensor bias_block3r_con8{};
	Tensor bias_block3r_con9{}; Tensor bias_block3r_con10{}; Tensor bias_block3r_con11{};
	Tensor bias_block3r_con12{}; Tensor bias_block3r_con13{}; Tensor bias_block3r_con14{};
	Tensor bias_block3r_con15{}; Tensor bias_block3r_con16{}; Tensor bias_block3r_con17{};
	Tensor bias_block3l_con0{}; 

	Tensor bias_block4r_con0{}; Tensor bias_block4r_con1{}; Tensor bias_block4r_con2{};
	Tensor bias_block4r_con3{}; Tensor bias_block4r_con4{}; Tensor bias_block4r_con5{};
	Tensor bias_block4r_con6{}; Tensor bias_block4r_con7{}; Tensor bias_block4r_con8{};
	Tensor bias_block4l_con0{}; 
	
	Tensor input_fc{};
	Tensor weights_con1{}, bias_con1{};


	Tensor out_con0{}; 
	Tensor out_act0{}; Tensor out_pool0{};
	/*block1*/
	Tensor out_block1r_con0{}; Tensor out_block1r_act0{};
	Tensor out_block1r_con1{};  Tensor out_block1r_act1{};
	Tensor out_block1r_con2{}; Tensor out_block1l_con0{}; 
	Tensor out_block1_add0{}; Tensor out_block1_act0{};

	Tensor out_block1r_con3{}; Tensor out_block1r_act2{};
	Tensor out_block1r_con4{};  Tensor out_block1r_act3{};
	Tensor out_block1r_con5{};
	Tensor out_block1_add1{}; Tensor out_block1_act1{};

	Tensor out_block1r_con6{}; Tensor out_block1r_act4{};
	Tensor out_block1r_con7{};  Tensor out_block1r_act5{};
	Tensor out_block1r_con8{}; 
	Tensor out_block1_add2{}; Tensor out_block1_act2{};
	/*block2*/
	Tensor out_block2r_con0{}; Tensor out_block2r_act0{};
	Tensor out_block2r_con1{};  Tensor out_block2r_act1{};
	Tensor out_block2r_con2{}; Tensor out_block2l_con0{}; 
	Tensor out_block2_add0{}; Tensor out_block2_act0{};

	Tensor out_block2r_con3{}; Tensor out_block2r_act2{};
	Tensor out_block2r_con4{};  Tensor out_block2r_act3{};
	Tensor out_block2r_con5{};
	Tensor out_block2_add1{}; Tensor out_block2_act1{};

	Tensor out_block2r_con6{}; Tensor out_block2r_act4{};
	Tensor out_block2r_con7{}; Tensor out_block2r_act5{};
	Tensor out_block2r_con8{}; 
	Tensor out_block2_add2{}; Tensor out_block2_act2{};

	Tensor out_block2r_con9{};  Tensor out_block2r_act6{};
	Tensor out_block2r_con10{};Tensor out_block2r_act7{};
	Tensor out_block2r_con11{}; 
	Tensor out_block2_add3{}; Tensor out_block2_act3{};
	/*block3*/
	Tensor out_block3r_con0{};  Tensor out_block3r_act0{};
	Tensor out_block3r_con1{};Tensor out_block3r_act1{};
	Tensor out_block3r_con2{}; Tensor out_block3l_con0{}; 
	Tensor out_block3_add0{}; Tensor out_block3_act0{};

	Tensor out_block3r_con3{};  Tensor out_block3r_act2{};
	Tensor out_block3r_con4{}; Tensor out_block3r_act3{};
	Tensor out_block3r_con5{}; 
	Tensor out_block3_add1{}; Tensor out_block3_act1{};

	Tensor out_block3r_con6{};  Tensor out_block3r_act4{};
	Tensor out_block3r_con7{};  Tensor out_block3r_act5{};
	Tensor out_block3r_con8{}; 
	Tensor out_block3_add2{}; Tensor out_block3_act2{};

	Tensor out_block3r_con9{}; Tensor out_block3r_act6{};
	Tensor out_block3r_con10{}; Tensor out_block3r_act7{};
	Tensor out_block3r_con11{}; 
	Tensor out_block3_add3{}; Tensor out_block3_act3{};

	Tensor out_block3r_con12{}; Tensor out_block3r_act8{};
	Tensor out_block3r_con13{};  Tensor out_block3r_act9{};
	Tensor out_block3r_con14{};
	Tensor out_block3_add4{}; Tensor out_block3_act4{};

	Tensor out_block3r_con15{}; Tensor out_block3r_act10{};
	Tensor out_block3r_con16{};  Tensor out_block3r_act11{};
	Tensor out_block3r_con17{}; 
	Tensor out_block3_add5{}; Tensor out_block3_act5{};

	/*block4*/
	Tensor out_block4r_con0{};  Tensor out_block4r_act0{};
	Tensor out_block4r_con1{};  Tensor out_block4r_act1{};
	Tensor out_block4r_con2{};  Tensor out_block4l_con0{};
	Tensor out_block4_add0{}; Tensor out_block4_act0{};

	Tensor out_block4r_con3{};  Tensor out_block4r_act2{};
	Tensor out_block4r_con4{};  Tensor out_block4r_act3{};
	Tensor out_block4r_con5{}; 
	Tensor out_block4_add1{}; Tensor out_block4_act1{};

	Tensor out_block4r_con6{};  Tensor out_block4r_act4{};
	Tensor out_block4r_con7{};  Tensor out_block4r_act5{};
	Tensor out_block4r_con8{}; 
	Tensor out_block4_add2{}; Tensor out_block4_act2{};

	Tensor out_pool1{}; Tensor out_con1{}; Tensor out_flatten{}; Tensor out_softmax{};


	/*type change tensor*/
	Tensor conv0sf{}; Tensor pool0fs{};

	Tensor b1rconv0sf{}; Tensor b1ract0fs{}; Tensor b1rconv1sf{}; Tensor b1ract1fs{}; Tensor b1add0sf{};
	Tensor b1act0fs{}; Tensor b1rconv3sf{}; Tensor b1ract2fs{}; Tensor b1rconv4sf{}; Tensor b1ract3fs{}; Tensor b1add1sf{};
	Tensor b1act1fs{}; Tensor b1rconv6sf{}; Tensor b1ract4fs{}; Tensor b1rconv7sf{}; Tensor b1ract5fs{}; Tensor b1add2sf{};

	Tensor b1act2fs{};Tensor b2rconv0sf{}; Tensor b2ract0fs{}; Tensor b2rconv1sf{}; Tensor b2ract1fs{}; Tensor b2add0sf{};
	Tensor b2act0fs{}; Tensor b2rconv3sf{}; Tensor b2ract2fs{}; Tensor b2rconv4sf{}; Tensor b2ract3fs{}; Tensor b2add1sf{};
	Tensor b2act1fs{}; Tensor b2rconv6sf{}; Tensor b2ract4fs{}; Tensor b2rconv7sf{}; Tensor b2ract5fs{}; Tensor b2add2sf{};
	Tensor b2act2fs{}; Tensor b2rconv9sf{}; Tensor b2ract6fs{}; Tensor b2rconv10sf{}; Tensor b2ract7fs{}; Tensor b2add3sf{};

	Tensor b2act3fs{};Tensor b3rconv0sf{}; Tensor b3ract0fs{}; Tensor b3rconv1sf{}; Tensor b3ract1fs{}; Tensor b3add0sf{};
	Tensor b3act0fs{}; Tensor b3rconv3sf{}; Tensor b3ract2fs{}; Tensor b3rconv4sf{}; Tensor b3ract3fs{}; Tensor b3add1sf{};
	Tensor b3act1fs{}; Tensor b3rconv6sf{}; Tensor b3ract4fs{}; Tensor b3rconv7sf{}; Tensor b3ract5fs{}; Tensor b3add2sf{};
	Tensor b3act2fs{}; Tensor b3rconv9sf{}; Tensor b3ract6fs{}; Tensor b3rconv10sf{}; Tensor b3ract7fs{}; Tensor b3add3sf{};
	Tensor b3act3fs{}; Tensor b3rconv12sf{}; Tensor b3ract8fs{}; Tensor b3rconv13sf{}; Tensor b3ract9fs{}; Tensor b3add4sf{};
	Tensor b3act4fs{}; Tensor b3rconv15sf{}; Tensor b3ract10fs{}; Tensor b3rconv16sf{}; Tensor b3ract11fs{}; Tensor b3add5sf{};

	Tensor b3act5fs{};Tensor b4rconv0sf{}; Tensor b4ract0fs{}; Tensor b4rconv1sf{}; Tensor b4ract1fs{}; Tensor b4add0sf{};
	Tensor b4act0fs{}; Tensor b4rconv3sf{}; Tensor b4ract2fs{}; Tensor b4rconv4sf{}; Tensor b4ract3fs{};Tensor b4add1sf{};
	Tensor b4act1fs{}; Tensor b4rconv6sf{}; Tensor b4ract4fs{}; Tensor b4rconv7sf{}; Tensor b4ract5fs{}; Tensor b4add2sf{};

	Tensor pool1fs{}; Tensor conv1sf{};

	Tensor test1{}; Tensor test2{}; Tensor test{}; Tensor out_block1_add02{};


	/*Layer*/
	NEABMConvolutionLayer con0{}; NEActivationLayer act0{}; NEPoolingLayer pool0{};
	/*block1*/
	NEABMConvolutionLayer  block1r_con0{};   NEActivationLayer  block1r_act0{};
	NEABMConvolutionLayer  block1r_con1{};  NEActivationLayer  block1r_act1{};
	NEABMConvolutionLayer  block1r_con2{};   NEABMConvolutionLayer block1l_con0{}; 
	NEFPAdditionLayer  block1_add0{}; NEActivationLayer  block1_act0{};


	NEABMConvolutionLayer  block1r_con3{}; NEActivationLayer  block1r_act2{};
	NEABMConvolutionLayer  block1r_con4{};   NEActivationLayer  block1r_act3{};
	NEABMConvolutionLayer  block1r_con5{}; 
	NEFPAdditionLayer  block1_add1{}; NEActivationLayer  block1_act1{};

	NEABMConvolutionLayer  block1r_con6{}; NEActivationLayer  block1r_act4{};
	NEABMConvolutionLayer  block1r_con7{};   NEActivationLayer  block1r_act5{};
	NEABMConvolutionLayer  block1r_con8{};  
	NEFPAdditionLayer  block1_add2{}; NEActivationLayer  block1_act2{};
	/*block2*/
	NEABMConvolutionLayer  block2r_con0{};  NEActivationLayer  block2r_act0{};
	NEABMConvolutionLayer  block2r_con1{};  NEActivationLayer  block2r_act1{};
	NEABMConvolutionLayer  block2r_con2{};  NEABMConvolutionLayer block2l_con0{}; 
	NEFPAdditionLayer  block2_add0{}; NEActivationLayer  block2_act0{};

	NEABMConvolutionLayer  block2r_con3{};   NEActivationLayer  block2r_act2{};
	NEABMConvolutionLayer  block2r_con4{};  NEActivationLayer  block2r_act3{};
	NEABMConvolutionLayer  block2r_con5{}; 
	NEFPAdditionLayer  block2_add1{}; NEActivationLayer  block2_act1{};

	NEABMConvolutionLayer  block2r_con6{}; NEActivationLayer  block2r_act4{};
	NEABMConvolutionLayer  block2r_con7{};  NEActivationLayer  block2r_act5{};
	NEABMConvolutionLayer  block2r_con8{}; 
	NEFPAdditionLayer  block2_add2{}; NEActivationLayer  block2_act2{};

	NEABMConvolutionLayer  block2r_con9{};   NEActivationLayer  block2r_act6{};
	NEABMConvolutionLayer  block2r_con10{}; NEActivationLayer  block2r_act7{};
	NEABMConvolutionLayer  block2r_con11{};  
	NEFPAdditionLayer  block2_add3{}; NEActivationLayer  block2_act3{};
	/*block3*/
	NEABMConvolutionLayer  block3r_con0{};  NEActivationLayer  block3r_act0{};
	NEABMConvolutionLayer  block3r_con1{};  NEActivationLayer  block3r_act1{};
	NEABMConvolutionLayer  block3r_con2{};  NEABMConvolutionLayer block3l_con0{}; 
	NEFPAdditionLayer  block3_add0{}; NEActivationLayer  block3_act0{};

	NEABMConvolutionLayer  block3r_con3{}; NEActivationLayer  block3r_act2{};
	NEABMConvolutionLayer  block3r_con4{};   NEActivationLayer  block3r_act3{};
	NEABMConvolutionLayer  block3r_con5{}; 
	NEFPAdditionLayer  block3_add1{}; NEActivationLayer  block3_act1{};

	NEABMConvolutionLayer  block3r_con6{};NEActivationLayer  block3r_act4{};
	NEABMConvolutionLayer  block3r_con7{};  NEActivationLayer  block3r_act5{};
	NEABMConvolutionLayer  block3r_con8{};  
	NEFPAdditionLayer  block3_add2{}; NEActivationLayer  block3_act2{};

	NEABMConvolutionLayer  block3r_con9{};  NEActivationLayer  block3r_act6{};
	NEABMConvolutionLayer  block3r_con10{}; NEActivationLayer  block3r_act7{};
	NEABMConvolutionLayer  block3r_con11{}; 
	NEFPAdditionLayer  block3_add3{}; NEActivationLayer  block3_act3{};

	NEABMConvolutionLayer  block3r_con12{};  NEActivationLayer  block3r_act8{};
	NEABMConvolutionLayer  block3r_con13{}; NEActivationLayer  block3r_act9{};
	NEABMConvolutionLayer  block3r_con14{}; 
	NEFPAdditionLayer  block3_add4{}; NEActivationLayer  block3_act4{};

	NEABMConvolutionLayer  block3r_con15{};  NEActivationLayer  block3r_act10{};
	NEABMConvolutionLayer  block3r_con16{};   NEActivationLayer  block3r_act11{};
	NEABMConvolutionLayer  block3r_con17{}; 
	NEFPAdditionLayer  block3_add5{}; NEActivationLayer  block3_act5{};
	/*block4*/
	NEABMConvolutionLayer  block4r_con0{};  NEActivationLayer  block4r_act0{};
	NEABMConvolutionLayer  block4r_con1{}; NEActivationLayer  block4r_act1{};
	NEABMConvolutionLayer  block4r_con2{}; NEABMConvolutionLayer block4l_con0{}; 
	NEFPAdditionLayer  block4_add0{}; NEActivationLayer  block4_act0{};

	NEABMConvolutionLayer  block4r_con3{}; NEActivationLayer  block4r_act2{};
	NEABMConvolutionLayer  block4r_con4{}; NEActivationLayer  block4r_act3{};
	NEABMConvolutionLayer  block4r_con5{};
	NEFPAdditionLayer  block4_add1{}; NEActivationLayer  block4_act1{};


	NEABMConvolutionLayer  block4r_con6{};  NEActivationLayer  block4r_act4{};
	NEABMConvolutionLayer  block4r_con7{};NEActivationLayer  block4r_act5{};
	NEABMConvolutionLayer  block4r_con8{}; 
	NEFPAdditionLayer  block4_add2{}; NEActivationLayer  block4_act2{};


	NEPoolingLayer pool1{}; NEABMConvolutionLayer con1{}; NEFlattenLayer flatten{}; NESoftmaxLayer softmax{};

	NES8toF32Layer lconv0sf{}; NEF32toS8Layer lpool0fs{};

	NES8toF32Layer lb1rconv0sf{}; NEF32toS8Layer lb1ract0fs{}; NES8toF32Layer lb1rconv1sf{}; NEF32toS8Layer lb1ract1fs{};NES8toF32Layer lb1add0sf{};
	NEF32toS8Layer lb1act0fs{}; NES8toF32Layer lb1rconv3sf{}; NEF32toS8Layer lb1ract2fs{}; NES8toF32Layer lb1rconv4sf{}; NEF32toS8Layer lb1ract3fs{}; NES8toF32Layer lb1add1sf{};
	NEF32toS8Layer lb1act1fs{}; NES8toF32Layer lb1rconv6sf{}; NEF32toS8Layer lb1ract4fs{}; NES8toF32Layer lb1rconv7sf{}; NEF32toS8Layer lb1ract5fs{}; NES8toF32Layer lb1add2sf{};

	NEF32toS8Layer lb1act2fs{};NES8toF32Layer lb2rconv0sf{}; NEF32toS8Layer lb2ract0fs{}; NES8toF32Layer lb2rconv1sf{}; NEF32toS8Layer lb2ract1fs{}; NES8toF32Layer lb2add0sf{};
	NEF32toS8Layer lb2act0fs{}; NES8toF32Layer lb2rconv3sf{}; NEF32toS8Layer lb2ract2fs{}; NES8toF32Layer lb2rconv4sf{}; NEF32toS8Layer lb2ract3fs{}; NES8toF32Layer lb2add1sf{};
	NEF32toS8Layer lb2act1fs{}; NES8toF32Layer lb2rconv6sf{}; NEF32toS8Layer lb2ract4fs{}; NES8toF32Layer lb2rconv7sf{}; NEF32toS8Layer lb2ract5fs{}; NES8toF32Layer lb2add2sf{};
	NEF32toS8Layer lb2act2fs{}; NES8toF32Layer lb2rconv9sf{}; NEF32toS8Layer lb2ract6fs{}; NES8toF32Layer lb2rconv10sf{}; NEF32toS8Layer lb2ract7fs{};NES8toF32Layer lb2add3sf{};

	NEF32toS8Layer lb2act3fs{};NES8toF32Layer lb3rconv0sf{}; NEF32toS8Layer lb3ract0fs{}; NES8toF32Layer lb3rconv1sf{}; NEF32toS8Layer lb3ract1fs{}; NES8toF32Layer lb3add0sf{};
	NEF32toS8Layer lb3act0fs{}; NES8toF32Layer lb3rconv3sf{}; NEF32toS8Layer lb3ract2fs{}; NES8toF32Layer lb3rconv4sf{}; NEF32toS8Layer lb3ract3fs{}; NES8toF32Layer lb3add1sf{};
	NEF32toS8Layer lb3act1fs{}; NES8toF32Layer lb3rconv6sf{}; NEF32toS8Layer lb3ract4fs{}; NES8toF32Layer lb3rconv7sf{}; NEF32toS8Layer lb3ract5fs{};NES8toF32Layer lb3add2sf{};
	NEF32toS8Layer lb3act2fs{}; NES8toF32Layer lb3rconv9sf{}; NEF32toS8Layer lb3ract6fs{}; NES8toF32Layer lb3rconv10sf{}; NEF32toS8Layer lb3ract7fs{};NES8toF32Layer lb3add3sf{};
	NEF32toS8Layer lb3act3fs{}; NES8toF32Layer lb3rconv12sf{}; NEF32toS8Layer lb3ract8fs{}; NES8toF32Layer lb3rconv13sf{}; NEF32toS8Layer lb3ract9fs{}; NES8toF32Layer lb3add4sf{};
	NEF32toS8Layer lb3act4fs{}; NES8toF32Layer lb3rconv15sf{}; NEF32toS8Layer lb3ract10fs{}; NES8toF32Layer lb3rconv16sf{}; NEF32toS8Layer lb3ract11fs{};NES8toF32Layer lb3add5sf{};

	NEF32toS8Layer lb3act5fs{};NES8toF32Layer lb4rconv0sf{}; NEF32toS8Layer lb4ract0fs{}; NES8toF32Layer lb4rconv1sf{}; NEF32toS8Layer lb4ract1fs{};NES8toF32Layer lb4add0sf{};
	NEF32toS8Layer lb4act0fs{}; NES8toF32Layer lb4rconv3sf{}; NEF32toS8Layer lb4ract2fs{}; NES8toF32Layer lb4rconv4sf{}; NEF32toS8Layer lb4ract3fs{}; NES8toF32Layer lb4add1sf{};
	NEF32toS8Layer lb4act1fs{}; NES8toF32Layer lb4rconv6sf{}; NEF32toS8Layer lb4ract4fs{}; NES8toF32Layer lb4rconv7sf{}; NEF32toS8Layer lb4ract5fs{}; NES8toF32Layer lb4add2sf{};

	NEF32toS8Layer lpool1fs{}; NES8toF32Layer lconv1sf{};

	NES8toF32Layer testsf1{}; NES8toF32Layer testsf2{}; NEF32toS8Layer testfs{};NEFP2AdditionLayer block1_add02{};


	
	ConvertPolicy A{};

};/*end of class*/
int main(int argc, char **argv)
{
	return utils::run_example<NEONRESNETExample>(argc, argv);
}
