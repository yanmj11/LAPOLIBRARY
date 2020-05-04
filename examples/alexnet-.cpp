#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/CPP/CPPScheduler.h"
#include "arm_compute/runtime/NEON/NEScheduler.h"


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
using namespace arm_compute;
using namespace utils;


class NEONALEXExample : public Example
{
public:

    bool do_setup(int argc, char **argv) override
    {
        /*---------------[init_model_alex]-----------------*/

        /* [Initialize tensors] */
        constexpr unsigned int width_src_image  = 227;
        constexpr unsigned int height_src_image = 227;
        constexpr unsigned int ifm_src_img      = 3;

        const TensorShape src_shape(width_src_image, height_src_image, ifm_src_img);
        src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));



        ///////////////  layer 1 ok
        // Initialize tensors of conv0
        constexpr unsigned int kernel_x_conv0 = 11;
        constexpr unsigned int kernel_y_conv0 = 11;
        constexpr unsigned int ofm_conv0      = 96;
        constexpr unsigned int out_x_conv0    = 55;
        constexpr unsigned int out_y_conv0    = 55;
        const TensorShape weights_shape_conv0(kernel_x_conv0, kernel_y_conv0, src_shape.z(), ofm_conv0);
        const TensorShape biases_shape_conv0(weights_shape_conv0[3]);
        const TensorShape out_shape_conv0(out_x_conv0, out_y_conv0, weights_shape_conv0[3]);
        weights0.allocator()->init(TensorInfo(weights_shape_conv0, 1, DataType::F32));
        biases0.allocator()->init(TensorInfo(biases_shape_conv0, 1, DataType::F32));
        out_conv0.allocator()->init(TensorInfo(out_shape_conv0, 1, DataType::F32));
        // Initialize tensor of act0
        out_act0.allocator()->init(TensorInfo(out_shape_conv0, 1, DataType::F32));
        //////norm0
        out_norm0.allocator()->init(TensorInfo(out_shape_conv0, 1, DataType::F32));
        // Initialize tensor of pool0
        TensorShape out_shape_pool0 = out_shape_conv0;
        out_shape_pool0.set(0, out_shape_pool0.x() / 2); //2 is stride
        out_shape_pool0.set(1, out_shape_pool0.y() / 2);
        out_pool0.allocator()->init(TensorInfo(out_shape_pool0, 1, DataType::F32));


        ///////////////  layer 2 ok
        // Initialize tensors of conv1
        constexpr unsigned int kernel_x_conv1 = 5;
        constexpr unsigned int kernel_y_conv1 = 5;
        constexpr unsigned int ofm_conv1      = 256;
        constexpr unsigned int out_x_conv1    = 27;
        constexpr unsigned int out_y_conv1    = 27;        
        const TensorShape weights_shape_conv1(kernel_x_conv1, kernel_y_conv1, out_shape_pool0.z(), ofm_conv1);
        const TensorShape biases_shape_conv1(weights_shape_conv1[3]);
        const TensorShape out_shape_conv1(out_x_conv1, out_y_conv1, weights_shape_conv1[3]);
        weights1.allocator()->init(TensorInfo(weights_shape_conv1, 1, DataType::F32));
        biases1.allocator()->init(TensorInfo(biases_shape_conv1, 1, DataType::F32));
        out_conv1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));
        // Initialize tensor of act1
        out_act1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));
        //////norm1
        out_norm1.allocator()->init(TensorInfo(out_shape_conv1, 1, DataType::F32));
        // Initialize tensor of pool1
        TensorShape out_shape_pool1 = out_shape_conv1;
        out_shape_pool1.set(0, out_shape_pool1.x() / 2);
        out_shape_pool1.set(1, out_shape_pool1.y() / 2);
        out_pool1.allocator()->init(TensorInfo(out_shape_pool1, 1, DataType::F32));


        ///////////////  layer 3
        // Initialize tensors of conv2
        constexpr unsigned int kernel_x_conv2 = 3;
        constexpr unsigned int kernel_y_conv2 = 3;
        constexpr unsigned int ofm_conv2      = 384;
        constexpr unsigned int out_x_conv2    = 13;
        constexpr unsigned int out_y_conv2    = 13;  
        const TensorShape weights_shape_conv2(kernel_x_conv2, kernel_y_conv2, out_shape_pool1.z(), ofm_conv2);
        const TensorShape biases_shape_conv2(weights_shape_conv2[3]);
        const TensorShape out_shape_conv2(out_x_conv2, out_y_conv2, weights_shape_conv2[3]);
        weights2.allocator()->init(TensorInfo(weights_shape_conv2, 1, DataType::F32));
        biases2.allocator()->init(TensorInfo(biases_shape_conv2, 1, DataType::F32));
        out_conv2.allocator()->init(TensorInfo(out_shape_conv2, 1, DataType::F32));
        // Initialize tensor of act2
        out_act2.allocator()->init(TensorInfo(out_shape_conv2, 1, DataType::F32));


        ///////////////  layer 4
        // Initialize tensors of conv3
        constexpr unsigned int kernel_x_conv3 = 3;
        constexpr unsigned int kernel_y_conv3 = 3;
        constexpr unsigned int ofm_conv3      = 384;
        constexpr unsigned int out_x_conv3    = 13;
        constexpr unsigned int out_y_conv3    = 13;
        const TensorShape weights_shape_conv3(kernel_x_conv3, kernel_y_conv3, out_shape_conv2.z(), ofm_conv3);
        const TensorShape biases_shape_conv3(weights_shape_conv3[3]);
        const TensorShape out_shape_conv3(out_x_conv3, out_y_conv3, weights_shape_conv3[3]);
        weights3.allocator()->init(TensorInfo(weights_shape_conv3, 1, DataType::F32 ));
        biases3.allocator()->init(TensorInfo(biases_shape_conv3, 1, DataType::F32));
        out_conv3.allocator()->init(TensorInfo(out_shape_conv3, 1, DataType::F32));
        // Initialize tensor of act3
        out_act3.allocator()->init(TensorInfo(out_shape_conv3, 1, DataType::F32));


        ///////////////  layer 5
        // Initialize tensors of conv4
        constexpr unsigned int kernel_x_conv4 = 3;
        constexpr unsigned int kernel_y_conv4 = 3;
        constexpr unsigned int ofm_conv4      = 256;
        constexpr unsigned int out_x_conv4    = 13;
        constexpr unsigned int out_y_conv4    = 13;
        const TensorShape weights_shape_conv4(kernel_x_conv4, kernel_y_conv4, out_shape_conv3.z(), ofm_conv4);
        const TensorShape biases_shape_conv4(weights_shape_conv4[3]);
        const TensorShape out_shape_conv4(out_x_conv4, out_y_conv4, weights_shape_conv4[3]);
        weights4.allocator()->init(TensorInfo(weights_shape_conv4, 1, DataType::F32));
        biases4.allocator()->init(TensorInfo(biases_shape_conv4, 1, DataType::F32));
        out_conv4.allocator()->init(TensorInfo(out_shape_conv4, 1, DataType::F32));
        // Initialize tensor of act4
        out_act4.allocator()->init(TensorInfo(out_shape_conv4, 1, DataType::F32));
        // Initialize tensor of pool2
        TensorShape out_shape_pool2 = out_shape_conv4;
        out_shape_pool2.set(0, out_shape_pool2.x() / 2);
        out_shape_pool2.set(1, out_shape_pool2.y() / 2);
        out_pool2.allocator()->init(TensorInfo(out_shape_pool2, 1, DataType::F32));


        ///////////////  layer 6
        // Initialize tensor of fc0
        unsigned int num_labels = 4096;

        const TensorShape weights_shape_fc0(out_shape_pool2.x() * out_shape_pool2.y() * out_shape_pool2.z(), 4096);
        const TensorShape biases_shape_fc0(4096);
        const TensorShape out_shape_fc0(4096);

        weights5.allocator()->init(TensorInfo(weights_shape_fc0, 1, DataType::F32));
        biases5.allocator()->init(TensorInfo(biases_shape_fc0, 1, DataType::F32));
        out_fc0.allocator()->init(TensorInfo(out_shape_fc0, 1, DataType::F32));

        // Initialize tensor of act5
        out_act5.allocator()->init(TensorInfo(out_shape_fc0, 1, DataType::F32));

        ///////////////  layer 7
        // Initialize tensor of fc1
        num_labels = 4096; 

        const TensorShape weights_shape_fc1(out_shape_fc0.x() * out_shape_fc0.y() * out_shape_fc0.z(), 4096);
        const TensorShape biases_shape_fc1(4096);
        const TensorShape out_shape_fc1(4096);

        weights6.allocator()->init(TensorInfo(weights_shape_fc1, 1, DataType::F32));
        biases6.allocator()->init(TensorInfo(biases_shape_fc1, 1, DataType::F32));
        out_fc1.allocator()->init(TensorInfo(out_shape_fc1, 1, DataType::F32));

        // Initialize tensor of act6
        out_act6.allocator()->init(TensorInfo(out_shape_fc1, 1, DataType::F32));

        ///////////////  layer 8
        // Initialize tensor of fc2
        num_labels = 1000;  

        const TensorShape weights_shape_fc2(out_shape_fc1.x() * out_shape_fc1.y() * out_shape_fc1.z(), 1000);
        const TensorShape biases_shape_fc2(1000);
        const TensorShape out_shape_fc2(1000);

        weights7.allocator()->init(TensorInfo(weights_shape_fc2, 1, DataType::F32));
        biases7.allocator()->init(TensorInfo(biases_shape_fc2, 1, DataType::F32));
        out_fc2.allocator()->init(TensorInfo(out_shape_fc2, 1, DataType::F32));

        // Initialize tensor of softmax
        const TensorShape out_shape_softmax(out_shape_fc2.x());
        out_softmax.allocator()->init(TensorInfo(out_shape_softmax, 1, DataType::F32));

        /* -----------------------End: [Initialize tensors] */

        /*-----------------BEGIN:[Configure Functions]--------------*/

        /* [Configure functions] */

        ///layer1
        conv0.configure(&src, &weights0, &biases0, &out_conv0, PadStrideInfo(4, 4, 0, 0));
        act0.configure(&out_conv0, &out_act0, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        norm0.configure(&out_act0, &out_norm0, NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f));
        pool0.configure(&out_norm0, &out_pool0, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)));

        ///layer2
        conv1.configure(&out_pool0, &weights1, &biases1, &out_conv1, PadStrideInfo(1, 1, 2, 2));
        act1.configure(&out_conv1, &out_act1, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        norm1.configure(&out_act1, &out_norm1, NormalizationLayerInfo(NormType::CROSS_MAP, 5, 0.0001f, 0.75f));
        pool1.configure(&out_act1, &out_pool1, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)));

        ///layer3
        conv2.configure(&out_pool1, &weights2, &biases2, &out_conv2, PadStrideInfo(1, 1, 1, 1));
        act2.configure(&out_conv2, &out_act2, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        ///layer4
        conv3.configure(&out_act2, &weights3, &biases3, &out_conv3, PadStrideInfo(1, 1, 1, 1));
        act3.configure(&out_conv3, &out_act3, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));

        ///layer5
        conv4.configure(&out_act3, &weights4, &biases4, &out_conv4, PadStrideInfo(1, 1, 1, 1));
        act4.configure(&out_conv4, &out_act4, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        pool2.configure(&out_act4, &out_pool2, PoolingLayerInfo(PoolingType::MAX, 3, PadStrideInfo(2, 2, 0, 0)));

        //layer6
        fc0.configure(&out_pool2, &weights5, &biases5, &out_fc0);
        act5.configure(&out_fc0, &out_act5, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        //layer7
        fc1.configure(&out_act5, &weights6, &biases6, &out_fc1);
        act6.configure(&out_fc1, &out_act6, ActivationLayerInfo(ActivationLayerInfo::ActivationFunction::RELU));
        //layer8
        fc2.configure(&out_act6, &weights7, &biases7, &out_fc2);
        softmax.configure(&out_fc2, &out_softmax);

        /* -----------------------End: [Configure functions] */

        /*---------------END:[Allocate tensors]---------------*/
        ///layer1
        out_conv0.allocator()->allocate();
        out_act0.allocator()->allocate();
        out_norm0.allocator()->allocate();
        out_pool0.allocator()->allocate();
        ///layer2
        out_conv1.allocator()->allocate();
        out_act1.allocator()->allocate();
        out_norm1.allocator()->allocate();
        out_pool1.allocator()->allocate();
        ///layer3
        out_conv2.allocator()->allocate();
        out_act2.allocator()->allocate();
        ///layer4
        out_conv3.allocator()->allocate();
        out_act3.allocator()->allocate();
        ///layer5
        out_conv4.allocator()->allocate();
        out_act4.allocator()->allocate();
        out_pool2.allocator()->allocate();

        ///layer6
        out_fc0.allocator()->allocate();
        out_act5.allocator()->allocate();
        ///layer7
        out_fc1.allocator()->allocate();
        out_act6.allocator()->allocate();
        ///layer8
        out_fc2.allocator()->allocate();
        out_softmax.allocator()->allocate();
        /* -----------------------End: [ Add tensors to memory manager ] */

        /* [Allocate tensors] */

        // Now that the padding requirements are known we can allocate all tensors
        src.allocator()->allocate();
        weights0.allocator()->allocate();
        weights1.allocator()->allocate();
        weights2.allocator()->allocate();
        weights3.allocator()->allocate();
        weights4.allocator()->allocate();
        weights5.allocator()->allocate();
        weights6.allocator()->allocate();
        weights7.allocator()->allocate();

        biases0.allocator()->allocate();
        biases1.allocator()->allocate();
        biases2.allocator()->allocate();
        biases3.allocator()->allocate();
        biases4.allocator()->allocate();
        biases5.allocator()->allocate();
        biases6.allocator()->allocate();
        biases7.allocator()->allocate();
        /* -----------------------End: [Allocate tensors] */
        return true;

    }

    void do_run() override
    {
            auto tbegin1 = std::chrono::high_resolution_clock::now();
            auto tbegin = std::chrono::high_resolution_clock::now();
            conv0.run();
            act0.run();
            norm0.run();
            pool0.run();

            conv1.run();
            act1.run();
            norm1.run();
            pool1.run();
            
            conv2.run();
            act2.run();
          

            conv3.run();
            act3.run();
           
            conv4.run();
            act4.run();
            pool2.run();
    
            
            fc0.run();
            act5.run();


            fc1.run();
            act6.run();
            

            fc2.run();
            softmax.run();

            auto tend = std::chrono::high_resolution_clock::now();
            double cost0 = std::chrono::duration_cast<std::chrono::duration<double>>(tend - tbegin).count();
            double cost = cost0;
            std::cout <<cost*1000<<std::endl;           

    }

private:
    // The src tensor should contain the input image
    Tensor src{};

    Tensor weights0{},weights1{},weights2{},weights3{},weights4{},weights5{},weights6{},weights7{};
    Tensor biases0{},biases1{},biases2{},biases3{},biases4{},biases5{},biases6{},biases7{};

    Tensor out_conv0{};
    Tensor out_conv1{};
    Tensor out_conv2{};
    Tensor out_conv3{};
    Tensor out_conv4{};

    Tensor out_act0{};
    Tensor out_act1{};
    Tensor out_act2{};
    Tensor out_act3{};
    Tensor out_act4{};
    Tensor out_act5{};
    Tensor out_act6{};

    Tensor out_norm0{};
    Tensor out_norm1{};

    Tensor out_pool0{};
    Tensor out_pool1{};
    Tensor out_pool2{};

    Tensor out_fc0{};
    Tensor out_fc1{};
    Tensor out_fc2{};

    Tensor out_softmax{};


    // Layers
    NEConvolutionLayer          conv0{};
    NEConvolutionLayer          conv1{};
    NEConvolutionLayer          conv2{};
    NEConvolutionLayer          conv3{};
    NEConvolutionLayer          conv4{};

    NEFullyConnectedLayer       fc0{};
    NEFullyConnectedLayer       fc1{};
    NEFullyConnectedLayer       fc2{};

    NESoftmaxLayer              softmax{};

    NENormalizationLayer        norm0{}; 
    NENormalizationLayer        norm1{};


    NEPoolingLayer              pool0{};
    NEPoolingLayer              pool1{};
    NEPoolingLayer              pool2{};

    NEActivationLayer           act0{};
    NEActivationLayer           act1{};
    NEActivationLayer           act2{};
    NEActivationLayer           act3{};
    NEActivationLayer           act4{};
    NEActivationLayer           act5{};
    NEActivationLayer           act6{};

};

int main(int argc, char **argv)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    CPU_SET(0, &cpuset);
    // CPU_SET(1, &cpuset);
    // CPU_SET(2, &cpuset);
    // CPU_SET(3, &cpuset);

    // CPU_SET(4, &cpuset);
    // CPU_SET(5, &cpuset);  
    // CPU_SET(6, &cpuset);
    // CPU_SET(7, &cpuset);

	int e = sched_setaffinity(getpid(), sizeof(cpuset), &cpuset);
	if(e !=0) {
		std::cout << "Error in setting sched_setaffinity \n";
	}
    int num=1;
    CPPScheduler::get().set_num_threads(num);
    std::cout<<"thread="<<CPPScheduler::get().num_threads()<<std::endl<<std::endl;

    return utils::run_example<NEONALEXExample>(argc, argv);
}
