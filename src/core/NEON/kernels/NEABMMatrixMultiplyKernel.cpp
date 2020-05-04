#include "arm_compute/core/NEON/kernels/NEABMMatrixMultiplyKernel.h"
#include "arm_compute/core/AccessWindowStatic.h"
#include "arm_compute/core/CPP/Validate.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/IAccessWindow.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/TensorInfo.h"

#include "arm_compute/core/Utils.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Window.h"
#include "arm_compute/core/utils/helpers/float_ops.h"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"
#include "arm_compute/core/NEON/NEFixedPoint.h"
#include "arm_compute/core/utils/helpers/float_ops.h"



#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <cmath>
#define num_processed_kernels 4

using namespace arm_compute;

namespace
{

inline Status validate_arguments(const ITensorInfo *input, const ITensorInfo *value, const ITensorInfo *quantity, const ITensorInfo *location,
                    const ITensorInfo *bias, const ITensorInfo *output, float alpha,const GEMMReshapeInfo &reshape_info)

{
        return Status{};
}

inline int8_t judge_overflow(float ope)
{
        int8_t real_ope=0;
        if(ope>127){
                real_ope=127;
        }
        else if(ope<-128){
                real_ope=-128;
        }
        else{
                real_ope=(int8_t)ope;
        }
        return real_ope;
}


inline std::pair<Status,Window> validate_and_configure_window(ITensorInfo *input, ITensorInfo *output)

{
        Window window=calculate_max_window(*input,Steps());
        if(input->dimension(1)<num_processed_kernels)
        {
                window.set(Window::DimY, Window::Dimension(0, input->dimension(1),input->dimension(1)));   
        }
        else{
                window.set(Window::DimY, Window::Dimension(0, input->dimension(1),num_processed_kernels));     
        }
        if(input->dimension(0)<8)
        {
                window.set(Window::DimX, Window::Dimension(0, input->dimension(0),input->dimension(0)));  
        }
        else{
                window.set(Window::DimX, Window::Dimension(0, input->dimension(0),8));         
        }     
        window.set(Window::DimZ, Window::Dimension(0, 1, 1));
        return std::make_pair(Status{}, window);
}

}

NEABMMatrixMultiplyKernel::NEABMMatrixMultiplyKernel()
        :_input(nullptr),_value(nullptr),_quantity(nullptr),_location(nullptr),_bias(nullptr), _output(nullptr)
         ,_weights_precision(0),_bias_precision(0),_input_precision(0),_output_precision(0),_a_mul_value(0),_b_mul_value(0),_x_index(0),_y_index(0),_z_index(0),_y_index_bit(0),_x_index_bit(0)
        ,_alpha(1.0f),_kernel_width(0),_kernel_area(0),_convolution_last(0),_kernel_last(0)
{
}
void NEABMMatrixMultiplyKernel::configure(const ITensor *input, ITensor *value, ITensor *quantity, ITensor *location,const ITensor *bias, 
                    ITensor *output,unsigned int precision[],unsigned int index[],float alpha, const GEMMReshapeInfo &reshape_info ,int kernel_width, int kernel_area, long long int total_number )
                    {
                                 ARM_COMPUTE_ERROR_ON_NULLPTR(input,output);
        long long int convolution_steps=output->info()->dimension(0);
        long long int kernel_numbers=output->info()->dimension(1);
        _input=input;
        _value=value;
        _quantity=quantity;
        _location=location;
        _bias=(bias!=nullptr)?bias:nullptr;
        _output=output;
        _weights_precision=precision[0];
        _bias_precision=precision[1];
        _input_precision=precision[2];
        _output_precision=precision[3];
        int temp1=_output_precision-_weights_precision-_input_precision;
        _a_mul_value=pow(2,temp1);
        int temp2=_output_precision-_bias_precision;
        _b_mul_value=pow(2,temp2);
        _x_index=index[0];
        _y_index=index[1];
        _z_index=index[2];
        _y_index_bit=(unsigned int )(pow(2,_y_index))-1;
        _x_index_bit=(unsigned int)(pow(2,_x_index))-1;
        _alpha=alpha;
        _kernel_width=kernel_width;
        _kernel_area=kernel_area;
        _convolution_last=convolution_steps%8;
        _kernel_last=kernel_numbers%num_processed_kernels;
        auto win_config = validate_and_configure_window(output->info(),output->info());
        ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
        INEKernel::configure(win_config.second);
}



Status NEABMMatrixMultiplyKernel::validate(const ITensorInfo *input,const ITensorInfo *value,  const ITensorInfo *quantity, const ITensorInfo *location, const ITensorInfo *bias,
                    const ITensorInfo *output, float alpha,const GEMMReshapeInfo &reshape_info)

{
        /*ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(input, quantity, location, alpha, reshape_info));*/
        /*ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(input->clone().get(), output->clone().get()).first);*/
        return Status{};
}



void NEABMMatrixMultiplyKernel::run(const Window &window, const ThreadInfo &info)
{
ARM_COMPUTE_UNUSED(info);
ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

Window wout(window);
wout.set(Window::DimX, Window::Dimension(0, 0, 0));
wout.set(Window::DimY, Window::Dimension(0, 0, 0));
wout.set(Window::DimZ, Window::Dimension(0, 0, 0));

Iterator out(_output,wout);

int16x8_t max_value=vdupq_n_s16(127);
int16x8_t min_value=vdupq_n_s16(-128);

//value 三维：z表示卷积核的个数；value_stride_z：一个卷积核所占的字节数（前两个维度表示一个卷积核）
uint8_t *value_buffer=_value->buffer();
const unsigned int value_stride_z=_value->info()->strides_in_bytes().z();

uint8_t *quantity_buffer=_quantity->buffer();
const unsigned int quantity_stride_z=_quantity->info()->strides_in_bytes().z();

uint8_t *location_buffer=_location->buffer();
const unsigned int location_stride_z=_location->info()->strides_in_bytes().z();

uint8_t *bias_buffer=(_bias!=nullptr)?_bias->buffer():nullptr;

float32x4_t a_mul=vdupq_n_f32(_a_mul_value);
float32x4_t b_mul=vdupq_n_f32(_b_mul_value);

execute_window_loop(window, [&](const Coordinates & id){
        size_t x_dimension=id[0], y_dimension=id[1]; 

        uint8_t *temp_input_ptr=_input->buffer();                            
        size_t input_ptr_offset=_input->info()->offset_element_in_bytes(Coordinates(0,x_dimension/8,0));

        uint8_t *temp_output_ptr=_output->buffer();
        int output_x_dimension=x_dimension;
        size_t output_ptr_offset=0;
        output_ptr_offset=_output->info()->offset_element_in_bytes(Coordinates(output_x_dimension,0,0));

        int start_kernel=0, end_kernel=0;

        if(_kernel_last==0)
        { 
                //_kernel_last=kernel_numbers%num_processed_kernels;
                // kernel数量能够被4除尽
                start_kernel=y_dimension;
                end_kernel=y_dimension+num_processed_kernels;
        }

        else{
                //判断是不是到了最后一个子窗口
                if(y_dimension/num_processed_kernels==_output->info()->dimension(1)/num_processed_kernels)
                {
                        start_kernel=y_dimension;
                        end_kernel=_output->info()->dimension(1);
                }
                else{
                        start_kernel=y_dimension;
                        end_kernel=y_dimension+num_processed_kernels;
                }
        }

        for(int j=start_kernel; j<end_kernel; j++)
        {  
                int16x8_t acc=vdupq_n_s16(0.f);

                //第j个kernel开始的地址
                uint8_t *value_start_ptr = value_buffer+j*value_stride_z;  
                uint8_t *quantity_start_ptr = quantity_buffer+j*quantity_stride_z;
                uint8_t *location_start_ptr = location_buffer+j*location_stride_z;

                int8_t *value_ptr = reinterpret_cast<int8_t *>(value_start_ptr);
                uint16_t *quantity_ptr = reinterpret_cast<uint16_t *>(quantity_start_ptr);
                uint16_t *location_ptr=reinterpret_cast<uint16_t *>(location_start_ptr);

                while((*value_ptr)!=0)
                {
                        int16x8_t onesum=vdupq_n_s16(0.f);

                        int8_t temp_value=*value_ptr; 
                        unsigned short temp_quantity=*quantity_ptr;

                        value_ptr++;
                        quantity_ptr++;

                        while(temp_quantity--)
                        {
                                unsigned short result=(*location_ptr);
                                unsigned short _z=(result>>(_x_index+_y_index));
                                unsigned short _y=((result>>_x_index)&_y_index_bit);
                                // 保留最后的位数
                                unsigned short _x=(result&_x_index_bit);
                                location_ptr++;
                                //_x_index_bit=(unsigned int)(pow(2,_x_index))-1;
                                
                                int16x8_t input_value=vdupq_n_s16(0.f); 

                                /*
                                temp_input_ptr为input的首地址,input_ptr_offset为需要在输入的第几行中找
                                
                                 */
                                int16_t *input_ptr = reinterpret_cast<int16_t *>(temp_input_ptr+input_ptr_offset+(8*2)*((int)_x*1+(int)_y*_kernel_width+(int)_z*_kernel_area));   

                                input_value=vld1q_s16(input_ptr);
                                onesum=vaddq_s16(onesum,input_value);
                        } 
                        int16_t real_value=(int16_t)(temp_value);     
                        int16x8_t onevalue = vdupq_n_s16(real_value);
                        acc=vaddq_s16(acc,vmulq_s16(onesum,onevalue)); 
                        //结束一个权重值所有的计算         
                } // end while 结束一个卷积核所有的计算

                //得到的acc 里面存着对于一个kernel的8次卷积操作

                if(_convolution_last==0)
                {
                        int8_t bias_value=0;
                        if(bias_buffer!=nullptr){
                                int8_t *bias_ptr= reinterpret_cast<int8_t *>(bias_buffer+(int)j);
                                bias_value=(*bias_ptr);
                        }
                        int16_t real_bias=(int16_t)bias_value;
                        int16x8_t acc_bias=vdupq_n_s16(real_bias);

                        int16x8_t ope_a=acc;
                        int16x8_t ope_b=acc_bias;

                        //考虑精度
                        int16x4_t ope_a_high=vget_high_s16(ope_a);
                        int16x4_t ope_a_low=vget_low_s16(ope_a);
                        int16x4_t ope_b_high=vget_high_s16(ope_b);
                        int16x4_t ope_b_low=vget_low_s16(ope_b);

                        int32x4_t ope_a_high_long=vmovl_s16(ope_a_high);
                        int32x4_t ope_a_low_long=vmovl_s16(ope_a_low);
                        int32x4_t ope_b_high_long=vmovl_s16(ope_b_high);
                        int32x4_t ope_b_low_long=vmovl_s16(ope_b_low);

                        float32x4_t a_high=vcvtq_f32_s32(ope_a_high_long);
                        float32x4_t a_low=vcvtq_f32_s32(ope_a_low_long);
                        float32x4_t b_high=vcvtq_f32_s32(ope_b_high_long);
                        float32x4_t b_low=vcvtq_f32_s32(ope_b_low_long);


                        a_high=vmulq_f32(a_high,a_mul); //a_mul浮点数
                        a_low=vmulq_f32(a_low,a_mul);
                        b_high=vmulq_f32(b_high,b_mul);
                        b_low=vmulq_f32(b_low,b_mul);

                        float32x4_t result1=vaddq_f32(a_high,b_high);  //加上bias
                        float32x4_t result2=vaddq_f32(a_low,b_low);


                        int32x4_t result1_s32=vcvtq_s32_f32(result1);
                        int32x4_t result2_s32=vcvtq_s32_f32(result2);

                        int16x4_t result1_s16=vmovn_s32(result1_s32);
                        int16x4_t result2_s16=vmovn_s32(result2_s32);

                        int16x8_t result=vcombine_s16(result2_s16,result1_s16);

                        // 也是处理overflow的方法 取127 -  -128的最大值 最小值
                        result=vminq_s16(max_value,vmaxq_s16(min_value,result));

                        int8x8_t real_result=vmovn_s16(result);

                        int8_t *out_addr=reinterpret_cast<int8_t*>(_output->buffer()+j*_output->info()->dimension(0)+output_ptr_offset);

                        vst1_s8(out_addr,real_result);

                }

                else{
                        if(x_dimension/8==_output->info()->dimension(0)/8)
                        {
                                //到了最后一次了
                                int16_t a[8];
                                vst1q_s16(a,acc);
                                float temp_a=0,temp_b=0;

                                int8_t bias_value=0;
                                if(bias_buffer!=nullptr){
                                        int8_t *bias_ptr= reinterpret_cast<int8_t *>(bias_buffer+(int)j);
                                        bias_value=(*bias_ptr);
                                }
                                temp_b=(float)bias_value;

                                if(_convolution_last==1){

                                        int8_t *myout_ptr3= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+j*_output->info()->strides_in_bytes().y());

                                        temp_a=(float)(a[0]);
                                        *myout_ptr3=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);

                                }

                                else if(_convolution_last==2){

                                        int8_t *myout_ptr3= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+j*_output->info()->strides_in_bytes().y());
                                        temp_a=(float)(a[0]);
                                        *myout_ptr3=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);

                                        temp_a=(float)(a[1]);
                                        int8_t *myout_ptr31= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+1+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr31=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                }

                                else if(_convolution_last==3){

                                        int8_t *myout_ptr3= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+j*_output->info()->strides_in_bytes().y());
                                        temp_a=(float)(a[0]);
                                        *myout_ptr3=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[1]);

                                        int8_t *myout_ptr31= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+1+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr31=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[2]);

                                        int8_t *myout_ptr32= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+2+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr32=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                }
                                else if(_convolution_last==4){

                                        int8_t *myout_ptr3= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+j*_output->info()->strides_in_bytes().y());
                                        temp_a=(float)(a[0]);
                                        *myout_ptr3=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[1]);

                                        int8_t *myout_ptr31= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+1+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr31=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[2]);

                                        int8_t *myout_ptr32= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+2+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr32=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[3]);

                                        int8_t *myout_ptr33= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+3+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr33=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                }
                                else if(_convolution_last==5){

                                        int8_t *myout_ptr3= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+j*_output->info()->strides_in_bytes().y());
                                        temp_a=(float)(a[0]);
                                        *myout_ptr3=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[1]);

                                        int8_t *myout_ptr31= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+1+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr31=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[2]);

                                        int8_t *myout_ptr32= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+2+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr32=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[3]);

                                        int8_t *myout_ptr33= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+3+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr33=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[4]);

                                        int8_t *myout_ptr34= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+4+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr34=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                }
                                else if(_convolution_last==6){
                                        int8_t *myout_ptr3= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+j*_output->info()->strides_in_bytes().y());
                                        temp_a=(float)(a[0]);
                                        *myout_ptr3=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[1]);

                                        int8_t *myout_ptr31= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+1+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr31=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[2]);

                                        int8_t *myout_ptr32= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+2+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr32=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[3]);

                                        int8_t *myout_ptr33= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+3+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr33=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[4]);

                                        int8_t *myout_ptr34= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+4+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr34=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[5]);
                                        int8_t *myout_ptr35= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+5+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr35=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);

                                }
                                else if(_convolution_last==7){

                                        int8_t *myout_ptr3= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+j*_output->info()->strides_in_bytes().y());
                                        temp_a=(float)(a[0]);
                                        *myout_ptr3=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[1]);

                                        int8_t *myout_ptr31= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+1+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr31=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[2]);

                                        int8_t *myout_ptr32= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+2+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr32=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[3]);

                                        int8_t *myout_ptr33= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+3+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr33=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[4]);

                                        int8_t *myout_ptr34= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+4+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr34=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[5]);
                                        int8_t *myout_ptr35= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+5+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr35=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                        temp_a=(float)(a[6]);


                                        int8_t *myout_ptr36= reinterpret_cast<int8_t *>(temp_output_ptr+(int)output_ptr_offset+6+j*_output->info()->strides_in_bytes().y());

                                        *myout_ptr36=judge_overflow(temp_a*_a_mul_value+temp_b*_b_mul_value);
                                }

                        }

                        else{
                                int8_t bias_value=0;
                                if(bias_buffer!=nullptr){
                                        int8_t *bias_ptr= reinterpret_cast<int8_t *>(bias_buffer+(int)j);
                                        bias_value=(*bias_ptr);
                                }
                                int16_t real_bias=(int16_t)bias_value;
                                int16x8_t acc_bias=vdupq_n_s16(real_bias);

                                int16x8_t ope_a=acc;int16x8_t ope_b=acc_bias;

                                int16x4_t ope_a_high=vget_high_s16(ope_a);
                                int16x4_t ope_a_low=vget_low_s16(ope_a);
                                int16x4_t ope_b_high=vget_high_s16(ope_b);
                                int16x4_t ope_b_low=vget_low_s16(ope_b);

                                int32x4_t ope_a_high_long=vmovl_s16(ope_a_high);
                                int32x4_t ope_a_low_long=vmovl_s16(ope_a_low);
                                int32x4_t ope_b_high_long=vmovl_s16(ope_b_high);
                                int32x4_t ope_b_low_long=vmovl_s16(ope_b_low);

                                float32x4_t a_high=vcvtq_f32_s32(ope_a_high_long);
                                float32x4_t a_low=vcvtq_f32_s32(ope_a_low_long);
                                float32x4_t b_high=vcvtq_f32_s32(ope_b_high_long);
                                float32x4_t b_low=vcvtq_f32_s32(ope_b_low_long);


                                a_high=vmulq_f32(a_high,a_mul);
                                a_low=vmulq_f32(a_low,a_mul);
                                b_high=vmulq_f32(b_high,b_mul);
                                b_low=vmulq_f32(b_low,b_mul);

                                float32x4_t result1=vaddq_f32(a_high,b_high);
                                float32x4_t result2=vaddq_f32(a_low,b_low);


                                int32x4_t result1_s32=vcvtq_s32_f32(result1);
                                int32x4_t result2_s32=vcvtq_s32_f32(result2);

                                int16x4_t result1_s16=vmovn_s32(result1_s32);
                                int16x4_t result2_s16=vmovn_s32(result2_s32);

                                int16x8_t result=vcombine_s16(result2_s16,result1_s16);

                                result=vminq_s16(max_value,vmaxq_s16(min_value,result));
                                int8x8_t real_result=vmovn_s16(result);
                                int8_t *out_addr=reinterpret_cast<int8_t*>(_output->buffer()+j*_output->info()->dimension(0)+output_ptr_offset);
                                vst1_s8(out_addr,real_result);

                        }
                }

        } //end 结束了这几个kernel
},out);

}




