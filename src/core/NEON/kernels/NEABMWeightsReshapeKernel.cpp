#include "arm_compute/core/NEON/kernels/NEABMWeightsReshapeKernel.h"

#include "arm_compute/core/Dimensions.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/ITensor.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/core/Validate.h"
#include "arm_compute/core/Rounding.h"

using namespace arm_compute;

namespace
{
// float keep4numbers(float n)
// {
//     return round(n*10000,RoundingPolicy::TO_ZERO)*0.0001;
// }
inline Status validate_arguments(const ITensorInfo *weights, const ITensorInfo *value,ITensorInfo *quantity, const ITensorInfo *location)
{
    return Status{};
}
std::pair<Status, Window> validate_and_configure_window(ITensorInfo *weights)
{
    Window window = calculate_max_window(*weights, Steps());
    window.set(Window::DimX, Window::Dimension(0, weights->dimension(0), weights->dimension(0)));
    window.set(Window::DimY, Window::Dimension(0, weights->dimension(1), weights->dimension(1)));
    window.set(Window::DimZ, Window::Dimension(0, weights->dimension(2), weights->dimension(2)));

    return std::make_pair(Status{}, window);
}
}//end namespace;

NEABMWeightsReshapeKernel::NEABMWeightsReshapeKernel()
        :_input(nullptr),_value(nullptr),_quantity(nullptr),_location(nullptr),_x_index(0),_y_index(0),_z_index(0),_location_y_size(0)
{
}
void NEABMWeightsReshapeKernel::configure(const ITensor *weights,ITensor *value,ITensor *quantity, ITensor *location,unsigned int index[],int location_y_size)
{
    ARM_COMPUTE_ERROR_ON_NULLPTR(weights);
    ARM_COMPUTE_ERROR_THROW_ON(validate_arguments(weights->info(),value,quantity,location));
    _input=weights;
    _value=value;
    _quantity=quantity;
    _location=location;
    _x_index=index[0];
    _y_index=index[1];
    _z_index=index[2];
    //std::cout<<_x_index<<_y_index<<_z_index<<std::endl;
    _location_y_size=location_y_size;
    auto win_config = validate_and_configure_window(weights->info());
    ARM_COMPUTE_ERROR_THROW_ON(win_config.first);
    INEKernel::configure(win_config.second);
}
Status NEABMWeightsReshapeKernel::validate(const ITensorInfo *weights,const ITensorInfo *value,  const ITensorInfo *quantity, const ITensorInfo *location)
{
    //ARM_COMPUTE_RETURN_ON_ERROR(validate_arguments(weights,value, quantity, location));
   // ARM_COMPUTE_RETURN_ON_ERROR(validate_and_configure_window(weights->clone().get()));

    return Status{};
}

void NEABMWeightsReshapeKernel::run(const Window &window, const ThreadInfo &info)
{
    //printf("NEABMWeightsReshapeKernel run start!\n");
    ARM_COMPUTE_UNUSED(info);
    ARM_COMPUTE_ERROR_ON_UNCONFIGURED_KERNEL(this);
    ARM_COMPUTE_ERROR_ON_INVALID_SUBWINDOW(INEKernel::window(), window);

    //_input means weights_tensor
    const unsigned int kernel_size_x   = _input->info()->dimension(0);               //weights_kernel_x;
    const unsigned int kernel_size_y   = _input->info()->dimension(1);               //weights_kernel_y;
    const unsigned int kernel_depth    = _input->info()->dimension(2);               //weights_kernel_z;
    const unsigned int input_stride_y  = _input->info()->strides_in_bytes().y();
    const unsigned int input_stride_z  = _input->info()->strides_in_bytes().z();
   //value 
    uint8_t *value_buffer=_value->buffer();
    //const unsigned int value_stride_y=_value->info()->strides_in_bytes().y();
    const unsigned int value_stride_z=_value->info()->strides_in_bytes().z();
    //quantity
    uint8_t *quantity_buffer=_quantity->buffer();
    //const unsigned int quantity_stride_y=_quantity->info()->strides_in_bytes().y();
    const unsigned int quantity_stride_z=_quantity->info()->strides_in_bytes().z();
    //location
    uint8_t *location_buffer=_location->buffer();
    //const unsigned int location_stride_x=_location->info()->strides_in_bytes().x();
    //const unsigned int location_stride_y=_location->info()->strides_in_bytes().y();
    const unsigned int location_stride_z=_location->info()->strides_in_bytes().z();
    // Create iterators
    Iterator in(_input, window);
    execute_window_loop(window, [&](const Coordinates & id)
    {
        //以第几个kernel为索引遍历的
        const int kernel_idx = id[3];
        uint8_t *value_start_ptr=value_buffer+kernel_idx*value_stride_z;
        uint8_t *quantity_start_ptr=quantity_buffer+kernel_idx*quantity_stride_z;
        uint8_t *location_start_ptr=location_buffer+kernel_idx*location_stride_z;
        int8_t *value_ptr = reinterpret_cast<int8_t *>(value_start_ptr);
        uint16_t *quantity_ptr = reinterpret_cast<uint16_t *>(quantity_start_ptr);
        uint16_t *location_ptr = reinterpret_cast<uint16_t *>(location_start_ptr);
        //取得一个卷积核的信息，保存在map<float,float> value_quantity中
        int total_valid_quantity=0;
        int total_valid_location=0;
        map<signed char,unsigned short> value_quantity;
        uint8_t *input_row_ptr=in.ptr();
        uint8_t *input_depth_ptr=in.ptr();
        int8_t *weights_ptr = reinterpret_cast<int8_t *>(input_row_ptr);
        for(unsigned int d=0; d<kernel_depth; ++d)
        {
            for(unsigned int j=0; j<kernel_size_y; ++j)
            {
                for(unsigned int i=0; i<kernel_size_x; ++i)
                {
                    //需要对量化后的结果进行分类！此处简易处理，保留四位小数！
                    signed char a=(*weights_ptr);
                    if(a!=0){value_quantity[a]++;}
                    weights_ptr++;
                }
                input_row_ptr+=input_stride_y;
                weights_ptr = reinterpret_cast<int8_t *>(input_row_ptr);
            }
            input_depth_ptr+=input_stride_z;
            input_row_ptr=input_depth_ptr;
            weights_ptr = reinterpret_cast<int8_t *>(input_row_ptr);
        }
        //迭代遍历map<float,float> value_quantity，获取value,quantity,location信息
        map<signed char,unsigned short>::iterator ite_map;
        for(ite_map=value_quantity.begin();ite_map!=value_quantity.end();++ite_map)
        {
            //统计生成value、quantity表(tensor)
            signed char a=ite_map->first;       //权重value
            unsigned short b=ite_map->second;   //权重value出现的次数
            if(a){
                total_valid_quantity++;  
                total_valid_location+=b;                            //统计有效值的个数及有效位置的个数
                *value_ptr=a;
                value_ptr++;
                *quantity_ptr=b;
                quantity_ptr++;
            }
            //统计生成location表(tensor)          使用location++生成   也可使用步长生成
            signed char temp_value;
            bool remain_flag=true;
            input_row_ptr=in.ptr();
            input_depth_ptr=in.ptr();
            weights_ptr = reinterpret_cast<int8_t *>(in.ptr());
            for(unsigned int d=0;d<kernel_depth;d++)
            {
                for(unsigned int j=0;j<kernel_size_y;j++)
                {
                    for(unsigned int i=0;i<kernel_size_x;i++)
                    {
                        temp_value=(*weights_ptr);
                        if(temp_value==a){
                            unsigned short dd=(unsigned short)d;
                            unsigned short jj=(unsigned short)j;
                            unsigned short ii=(unsigned short)i;
                            unsigned short result=(dd<<(_x_index+_y_index))|(jj<<_x_index)|ii;
                            *location_ptr=result;
                            location_ptr++;
                            b--;
                        }
                        if(b){weights_ptr++;}
                        else{remain_flag=false;break;}
                    }
                    if(remain_flag){
                        input_row_ptr+=input_stride_y;
                        weights_ptr = reinterpret_cast<int8_t *>(input_row_ptr);
                    }
                    else{break;}
                }
                if(remain_flag){
                    input_depth_ptr+=input_stride_z;
                    input_row_ptr=input_depth_ptr;
                    weights_ptr = reinterpret_cast<int8_t *>(input_row_ptr);
                }
                else{break;}
            }
        }
        //处理剩余位置    使用个数处理   也可使用指针终点处理
        for(int i=total_valid_quantity;i<256;i++)
        {
            *value_ptr=0;*quantity_ptr=0;
            value_ptr++;quantity_ptr++;
        }
        for(int i=total_valid_location;i<_location_y_size;i++)
        {
            *location_ptr=0;location_ptr++;
        }
    },
    in);
    //printf("NEABMWeightsReshapeKernel run end!\n");
}
