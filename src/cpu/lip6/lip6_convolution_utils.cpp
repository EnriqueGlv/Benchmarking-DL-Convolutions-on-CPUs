#include "cpu/lip6/lip6_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

void print_tensor(float* tensor, int num_dims, ...) {
    // Initialize va_list and retrieve dimensions
    va_list dims;
    va_start(dims, num_dims);
    // Retrieve dimensions
    int dimensions[num_dims];
    for (int i = 0; i < num_dims; ++i) {
        dimensions[i] = va_arg(dims, int);
    }
    if(num_dims == 1){
        for (int i = 0; i < dimensions[0]; i++){
            Dn(i)
            std::cout << tensor[i] << " ";
        }
        std::cout << std::endl;

    } else {
        // Calculate total number of elements
        int outter_dims = 1;
        for (int i = 0; i < num_dims-1; ++i) {
            outter_dims *= dimensions[i];
        }

        // Print tensor elements
        int idx = 0;
        for (int i = 0; i < outter_dims; ++i) {
            for (int j = 0; j < dimensions[num_dims-1]; j++)
                std::cout << tensor[idx++] << " ";
            std::cout << std::endl;
        }
    }

    // Clean up va_list
    va_end(dims);
}

}
}
}