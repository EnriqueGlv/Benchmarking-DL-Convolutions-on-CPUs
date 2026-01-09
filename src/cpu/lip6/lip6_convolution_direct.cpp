/*******************************************************************************
* Copyright 2016-2021 Intel Corporation
* Copyright 2025 Sorbonne Université, LIP6
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/dnnl_traits.hpp"
#include "common/math_utils.hpp"
#include "common/type_helpers.hpp"

#include "cpu/cpu_primitive.hpp"
#include "cpu/ref_io_helper.hpp"

#include <iostream>

#include "cpu/lip6/lip6_convolution_direct.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

status_t lip6_convolution_direct_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
    status_t status = status::success;
    auto src = CTX_IN_MEM(const void *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const void *, DNNL_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const void *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_CLEAN_MEM(void *, DNNL_ARG_DST, status);
    CHECK(status);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    // const bool with_groups = pd()->with_groups();

    const auto G = pd()->G();
    const auto MB = pd()->MB();
    const auto OH = pd()->OH();
    const auto OW = pd()->OW();
    const auto IH = pd()->IH();
    const auto IW = pd()->IW();

    const auto OC = pd()->OC() / G;
    const auto IC = pd()->IC() / G;
    const auto KH = pd()->KH();
    const auto KW = pd()->KW();

    const auto KSH = pd()->KSH();
    const auto KSW = pd()->KSW();

    const auto KDH = pd()->KDH() + 1;
    const auto KDW = pd()->KDW() + 1;

    const auto padT = pd()->padT();
    const auto padL = pd()->padL();

    const dim_t OW_start = 0;//ceil((float)(padL)/KSW);
    const dim_t OH_start = 0;//ceil((float)(padT)/KSH);
    const dim_t OW_end = OW;//MIN((IW + padL - KW*KDW) / KSW, OW);
    const dim_t OH_end = OH;//MIN((IH + padT - KH*KDH) / KSH, OH);

    // const dim_t BLOCK_SIZE = 128*28;

    const dim_t OWB = 56;
    const dim_t OW_ = ceil((float)(OW_end - OW_start) / OWB);
    const dim_t OCB = OC;//sqrt(BLOCK_SIZE/MIN(OW,OWB));
    const dim_t OC_ = ceil((float)OC / OCB);
    const dim_t ICB = 32;//OCB;
    const dim_t IC_ = ceil((float)IC / ICB);

    // start benchmark
    // int gpio1_fd = open("/sog/gpio/pin1",O_WRONLY);
    // write(gpio1_fd, "1", 1);

    // add more reps
    const int NBREPS = getenv_int("LIP6_NB_REPS", 1);

    for (int rep = 0; rep < NBREPS; rep++){

    // const dim_t OW_ = 1, OC_ = 1, IC_ = 1;
    // const dim_t OWB = OW, OCB = OC, ICB = IC;

    // for_(dim_t mb = 0; mb < MB; mb++)
    // for_(dim_t oc_ = 0; oc_ < OC_; oc_++)
    // for_(dim_t oh = OH_start; oh < OH_end; oh++)
    // for (dim_t ow_ = 0; ow_ < OW_; ow_++) {
    parallel_nd(MB,OC_,OH,OW_,[&](dim_t mb, dim_t oc_, dim_t oh, dim_t ow_){

        for_(dim_t ic_ = 0; ic_ < IC_; ++ic_)
        for (dim_t kh = 0; kh < KH; ++kh){

            const dim_t ih = oh * KSH - padT + kh * KDH;
            
            for_(dim_t kw = 0; kw < KW; ++kw)
            for_(dim_t ic = ic_*ICB; ic < MIN((ic_+1)*ICB,IC); ++ic)
            for (dim_t ow = OW_start + ow_*OWB; ow < MIN(OW_start + (ow_+1)*OWB,OW_end); ++ow){

                const dim_t dst_off_0 = ((mb * OH + oh) * OW + ow) * OC;
                
                const dim_t iw = ow * KSW - padL + kw * KDW;
                if(iw < 0 || iw >= IW || ih < 0 || ih >= IH){
                    for (dim_t oc = oc_*OCB; oc < MIN((oc_+1)*OCB, OC); ++oc){
                        const dim_t dst_off = dst_off_0 + oc;
                        if(ic == 0 && kh == 0 && kw == 0) ((float*)dst)[dst_off] = 0;
                    }
                    continue;
                }

                const auto src_off = ((mb * IH + ih) * IW + iw) * IC + ic;
                const float s = ((float*)src)[src_off];
            
                for (dim_t oc = oc_*OCB; oc < MIN((oc_+1)*OCB, OC); ++oc){
                    const dim_t dst_off = dst_off_0 + oc;
                    const auto wei_off = ((ic * KH + kh) * KW + kw) * OC + oc;

                    const float w = ((float*)weights)[wei_off];

                    const float tmp_dst = (ic == 0 && kh == 0 && kw == 0) ? 0 : ((float*)dst)[dst_off]; 
                    ((float*)dst)[dst_off] = s*w + tmp_dst;
                }
            }
        }
    });

    auto sum_dt = pd()->attr()->post_ops_.get_sum_dt(dst_d.data_type());
    parallel_nd(MB, OH, OW, OC,
        [&](dim_t mb, dim_t oh, dim_t ow, dim_t oc) {
            // physical memory offset in NHWC order
            dim_t dst_off = dst_d.off(mb, oc, oh, ow);
            
            // load conv result currently stored in dst
            float conv_val = io::load_float_value(dst_d.data_type(), dst, dst_off);
            
            // add bias
            if (bias) {
                const auto bias_off = bias_d.off(oc);
                const float b = io::load_float_value(bias_d.data_type(), bias, bias_off);
                conv_val += b;
            }

            // Prepare args: args.dst_val must be the *previous* destination value
            ref_post_ops_t::args_t args;
            args.dst_val = io::load_float_value(sum_dt, dst, dst_off);
            args.ctx = &ctx;
            args.l_offset = ((mb*OH + oh)*OW + ow)*OC + oc; //logical_off_nhwc(mb, OH, OW, oh, ow, oc, OC);
            args.dst_md = pd()->dst_md();

            float out_val = conv_val;
            ref_post_ops->execute(out_val, args);

            // store post-op result back
            io::store_float_value(dst_d.data_type(), out_val, dst, dst_off);
    });
        
    }
    // stop benchmark
    // write(gpio1_fd, "0", 1);
    // close(gpio1_fd);

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl