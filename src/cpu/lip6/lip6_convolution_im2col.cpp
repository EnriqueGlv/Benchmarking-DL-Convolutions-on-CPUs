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

#include "cpu/lip6/lip6_convolution_im2col.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

status_t lip6_convolution_im2col_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
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

    // we don't support groups for this version (as ResNet50 doesn't need it)
    assert(!pd()->with_groups());

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

    // initialize the memory needed to store im2col result
    const auto im2col  = pd()->im2col;

    // transform image
    for (int mb=0; mb < MB; mb++){

        // kernel to compute im2col in parallel for a slice of mb
        auto ker_im2col = [=](dim_t oh, dim_t ow) {
            const auto im2col_off_0 = (OW * oh + ow);

            for_(dim_t kh = 0; kh < KH; ++kh)
            for (dim_t kw = 0; kw < KW; ++kw) {
                const dim_t ih = oh * KSH - padT + kh * KDH;
                const dim_t iw = ow * KSW - padL + kw * KDW;
                for (dim_t ic = 0; ic < IC; ++ic) {
                    const auto im2col_off = im2col_off_0 + (KW * (KH * ic + kh) + kw) * OH * OW;
                    if (ih < 0 || ih >= IH || iw < 0 || iw >= IW){
                        im2col[im2col_off] = 0;
                        continue;
                    }
                    const auto src_off = src_d.off(mb, ic, ih, iw);
                    const float s = io::load_float_value(src_d.data_type(), src, src_off);
                    im2col[im2col_off] = s;
                }
            }
        };
        
        parallel_nd(OH, OW, ker_im2col);

        const float onef = 1, zerof = 0;
        const dim_t M = OH*OW, N = OC, K = IC*KH*KW;

        extended_sgemm("N", "N", &M, &N, &K, &onef, 
            im2col, &M, (float*)weights, &K, &zerof, &((float*)dst)[mb*OC*OH*OW], &M);
    }

    auto sum_dt = pd()->attr()->post_ops_.get_sum_dt(dst_d.data_type());
    parallel_nd(MB, OH, OW, OC,
        [&](dim_t mb, dim_t oh, dim_t ow, dim_t oc) {
            // physical memory offset in NHWC order
            dim_t dst_off = dst_d.off(mb, oc, oh, ow); //maybe here error
            
            // load conv result currently stored in dst
            float conv_val = io::load_float_value(dst_d.data_type(), dst, dst_off);
            
            if (bias) {
                // add bias (bias_d.off likely expects index 'oc' only)
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

            // store the post-op result back
            io::store_float_value(dst_d.data_type(), out_val, dst, dst_off);
    });

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
