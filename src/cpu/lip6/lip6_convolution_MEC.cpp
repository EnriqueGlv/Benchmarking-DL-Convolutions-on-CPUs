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

#include "cpu/lip6/lip6_convolution_MEC.hpp"
#include "cpu/lip6/lip6_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

status_t lip6_convolution_mec_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
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

    // const auto KDH = pd()->KDH() + 1;
    const auto KDW = pd()->KDW() + 1;

    const auto padT = pd()->padT();
    const auto padB = MAX(pd()->padB(),0);
    const auto padL = pd()->padL();

    // initialize the memory needed to store im2row result
    const auto im_buf  = pd()->im_buf;
    const auto dst_buf = (MB > 1) ? pd()->dst_buf : ((float*) dst);

    const int NBREPS = getenv_int("LIP6_NB_REPS", 1);

    for (int rep = 0; rep < NBREPS; rep++){

    // init ih padding
    auto ker_mec_ih_padT = [=](dim_t mb, dim_t ow, dim_t ih, dim_t kw){
        const auto mec_off_0 = ((ih * KW + kw) * IC * MB + mb) * OW + ow; 
        for (dim_t ic = 0; ic < IC; ++ic){
            const auto mec_off = mec_off_0 + ic * MB * OW;
            im_buf[mec_off] = 0;
        }
    };
    parallel_nd(MB, OW, padT, KW, ker_mec_ih_padT);

    auto ker_mec_ih_padB = [=](dim_t mb, dim_t ow, dim_t ih, dim_t kw){
        const auto mec_off_0 = (((ih+padT+IH) * KW + kw) * IC * MB + mb) * OW + ow; 
        for (dim_t ic = 0; ic < IC; ++ic){
            const auto mec_off = mec_off_0 + ic * MB * OW;
            im_buf[mec_off] = 0;
        }
    };
    parallel_nd(MB, OW, padB, KW, ker_mec_ih_padB);

    // create mec compressed buffer
    auto ker_mec = [=](dim_t mb, dim_t ow, dim_t ih) {
        for (dim_t kw = 0; kw < KW; kw++) {
            const dim_t iw = ow * KSW - padL + kw * KDW;

            const auto mec_off_0 = (((ih + padT) * KW + kw) * IC * MB + mb) * OW + ow;

            if (iw < 0 || iw >= IW){
                for (dim_t ic = 0; ic < IC; ++ic){
                    const auto mec_off = mec_off_0 + ic * MB * OW;
                    im_buf[mec_off] = 0;
                }
                continue;
            }

            for (dim_t ic = 0; ic < IC; ic++) {
                const auto src_off = src_d.off(mb, ic, ih, iw);
                const auto mec_off = mec_off_0 + ic * MB * OW;
                const float s = io::load_float_value(src_d.data_type(), src, src_off);
                im_buf[mec_off] = s;
            }
        }
    };
    parallel_nd(MB, OW, IH, ker_mec);

    // perform OH GEMMs to generate NWC tensors
    parallel_nd(OH, [&](dim_t oh){
            const float onef = 1, zerof = 0;
            const dim_t M = MB * OW, N = OC, K = KH*KW*IC;
            const auto l_off = KSH*KW*IC*oh * MB * OW;
            const auto dst_off = oh*MB*OW*OC;
            
            extended_sgemm("T", "T", &N, &M, &K, &onef, (float*)weights, &K,
                &((float*)im_buf)[l_off], &M, &zerof, &((float*)dst_buf)[dst_off], &N);
        });

    // in the case MB > 1, a reshap eis needed to get dst in NHWC instead of HNWC
    if(MB > 1){
        parallel_nd(MB, OC, OH, OW,
            [&](dim_t mb, dim_t oc, dim_t oh, dim_t ow) {
                    const auto dst_buf_off = ((oh*MB + mb)*OW + ow)*OC + oc;
                    const auto b = dst_buf[dst_buf_off]; 
                    const auto dst_off = dst_d.off(mb, oc, oh, ow);
                    io::store_float_value(dst_d.data_type(), b, dst, dst_off);
        });
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

    }

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
