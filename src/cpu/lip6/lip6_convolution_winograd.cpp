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

#include "cpu/lip6/lip6_convolution_winograd.hpp"
#include "cpu/lip6/lip6_convolution_utils.hpp"

namespace dnnl {
namespace impl {
namespace cpu {

status_t lip6_convolution_wino_fwd_t::execute_forward(const exec_ctx_t &ctx) const {
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

    const auto MB = pd()->MB();
    const auto OH = pd()->OH();
    const auto OW = pd()->OW();
    const auto IH = pd()->IH();
    const auto IW = pd()->IW();

    const auto OC = pd()->OC();
    const auto IC = pd()->IC();
    const auto KH = pd()->KH();
    const auto KW = pd()->KW();

    const auto padT = pd()->padT();
    const auto padL = pd()->padL();

    const auto al = pd()->alpha;
    const auto m  = pd()->m;
    const auto P  = pd()->P;
    const auto h_tiles = pd()->h_tiles;
    const auto w_tiles = pd()->w_tiles;
    const auto batch_tiles = h_tiles * w_tiles;

    auto U  = pd()->U;
    auto V  = pd()->V;
    auto M  = pd()->M;

    const float onef = 1, zerof = 0;

    // start benchmark
    // int gpio1_fd = open("/sog/gpio/pin1",O_WRONLY);
    // write(gpio1_fd, "1", 1);


    // add more reps
    const int NBREPS = getenv_int("LIP6_NB_REPS", 1);

    for (int rep = 0; rep < NBREPS; rep++){

    #define U_off(x,y,z,t) (((x*al+y)*OC+z)*IC+t)

    // Compute U^T(k,c) = Gg(k,c)G^T
    // with G[12] = {  1,   0,   0, 
    //               0.5, 0.5, 0.5,
    //               0.5,-0.5, 0.5, 
    //                 0,   0,   1};
    auto ker_U = [=](dim_t oc, dim_t ic){
    // for_(dim_t oc = 0; oc < OC; oc++)
    // for (dim_t ic = 0; ic < IC; ++ic) {
        const auto wei_off_0 = (oc*IC+ic)*KH*KW;
        float* __restrict _wei = (float*)weights + wei_off_0;

        #define _U(a,b) U[U_off(a,b,oc,ic)]

        _U(0,0) = _wei[0];
        float g0pg6 = _wei[0] + _wei[6];
        _U(1,0) = (g0pg6 + _wei[3]) * 0.5;
        _U(2,0) = (g0pg6 - _wei[3]) * 0.5;
        _U(3,0) = _wei[6];

        _U(0,1) = (_wei[0] + _wei[1] + _wei[2]) * 0.5;
        float g0pg1pg2 = _wei[0] + _wei[1] + _wei[2];
        float g3pg4pg5 = _wei[3] + _wei[4] + _wei[5];
        float g6pg7pg8 = _wei[6] + _wei[7] + _wei[8];
        _U(1,1) = (g0pg1pg2 + g3pg4pg5 + g6pg7pg8) * 0.25;
        _U(2,1) = (g0pg1pg2 - g3pg4pg5 + g6pg7pg8) * 0.25;
        _U(3,1) = (_wei[6] + _wei[7] + _wei[8]) * 0.5;

        _U(0,2) = (_wei[0] - _wei[1] + _wei[2]) * 0.5;
        float g0mg1pg2 = _wei[0] - _wei[1] + _wei[2];
        float g3mg4pg5 = _wei[3] - _wei[4] + _wei[5];
        float g6mg7pg8 = _wei[6] - _wei[7] + _wei[8];
        _U(1,2) = (g0mg1pg2 + g3mg4pg5 + g6mg7pg8) * 0.25;
        _U(2,2) = (g0mg1pg2 - g3mg4pg5 + g6mg7pg8) * 0.25;
        _U(3,2) = (_wei[6] - _wei[7] + _wei[8]) * 0.5;

        _U(0,3) = _wei[2];
        float g2pg8 = _wei[2] + _wei[8];
        _U(1,3) = (g2pg8 + _wei[5]) * 0.5;
        _U(2,3) = (g2pg8 - _wei[5]) * 0.5;
        _U(3,3) = _wei[8];
    // }
    };
    parallel_nd(OC, IC, ker_U);

    // Allow smart access of src including stride and padding
    auto src_get = [=](dim_t ic, dim_t b, dim_t al1, dim_t al2){
        const dim_t htile = (b%batch_tiles) / w_tiles;
        const dim_t h0    = m * htile;
        const dim_t wtile = (b%batch_tiles) % w_tiles;
        const dim_t w0    = m * wtile;

        const auto mb = (int)(b/batch_tiles);
        const auto ih = h0 + al1 - padT;
        const auto iw = w0 + al2 - padL;
        
        if(ih < 0 || ih >= IH || iw < 0 || iw >= IW){
            return (float)0;
        } else {
            const auto src_off = ((mb*IC + ic)*IH + ih)*IW + iw;
            return ((float*)src)[src_off];
        }
    };

    #define V_off(x,y,z,t) (((x*al+y)*P+z)*IC+t)

    // Compute V^T(k,c) = Bd(k,c)B^T
    // with BT[16] = { 1, 0,-1, 0,
    //                 0, 1, 1, 0,
    //                 0,-1, 1, 0,
    //                 0, 1, 0,-1};
    auto ker_V = [=](dim_t b, dim_t ic){
    // for_(dim_t b = 0; b < P; b++)
    // for (dim_t ic = 0; ic < IC; ++ic) {
        #define _src(x) src_get(ic,b,(int)x/al,x%al)
        #define _V(x,y) V[V_off(x,y,b,ic)]

        _V(0,0) = _src(0) - _src(2) - _src(8) + _src(10);
        _V(1,0) = _src(4) - _src(6) + _src(8) - _src(10);
        _V(2,0) = _src(6) - _src(4) + _src(8) - _src(10);
        _V(3,0) = _src(4) - _src(6) - _src(12) + _src(14);

        _V(0,1) = _src( 1) + _src(2) - _src( 9) - _src(10);
        _V(1,1) = _src( 5) + _src(6) + _src( 9) + _src(10);
        _V(2,1) = _src(10) - _src(5) - _src( 6) + _src( 9);
        _V(3,1) = _src( 5) + _src(6) - _src(13) - _src(14);

        _V(0,2) = _src( 2) + _src( 9) - _src( 1) - _src(10);
        _V(1,2) = _src(10) - _src( 5) + _src( 6) - _src( 9);
        _V(2,2) = _src( 5) - _src( 6) + _src(10) - _src( 9);
        _V(3,2) = _src(13) - _src(14) - _src( 5) + _src( 6);

        _V(0,3) = _src(1) - _src(3) - _src( 9) + _src(11);
        _V(1,3) = _src(5) - _src(7) + _src( 9) - _src(11);
        _V(2,3) = _src(7) - _src(5) + _src( 9) - _src(11);
        _V(3,3) = _src(5) - _src(7) - _src(13) + _src(15);
    // }
    };
    parallel_nd(P, IC, ker_V);
    
    // Compute M(xi,nu)=U(xi,nu)V(xi,nu)
    auto ker_M = [=](dim_t al1, dim_t al2){
    // for_(dim_t al1 = 0; al1 < al; al1++)
    // for (dim_t al2 = 0; al2 < al; ++al2) {
        const auto U_off = (al*al1 + al2)*OC*IC;
        const auto V_off = (al*al1 + al2)*P*IC;
        const auto M_off = (al*al1 + al2)*OC*P;
        extended_sgemm("T", "N", &P, &OC, &IC, &onef, &V[V_off], &IC, &U[U_off], &IC, &zerof, &M[M_off], &P);
    // }
    };
    parallel_nd(al, al, ker_M);

    // allow smart storage into dst, avoiding overflow
    auto dst_off = [=](dim_t oc, dim_t b, dim_t m1, dim_t m2){
        const dim_t htile = (b%batch_tiles) / w_tiles;
        const dim_t h0   = m * htile;
        const dim_t wtile = (b%batch_tiles) % w_tiles;
        const dim_t w0   = m * wtile;

        const auto mb = (int)(b/batch_tiles);
        const auto oh = h0 + m1;
        const auto ow = w0 + m2;

        if(oh >= OH || ow >= OW) return (dim_t)-1;

        return ((mb*OC+oc)*OH+oh)*OW+ow;
    };

    #define M_off(x,y,z,t) (((x*al+y)*OC+z)*P+t)

    // Compute Y(k,b) = A^TmA
    // with AT[8] = { 1, 1, 1, 0,
    //                0, 1,-1,-1};
    auto ker_dst = [=](dim_t oc, dim_t b){
    // for_(dim_t oc = 0; oc < OC; ++oc)
    // for (dim_t b = 0; b < P; b++){
        #define _M(x) M[M_off((int)x/al,x%al,oc,b)]

        dim_t _dst00 = dst_off(oc,b,0,0);
        if(_dst00 > -1){
            ((float*)dst)[_dst00] = _M(0) + _M(1) + _M(2) 
                                  + _M(4) + _M(5) + _M(6) 
                                  + _M(8) + _M(9) + _M(10);
        }
        dim_t _dst10 = dst_off(oc,b,1,0);
        if(_dst10 > -1){
            ((float*)dst)[_dst10] = _M( 4) + _M( 5) + _M( 6) 
                                  - _M( 8) - _M( 9) - _M(10)
                                  - _M(12) - _M(13) - _M(14) ;
        }
        dim_t _dst01 = dst_off(oc,b,0,1);
        if(_dst01 > -1){
            ((float*)dst)[_dst01] = _M( 1) - _M( 2) - _M( 3) 
                                  + _M( 5) - _M( 6) - _M( 7)
                                  + _M( 9) - _M(10) - _M(11) ;
        }
        dim_t _dst11 = dst_off(oc,b,1,1);
        if(_dst11 > -1){
            ((float*)dst)[_dst11] = _M( 5) - _M( 6) - _M( 7) 
                                  + _M(10) - _M( 9) + _M(11)
                                  + _M(14) - _M(13) + _M(15) ;
        }
    // }
    };
    parallel_nd(OC, P, ker_dst);

    // Manage Bias and post opps
    auto sum_dt = pd()->attr()->post_ops_.get_sum_dt(dst_d.data_type());
    parallel_nd(MB, OH, OW, OC,
        [&](dim_t mb, dim_t oh, dim_t ow, dim_t oc) {
            // physical memory offset in NHWC order
            dim_t dst_off = dst_d.off(mb, oc, oh, ow);
            
            // load conv result currently stored in dst
            float conv_val = io::load_float_value(dst_d.data_type(), dst, dst_off);
            
            if (bias) {
                // add bias
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

    // stop benchmark
    // write(gpio1_fd, "0", 1);
    // close(gpio1_fd);

    return status::success;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
