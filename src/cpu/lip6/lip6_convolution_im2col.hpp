/*******************************************************************************
* Copyright 2016-2023 Intel Corporation
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

#ifndef CPU_LIP6_CONVOLUTION_IM2COL_HPP
#define CPU_LIP6_CONVOLUTION_IM2COL_HPP

#include <assert.h>

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/type_helpers.hpp"
#include "common/utils.hpp"

#include "cpu/cpu_convolution_pd.hpp"
#include "cpu/primitive_attr_postops.hpp"
#include "cpu/gemm_convolution.hpp"

#include "cpu/lip6/lip6_convolution_utils.hpp" 

namespace dnnl {
namespace impl {
namespace cpu {

struct lip6_convolution_im2col_fwd_t : public primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        using cpu_convolution_fwd_pd_t::cpu_convolution_fwd_pd_t;

        DECLARE_COMMON_PD_T("lip6:im2col", lip6_convolution_im2col_fwd_t);

        float* im2col;
        
        status_t init(engine_t *engine) {
            using namespace data_type;
            // const auto dst_type = dst_md(0)->data_type;
            const memory_desc_wrapper src_d(src_md());
            const memory_desc_wrapper weights_d(weights_md(0));

            bool ok = is_fwd()
                    && set_default_alg_kind(alg_kind::convolution_direct)
                    && expect_data_types(f32, f32, f32, f32, f32)
                    && set_default_formats()
                    && !has_zero_dim_memory()
                    && attr()->has_default_values(
                            primitive_attr_t::skip_mask_t::post_ops, f32)
                    // && attr()->post_ops_.check_sum_consistent_dt(dst_type)
                    && post_ops_ok()
                    && attr_.set_default_formats(dst_md(0)) == status::success
                    // exclude 3D images
                    && ndims() == 4
                    // exclude blocked layouts
                    && src_d.is_plain() && weights_d.is_plain()
                    && !getenv_int("LIP6_DISABLE_IM2COL", 0)
                    ;

            if(!ok) return status::unimplemented;

            alloc_mem(im2col);

            return status::success;
        }

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto dat_tag = nchw;
            auto wei_tag = with_groups() ? goihw : oihw;
            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

        bool post_ops_ok() const {
            return attr()->post_ops_.find(primitive_kind::convolution) == -1;
        }
        
        status_t alloc_mem(float* &im2col){
            const memory_desc_wrapper src_d(src_md());

            const auto com_size = IC() * KH() * KW();
            const auto im2col_size = OH() * OW() * com_size;
            im2col = new float[im2col_size];
            
            return status::success;
        }
    };

    lip6_convolution_im2col_fwd_t(const pd_t *apd) : primitive_t(apd) {}

    status_t init(engine_t *engine) override {
        ref_post_ops
                = utils::make_unique<ref_post_ops_t>(pd()->attr()->post_ops_);
        if (!ref_post_ops) return status::out_of_memory;
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }


private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    status_t execute_forward(const exec_ctx_t &ctx) const;
    std::unique_ptr<ref_post_ops_t> ref_post_ops;
};

} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif

// vim: et ts=4 sw=4 cindent cino+=l0,\:4,N-s
