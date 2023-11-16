/*
 * This file is part of Multem.
 * Copyright 2023 Ivan Lobato <Ivanlh20@gmail.com>
 *
 * Multem is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version of the License, or
 * (at your option) any later version.
 *
 * Multem is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Multem. If not, see <http:// www.gnu.org/licenses/>.
 */

#define MATLAB_BLAS_LAPACK

#include "math_mt.h"
#include "types.cuh"
#include "type_traits_gen.h"
#include "cgpu_stream.cuh"

#include "cgpu_classes.cuh"

#include <mex.h>
#include "matlab_mex.h"

using mt::pMLD;
using mt::pMx_c;

template <class T, mt::eDev Dev>
void run_rs_2d(mt::System_Config &system_config, pMLD &rM_i, 
T bs_x, T bs_y, T fxy, pMLD &rM_o)
{
	mt::Grid_2d<T> grid_2d(rM_i.cols, rM_i.rows, bs_x, bs_y);
	mt::Stream<Dev> stream(system_config.n_stream);

	mt::Vctr<T, Dev> M(rM_i.begin(), rM_i.end());
	mt::Vctr<T, Dev> mx_o(rM_o.size());

	mt::Rs_2d<T, Dev> rs_2d(&stream, grid_2d);
	rs_2d(M, fxy, mx_o);

	thrust::copy(mx_o.begin(), mx_o.end(), rM_o.begin());
}

void mexFunction(dt_int32 nlhs, mxArray* plhs[], dt_int32 nrhs, const mxArray* prhs[]) 
{
	auto system_config = mex_read_system_config(prhs[0]);
	system_config.set_gpu();

	dt_int32 idx_0 = (system_config.active)?1:0;

	auto rM_i = mex_get_pvctr<pMLD>(prhs[idx_0+0]);
	auto bs_x = rM_i.cols*mex_get_num<dt_float64>(prhs[idx_0+1]);
	auto bs_y = rM_i.rows*mex_get_num<dt_float64>(prhs[idx_0+2]);
	auto fxy = mex_get_num<dt_float64>(prhs[idx_0+3]);

	/***************************************************************************************/
	dt_int32 cols_o = mt::fcn_sc_size_c(rM_i.cols, fxy);
	dt_int32 rows_o = mt::fcn_sc_size_c(rM_i.rows, fxy);
	auto rM_o = mex_create_pVctr<pMLD>(rows_o, cols_o, plhs[0]);

	if (system_config.is_float32_cpu())
	{
		run_rs_2d<dt_float32, mt::edev_cpu>(system_config, rM_i, bs_x, bs_y, fxy, rM_o);
	}
	else if (system_config.is_float64_cpu())
	{
		run_rs_2d<dt_float64, mt::edev_cpu>(system_config, rM_i, bs_x, bs_y, fxy, rM_o);
	}
	else if (system_config.is_float32_gpu())
	{
		run_rs_2d<dt_float32, mt::edev_gpu>(system_config, rM_i, bs_x, bs_y, fxy, rM_o);
	}
	else if (system_config.is_float64_gpu())
	{
		run_rs_2d<dt_float64, mt::edev_gpu>(system_config, rM_i, bs_x, bs_y, fxy, rM_o);
	}
}