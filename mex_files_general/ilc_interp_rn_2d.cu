/*
 * This file is part of Multem.
 * Copyright 2022 Ivan Lobato <Ivanlh20@gmail.com>
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

#include "math_mt.h"
#include "types.cuh"
#include "type_traits_gen.h"
#include "cgpu_stream.cuh"

#include "cgpu_classes.cuh"

#include <mex.h>
#include "matlab_mex.h"

using mt::pMLD;

template <class T, mt::eDev Dev>
void run_intrplrn_2d(mt::System_Config &system_config, pMLD &rM_i, 
pMLD &rRx_i, pMLD &rRy_i, mt::eFil_Sel_Typ bg_opt, dt_float64 &bg, pMLD &rM_o)
{
	mt::Grid_2d<T> grid_2d(rM_i.cols, rM_i.rows);
	mt::Grid_2d<T> grid_2d_o(rRx_i.cols, rRx_i.rows);
	mt::Stream<Dev> stream(system_config.n_stream);

	mt::Vctr<T, Dev> M(rM_i.begin(), rM_i.end());
	mt::Vctr<T, Dev> rx(rRx_i.begin(), rRx_i.end());
	mt::Vctr<T, Dev> ry(rRy_i.begin(), rRy_i.end());
	mt::Vctr<T, Dev> mx_o(rM_o.size());

	mt::Interp_rn_2d<T, Dev> fcn_intrpl_bl_rg_2d(&stream, grid_2d, grid_2d_o, bg_opt, bg);
	bg = fcn_intrpl_bl_rg_2d(M, Rx, Ry, mx_o);

	thrust::copy(mx_o.begin(), mx_o.end(), rM_o.begin());
}

void mexFunction(dt_int32 nlhs, mxArray* plhs[], dt_int32 nrhs, const mxArray* prhs[]) 
{
	auto system_config = mex_read_system_config(prhs[0]);
	system_config.set_gpu();

	dt_int32 idx_0 = (system_config.active)?1:0;

	auto rM_i = mex_get_pvctr<pMLD>(prhs[idx_0+0]);
	auto rRx_i = mex_get_pvctr<pMLD>(prhs[idx_0+1]);
	auto rRy_i = mex_get_pvctr<pMLD>(prhs[idx_0+2]);
	auto bg_opt = (nrhs>idx_0+3)?mex_get_num<mt::eFil_Sel_Typ>(prhs[idx_0+3]):mt::efst_min;
	auto bg = (nrhs>idx_0+4)? mex_get_num<dt_float64>(prhs[idx_0+4]):0;

	/***************************************************************************************/
	auto rM_o = mex_create_pVctr<pMLD>(rRx_i.rows, rRy_i.cols, plhs[0]);

	if (system_config.is_float32_cpu())
	{
		run_intrplrn_2d<dt_float32, mt::edev_cpu>(system_config, rM_i, rRx_i, rRy_i, bg_opt, bg, rM_o);
	}
	else if (system_config.is_float64_cpu())
	{
		run_intrplrn_2d<dt_float64, mt::edev_cpu>(system_config, rM_i, rRx_i, rRy_i, bg_opt, bg, rM_o);
	}
	else if (system_config.is_float32_gpu())
	{
		run_intrplrn_2d<dt_float32, mt::edev_gpu>(system_config, rM_i, rRx_i, rRy_i, bg_opt, bg, rM_o);
	}
	else if (system_config.is_float64_gpu())
	{
		run_intrplrn_2d<dt_float64, mt::edev_gpu>(system_config, rM_i, rRx_i, rRy_i, bg_opt, bg, rM_o);
	}

	if (nlhs == 2)
	{
		mex_create_set_num<pMLD>(plhs[1], bg);
	}
}