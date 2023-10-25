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
void run_fd_sft_1d(mt::System_Config &system_config, pMLD &rM1_i, pMLD &rM2_i, 
T bs_x, T p, T sigma_g, T dx, pMLD &rbd, pMLD &rdx_o)
{
	mt::Grid_1d<T> grid_1d(rM1_i.size(), bs_x);
	mt::FFT<T, Dev> fft_1d;

	mt::Region _Rect_1d<T> bd(grid_1d.bs_x, rbd.size(), rbd.real);

	mt::Vctr<T, Dev> M1(rM1_i.begin(), rM1_i.end());
	mt::Vctr<T, Dev> M2(rM2_i.begin(), rM2_i.end());

	sigma_g = sigma_g*grid_1d.dgx;		// pixels to g

	mt::Fd_Sft_1d<T, Dev> fd_sft_1d(&fft_1d, grid_1d);
	fd_sft_1d.set_fft_plan();
	auto dx_o = fd_sft_1d(M1, M2, p, sigma_g, dx, bd, 2);
	fd_sft_1d.cleanup();

	rdx_o[0] = dx_o;
}

void mexFunction(dt_int32 nlhs, mxArray* plhs[], dt_int32 nrhs, const mxArray* prhs[]) 
{
	auto system_config = mex_read_system_config(prhs[0]);
	system_config.set_gpu();

	dt_int32 idx_0 = (system_config.active)?1:0;

	auto rM_1i = mex_get_pvctr<pMLD>(prhs[idx_0+0]);
	auto rM_2i = mex_get_pvctr<pMLD>(prhs[idx_0+1]);
	auto bs_x = rM_1i.size()*mex_get_num<dt_float64>(prhs[idx_0+2]);
	auto p = mex_get_num<dt_float64>(prhs[idx_0+3]);
	auto sigma_g = mex_get_num<dt_float64>(prhs[idx_0+4]);
	auto dx = (nrhs>idx_0+5)?mex_get_num<dt_float64>(prhs[idx_0+5]):0;
	auto rbd = (nrhs>idx_0+6)?mex_get_pvctr<pMLD>(prhs[idx_0+6]):pMLD();

	/***************************************************************************************/
	auto rdx_o = mex_create_pVctr<pMLD>(1, 1, plhs[0]);

	if (system_config.is_float32_cpu())
	{
		run_fd_sft_1d<dt_float32, mt::edev_cpu>(system_config, rM_1i, rM_2i, bs_x, p, sigma_g, dx, rbd, rdx_o);
	}
	else if (system_config.is_float64_cpu())
	{
		run_fd_sft_1d<dt_float64, mt::edev_cpu>(system_config, rM_1i, rM_2i, bs_x, p, sigma_g, dx, rbd, rdx_o);
	}
	else if (system_config.is_float32_gpu())
	{
		run_fd_sft_1d<dt_float32, mt::edev_gpu>(system_config, rM_1i, rM_2i, bs_x, p, sigma_g, dx, rbd, rdx_o);
	}
	else if (system_config.is_float64_gpu())
	{
		run_fd_sft_1d<dt_float64, mt::edev_gpu>(system_config, rM_1i, rM_2i, bs_x, p, sigma_g, dx, rbd, rdx_o);
	}
}