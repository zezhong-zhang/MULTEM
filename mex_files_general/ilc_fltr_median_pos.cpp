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
#include "vctr_cpu.h"
#include "fcns_image_cpu.h"

#include <mex.h>
#include "matlab_mex.h"

template <class T>
void mex_run(dt_int32 nlhs, mxArray* plhs[], dt_int32 nrhs, const mxArray* prhs[])
{
	auto pmx_i = mex_get_pvctr<T>(prhs[0]);
	auto nkr = (nrhs>1)?mex_get_num<dt_int32>(prhs[1]):1;

	/***************************************************************************************/
	mt::Vctr_cpu<T> mx_o(pmx_i.shape());
	auto pmx_o = mx_o.ptr_64();

	if (pmx_i.is_1d())
	{
		auto vctr_idx = (nrhs>2)?mex_get_vctr<dt_int32>(prhs[2]):mt::Vctr_cpu<dt_int32>();
		mt::fcns_image_cpu::fltr_median_1d_pos(pmx_i, nkr, vctr_idx.ptr_64(), pmx_o);
	}
	else
	{
		auto vctr_idx = (nrhs>2)?mex_get_vctr_r_2d<dt_int32>(prhs[2]):mt::Vctr_r_2d_cpu<dt_int32>();
		mt::fcns_image_cpu::fltr_median_2d_pos(pmx_i, nkr, vctr_idx.ptr_64(), pmx_o);
	}

	mex_create_set_pVctr<T>(plhs[0], pmx_o);
}

void mexFunction(dt_int32 nlhs, mxArray* plhs[], dt_int32 nrhs, const mxArray* prhs[])
{
	MEX_RUN_FCN_REAL(mex_run, 0);
}