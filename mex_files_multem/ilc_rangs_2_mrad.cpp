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

#include "fcns_cgpu_gen.h"

#include <mex.h>
#include "matlab_mex.h"

void mexFunction(dt_int32 nlhs, mxArray* plhs[], dt_int32 nrhs, const mxArray* prhs[])
{
	auto pE_0 = mex_get_pvctr<dt_float64>(prhs[0]);
	auto prangs = mex_get_pvctr<dt_float64>(prhs[1]);

	/***************************************************************************************/
	auto pmrad = mex_create_pVctr<dt_float64>(pE_0.shape(), plhs[0]);

	for (auto ik = 0; ik < prangs.size(); ik++)
	{
		pmrad[ik] = mt::fcn_rangs_2_rad(pE_0[ik], prangs[ik])/mt::c_mrad_2_rad<dt_float64>;
	}
}