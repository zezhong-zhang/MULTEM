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
* GNU General Public License for more details.Gauss_wd_
*
* You should have received a copy of the GNU General Public License
* along with Multem. If not, see <http:// www.gnu.org/licenses/>.
*/

#pragma once

#include "fcn_fermi.h"
#include "wd_fcn.h"

/* template definition */
namespace mt
{
#ifndef WD_FCN_DEC
	#define WD_FCN_DEC
	template <class T, eDim Dim, eFcn_typ Fcn_typ> class Wd_fcn_xd;
#endif
}	

/* derived class */
namespace mt
{
	template <class T, eDim Dim>
	using Wd_Fermi_xd = Wd_fcn_xd<T, Dim, efcn_fermi>;

	template <class T>
	using Wd_Fermi_1d = Wd_fcn_xd<T, edim_1, efcn_fermi>;

	template <class T>
	using Wd_Fermi_2d = Wd_fcn_xd<T, edim_2, efcn_fermi>;

	template <class T>
	using Wd_Fermi_3d = Wd_fcn_xd<T, edim_3, efcn_fermi>;
}

/* template specialization */
namespace mt
{
	template <class T, eDim Dim>
	class Wd_fcn_xd<T, Dim, efcn_fermi>: public Wdb_fcn_xd<T, Dim, efcn_fermi>
	{
	public:
		using value_type = T;

		/************************************* constructors ************************************/
		CGPU_EXEC
		Wd_fcn_xd();

		Wd_fcn_xd(const R_xd<T, Dim>& r, const T& alpha, const T& r_wd, const T& r_max);

		/* copy constructor */
		CGPU_EXEC
		Wd_fcn_xd(const Wd_fcn_xd<T, Dim, efcn_fermi>& wd);

		/******************************** assignment operators *********************************/
		/* copy assignment operator */
		CGPU_EXEC 
		Wd_fcn_xd<T, Dim, efcn_fermi>& operator=(const Wdb_fcn_xd<T, Dim, efcn_fermi>& wd);

		/***************************************************************************************/
		void set_in_data(const R_xd<T, Dim>& r, const T& alpha, const T& r_wd, const T& r_max);
	};
}

#include "../src/wd_fermi.inl"