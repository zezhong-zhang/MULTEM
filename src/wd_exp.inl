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
* GNU General Public License for more details.Gauss_wd_
*
* You should have received a copy of the GNU General Public License
* along with Multem. If not, see <http:// www.gnu.org/licenses/>.
*/

#include "wd_exp.h"

namespace mt
{
	/************************************* constructors ************************************/
	template <class T, eDim Dim>
	CGPU_EXEC
	Wd_fcn_xd<T, Dim, efcn_exp>::Wd_fcn_xd(): Wdb_fcn_xd<T, Dim, efcn_exp>() {}

	template <class T, eDim Dim>
	Wd_fcn_xd<T, Dim, efcn_exp>::Wd_fcn_xd(const R_xd<T, Dim>& r, const T& beta, const T& r_wd, const T& r_max) 
	{
		set_in_data(r, beta, r_wd, r_max);
	}

	/* copy constructor */
	template <class T, eDim Dim>
	CGPU_EXEC
	Wd_fcn_xd<T, Dim, efcn_exp>::Wd_fcn_xd(const Wd_fcn_xd<T, Dim, efcn_exp>& wd): Wdb_fcn_xd<T, Dim, efcn_exp>(wd){}

	/******************************** assignment operators *********************************/
	/* copy assignment operator */
	template <class T, eDim Dim>
	CGPU_EXEC 
	Wd_fcn_xd<T, Dim, efcn_exp>& Wd_fcn_xd<T, Dim, efcn_exp>::operator=(const Wdb_fcn_xd<T, Dim, efcn_exp>& wd)
	{
		Wdb_fcn_xd<T, Dim, efcn_exp>::operator=(wd);

		return *this;
	}

	/***************************************************************************************/
	template <class T, eDim Dim>
	void Wd_fcn_xd<T, Dim, efcn_exp>::set_in_data(const R_xd<T, Dim>& r, const T& beta, const T& r_wd, const T& r_max)
	{
		Fcn_Elem<T, efcn_exp>::set_in_data(T(1), beta);
		Wdb_fcn_xd<T, Dim, efcn_exp>::set_in_data(r, r_wd, r_max);
	}
}