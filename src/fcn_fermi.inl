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

#include "fcn_fermi.h"

namespace mt
{
	/************************************* constructors ************************************/
	template <class T>
	CGPU_EXEC
	Fcn_Elem<T, efcn_fermi>::Fcn_Elem(): a_1(0), b_1(0), b_2(0) {}

	template <class T>
	Fcn_Elem<T, efcn_fermi>::Fcn_Elem(const T& a, const T& alpha, const T& radius)
	{
		set_in_data(a, alpha, radius);
	}

	/* copy constructor */
	template <class T>
	CGPU_EXEC
	Fcn_Elem<T, efcn_fermi>::Fcn_Elem(const Fcn_Elem<T, efcn_fermi>& parm)
	{
		*this = parm;
	}

	/******************************** assignment operators *********************************/
	/* copy assignment operator */
	template <class T>
	CGPU_EXEC
	Fcn_Elem<T, efcn_fermi>& Fcn_Elem<T, efcn_fermi>::operator=(const Fcn_Elem<T, efcn_fermi>& parm)
	{
		if (this != &parm)
		{
			a_1 = parm.a_1;
			b_1 = parm.b_1;
			b_2 = parm.b_2;
		}

		return *this;
	}

	/***************************************************************************************/
	template <class T>
	void Fcn_Elem<T, efcn_fermi>::set_in_data(const T& a, const T& alpha, const T& radius)
	{
		a_1 = a;
		b_1 = alpha;
		b_2 = ::square(radius);
	}

	template <class T>
	CGPU_EXEC
	void Fcn_Elem<T, efcn_fermi>::clear()
	{
		a_1 = T(0);
		b_1 = T(0);
		b_2 = T(0);
	}

	/***************************************************************************************/
	template <class T>
	CGPU_EXEC
	T Fcn_Elem<T, efcn_fermi>::eval_r2(const T& r2, const T& radius) const
	{ 
		return a_1/(T(1) + exp(b_1*(r2-::square(radius))));
	}

	template <class T>
	CGPU_EXEC
	T Fcn_Elem<T, efcn_fermi>::eval_r2(const T& r2) const
	{ 
		return a_1/(T(1) + exp(b_1*(r2-b_2)));
	}

	template <class T>
	CGPU_EXEC
	R_2d<T> Fcn_Elem<T, efcn_fermi>::eval_r2(const R_2d<T>& r2) const
	{ 
		return {eval_r2(r2.x), eval_r2(r2.y)};
	}

	template <class T>
	CGPU_EXEC
	R_3d<T> Fcn_Elem<T, efcn_fermi>::eval_r2(const R_3d<T>& r2) const
	{ 
		return {eval_r2(r2.x), eval_r2(r2.y), eval_r2(r2.z)};
	}

	/***************************************************************************************/
	template <class T>
	CGPU_EXEC
	T Fcn_Elem<T, efcn_fermi>::eval_r(const T& r, const T& radius) const
	{ 
		return eval_r2(::square(r), radius);
	}

	template <class T>
	CGPU_EXEC
	T Fcn_Elem<T, efcn_fermi>::eval_r(const T& r) const
	{ 
		return eval_r2(::square(r));
	}

	template <class T>
	CGPU_EXEC
	R_2d<T> Fcn_Elem<T, efcn_fermi>::eval_r(const R_2d<T>& r) const
	{ 
		return eval_r2(square(r));
	}

	template <class T>
	CGPU_EXEC
	R_3d<T> Fcn_Elem<T, efcn_fermi>::eval_r(const R_3d<T>& r) const
	{ 
		return eval_r2(square(r));
	}

	/***************************************************************************************/
	template <class T>
	CGPU_EXEC
	T Fcn_Elem<T, efcn_fermi>::operator()(const T& r2, const T& radius) const
	{ 
		return eval_r2(r2, radius);
	}

	template <class T>
	CGPU_EXEC
	T Fcn_Elem<T, efcn_fermi>::operator()(const T& r2) const
	{ 
		return eval_r2(r2);
	}

	template <class T>
	CGPU_EXEC
	R_2d<T> Fcn_Elem<T, efcn_fermi>::operator()(const R_2d<T>& r2) const
	{ 
		return eval_r2(r2);
	}

	template <class T>
	CGPU_EXEC
	R_3d<T>Fcn_Elem<T, efcn_fermi>:: operator()(const R_3d<T>& r2) const
	{ 
		return eval_r2(r2);
	}
}