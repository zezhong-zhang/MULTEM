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

#pragma once

#include "const_enum.h"
#include "math_mt.h"
#include "r_2d.h"
#include "r_3d.h"

/* template definition */
namespace mt
{
#ifndef FCN_ELEM_DEC
	#define FCN_ELEM_DEC
	template <class T, eFcn_typ Fcn_typ> class Fcn_Elem;
#endif
}

/* derived class */
namespace mt
{
	template <class T>
	using Fcn_Cos_Tap = Fcn_Elem<T, efcn_cos_tap>;
}

namespace mt
{
	template <class T>
	class Fcn_Elem<T, efcn_cos_tap>
	{
	public:
		using value_type = T;

		T r_tap;
		T r_max;
		T coef_tap;

		/************************************* constructors ************************************/
		CGPU_EXEC
		Fcn_Elem();

		Fcn_Elem(const T& r_tap, const T& r_max);

		/* copy constructor */
		CGPU_EXEC
		Fcn_Elem(const Fcn_Elem<T, efcn_cos_tap>& parm);

		/* converting constructor */
		template <class U>
		CGPU_EXEC
		Fcn_Elem(const Fcn_Elem<U, efcn_cos_tap>& parm);

		/******************************** assignment operators *********************************/
		/* copy assignment operator */
		CGPU_EXEC
		Fcn_Elem<T, efcn_cos_tap>& operator=(const Fcn_Elem<T, efcn_cos_tap>& parm);
			
		/* converting assignment operator */
		template <class U>
		CGPU_EXEC
		Fcn_Elem<T, efcn_cos_tap>& operator=(const Fcn_Elem<U, efcn_cos_tap>& parm);

		template <class U>
		CGPU_EXEC
		void assign(const Fcn_Elem<U, efcn_cos_tap>& parm);

		/***************************************************************************************/
		void set_in_data(const T& r_tap, const T& r_max);

		CGPU_EXEC
		void clear();

		/***************************************************************************************/
		CGPU_EXEC
		T eval_r(const T& r, const T& r_tap, const T& r_max) const;

		CGPU_EXEC
		T eval_r(const T& r) const;

		CGPU_EXEC
		R_2d<T> eval_r(const R_2d<T>& r) const;

		CGPU_EXEC
		R_3d<T> eval_r(const R_3d<T>& r) const;

		/***************************************************************************************/
		CGPU_EXEC
		T operator()(const T& r, const T& r_tap, const T& r_max) const;

		CGPU_EXEC
		T operator()(const T& r) const;

		CGPU_EXEC
		R_2d<T> operator()(const R_2d<T>& r) const;

		CGPU_EXEC
		R_3d<T> operator()(const R_3d<T>& r) const;
	};
}


#include "../src/fcn_cos_tap.inl"