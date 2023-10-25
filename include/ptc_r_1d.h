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

#pragma once

#include "macros.h"
#include "math_mt.h"
#include "pvctr.h"
#include "vctr_cpu.h"
#include "fcns_cpu.h"

/* template definition */
namespace mt
{
#ifndef PTC_R_DX_DEC
	#define PTC_R_DX_DEC
	template <class T, eDim Dim> class Ptc_R_xd;
#endif
}

/* derived class */
namespace mt
{
	template <class T>
	using Ptc_R_1d = Ptc_R_xd<T, edim_1>;
}

/* template specialization 1d */
namespace mt
{
	template <class T>
	class Ptc_R_xd<T, edim_1>
	{
	public:
		using value_type = T;
		using size_type = dt_int64;

		R_1d<T> bs;				// box size

		mutable Vctr_cpu<T> x;

		R_2d<T> x_lim;

		R_1d<T> r_mean;			// mean position
		R_1d<T> r_std;			// standard deviation
		R_1d<T> sz;				// size

		/************************************* constructors ************************************/
		Ptc_R_xd();

		template <class U>
		Ptc_R_xd(const pVctr_cpu_64<U>& ptc, const dt_int64& icol, const R_1d<U>& bs, dt_bool pbc_x = false, dt_bool b_statistic = true);

		/* copy constructor */
		Ptc_R_xd(const Ptc_R_xd<T, edim_1>& ptc);

		/* converting constructor */
		template <class U>
		Ptc_R_xd(const Ptc_R_xd<U, edim_1>& ptc);

		/******************************** assignment operators *********************************/
		/* copy assignment operator */
		Ptc_R_xd<T, edim_1>& operator=(const Ptc_R_xd<T, edim_1>& ptc);

		/* converting assignment operator */
		template <class U> 
		Ptc_R_xd<T, edim_1>& operator=(const Ptc_R_xd<U, edim_1>& ptc);

		template <class U>
		void assign(const Ptc_R_xd<U, edim_1>& ptc);

		/***************************************************************************************/
		dt_shape_st<size_type> shape() const;

		size_type size() const;

		dt_int32 size_32() const;	

		virtual size_type cols() const;

		dt_bool empty() const;

		void clear();

		void resize(size_type new_size);

		void reserve(size_type new_size);

		void shrink_to_fit();

		void push_back(const R_1d<T>& r);

		template <class U>
		void set_bs(const R_2d<U>& bs);

		R_1d<T> get(const size_type& ia) const;

		void set(const size_type& ia, const R_1d<T>& r);

		R_1d<T> get_pos(const size_type& ia) const;

		void set_pos(const size_type& ia, const R_1d<T>& r);

		template <class U>
		void set_ptc(const Ptc_R_xd<U, edim_1>& ptc, dt_bool pbc_x = false, dt_bool b_statistic = true);

		template <class U>
		void set_ptc(const pVctr_cpu_64<U>& ptc, const dt_int64& icol, const R_1d<U>& bs, dt_bool pbc_x = false, dt_bool b_statistic = true);
		
		/* copy data to pointer */
		template <class U>
		dt_int32 cpy_to_ptr(U *ptc, size_type n_ptc, dt_int32 is_0=0, dt_int32 is_e=1) const;

		// sort by x
		void sort_by_x();

		// sort by idx
		virtual void sort_by_idx(const dt_int32& idx);

		/***************************************************************************************/
		T norm_2_pbc_x(const size_type& ia, const R_1d<T>& r_0) const;			
				
		T norm_2_pbc(const size_type& ia, const R_1d<T>& r_0) const;

		T norm_2(const size_type& ia, const R_1d<T>& r_0) const;

		T norm_2(const size_type& ia_0, const size_type& ia_e) const;

		/***************************************************************************************/
		T norm_pbc_x(const size_type& ia, const R_1d<T>& r_0) const;			
				
		T norm_pbc(const size_type& ia, const R_1d<T>& r_0) const;

		T norm(const size_type& ia, const R_2d<T>& r_0) const;

		T norm(const size_type& ia_0, const size_type& ia_e) const;

		/***************************************************************************************/
		virtual void get_statistic();

		void shift(const R_1d<T>& r_sft);

		void recenter(const R_1d<T>& bs);

		void recenter();

		void apply_ltf(const T mx, const R_1d<T>& p);
	};
}

/* fcns */
namespace mt
{
	template <class T, class U>
	void set_ptc_pbc_x(const Ptc_R_1d<T>& ptc_i, const dt_bool& pbc_x, const dt_bool& b_statistic, Ptc_R_1d<U>& ptc_o);

	template <class T, class U>
	void set_ptc_pbc_x(const pVctr_cpu_64<T>& pvctr, const dt_int64& icol, const R_1d<U>& bs, const dt_bool& pbc_x, const dt_bool& b_statistic, Ptc_R_1d<U>& ptc_o);

	template <class T>
	void fcn_ptc_pos_statistic(Ptc_R_1d<T>& ptc);

	template <class T>
	void fcn_ptc_pos_shift(const R_1d<T>& r_sft, Ptc_R_1d<T>& ptc);

	template <class T>
	void fcn_ptc_pos_recenter(const R_1d<T>& bs, Ptc_R_1d<T>& ptc);

	template <class T>
	void fcn_ptc_pos_apply_ltf(const T& mx, const R_1d<T>& p, Ptc_R_1d<T>& ptc);
}

#include "../src/ptc_r_1d.inl"