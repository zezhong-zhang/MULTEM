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

#include "grid_2d.h"

/* template specialization 2d */
namespace mt
{
	/************************************* constructors ************************************/
	template <class T, class ST>
	Grid_sxd<T, ST, edim_2>::Grid_sxd(): iGrid_sxd<ST, edim_2>(), bs_x(0), bs_y(0), 
		rx_0(0), ry_0(0), pbc_x(true), pbc_y(true), bwl(false), sli_thick(0), 
		drx(0), dry(0), dgx(0), dgy(0) {}

	template <class T, class ST>
	Grid_sxd<T, ST, edim_2>::Grid_sxd(const ST& nx, const ST& ny)
	{
		set_in_data(nx, ny);
	}

	template <class T, class ST>
	template <class U, class SU>
	Grid_sxd<T, ST, edim_2>::Grid_sxd(const U& bs_x, const U& bs_y, const SU& nx, const SU& ny)
	{
		set_in_data(bs_x, bs_y, nx, ny);
	}

	template <class T, class ST>
	template <class U, class SU>
	Grid_sxd<T, ST, edim_2>::Grid_sxd(const U& bs_x, const U& bs_y, const SU& nx, const SU& ny, const U& rx_0, const U& ry_0, 
	dt_bool pbc_x, dt_bool pbc_y, dt_bool bwl, U sli_thick)
	{
		set_in_data(bs_x, bs_y, nx, ny, rx_0, ry_0, pbc_x, pbc_y, bwl, sli_thick);
	}

	/* copy constructor */
	template <class T, class ST>
	CGPU_EXEC
	Grid_sxd<T, ST, edim_2>::Grid_sxd(const Grid_sxd<T, ST, edim_2>& grid)
	{
		*this = grid;
	}

	/* converting constructor */
	template <class T, class ST>
	template <class U, class SU>
	CGPU_EXEC
	Grid_sxd<T, ST, edim_2>::Grid_sxd(const Grid_sxd<U, SU, edim_2>& grid)
	{
		*this = grid;
	}

	/******************************** assignment operators *********************************/
	/* copy assignment operator */
	template <class T, class ST>
	CGPU_EXEC
	Grid_sxd<T, ST, edim_2>& Grid_sxd<T, ST, edim_2>::operator=(const Grid_sxd<T, ST, edim_2>& grid)
	{
		if (this != &grid)
		{
			iGrid_sxd<ST, edim_2>::operator=(grid);

			bs_x = grid.bs_x;
			bs_y = grid.bs_y;
			rx_0 = grid.rx_0;
			ry_0 = grid.ry_0;
			pbc_x = grid.pbc_x;
			pbc_y = grid.pbc_y;
			bwl = grid.bwl;
			sli_thick = grid.sli_thick;

			drx = grid.drx;
			dry = grid.dry;
			dgx = grid.dgx;
			dgy = grid.dgy;
		}

		return *this;
	}

	/* converting assignment operator */
	template <class T, class ST>
	template <class U, class SU>
	CGPU_EXEC
	Grid_sxd<T, ST, edim_2>& Grid_sxd<T, ST, edim_2>::operator=(const Grid_sxd<U, SU, edim_2>& grid)
	{
		iGrid_sxd<ST, edim_2>::operator=(grid);

		bs_x = T(grid.bs_x);
		bs_y = T(grid.bs_y);
		rx_0 = T(grid.rx_0);
		ry_0 = T(grid.ry_0);
		pbc_x = grid.pbc_x;
		pbc_y = grid.pbc_y;
		bwl = grid.bwl;
		sli_thick = T(grid.sli_thick);

		drx = T(grid.drx);
		dry = T(grid.dry);
		dgx = T(grid.dgx);
		dgy = T(grid.dgy);

		return *this;
	}

	template <class T, class ST>
	template <class U, class SU> 
	CGPU_EXEC
	void Grid_sxd<T, ST, edim_2>::assign(const Grid_sxd<U, SU, edim_2>& grid)
	{
		*this = grid;
	}

	/************************** user define conversion operators ***************************/
	template <class T, class ST>
	Grid_sxd<T, ST, edim_2>::operator iRegion_Rect_xd<edim_2>() const
	{
		return {ST(0), this->nx, ST(0), this->ny};
	}

	/***************************************************************************************/
	template <class T, class ST>
	void Grid_sxd<T, ST, edim_2>::set_in_data(const ST& nx, const ST& ny)
	{
		set_in_data(T(nx), T(ny), nx, ny, T(0), T(0));
	}

	template <class T, class ST>
	template <class U, class SU>
	void Grid_sxd<T, ST, edim_2>::set_in_data(const U& bs_x, const U& bs_y, const SU& nx, const SU& ny)
	{
		set_in_data(T(bs_x), T(bs_y), ST(nx), ST(ny), T(0), T(0));
	}

	template <class T, class ST>
	template <class U, class SU>
	void Grid_sxd<T, ST, edim_2>::set_in_data(const U& bs_x, const U& bs_y, const SU& nx, const SU& ny, 
	const U& rx_0, const U& ry_0, dt_bool pbc_x, dt_bool pbc_y, dt_bool bwl, U sli_thick)
	{
		this->set_size(nx, ny);

		this->bs_x = T(bs_x);
		this->bs_y = T(bs_y);
		this->rx_0 = T(rx_0);
		this->ry_0 = T(ry_0);
		this->pbc_x = pbc_x;
		this->pbc_y = pbc_y;
		this->bwl = bwl;
		this->sli_thick = T(sli_thick);

		set_dep_var();
	}

	template <class T, class ST>
	void Grid_sxd<T, ST, edim_2>::set_dep_var()
	{
		drx = mt::fcn_div(bs_x, this->nx);
		dry = mt::fcn_div(bs_y, this->ny);
		dgx = mt::fcn_div(T(1), bs_x);
		dgy = mt::fcn_div(T(1), bs_y);
	}

	template <class T, class ST>
	CGPU_EXEC
	void Grid_sxd<T, ST, edim_2>::set_r_0(const T& rx_0, const T& ry_0) 
	{	
		this->rx_0 = rx_0;
		this->ry_0 = ry_0;
	}

	template <class T, class ST>
	CGPU_EXEC
	void Grid_sxd<T, ST, edim_2>::set_r_0(const R_2d<T>& r_0) 
	{	
		set_r_0(r_0.x, r_0.y);
	}

	/***************************************************************************************/
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::nx_r() const
	{ 
		return T(this->nx);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::ny_r() const 
	{ 
		return T(this->ny);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::size_r() const 
	{ 
		return T(this->size());
	}

	template <class T, class ST>
	T Grid_sxd<T, ST, edim_2>::isize_r() const
	{ 
		return T(1)/size_r();
	}

	/***************************************************************************************/
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::bs_x_h() const 
	{ 
		return T(0.5)*bs_x;
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::bs_y_h() const 
	{ 
		return T(0.5)*bs_y;
	}

	template <class T, class ST>
	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::bs_h() const 
	{ 
		return {bs_x_h(), bs_y_h()};
	}

	/***************************************************************************************/
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::bs_min() const 
	{ 
		return ::fmin(bs_x, bs_y);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::bs_max() const 
	{ 
		return ::fmax(bs_x, bs_y);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::bs_h_min() const 
	{ 
		return T(0.5)*bs_min();
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::bs_h_max() const 
	{ 
		return T(0.5)*bs_max();
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::rx_c() const 
	{ 
		return rx_0 + bs_x_h();
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::ry_c() const 
	{ 
		return ry_0 + bs_y_h();
	}

	template <class T, class ST>
	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::rv_c() const 
	{ 
		return {rx_c(), ry_c()};
	}

	/***************************************************************************************/
	// maximum frequency
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::g_max() const 
	{ 
		return ::fmin(gx_back(), gy_back());
	}

	// maximum square frequency
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::g2_max() const 
	{ 
		return ::square(g_max());
	}

	// maximum allowed frequency
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gl_max() const
	{
		return g_max()*T(2.0/3.0);
	}

	// maximum square allowed frequency
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gl2_max() const
	{
		return ::square(gl_max());
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::r_0_min() const 
	{ 
		return ::fmin(rx_0, ry_0);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::dr_min() const 
	{ 
		return ::fmin(drx, dry);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::dg_min() const 
	{ 
		return ::fmin(dgx, dgy);
	}

	/***************************************************************************************/
	/***************************** Fourier space positions *********************************/
	/***************************************************************************************/
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gx(const ST& ix) const 
	{ 
		return T(this->igx(ix))*dgx;
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gy(const ST& iy) const 
	{ 
		return T(this->igy(iy))*dgy;
	}

	template <class T, class ST>
	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::gv(const ST& ix, const ST& iy) const 
	{ 
		return {gx(ix), gy(iy)};
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gx2(const ST& ix) const 
	{ 
		return ::square(gx(ix));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gy2(const ST& iy) const 
	{ 
		return ::square(gy(iy));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::g2(const ST& ix, const ST& iy) const 
	{ 
		return gx2(ix) + gy2(iy);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::g(const ST& ix, const ST& iy) const 
	{ 
		return ::sqrt(g2(ix, iy));
	}

	/***************************************************************************************/
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gx(const ST& ix, const T& gx_0) const 
	{ 
		return gx(ix) - gx_0;
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gy(const ST& iy, const T& gy_0) const 
	{ 
		return gy(iy) - gy_0;
	}

	template <class T, class ST>
	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::gv(const ST& ix, const ST& iy, const T& gx_0, const T& gy_0) const 
	{ 
		return {gx(ix, gx_0), gy(iy, gy_0)};
	}

	template <class T, class ST>
	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::gv(const ST& ix, const ST& iy, const R_2d<T>& g_0) const 
	{ 
		return gv(ix, iy, g_0.x, g_0.y);
	}
			
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gx2(const ST& ix, const T& gx_0) const 
	{ 
		return ::square(gx(ix, gx_0));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gy2(const ST& iy, const T& gy_0) const 
	{ 
		return ::square(gy(iy, gy_0));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::g2(const ST& ix, const ST& iy, const T& gx_0, const T& gy_0) const 
	{ 
		return gx2(ix, gx_0) + gy2(iy, gy_0);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::g2(const ST& ix, const ST& iy, const R_2d<T>& g0) const 
	{ 
		return gx2(ix, iy, g0.x, g0.y);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::g(const ST& ix, const ST& iy, const T& gx_0, const T& gy_0) const 
	{ 
		return ::sqrt(g2(ix, iy, gx_0, gy_0));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::g(const ST& ix, const ST& iy, const R_2d<T>& g0) const 
	{ 
		return ::sqrt(g2(ix, iy, g0));
	}

	/************************************* shift *******************************************/
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gx_sft(const ST& ix) const 
	{ 
		return T(this->igx_sft(ix))*dgx;
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gy_sft(const ST& iy) const 
	{ 
		return T(this->igy_sft(iy))*dgy;
	}

	template <class T, class ST>
	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::gv_sft(const ST& ix, const ST& iy) const 
	{ 
		return {gx_sft(ix), gy_sft(iy)};
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gx2_sft(const ST& ix) const 
	{ 
		return ::square(gx_sft(ix));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gy2_sft(const ST& iy) const 
	{ 
		return ::square(gy_sft(iy));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::g2_sft(const ST& ix, const ST& iy) const 
	{ 
		return gx2_sft(ix) + gy2_sft(iy);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::g_sft(const ST& ix, const ST& iy) const 
	{ 
		return ::sqrt(g2_sft(ix, iy));
	}

	/***************************************************************************************/
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gx_sft(const ST& ix, const T& gx_0) const 
	{ 
		return gx_sft(ix) - gx_0;
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gy_sft(const ST& iy, const T& gy_0) const 
	{ 
		return gy_sft(iy) - gy_0;
	}

	template <class T, class ST>
	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::gv_sft(const ST& ix, const ST& iy, const T& gx_0, const T& gy_0) const 
	{ 
		return {gx_sft(ix, gx_0), gy_sft(iy, gy_0)};
	}

	template <class T, class ST>
	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::gv_sft(const ST& ix, const ST& iy, const R_2d<T>& g_0) const 
	{ 
		return gv_sft(ix, iy, g_0.x, g_0.y);
	}
			
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gx2_sft(const ST& ix, const T& gx_0) const 
	{ 
		return ::square(gx_sft(ix, gx_0));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gy2_sft(const ST& iy, const T& gy_0) const 
	{ 
		return ::square(gy_sft(iy, gy_0));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::g2_sft(const ST& ix, const ST& iy, const T& gx_0, const T& gy_0) const 
	{ 
		return gx2_sft(ix, gx_0) + gy2_sft(iy, gy_0);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::g2_sft(const ST& ix, const ST& iy, const R_2d<T>& g0) const 
	{ 
		return g2_sft(ix, iy, g0.z, g0.y);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::g_sft(const ST& ix, const ST& iy, const T& gx_0, const T& gy_0) const 
	{ 
		return ::sqrt(g2_sft(ix, iy, gx_0, gy_0));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::g_sft(const ST& ix, const ST& iy, const R_2d<T>& g0) const 
	{ 
		return ::sqrt(g2_sft(ix, iy, g0));
	}

	/***************************************************************************************/
	/******************************* real space positions **********************************/
	/***************************************************************************************/
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::rx(const ST& ix) const 
	{ 
		return T(this->irx(ix))*drx + rx_0;
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::ry(const ST& iy) const 
	{ 
		return T(this->iry(iy))*dry + ry_0;
	}

	template <class T, class ST>
	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::rv(const ST& ix, const ST& iy) const 
	{ 
		return {rx(ix), ry(iy)};
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::rx2(const ST& ix) const 
	{ 
		return ::square(rx(ix));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::ry2(const ST& iy) const 
	{ 
		return ::square(ry(iy));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::r2(const ST& ix, const ST& iy) const 
	{ 
		return rx2(ix) + ry2(iy);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::r(const ST& ix, const ST& iy) const 
	{ 
		return ::sqrt(r2(ix, iy));
	}

	/***************************************************************************************/
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::rx(const ST& ix, const T& x0) const 
	{ 
		return rx(ix) - x0;
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::ry(const ST& iy, const T& y0) const 
	{ 
		return ry(iy) - y0;
	}

	template <class T, class ST>
	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::rv(const ST& ix, const ST& iy, const T& x0, const T& y0) const 
	{ 
		return {rx(ix, x0), ry(iy, y0)};
	}

	template <class T, class ST>
	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::rv(const ST& ix, const ST& iy, const R_2d<T>& r_0) const 
	{ 
		return rv(ix, iy, r_0.x, r_0.y);
	}
		
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::rx2(const ST& ix, const T& x0) const 
	{ 
		return ::square(rx(ix, x0));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::ry2(const ST& iy, const T& y0) const 
	{ 
		return ::square(ry(iy, y0));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::r2(const ST& ix, const ST& iy, const T& x0, const T& y0) const 
	{ 
		return rx2(ix, x0) + ry2(iy, y0);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::r2(const ST& ix, const ST& iy, const R_2d<T>& r_0) const 
	{ 
		return r2(ix, iy, r_0.x, r_0.y);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::r(const ST& ix, const ST& iy, const T& x0, const T& y0) const 
	{ 
		return ::sqrt(r2(ix, iy, x0, y0));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::r(const ST& ix, const ST& iy, const R_2d<T>& r_0) const 
	{ 
		return ::sqrt(r2(ix, iy, r_0));
	}

	/************************************* shift *******************************************/
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::rx_sft(const ST& ix) const 
	{ 
		return T(this->irx_sft(ix))*drx + rx_0;
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::ry_sft(const ST& iy) const 
	{ 
		return T(this->iry_sft(iy))*dry + ry_0;
	}

	template <class T, class ST>
	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::rv_sft(const ST& ix, const ST& iy) const 
	{ 
		return {rx_sft(ix), ry_sft(iy)};
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::rx2_sft(const ST& ix) const 
	{ 
		return ::square(rx_sft(ix));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::ry2_sft(const ST& iy) const 
	{ 
		return ::square(ry_sft(iy));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::r2_sft(const ST& ix, const ST& iy) const 
	{ 
		return rx2_sft(ix) + ry2_sft(iy);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::r_sft(const ST& ix, const ST& iy) const 
	{ 
		return ::sqrt(r2_sft(ix, iy));
	}

	/***************************************************************************************/
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::rx_sft(const ST& ix, const T& x0) const 
	{ 
		return rx_sft(ix) - x0;
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::ry_sft(const ST& iy, const T& y0) const 
	{ 
		return ry_sft(iy) - y0;
	}

	template <class T, class ST>
	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::rv_sft(const ST& ix, const ST& iy, const T& x0, const T& y0) const 
	{ 
		return {rx_sft(ix, x0), ry_sft(iy, y0)};
	}
			
	template <class T, class ST>
	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::rv_sft(const ST& ix, const ST& iy, const R_2d<T>& r_0) const 
	{ 
		return rv_sft(ix, iy, r_0.x, r_0.y);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::rx2_sft(const ST& ix, const T& x0) const 
	{ 
		return ::square(rx_sft(ix, x0));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::ry2_sft(const ST& iy, const T& y0) const 
	{ 
		return ::square(ry_sft(iy, y0));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::r2_sft(const ST& ix, const ST& iy, const T& x0, const T& y0) const 
	{ 
		return rx2_sft(ix, x0) + ry2_sft(iy, y0);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::r2_sft(const ST& ix, const ST& iy, const R_2d<T>& r_0) const 
	{ 
		return r2_sft(ix, iy, r_0.x, r_0.y);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::r_sft(const ST& ix, const ST& iy, const T& x0, const T& y0) const 
	{ 
		return ::sqrt(r2_sft(ix, iy, x0, y0));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::r_sft(const ST& ix, const ST& iy, const R_2d<T>& r_0) const 
	{ 
		return ::sqrt(r2_sft(ix, iy, r_0));
	}

	/***************************************************************************************/
	/***************************** from position to index **********************************/
	/***************************************************************************************/
	template <class T, class ST>
	template <class SU>
	void Grid_sxd<T, ST, edim_2>::ix_0_ix_n(const T& x, const T& x_max, SU& ix_0, SU& ix_n) const 
	{
		fcn_get_idx_0_idx_n(x, x_max, drx, pbc_x, this->nx-1, ix_0, ix_n);
	}

	template <class T, class ST>
	template <class SU>
	void Grid_sxd<T, ST, edim_2>::iy_0_iy_n(const T& y, const T& y_max, SU& iy_0, SU& iy_n) const 
	{
		fcn_get_idx_0_idx_n(y, y_max, dry, pbc_y, this->ny-1, iy_0, iy_n);
	}

	template <class T, class ST>
	template <class SU>
	void Grid_sxd<T, ST, edim_2>::ix_0_ix_e(const T& x, const T& x_max, SU& ix_0, SU& ix_e) const 
	{
		fcn_get_idx_0_idx_n(x, x_max, drx, pbc_x, this->nx-1, ix_0, ix_e);
		ix_e += ix_0;
	}

	template <class T, class ST>
	template <class SU>
	void Grid_sxd<T, ST, edim_2>::iy_0_iy_e(const T& y, const T& y_max, SU& iy_0, SU& iy_e) const 
	{
		fcn_get_idx_0_idx_n(y, y_max, dry, pbc_y, this->ny-1, iy_0, iy_e);
		iy_e += iy_0;
	}

	/************ fds = floor/division by pixel size ************/
	// locate x -> irx
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rx_2_irx_fd(const T& x) const 
	{ 
		return fcn_cfloor<ST>(x/drx);
	}

	// locate y -> iry
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::ry_2_iry_fd(const T& y) const 
	{ 
		return fcn_cfloor<ST>(y/dry);
	}

	// locate x, y -> ind
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rv_2_ir_fd(const T& x, const T& y) const 
	{ 
		return this->sub_2_ind(rx_2_irx_fd(x), ry_2_iry_fd(y));
	}

	// locate r -> ir using dr_min
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::r_2_ir_fd_dr_min(const T& x) const 
	{ 
		return fcn_cfloor<ST>(x/dr_min());
	}

	/********* bfds = bound/floor/division by pixel size ********/
	// locate x -> irx
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rx_2_irx_bfd(const T& x) const 
	{ 
		return fcn_set_bound(rx_2_irx_fd(x), ST(0), this->nx-1);
	}

	// locate y -> iry
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::ry_2_iry_bfd(const T& y) const 
	{ 
		return fcn_set_bound(ry_2_iry_fd(y), ST(0), this->ny-1);
	}

	// locate x, y -> ind
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rv_2_ir_bfd(const T& x, const T& y) const 
	{ 
		return this->sub_2_ind(rx_2_irx_bfd(x), ry_2_iry_bfd(y));
	}

	/********* cds = ceil/division by pixel size/shift **********/
	// locate x -> irx
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rx_2_irx_cd(const T& x) const 
	{ 
		return fcn_cceil<ST>(x/drx);
	}

	// locate y -> iry
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::ry_2_iry_cd(const T& y) const 
	{ 
		return fcn_cceil<ST>(y/dry);
	}

	// locate x, y -> ind
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rv_2_ir_cd(const T& x, const T& y) const 
	{ 
		return this->sub_2_ind(rx_2_irx_cd(x), ry_2_iry_cd(y));
	}

	// locate r -> ir using dr_min
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::r_2_ir_cd_dr_min(const T& x) const 
	{ 
		return static_cast<ST>(::ceil(x/dr_min()));
	}

	/****** bcds = bound/ceil/division by pixel size/shift ******/
	// locate x -> irx
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rx_2_irx_bcd(const T& x) const 
	{ 
		return fcn_set_bound(rx_2_irx_cd(x), ST(0), this->nx-1);
	}

	// locate y -> iry
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::ry_2_iry_bcd(const T& y) const 
	{ 
		return fcn_set_bound(ry_2_iry_cd(y), ST(0), this->ny-1);
	}

	// locate x, y -> ind
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rv_2_ir_bcd(const T& x, const T& y) const 
	{ 
		return this->sub_2_ind(rx_2_irx_bcd(x), ry_2_iry_bcd(y));
	}

	/********* fds = floor/division by pixel size/shift *********/
	// locate x -> irx
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rx_2_irx_fds(const T& x) const 
	{ 
		return fcn_cfloor<ST>((x - rx_0)/drx);
	}

	// locate y -> iry
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::ry_2_iry_fds(const T& y) const 
	{ 
		return fcn_cfloor<ST>((y - ry_0)/dry);
	}

	// locate x, y -> ind
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rv_2_ir_fds(const T& x, const T& y) const 
	{ 
		return this->sub_2_ind(rx_2_irx_fds(x), ry_2_iry_fds(y));
	}

	// locate r -> ir using dr_min
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::r_2_ir_fds_dr_min(const T& x) const 
	{ 
		return fcn_cfloor<ST>((x - r_0_min())/dr_min());
	}

	/****** bfds = bound/floor/division by pixel size/shift ******/
	// locate x -> irx
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rx_2_irx_bfds(const T& x) const 
	{ 
		return fcn_set_bound(rx_2_irx_fds(x), ST(0), this->nx-1);
	}

	// locate y -> iry
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::ry_2_iry_bfds(const T& y) const 
	{ 
		return fcn_set_bound(ry_2_iry_fds(y), ST(0), this->ny-1);
	}

	// locate x, y -> ind
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rv_2_ir_bfds(const T& x, const T& y) const 
	{ 
		return this->sub_2_ind(rx_2_irx_bfds(x), ry_2_iry_bfds(y));
	}

	/********* cds = ceil/division by pixel size/shift **********/
	// locate x -> irx
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rx_2_irx_cds(const T& x) const 
	{ 
		return fcn_cceil<ST>((x - rx_0)/drx);
	}

	// locate y -> iry
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::ry_2_iry_cds(const T& y) const 
	{ 
		return fcn_cceil<ST>((y - ry_0)/dry);
	}

	// locate x, y -> ind
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rv_2_ir_cds(const T& x, const T& y) const 
	{ 
		return this->sub_2_ind(rx_2_irx_cds(x), ry_2_iry_cds(y));
	}

	// locate r -> ir using dr_min
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::r_2_ir_cds_dr_min(const T& x) const 
	{ 
		return fcn_cceil<ST>((x - r_0_min())/dr_min());
	}

	/****** bcds = bound/ceil/division by pixel size/shift *******/
	// locate x -> irx
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rx_2_irx_bcds(const T& x) const 
	{ 
		return fcn_set_bound(rx_2_irx_cds(x), ST(0), this->nx-1);
	}

	// locate y -> iry
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::ry_2_iry_bcds(const T& y) const 
	{ 
		return fcn_set_bound(ry_2_iry_cds(y), ST(0), this->ny-1);
	}

	// locate x, y -> ind
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rv_2_ir_bcds(const T& x, const T& y) const 
	{ 
		return this->sub_2_ind(rx_2_irx_bcds(x), ry_2_iry_bcds(y));
	}

	/************ From position to index by searching ***********/
	// locate x -> irx
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::rx_2_irx_b(const T& x, ST ix_min, ST ix_max) const 
	{
		if (ix_min == ix_max)
		{
			ix_min = ST(0);
			ix_max = this->nx-1;
		}

		return fcn_r_2_ir_b_by_fcn(rx, ix_min, ix_max);
	}

	// locate y -> iry
	template <class T, class ST>
	CGPU_EXEC
	ST Grid_sxd<T, ST, edim_2>::ry_2_iry_b(const T& y, ST iy_min, ST iy_max) const 
	{
		if (iy_min == iy_max)
		{
			iy_min = ST(0);
			iy_max = this->ny-1;
		}

		return fcn_r_2_ir_b_by_fcn(ry, iy_min, iy_max);
	}

	/***************************************************************************************/
	/********************************** check bounds ***************************************/
	/***************************************************************************************/
	template <class T, class ST>
	CGPU_EXEC
	dt_bool Grid_sxd<T, ST, edim_2>::chk_bound_x(const T& x) const 
	{ 
		return fcn_chk_bound(x, rx_front(), rx_back());
	}

	template <class T, class ST>
	CGPU_EXEC
	dt_bool Grid_sxd<T, ST, edim_2>::chk_bound_y(const T& y) const 
	{ 
		return fcn_chk_bound(y, ry_front(), ry_back());
	}

	template <class T, class ST>
	CGPU_EXEC
	dt_bool Grid_sxd<T, ST, edim_2>::chk_bound(const R_2d<T>& r) const 
	{ 
		return chk_bound_x(r.x) && chk_bound_y(r.y);
	}

	template <class T, class ST>
	CGPU_EXEC
	dt_bool Grid_sxd<T, ST, edim_2>::chk_bound_x_eps(const T& x) const 
	{ 
		return fcn_chk_bound_eps(x, rx_front(), rx_back());
	}

	template <class T, class ST>
	CGPU_EXEC
	dt_bool Grid_sxd<T, ST, edim_2>::chk_bound_y_eps(const T& y) const 
	{ 
		return fcn_chk_bound_eps(y, ry_front(), ry_back());
	}

	template <class T, class ST>
	CGPU_EXEC
	dt_bool Grid_sxd<T, ST, edim_2>::chk_bound_eps(const R_2d<T>& r) const 
	{ 
		return chk_bound_x_eps(r.x) && chk_bound_y_eps(r.y);
	}

	/***************************************************************************************/
	/*********************************** set bounds ****************************************/
	/***************************************************************************************/
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::set_bound_x(const T& x) const 
	{ 
		return fcn_set_bound(x, rx_front(), rx_back());
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::set_bound_y(const T& y) const 
	{ 
		return fcn_set_bound(y, ry_front(), ry_back());
	}

	template <class T, class ST>
	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::set_bound(const R_2d<T>& r) const 
	{ 
		return {set_bound_x(r.x), set_bound_y(r.y)};
	}

	/***************************************************************************************/
	/************************************ front/back ***************************************/
	/***************************************************************************************/

	template <class T, class ST>	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gx_front() const 
	{ 
		return gx(ST(0));
	}

	template <class T, class ST>	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gx_back() const 
	{ 
		return gx(this->nx-1);
	}

	template <class T, class ST>	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gy_front() const 
	{ 
		return gy(ST(0));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::gy_back() const 
	{ 
		return gy(this->ny-1);
	}

	/***************************************************************************************/	
	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::rx_front() const 
	{ 
		return rx(ST(0));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::rx_back() const 
	{ 
		return rx(this->nx-1);
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::ry_front() const 
	{ 
		return ry(ST(0));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::ry_back() const 
	{ 
		return ry(this->ny-1);
	}

	/***************************************************************************************/
	/*********************************** factors *******************************************/
	/* calculate fermi low-pass filter alpha parameter */
	template <class T, class ST>	
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::fermi_lpf_alpha() const
	{ 
		return fcn_fermi_lpf_alpha(gl_max(), T(0.25), T(1e-02));
	}

	template <class T, class ST>
	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::factor_2pi_rx_ctr(const T& x) const 
	{ 
		return fcn_n_2pi_sft(x, bs_x_h());
	}


	template <class T, class ST>	CGPU_EXEC
	T Grid_sxd<T, ST, edim_2>::factor_2pi_ry_ctr(const T& y) const 
	{ 
		return fcn_n_2pi_sft(y, bs_y_h());
	}


	template <class T, class ST>	CGPU_EXEC
	R_2d<T> Grid_sxd<T, ST, edim_2>::factor_2pi_rv_ctr(const R_2d<T>& r) const
	{
		return {factor_2pi_rx_ctr(r.x), factor_2pi_ry_ctr(r.y)};
	}


	template <class T, class ST>	CPU_EXEC
	Vctr<R_2d<T>, edev_cpu> Grid_sxd<T, ST, edim_2>::factor_2pi_rv_ctr(const Vctr<R_2d<T>, edev_cpu>& rv) const
	{
		Vctr<R_2d<T>, edev_cpu> rv_o(rv.size());

		for(auto ik = 0; ik<rv.size(); ik++)
		{
			rv_o[ik] = factor_2pi_rv_ctr(rv[ik]);
		}

		return rv_o;
	}

	/***************************************************************************************/
	template <class T, class ST>
	iRegion_Rect_xd<edim_2> Grid_sxd<T, ST, edim_2>::iregion_rect(const R_2d<T>& r, const T& radius) const 
	{ 
		return {rx_2_irx_bfds(r.x - radius), rx_2_irx_bcds(r.x + radius), ry_2_iry_bfds(r.y - radius), ry_2_iry_bcds(r.y + radius)};
	}

	template <class T, class ST>	
	iRegion_Rect_xd<edim_2> Grid_sxd<T, ST, edim_2>::iregion_rect(const R_2d<T>& r, const T& f0, const T& a, const T& b, const T& c)
	{
		const T d = log(f0);
		const T dd = c*c-T(4)*a*b;

		const T radius_x = ::sqrt(T(4)*b*d/dd);
		const T radius_y = ::sqrt(T(4)*a*d/dd);

		return {rx_2_irx_bfds(r.x - radius_x), rx_2_irx_bcds(r.x + radius_x), ry_2_iry_bfds(r.y - radius_y), ry_2_iry_bcds(r.y + radius_y)};
	}
}

/* traits */
namespace mt
{
	template <class T>
	struct is_grid_2d: std::integral_constant<dt_bool, std::is_same<T, Grid_2d_st<typename T::value_type, typename T::size_type>>::value> {};	

	/***************************************************************************************/
	template <class T, class U>
	struct is_grid_2d_and_vctr_cpu: std::integral_constant<dt_bool, is_grid_2d<T>::value && is_vctr_cpu<U>::value> {};

	template <class T, class U>
	struct is_grid_2d_and_vctr_gpu: std::integral_constant<dt_bool, is_grid_2d<T>::value && is_vctr_gpu<U>::value> {};

	/***************************************************************************************/
	template <class T, class U>
	struct is_grid_2d_and_cvctr_cpu: std::integral_constant<dt_bool, is_grid_2d<T>::value && is_cvctr_cpu<U>::value> {};

	template <class T, class U>
	struct is_grid_2d_and_cvctr_gpu: std::integral_constant<dt_bool, is_grid_2d<T>::value && is_cvctr_gpu<U>::value> {};

	/***************************************************************************************/
	template <class T, class U, class V=void>
	using enable_if_grid_2d_and_vctr_cpu = typename std::enable_if<is_grid_2d_and_vctr_cpu<T, U>::value, V>::type;

	template <class T, class U, class V=void>
	using enable_if_grid_2d_and_vctr_gpu = typename std::enable_if<is_grid_2d_and_vctr_gpu<T, U>::value, V>::type;

	/***************************************************************************************/
	template <class T, class U, class V=void>
	using enable_if_grid_2d_and_cvctr_cpu = typename std::enable_if<is_grid_2d_and_cvctr_cpu<T, U>::value, V>::type;

	template <class T, class U, class V=void>
	using enable_if_grid_2d_and_cvctr_gpu = typename std::enable_if<is_grid_2d_and_cvctr_gpu<T, U>::value, V>::type;
}