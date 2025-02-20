/**
 * @file border_1d.h
 * @brief One-dimensional border handling for rectangular regions
 * @details Provides template classes and type definitions for handling 1D borders
 *          in rectangular regions. Supports various data types and dimensions.
 *
 * @author Ivan Lobato <Ivanlh20@gmail.com>
 * @copyright 2023 Ivan Lobato
 * @license GNU General Public License
 */

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

#include "math_mt.h"
#include "const_enum.h"
#include "fcns_cgpu_gen.h"
#include "vctr_cpu.h"

/* template definition */
namespace mt
{
    /**
     * @brief Base template for rectangular border handling in x dimensions
     * @tparam T Data type (int32, int64, etc.)
     * @tparam Dim Dimension enumeration (edim_1, edim_2, etc.)
     */
    template <class T, eDim Dim> class Border_Rect_xd;

    /**
     * @brief Specialized integer border handling for x dimensions
     * @tparam Dim Dimension enumeration
     */
    template <eDim Dim>
    using iBorder_Rect_xd = Border_Rect_xd<dt_int32, Dim>;
}

/* derived class */
namespace mt
{
    /**
     * @brief One-dimensional rectangular border handling
     * @tparam T Data type for border coordinates
     */
    template <class T>
    using Border_Rect_1d = Border_Rect_xd<T, edim_1>;

    /**
     * @brief 32-bit integer specialized 1D border handling
     */
    using iBorder_Rect_1d = Border_Rect_xd<dt_int32, edim_1>;

    /**
     * @brief 64-bit integer specialized 1D border handling
     */
    using iBorder_Rect_1d_64 = Border_Rect_xd<dt_int64, edim_1>;
}	

/***************************************************************************************/
/******************************** rectangular border ***********************************/
/***************************************************************************************/
/* template specialization 1d */
namespace mt
{
    /**
     * @brief One-dimensional rectangular border handling
     * @tparam T Data type for border coordinates
     */
    template <class T>
    class Border_Rect_xd<T, edim_1>
    {
    public:
        using value_type = T;
        using size_type = dt_int32;

        T bx_0;		// initial x position
        T bx_e;		// final x position

        /**
         * @brief Default constructor
         */
        CGPU_EXEC
        Border_Rect_xd();

        /**
         * @brief Constructor with initial and final x positions
         * @param bx_0 Initial x position
         * @param bx_e Final x position
         */
        Border_Rect_xd(const T& bx_0, const T& bx_e);

        /**
         * @brief Constructor with initialization list
         * @tparam U Data type of initialization list
         * @param list Initialization list
         */
        template <class U>
        Border_Rect_xd(const dt_init_list<U>& list);

        /**
         * @brief Constructor with vector
         * @tparam U Data type of vector
         * @param vctr Vector
         */
        template <class U>
        Border_Rect_xd(const Vctr_cpu<U>& vctr);

        /**
         * @brief Copy constructor
         * @param border Border to copy
         */
        CGPU_EXEC
        Border_Rect_xd(const Border_Rect_xd<T, edim_1>& border);

        /**
         * @brief Converting constructor
         * @tparam U Data type of border to convert
         * @param border Border to convert
         */
        template <class U>
        CGPU_EXEC
        Border_Rect_xd(const Border_Rect_xd<U, edim_1>& border);

        /**
         * @brief Copy assignment operator
         * @param border Border to assign
         * @return Reference to this border
         */
        CGPU_EXEC
        Border_Rect_xd<T, edim_1>& operator=(const Border_Rect_xd<T, edim_1>& border);		
            
        /**
         * @brief Converting assignment operator
         * @tparam U Data type of border to convert
         * @param border Border to convert and assign
         * @return Reference to this border
         */
        template <class U>
        CGPU_EXEC
        Border_Rect_xd<T, edim_1>& operator=(const Border_Rect_xd<U, edim_1>& border);

        /**
         * @brief Assign border from another border
         * @tparam U Data type of border to assign
         * @param border Border to assign
         */
        template <class U> 
        CGPU_EXEC
        void assign(const Border_Rect_xd<U, edim_1>& border);

        /**
         * @brief Set border data from initial and final x positions
         * @tparam U Data type of initial and final x positions
         * @param bx_0 Initial x position
         * @param bx_e Final x position
         */
        template <class U>
        void set_in_data(const U& bx_0, const U& bx_e);

        /**
         * @brief Set border data from initialization list
         * @tparam U Data type of initialization list
         * @param list Initialization list
         */
        template <class U> 
        void set_in_data(const dt_init_list<U>& list);

        /**
         * @brief Set border data from vector
         * @tparam U Data type of vector
         * @param vctr Vector
         */
        template <class U>
        void set_in_data(const Vctr_cpu<U>& vctr);

        /**
         * @brief Clear border data
         */
        CGPU_EXEC
        void clear();
            
        /**
         * @brief Calculate sum of border coordinates
         * @return Sum of border coordinates
         */
        CGPU_EXEC
        T bx_sum() const;

        /**
         * @brief Calculate minimum border coordinate
         * @return Minimum border coordinate
         */
        CGPU_EXEC
        T bx_min() const;

        /**
         * @brief Calculate maximum border coordinate
         * @return Maximum border coordinate
         */
        CGPU_EXEC
        T bx_max() const;

        /**
         * @brief Calculate border size in x direction
         * @param bs_x_i Border size in x direction
         * @return Border size in x direction
         */
        CGPU_EXEC
        T bs_x(const T& bs_x_i) const;

        /**
         * @brief Calculate border size
         * @param bs_i Border size
         * @return Border size
         */
        CGPU_EXEC
        T bs(const T& bs_i) const;

        /**
         * @brief Calculate minimum border size
         * @param bs Border size
         * @return Minimum border size
         */
        CGPU_EXEC
        T bs_min(const T& bs) const;

        /**
         * @brief Calculate maximum border size
         * @param bs Border size
         * @return Maximum border size
         */
        CGPU_EXEC
        T bs_max(const T& bs) const;

        /**
         * @brief Calculate half border size in x direction
         * @param bs_x Border size in x direction
         * @return Half border size in x direction
         */
        CGPU_EXEC
        T bs_x_h(const T& bs_x) const;

        /**
         * @brief Calculate half border size
         * @param bs Border size
         * @return Half border size
         */
        CGPU_EXEC
        T bs_h(const T& bs) const;

        /**
         * @brief Calculate center coordinate in x direction
         * @param bs_x Border size in x direction
         * @return Center coordinate in x direction
         */
        CGPU_EXEC
        T rx_c(const T& bs_x) const;

        /**
         * @brief Calculate center coordinate
         * @param bs Border size
         * @return Center coordinate
         */
        CGPU_EXEC
        T r_c(const T& bs) const;

        /**
         * @brief Calculate radius in x direction
         * @param bs_x Border size in x direction
         * @return Radius in x direction
         */
        CGPU_EXEC
        T radius_x(const T& bs_x) const;

        /**
         * @brief Calculate radius
         * @param bs Border size
         * @return Radius
         */
        CGPU_EXEC
        T radius(const T& bs) const;

        /**
         * @brief Calculate radius in x direction with power scaling
         * @details This function calculates the radius in x direction scaled by (1-p),
         *          where p is a power factor. The result is (1-p) * radius_x(bs_x).
         * @param bs_x Border size in x direction
         * @param p Power scaling factor
         * @return Scaled radius in x direction: (1-p) * radius_x(bs_x)
         * @see radius_x
         */
        CGPU_EXEC
        T radius_x_p(const T& bs_x, const T& p) const;

        /**
         * @brief Calculate radius with power
         * @param bs Border size
         * @param p Power
         * @return Radius with power
         */
        CGPU_EXEC
        T radius_p(const T& bs, const T& p) const;

        /**
         * @brief Set border by shifting
         * @param bs Border size
         * @param dr Shift amount
         */
        void set_by_sft(const T& bs, const T& dr);

        /**
         * @brief Shift border
         * @param bs Border size
         * @param dr Shift amount
         */
        void sft_bdr(const T& bs, const T& dr);
    };
}

#include "../src/border_1d.inl"