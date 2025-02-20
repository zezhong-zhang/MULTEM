/**
 * @file border_2d.h
 * @brief Two-dimensional border handling for electron microscopy simulations
 * @details Provides classes and utilities for handling 2D borders and boundaries
 *          in both real and reciprocal space. This is essential for proper boundary
 *          conditions in electron microscopy simulations.
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

#include "r_2d.h"
#include "border_1d.h"

/* derived class */
namespace mt
{
    /**
     * @brief Two-dimensional rectangular border with custom value type
     * @tparam T Value type for border coordinates
     */
    template <class T>
    using Border_Rect_2d = Border_Rect_xd<T, edim_2>;

    /**
     * @brief Two-dimensional rectangular border with 32-bit integer coordinates
     */
    using iBorder_Rect_2d = Border_Rect_xd<dt_int32, edim_2>;

    /**
     * @brief Two-dimensional rectangular border with 64-bit integer coordinates
     */
    using iBorder_Rect_2d_64 = Border_Rect_xd<dt_int64, edim_2>;
}

/* template specialization 2d */
namespace mt
{
    /**
     * @brief Specialized implementation of two-dimensional rectangular border
     * @details Extends the one-dimensional border to handle 2D boundaries with
     *          additional functionality for y-direction operations.
     *
     * @tparam T Value type for border coordinates
     */
    template <class T>
    class Border_Rect_xd<T, edim_2>: public Border_Rect_xd<T, edim_1>
    {       
    public:         
        /** @brief Value type for border coordinates */
        using value_type = T;
        
        /** @brief Size type for indexing */
        using size_type = dt_int32;

        /** @brief Initial y position of the border */
        T by_0;

        /** @brief Final y position of the border */
        T by_e;

        /**
         * @brief Default constructor
         */
        CGPU_EXEC
        Border_Rect_xd();

        /**
         * @brief Construct border with specified dimensions
         * @param bx_0 Initial x position
         * @param bx_e Final x position
         * @param by_0 Initial y position
         * @param by_e Final y position
         */
        Border_Rect_xd(const T& bx_0, const T& bx_e, const T& by_0, const T& by_e);

        /**
         * @brief Construct border from initializer list
         * @tparam U Value type of initializer list
         * @param list Initializer list with border dimensions
         */
        template <class U> 
        Border_Rect_xd(const dt_init_list<U>& list);

        /**
         * @brief Construct border from CPU vector
         * @tparam U Value type of CPU vector
         * @param vctr CPU vector with border dimensions
         */
        template <class U> 
        Border_Rect_xd(const Vctr_cpu<U>& vctr);

        /**
         * @brief Copy constructor
         * @param border Border to copy from
         */
        CGPU_EXEC
        Border_Rect_xd(const Border_Rect_xd<T, edim_2>& border);

        /**
         * @brief Converting constructor
         * @tparam U Value type of border to convert from
         * @param border Border to convert from
         */
        template <class U>
        CGPU_EXEC
        Border_Rect_xd(const Border_Rect_xd<U, edim_2>& border);

        /**
         * @brief Copy assignment operator
         * @param border Border to assign from
         * @return Reference to this border
         */
        CGPU_EXEC
        Border_Rect_xd<T, edim_2>& operator=(const Border_Rect_xd<T, edim_2>& border);        
            
        /**
         * @brief Converting assignment operator
         * @tparam U Value type of border to assign from
         * @param border Border to assign from
         * @return Reference to this border
         */
        template <class U>
        CGPU_EXEC
        Border_Rect_xd<T, edim_2>& operator=(const Border_Rect_xd<U, edim_2>& border);

        /**
         * @brief Assign border from another border
         * @tparam U Value type of border to assign from
         * @param border Border to assign from
         */
        template <class U> 
        CGPU_EXEC
        void assign(const Border_Rect_xd<U, edim_2>& border);

        /**
         * @brief Set border dimensions from values
         * @tparam U Value type of dimensions
         * @param bx_0 Initial x position
         * @param bx_e Final x position
         * @param by_0 Initial y position
         * @param by_e Final y position
         */
        template <class U>
        void set_in_data(const U& bx_0, const U& bx_e, const U& by_0, const U& by_e);

        /**
         * @brief Set border dimensions from initializer list
         * @tparam U Value type of initializer list
         * @param list Initializer list with border dimensions
         */
        template <class U> 
        void set_in_data(const dt_init_list<U>& list);

        /**
         * @brief Set border dimensions from CPU vector
         * @tparam U Value type of CPU vector
         * @param vctr CPU vector with border dimensions
         */
        template <class U>
        void set_in_data(const Vctr_cpu<U>& vctr);

        /**
         * @brief Clear border dimensions
         */
        CGPU_EXEC
        void clear();
    
        /**
         * @brief Get sum of y-direction border dimensions
         * @return Sum of y-direction border dimensions
         */
        CGPU_EXEC
        T by_sum() const;

        /**
         * @brief Get minimum y-direction border dimension
         * @return Minimum y-direction border dimension
         */
        CGPU_EXEC
        T by_min() const;

        /**
         * @brief Get maximum y-direction border dimension
         * @return Maximum y-direction border dimension
         */
        CGPU_EXEC
        T by_max() const;

        /**
         * @brief Get y-direction border dimension at specified index
         * @param bs_y_i Index of y-direction border dimension
         * @return Y-direction border dimension at specified index
         */
        CGPU_EXEC
        T bs_y(const T& bs_y_i) const;

        /**
         * @brief Get border dimensions at specified 2D position
         * @param bs_i 2D position
         * @return Border dimensions at specified 2D position
         */
        CGPU_EXEC
        R_2d<T> bs(const R_2d<T>& bs_i) const;

        /**
         * @brief Get minimum border dimension at specified 2D position
         * @param bs 2D position
         * @return Minimum border dimension at specified 2D position
         */
        CGPU_EXEC
        T bs_min(const R_2d<T>& bs) const;

        /**
         * @brief Get maximum border dimension at specified 2D position
         * @param bs 2D position
         * @return Maximum border dimension at specified 2D position
         */
        CGPU_EXEC
        T bs_max(const R_2d<T>& bs) const;

        /**
         * @brief Get y-direction border dimension at specified 2D position
         * @param bs_y 2D position
         * @return Y-direction border dimension at specified 2D position
         */
        CGPU_EXEC
        T bs_y_h(const T& bs_y) const;

        /**
         * @brief Get border dimensions at specified 2D position
         * @param bs 2D position
         * @return Border dimensions at specified 2D position
         */
        CGPU_EXEC
        R_2d<T> bs_h(const R_2d<T>& bs) const;

        /**
         * @brief Get y-direction radius at specified 2D position
         * @param bs_y 2D position
         * @return Y-direction radius at specified 2D position
         */
        CGPU_EXEC
        T ry_c(const T& bs_y) const;

        /**
         * @brief Get radius at specified 2D position
         * @param bs 2D position
         * @return Radius at specified 2D position
         */
        CGPU_EXEC
        R_2d<T> r_c(const R_2d<T>& bs) const;

        /**
         * @brief Get y-direction radius at specified 2D position
         * @param bs_y 2D position
         * @return Y-direction radius at specified 2D position
         */
        CGPU_EXEC
        T radius_y(const T& bs_y) const;

        /**
         * @brief Get radius at specified 2D position
         * @param bs 2D position
         * @return Radius at specified 2D position
         */
        CGPU_EXEC
        R_2d<T> radius(const R_2d<T>& bs) const;

        /**
         * @brief Get y-direction radius at specified 2D position with power
         * @param bs_y 2D position
         * @param p Power
         * @return Y-direction radius at specified 2D position with power
         */
        CGPU_EXEC
        T radius_y_p(const T& bs_y, const T& p) const;

        /**
         * @brief Get radius at specified 2D position with power
         * @param bs 2D position
         * @param p Power
         * @return Radius at specified 2D position with power
         */
        CGPU_EXEC
        R_2d<T> radius_p(const R_2d<T>& bs, const T& p) const;

        /**
         * @brief Set y-direction border dimensions from 2D position and shift
         * @param bs 2D position
         * @param dr Shift
         */
        void set_by_sft(const R_2d<T>& bs, const R_2d<T>& dr);

        /**
         * @brief Shift border by specified 2D position and shift
         * @param bs 2D position
         * @param dr Shift
         */
        void sft_bdr(const R_2d<T>& bs, const R_2d<T>& dr);
    };
}

#include "../src/border_2d.inl"