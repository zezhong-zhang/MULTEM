/**
 * @file wd_fermi.h
 * @brief Fermi window function implementation for electron microscopy
 * @details Provides template classes for implementing Fermi-function based window
 *          functions in 1D, 2D, and 3D. These window functions are used for
 *          smooth aperture functions and energy filtering in electron microscopy
 *          simulations.
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
    /**
     * @brief Base template for window functions
     * @tparam T Value type (e.g., float, double)
     * @tparam Dim Dimensionality of the window function
     * @tparam Fcn_typ Type of window function
     */
    template <class T, eDim Dim, eFcn_typ Fcn_typ> class Wd_fcn_xd;
#endif
}   

/* derived class */
namespace mt
{
    /**
     * @brief N-dimensional Fermi window function
     * @tparam T Value type (e.g., float, double)
     * @tparam Dim Dimensionality (1D, 2D, or 3D)
     */
    template <class T, eDim Dim>
    using Wd_Fermi_xd = Wd_fcn_xd<T, Dim, efcn_fermi>;

    /**
     * @brief One-dimensional Fermi window function
     * @tparam T Value type (e.g., float, double)
     */
    template <class T>
    using Wd_Fermi_1d = Wd_fcn_xd<T, edim_1, efcn_fermi>;

    /**
     * @brief Two-dimensional Fermi window function
     * @tparam T Value type (e.g., float, double)
     */
    template <class T>
    using Wd_Fermi_2d = Wd_fcn_xd<T, edim_2, efcn_fermi>;

    /**
     * @brief Three-dimensional Fermi window function
     * @tparam T Value type (e.g., float, double)
     */
    template <class T>
    using Wd_Fermi_3d = Wd_fcn_xd<T, edim_3, efcn_fermi>;
}

/* template specialization */
namespace mt
{
    /**
     * @brief Specialized implementation of Fermi window function
     * @details Implements a smooth window function based on the Fermi-Dirac distribution:
     *          f(x) = 1 / (1 + exp((|x| - x_c) / alpha))
     *          where x_c is the cutoff value and alpha controls the smoothness
     *
     * @tparam T Value type (e.g., float, double)
     * @tparam Dim Dimensionality of the window function
     */
    template <class T, eDim Dim>
    class Wd_fcn_xd<T, Dim, efcn_fermi>: public Wdb_fcn_xd<T, Dim, efcn_fermi>
    {
    public:
        /** @brief Value type for calculations */
        using value_type = T;

        /**
         * @brief Default constructor
         */
        CGPU_EXEC
        Wd_fcn_xd();

        /**
         * @brief Construct window function with parameters
         * @param r Input data
         * @param alpha Smoothness parameter
         * @param r_wd Window radius
         * @param r_max Maximum radius
         */
        Wd_fcn_xd(const R_xd<T, Dim>& r, const T& alpha, const T& r_wd, const T& r_max);

        /**
         * @brief Copy constructor
         * @param wd Window function to copy
         */
        CGPU_EXEC
        Wd_fcn_xd(const Wd_fcn_xd<T, Dim, efcn_fermi>& wd);

        /**
         * @brief Copy assignment operator
         * @param wd Window function to copy
         * @return Reference to this window function
         */
        CGPU_EXEC 
        Wd_fcn_xd<T, Dim, efcn_fermi>& operator=(const Wdb_fcn_xd<T, Dim, efcn_fermi>& wd);

        /**
         * @brief Initialize window function parameters
         * @param r Input data
         * @param alpha Smoothness parameter
         * @param r_wd Window radius
         * @param r_max Maximum radius
         */
        void set_in_data(const R_xd<T, Dim>& r, const T& alpha, const T& r_wd, const T& r_max);
    };
}

#include "../src/wd_fermi.inl"