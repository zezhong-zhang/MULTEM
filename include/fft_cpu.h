/**
 * @file fft_cpu.h
 * @brief CPU-based Fast Fourier Transform implementation
 * @details Provides template classes for performing FFT operations on CPU using FFTW3.
 *          Supports both single and double precision, 1D and 2D transforms, and batch processing.
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

#include <fftw3.h>

#include "macros.h"
#include "const_enum.h"
#include "math_mt.h"
#include "vctr_cpu.h"
#include "stream_cpu.h"

/* forward declaration */
namespace mt
{
#ifndef FFT_H
    #define FFT_H
    /**
     * @brief Base template for FFT operations
     * @tparam T Data type (float/double)
     * @tparam Dev Device type (CPU/GPU)
     */
    template <class T, eDev Dev> class FFT;
#endif
}

/* derive classes */
namespace mt
{
    /**
     * @brief CPU FFT specialization
     * @tparam T Data type (float/double)
     */
    template <class T>
    using FFT_cpu = FFT<T, edev_cpu>;

    /**
     * @brief GPU FFT specialization
     * @tparam T Data type (float/double)
     */
    template <class T>
    using FFT_gpu = FFT<T, edev_gpu>;
}

/* cpu fourier transform - dt_float32 */
namespace mt
{
    /**
     * @brief Single precision FFT implementation for CPU
     */
    template <>
    class FFT<dt_float32, edev_cpu>
    {
    public:
        /** @brief Value type for computations */
        using value_type = dt_float32;
        
        /** @brief Device type identifier */
        static const eDev device = edev_cpu;
        
        /** @brief Complex vector type */
        using TVctr_c = Vctr<complex<value_type>, edev_cpu>;

        /**
         * @brief Default constructor
         */
        FFT();

        /**
         * @brief Constructor for 1D FFT
         * @param s0 Size of the transform
         * @param pstream Optional CPU stream for execution
         */
        FFT(const dt_int32& s0, Stream_cpu* pstream = nullptr);

        /**
         * @brief Constructor for 2D FFT
         * @param s0 Size in first dimension
         * @param s1 Size in second dimension
         * @param pstream Optional CPU stream for execution
         */
        FFT(const dt_int32& s0, const dt_int32& s1, Stream_cpu* pstream = nullptr);

        /**
         * @brief Destructor
         */
        ~FFT();

        /**
         * @brief Clean up FFT plans and resources
         */
        void cleanup();

        /**
         * @brief Destroy existing FFT plans
         */
        void destroy_plan();

        /**
         * @brief Create plan for 1D FFT
         * @param s0 Size of the transform
         * @param n_thread Number of threads to use
         */
        void create_plan_1d(const dt_int32& s0, dt_int32 n_thread=1);

        /**
         * @brief Create plan for batch 1D FFT
         * @param s0 Size of each transform
         * @param s1 Number of transforms
         * @param n_thread Number of threads to use
         */
        void create_plan_1d_batch(const dt_int32& s0, const dt_int32& s1, dt_int32 n_thread=1);

        /**
         * @brief Create plan for 2D FFT
         * @param s0 Size in first dimension
         * @param s1 Size in second dimension
         * @param n_thread Number of threads to use
         */
        void create_plan_2d(const dt_int32& s0, const dt_int32& s1, dt_int32 n_thread=1);

        /**
         * @brief Create plan for batch 2D FFT
         * @param s0 Size in first dimension
         * @param s1 Size in second dimension
         * @param s2 Number of transforms
         * @param n_thread Number of threads to use
         */
        void create_plan_2d_batch(const dt_int32& s0, const dt_int32& s1, const dt_int32& s2, dt_int32 n_thread=1);

        /**
         * @brief Perform forward FFT in-place
         * @param mx Input/output vector
         */
        void forward(TVctr_c& mx);

        /**
         * @brief Perform inverse FFT in-place
         * @param mx Input/output vector
         */
        void inverse(TVctr_c& mx);

        /**
         * @brief Perform forward FFT out-of-place
         * @param mx_i Input vector
         * @param mx_o Output vector
         */
        void forward(TVctr_c& mx_i, TVctr_c& mx_o);

        /**
         * @brief Perform inverse FFT out-of-place
         * @param mx_i Input vector
         * @param mx_o Output vector
         */
        void inverse(TVctr_c& mx_i, TVctr_c& mx_o);

    private:
        /** @brief FFTW plan for forward transform */
        fftwf_plan plan_forward;
        
        /** @brief FFTW plan for backward transform */
        fftwf_plan plan_backward;
    };
}

/* cpu fourier transform - dt_float64 */
namespace mt
{
    /**
     * @brief Double precision FFT implementation for CPU
     */
    template <>
    class FFT<dt_float64, edev_cpu>
    {
    public:
        /** @brief Value type for computations */
        using value_type = dt_float64;
        
        /** @brief Device type identifier */
        static const eDev device = edev_cpu;
        
        /** @brief Complex vector type */
        using TVctr_c = Vctr<complex<value_type>, edev_cpu>;

        /**
         * @brief Default constructor
         */
        FFT();

        /**
         * @brief Constructor for 1D FFT
         * @param s0 Size of the transform
         * @param pstream Optional CPU stream for execution
         */
        FFT(const dt_int32& s0, Stream_cpu* pstream = nullptr);

        /**
         * @brief Constructor for 2D FFT
         * @param s0 Size in first dimension
         * @param s1 Size in second dimension
         * @param pstream Optional CPU stream for execution
         */
        FFT(const dt_int32& s0, const dt_int32& s1, Stream_cpu* pstream = nullptr);

        /**
         * @brief Destructor
         */
        ~FFT();

        /**
         * @brief Clean up FFT plans and resources
         */
        void cleanup();

        /**
         * @brief Destroy existing FFT plans
         */
        void destroy_plan();

        /**
         * @brief Create plan for 1D FFT
         * @param s0 Size of the transform
         * @param n_thread Number of threads to use
         */
        void create_plan_1d(const dt_int32& s0, dt_int32 n_thread=1);

        /**
         * @brief Create plan for batch 1D FFT
         * @param s0 Size of each transform
         * @param s1 Number of transforms
         * @param n_thread Number of threads to use
         */
        void create_plan_1d_batch(const dt_int32& s0, const dt_int32& s1, dt_int32 n_thread=1);

        /**
         * @brief Create plan for 2D FFT
         * @param s0 Size in first dimension
         * @param s1 Size in second dimension
         * @param n_thread Number of threads to use
         */
        void create_plan_2d(const dt_int32& s0, const dt_int32& s1, dt_int32 n_thread=1);

        /**
         * @brief Create plan for batch 2D FFT
         * @param s0 Size in first dimension
         * @param s1 Size in second dimension
         * @param s2 Number of transforms
         * @param n_thread Number of threads to use
         */
        void create_plan_2d_batch(const dt_int32& s0, const dt_int32& s1, const dt_int32& s2, dt_int32 n_thread=1);

        /**
         * @brief Perform forward FFT in-place
         * @param mx Input/output vector
         */
        void forward(TVctr_c& mx);

        /**
         * @brief Perform inverse FFT in-place
         * @param mx Input/output vector
         */
        void inverse(TVctr_c& mx);

        /**
         * @brief Perform forward FFT out-of-place
         * @param mx_i Input vector
         * @param mx_o Output vector
         */
        void forward(TVctr_c& mx_i, TVctr_c& mx_o);

        /**
         * @brief Perform inverse FFT out-of-place
         * @param mx_i Input vector
         * @param mx_o Output vector
         */
        void inverse(TVctr_c& mx_i, TVctr_c& mx_o);

    private:
        /** @brief FFTW plan for forward transform */
        fftw_plan plan_forward;
        
        /** @brief FFTW plan for backward transform */
        fftw_plan plan_backward;
    };
}

#include "../src/fft_cpu.inl"