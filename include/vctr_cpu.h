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

#include <vector>

#include "memcpy.cuh"
#include "r_2d.h"
#include "r_3d.h"
#include "mx_2x2.h"
#include "mx_3x3.h"

#include "pvctr.h"

#ifdef __CUDACC__
	#include <thrust/detail/raw_pointer_cast.h>
	#include <thrust/device_vector.h>
	#include <thrust/host_vector.h>
#endif

/* template definition */
namespace mt
{
#ifndef VCTR_DEC
	#define VCTR_DEC
    /**
     * @brief Forward declaration of the Vctr template class.
     * 
     * The `Vctr` template class represents a vector-like data structure and is
     * a fundamental component of the Multem library. It is designed to handle
     * vectors on different devices, such as CPUs and GPUs.
     * 
     * @tparam T The type of data stored in the vector.
     * @tparam Dev The device type (e.g., edev_cpu or edev_gpu) where the vector
     *             resides.
     */	
	template <class T, eDev Dev> class Vctr;

	/**
	 * @brief Forward declaration of the pVctr template class.
	 * 
	 * The `pVctr` template class is used in conjunction with the `Vctr` class and
	 * represents a pointer to a vector. It is also an essential part of the Multem
	 * library and can be used on various devices, including CPUs and GPUs.
	 * 
	 * @tparam T The type of data stored in the vector.
	 * @tparam Dev The device type (e.g., edev_cpu or edev_gpu) where the vector
	 *             resides.
	 * @tparam ST A template parameter for specifying additional properties or
	 *            transformations applied to the vector.
	 */
	template <class T, eDev Dev, class ST> class pVctr;
#endif
}

/* derived class */
namespace mt
{
	/**
	 * @brief Specialization of Vctr using std::vector on CPU.
	 * 
	 * The `Vctr_std` type alias represents a specialization of the `Vctr` class
	 * using the `std::vector` container on the CPU. It is a convenient way to
	 * work with CPU-based vectors in the Multem library.
	 * 
	 * @tparam T The type of data stored in the vector.
	 */
	template <class T>
	using Vctr_std = std::vector<T>;

	/**
	 * @brief Specialization of Vctr for 2D matrices on CPU.
	 * 
	 * The `Vctr_cpu` type alias represents a specialization of the `Vctr` class
	 * designed to handle 2D matrices on the CPU in the Multem library.
	 * 
	 * @tparam T The type of data stored in the matrix.
	 */
	template <class T>
	using Vctr_cpu = Vctr<T, edev_cpu>;

	/***************************************************************************************/

	/**
	 * @brief Alias for a CPU vector of 32-bit unsigned integers.
	 */
	using Vctr_uint32_cpu = Vctr_cpu<dt_uint32>;

	/**
	 * @brief Alias for a CPU vector of 32-bit signed integers.
	 */
	using Vctr_int32_cpu = Vctr_cpu<dt_int32>;

	/**
	 * @brief Alias for a CPU vector of 64-bit unsigned integers.
	 */
	using Vctr_uint64_cpu = Vctr_cpu<dt_uint64>;

	/**
	 * @brief Alias for a CPU vector of 64-bit signed integers.
	 */
	using Vctr_int64_cpu = Vctr_cpu<dt_int64>;

	/***************************************************************************************/

	/**
	 * @brief Specialization of Vctr for 2D matrices on a specified device.
	 * 
	 * The `Vctr_r_2d` type alias represents a specialization of the `Vctr` class
	 * designed to handle 2D matrices of type `R_2d<T>` on a specified device in
	 * the Multem library.
	 * 
	 * @tparam T The type of data stored in the matrix.
	 * @tparam Dev The device type (e.g., edev_cpu or edev_gpu).
	 */
	template <class T, eDev Dev>
	using Vctr_r_2d = Vctr<R_2d<T>, Dev>;

	/**
	 * @brief Specialization of Vctr for 2D matrices on CPU.
	 * 
	 * The `Vctr_r_2d_cpu` type alias represents a specialization of the `Vctr` class
	 * designed to handle 2D matrices of type `R_2d<T>` on the CPU in the Multem library.
	 * 
	 * @tparam T The type of data stored in the matrix.
	 */
	template <class T>
	using Vctr_r_2d_cpu = Vctr<R_2d<T>, edev_cpu>;

	/***************************************************************************************/

	/**
	 * @brief Specialization of Vctr for 3D matrices on a specified device.
	 * 
	 * The `Vctr_r_3d` type alias represents a specialization of the `Vctr` class
	 * designed to handle 3D matrices of type `R_3d<T>` on a specified device in
	 * the Multem library.
	 * 
	 * @tparam T The type of data stored in the matrix.
	 * @tparam Dev The device type (e.g., edev_cpu or edev_gpu).
	 */
	template <class T, eDev Dev>
	using Vctr_r_3d = Vctr<R_3d<T>, Dev>;

	/**
	 * @brief Specialization of Vctr for 3D matrices on CPU.
	 * 
	 * The `Vctr_r_3d_cpu` type alias represents a specialization of the `Vctr` class
	 * designed to handle 3D matrices of type `R_3d<T>` on the CPU in the Multem library.
	 * 
	 * @tparam T The type of data stored in the matrix.
	 */
	template <class T>
	using Vctr_r_3d_cpu = Vctr<R_3d<T>, edev_cpu>;

	/***************************************************************************************/

	/**
	 * @brief Specialization of Vctr for 2x2 matrices on a specified device.
	 * 
	 * The `Vctr_Mx_2x2` type alias represents a specialization of the `Vctr` class
	 * designed to handle 2x2 matrices of type `Mx_2x2<T>` on a specified device in
	 * the Multem library.
	 * 
	 * @tparam T The type of data stored in the matrix.
	 * @tparam Dev The device type (e.g., edev_cpu or edev_gpu).
	 */
	template <class T, eDev Dev>
	using Vctr_Mx_2x2 = Vctr<Mx_2x2<T>, Dev>;

	/**
	 * @brief Specialization of Vctr for 2x2 matrices on CPU.
	 * 
	 * The `Vctr_Mx_2x2_cpu` type alias represents a specialization of the `Vctr` class
	 * designed to handle 2x2 matrices of type `Mx_2x2<T>` on the CPU in the Multem library.
	 * 
	 * @tparam T The type of data stored in the matrix.
	 */
	template <class T>
	using Vctr_Mx_2x2_cpu = Vctr<Mx_2x2<T>, edev_cpu>;

	/***************************************************************************************/

	/**
	 * @brief Specialization of Vctr for 3x3 matrices on a specified device.
	 * 
	 * The `Vctr_Mx_3x3` type alias represents a specialization of the `Vctr` class
	 * designed to handle 3x3 matrices of type `Mx_3x3<T>` on a specified device in
	 * the Multem library.
	 * 
	 * @tparam T The type of data stored in the matrix.
	 * @tparam Dev The device type (e.g., edev_cpu or edev_gpu).
	 */
	template <class T, eDev Dev>
	using Vctr_Mx_3x3 = Vctr<Mx_3x3<T>, Dev>;

	/**
	 * @brief Specialization of Vctr for 3x3 matrices on CPU.
	 * 
	 * The `Vctr_Mx_3x3_cpu` type alias represents a specialization of the `Vctr` class
	 * designed to handle 3x3 matrices of type `Mx_3x3<T>` on the CPU in the Multem library.
	 * 
	 * @tparam T The type of data stored in the matrix.
	 */
	template <class T>
	using Vctr_Mx_3x3_cpu = Vctr<Mx_3x3<T>, edev_cpu>;

}

/* cpu vector */
namespace mt
{
	/**
     * @brief A class representing a vector on the CPU.
     * 
     * This class provides various methods and operations for working with vectors
     * on the CPU.
     * 
     * @tparam T The type of data stored in the vector.
     */
	template <class T>
	class Vctr<T, edev_cpu>
	{
	public:
		using value_type = T;
		using size_type = dt_int64;
		static const eDev device = edev_cpu;

		mutable T* m_data;
		size_type m_s0;
		size_type m_s1;
		size_type m_s2;
		size_type m_s3;
		size_type m_size;
		size_type m_capacity;

		size_type m_pitch_s1;
		size_type m_pitch_s2;
		size_type m_pitch_s3;

		/************************************* constructors ************************************/
		/**
		 * @brief Default constructor for Vctr.
		 */		
		explicit Vctr();

		/**
		 * @brief Constructor for Vctr that initializes from a list of values (float 64).
		 * 
		 * @param data The list of values to initialize the vector with.
		 */
		Vctr(const dt_init_list_f64& data);

		/**
		 * @brief Constructor for Vctr with a specified size.
		 * 
		 * @param s0 The size of the vector.
		*/
		Vctr(size_type s0);

		/**
		 * @brief Constructor for Vctr with a specified size and initial value.
		 * 
		 * @param s0 The size of the vector.
		 * @param value The initial value to fill the vector with.
		 */
		Vctr(size_type s0, const T& value);

		/**
		 * @brief Constructor for Vctr with a specified shape.
		 * 
		 * @param shape The shape of the vector.
		 */
		explicit Vctr(const dt_shape_st<size_type>& shape);

		/**
		 * @brief Constructor for Vctr with a specified shape and initial value.
		 * 
		 * @param shape The shape of the vector.
		 * @param value The initial value to fill the vector with.
		 */
		explicit Vctr(const dt_shape_st<size_type>& shape, const T& value);

		/* copy constructor */
		Vctr(const Vctr<T, edev_cpu>& vctr);

		/* Move constructor */
		Vctr(Vctr<T, edev_cpu>&& vctr);

		/* converting constructor */

		/**
		 * @brief Converting constructor that constructs a new Vctr from a different type of Vctr on the CPU.
		 * 
		 * This constructor allows you to create a new `Vctr` object from an existing `Vctr` of a different data type
		 * while keeping the same execution device (CPU).
		 * 
		 * @tparam U The data type of the source `Vctr`.
		 * @param vctr The source `Vctr` to be converted.
		 */
		template <class U>
		Vctr(const Vctr<U, edev_cpu>& vctr);

		/**
		 * @brief Converting constructor that constructs a new Vctr from a range of elements specified by iterators.
		 * 
		 * This constructor allows you to create a new `Vctr` object from a range of elements specified by iterators.
		 * 
		 * @tparam U The data type of the elements in the range.
		 * @param first An iterator pointing to the first element of the range.
		 * @param last An iterator pointing one past the last element of the range.
		 */		
		template <class U>
		Vctr(U* first, U* last);

		/**
		 * @brief Converting constructor that constructs a new Vctr from a memory buffer with optional column offset.
		 * 
		 * This constructor allows you to create a new `Vctr` object from a memory buffer with an optional column offset.
		 * 
		 * @tparam U The data type of the elements in the memory buffer.
		 * @tparam V The data type of the elements in the new `Vctr`.
		 * @param p A pointer to the memory buffer.
		 * @param n_p The number of elements in the memory buffer.
		 * @param icol An optional column offset (default is 0).
		 */
		template <class U, class V=T, class = enable_if_r_nd<V>>
		Vctr(U *p, dt_int64 n_p, dt_int64 icol=0);

		/**
		 * @brief Converting constructor that constructs a new Vctr from a std::vector.
		 * 
		 * This constructor allows you to create a new `Vctr` object from a std::vector of a different data type.
		 * 
		 * @tparam U The data type of the elements in the std::vector.
		 * @param vctr The std::vector to be converted.
		 */	
		template <class U>
		Vctr(const std::vector<U>& vctr);

		/**
		 * @brief Converting constructor that constructs a new Vctr from a CPU pVctr with optional storage type.
		 * 
		 * This constructor allows you to create a new `Vctr` object from a CPU pVctr with an optional storage type.
		 * 
		 * @tparam U The data type of the elements in the source pVctr.
		 * @tparam STU The storage type of the source pVctr.
		 * @param pvctr The source pVctr to be converted.
		 */
		template <class U, class STU>
		Vctr(const pVctr<U, edev_cpu, STU>& pvctr);

#ifdef __CUDACC__
		/**
		 * @brief Converting constructor that constructs a new Vctr from a different type of Vctr or GPU-specific containers.
		 * 
		 * This constructor allows you to create a new `Vctr` object by converting data from various sources, including:
		 * - Another `Vctr` of a different data type on the GPU.
		 * - A `thrust::host_vector` on the GPU.
		 * - A `thrust::device_vector` on the GPU.
		 * - A GPU Vctr pointer with an optional storage type.
		 * 
		 * @tparam U The data type of the source data or container elements.
		 * @tparam STU The storage type (only for pVctr conversion).
		 * @param vctr The source data or container to be converted.
		 */

		template <class U>
		Vctr(const Vctr<U, edev_gpu>& vctr);

		template <class U>
		Vctr(const thrust::host_vector<U>& vctr);

		template <class U>
		Vctr(const thrust::device_vector<U>& vctr);

		// from gpu pVctr to Vctr
		template <class U, class STU>
		Vctr(const pVctr<U, edev_gpu, STU>& pvctr);
#endif
		/**
		 * @brief Destructor for the Vctr class.
		 */
		~Vctr();

		/******************************** assignment operators *********************************/
		/**
		 * @brief Copy assignment operator that assigns the contents of another Vctr of the same type.
		 * 
		 * This operator allows you to copy the contents of another `Vctr` object of the same data type on the CPU.
		 * 
		 * @param vctr The source `Vctr` object to copy data from.
		 * @return A reference to the modified Vctr object.
		 */
		Vctr<T, edev_cpu>& operator=(const Vctr<T, edev_cpu>& vctr);

		/**
		 * @brief Move assignment operator that transfers ownership of another Vctr of the same type.
		 * 
		 * This operator allows you to move the contents and ownership of another `Vctr` object of the same data type on the CPU.
		 * 
		 * @param vctr The source `Vctr` object to move data from.
		 * @return A reference to the modified Vctr object.
		 */
		Vctr<T, edev_cpu>& operator=(Vctr<T, edev_cpu>&& vctr);

		/**
		 * @brief Converting assignment operator that assigns data from another Vctr or a std::vector of a different type.
		 * 
		 * This operator allows you to assign data from various sources, including:
		 * - Another `Vctr` of a different data type on the CPU.
		 * - A std::vector of a different data type.
		 * 
		 * @tparam U The data type of the source data or container elements.
		 * @param vctr The source data or container to be assigned.
		 * @return A reference to the modified Vctr object.
		 */
		template <class U>
		Vctr<T, edev_cpu>& operator=(const Vctr<U, edev_cpu>& vctr);

		template <class U>
		Vctr<T, edev_cpu>& operator=(const std::vector<U>& vctr);

#ifdef __CUDACC__
		template <class U>
		Vctr<T, edev_cpu>& operator=(const Vctr<U, edev_gpu>& vctr);

		template <class U>
		Vctr<T, edev_cpu>& operator=(const thrust::host_vector<U>& vctr);

		template <class U>
		Vctr<T, edev_cpu>& operator=(const thrust::device_vector<U>& vctr);
#endif
		/**
		 * @brief Assigns the content of another Vctr object with a different data type to this Vctr.
		 * 
		 * This function copies the content of the provided Vctr object with a potentially different data type
		 * into this Vctr object, ensuring that both objects have compatible data types. This operation makes
		 * a deep copy of the data.
		 * 
		 * @tparam U The data type of the source Vctr object.
		 * @param vctr The source Vctr object to copy data from.
		 * @param pvctr_cpu A pointer to an optional CPU storage location where the data can be copied.
		 */
		template <class U>
		void assign(const Vctr<U, edev_cpu>& vctr, U* pvctr_cpu = nullptr);

		/**
		 * @brief Assigns a range of data from a CPU pointer to this Vctr.
		 * 
		 * This function copies data from a range specified by two pointers, 'first' and 'last', from a CPU
		 * memory location to this Vctr object. It ensures that the Vctr object is appropriately resized to hold
		 * the copied data.
		 * 
		 * @tparam U The data type of the CPU memory.
		 * @param first A pointer to the first element of the source data range.
		 * @param last A pointer one-past-the-end of the source data range.
		 */
		template <class U>
		void assign(U* first, U* last);

		/**
		 * @brief Assigns the content of a CPU pVctr object to this Vctr.
		 * 
		 * This function copies the content of a CPU pVctr (pointer for vector) object to this Vctr object.
		 * It ensures that the data is appropriately copied and that the Vctr object is resized as needed.
		 * 
		 * @tparam U The data type of the elements in the pVctr.
		 * @tparam STU The storage type of the pVctr (e.g., contiguous or strided).
		 * @param pvctr The source pVctr object to copy data from.
		 */
		template <class U, class STU>
		void assign(const pVctr<U, edev_cpu, STU>& pvctr);

#ifdef __CUDACC__
		template <class U>
		void assign(const thrust::device_ptr<U>& first, const thrust::device_ptr<U>& last, U* pvctr_cpu = nullptr);

		// from gpu pVctr to Vctr
		template <class U, class STU>
		void assign(const pVctr<U, edev_gpu, STU>& pvctr);
#endif

		/**
		 * @brief Assigns the content of a std::vector to this Vctr.
		 * 
		 * This method allows for the assignment of data from a standard C++ vector (`std::vector`)
		 * to this Vctr instance. It is particularly useful for initializing or updating the Vctr
		 * with a collection of elements that are already stored in a std::vector. The method
		 * ensures that the data types are compatible and handles the allocation and copying of
		 * data internally.
		 * 
		 * @tparam U The data type of the elements in the std::vector. This type must be compatible
		 *           with the data type of the Vctr instance.
		 * @param vctr The std::vector from which to copy the data. The elements of this vector will
		 *             be copied into the Vctr.
		 * @param pvctr_cpu Optional parameter. If provided, it should be a pointer to a pre-allocated
		 *                  memory space on the CPU where the data from the std::vector can be copied.
		 *                  If nullptr (default), the method will manage the allocation.
		 */
		template <class U>
		void assign(const std::vector<U>& vctr, U* pvctr_cpu = nullptr);

#ifdef __CUDACC__
		template <class U>
		void assign(const Vctr<U, edev_gpu>& vctr, U* pvctr_cpu = nullptr);

		template <class U>
		void assign(const thrust::host_vector<U>& vctr, U* pvctr_cpu = nullptr);

		template <class U>
		void assign(const thrust::device_vector<U>& vctr, U* pvctr_cpu = nullptr);
#endif
		/**************** user define conversion operators *******************/
		pVctr_cpu_32<T> ptr_32() const;

		pVctr_cpu_64<T> ptr_64() const;

		operator pVctr_cpu_32<T>() const;

		operator pVctr_cpu_64<T>() const;

		/* user define conversion */
		operator std::vector<T>() const;

		/* user define conversion in which T is the complemented precision */
		template <class U = chg_2_compl_float_type<T>>
		operator std::vector<U>() const;

#ifdef __CUDACC__
		/* user define conversion to output type std::vector<thrust::complex<dt_float32>> */
		template <class U=T, class = enable_if_std_cfloat<U>>
		operator std::vector<thrust::complex<dt_float32>>() const;

		/* user define conversion to output type std::vector<thrust::complex<dt_float64>> */
		template <class U=T, class = enable_if_std_cfloat<U>>
		operator std::vector<thrust::complex<dt_float64>>() const;

		/***************************************************************************************/
		/* user define conversion */
		operator thrust::host_vector<T>() const;

		/* user define conversion in which T is the complemented precision */
		template <class U = chg_2_compl_float_type<T>>
		operator thrust::host_vector<U>() const;

		/* user define conversion to output type thrust::host_vector<std::complex<dt_float32>> */
		template <class U=T, class = enable_if_thr_cfloat<U>>
		operator thrust::host_vector<std::complex<dt_float32>>() const;

		/* user define conversion to output type thrust::host_vector<std::complex<dt_float64>> */
		template <class U=T, class = enable_if_thr_cfloat<U>>
		operator thrust::host_vector<std::complex<dt_float64>>() const;

		/***************************************************************************************/
		/* user define conversion */
		operator thrust::device_vector<T>() const;

		/* user define conversion in which T is the complemented precision */
		template <class U = chg_2_compl_float_type<T>>
		operator thrust::device_vector<U>() const;
#endif
		/***************************************************************************************/
		template <class U>
		void cpy_to_cpu_ptr(U* pdata, size_type n_data, T* pvctr_cpu = nullptr);

		template <class U>
		void cpy_to_cpu_ptr(U* first, U* last, T* pvctr_cpu = nullptr);

#ifdef __CUDACC__
		template <class U>
		void cpy_to_gpu_ptr(U* pdata, size_type n_data, U* pvctr_cpu = nullptr);

		template <class U>
		void cpy_to_gpu_ptr(U* first, U* last, U* pvctr_cpu = nullptr);
#endif

		/***************************************************************************************/
		template <class U, class V=T, class = enable_if_cmplx<V>>
		void cpy_real_to_cpu_ptr(U* pdata, size_type n_data, T* pvctr_cpu = nullptr);

		template <class U, class V=T, class = enable_if_cmplx<V>>
		void cpy_real_to_cpu_ptr(U* first, U* last, T* pvctr_cpu = nullptr);

#ifdef __CUDACC__
		template <class U, class V=T, class = enable_if_cmplx<V>>
		void cpy_real_to_gpu_ptr(U* pdata, size_type n_data, U* pvctr_cpu = nullptr);

		template <class U, class V=T, class = enable_if_cmplx<V>>
		void cpy_real_to_gpu_ptr(U* first, U* last, U* pvctr_cpu = nullptr);
#endif

		/***************************************************************************************/
		template <class U, class V=T, class = enable_if_cmplx<V>>
		void cpy_imag_to_cpu_ptr(U* pdata, size_type n_data, T* pvctr_cpu = nullptr);

		template <class U, class V=T, class = enable_if_cmplx<V>>
		void cpy_imag_to_cpu_ptr(U* first, U* last, T* pvctr_cpu = nullptr);

#ifdef __CUDACC__
		template <class U, class V=T, class = enable_if_cmplx<V>>
		void cpy_imag_to_gpu_ptr(U* pdata, size_type n_data, U* pvctr_cpu = nullptr);

		template <class U, class V=T, class = enable_if_cmplx<V>>
		void cpy_imag_to_gpu_ptr(U* first, U* last, U* pvctr_cpu = nullptr);
#endif

		/***************************************************************************************/
		template <class U, class V=T, class = enable_if_r_nd<V>>
		void set_pos_from_cpu_ptr(U *p, size_type n_p, dt_int64 icol=0);

		template <class U, class V=T, class = enable_if_r_nd<V>>
		void cpy_pos_to_cpu_ptr(U *p, size_type icol=0);

		/***************************************************************************************/
		void resize(const dt_shape_st<size_type>& shape);

		void resize(const dt_shape_st<size_type>& shape, const T& value);

		void reserve(const dt_shape_st<size_type>& shape);

		void shrink_to_fit();

		template <class U>
		void push_back(const U& val);

		template <class U>
		void push_back(const Vctr<U, edev_cpu>& vctr);

		void pop_back();

		void fill(T val);

		/***************************************************************************************/
		size_type s0() const;

		size_type s1() const;

		size_type s2() const;

		size_type s3() const;			
			
		dt_int32 s0_32() const;

		dt_int32 s1_32() const;

		dt_int32 s2_32() const;

		dt_int32 s3_32() const;
			
		dt_int64 s0_64() const;

		dt_int64 s1_64() const;

		dt_int64 s2_64() const;

		dt_int64 s3_64() const;

		size_type s0h() const;

		size_type s1h() const;

		size_type s2h() const;

		size_type s3h() const;

		dt_shape_st<size_type> shape() const;

		dt_shape_st<size_type> shape_2d_trs() const;

		size_type shape_size() const;

		size_type pitch_s1() const;

		size_type pitch_s2() const;

		size_type pitch_s3() const;

		size_type size() const;

		dt_int32 size_32() const;

		dt_int64 size_64() const;

		iGrid_1d igrid_1d() const;

		iGrid_2d igrid_2d() const;

		iGrid_3d igrid_3d() const;

		iGrid_1d_64 igrid_1d_64() const;

		iGrid_2d_64 igrid_2d_64() const;

		iGrid_3d_64 igrid_3d_64() const;

		size_type capacity() const;

		dt_bool empty() const;

		dt_bool is_1d() const;

		void clear();

		void clear_shrink_to_fit();

		size_type sub_2_ind(const size_type& ix_0) const;

		size_type sub_2_ind(const size_type& ix_0, const size_type& ix_1) const;

		size_type sub_2_ind(const size_type& ix_0, const size_type& ix_1, const size_type& ix_2) const;

		size_type sub_2_ind(const size_type& ix_0, const size_type& ix_1, const size_type& ix_2, const size_type& ix_3) const;

		T& operator[](const size_type& iy);

		const T& operator[](const size_type& iy) const;

		T& operator()(const size_type& iy);

		const T& operator()(const size_type& iy) const;

		T& operator()(const size_type& ix_0, const size_type& ix_1);

		const T& operator()(const size_type& ix_0, const size_type& ix_1) const;

		T& operator()(const size_type& ix_0, const size_type& ix_1, const size_type& ix_2);

		const T& operator()(const size_type& ix_0, const size_type& ix_1, const size_type& ix_2) const;

		T& operator()(const size_type& ix_0, const size_type& ix_1, const size_type& ix_2, const size_type& ix_3);

		const T& operator()(const size_type& ix_0, const size_type& ix_1, const size_type& ix_2, const size_type& ix_3) const;

		T* begin();

		const T* begin() const;

		T* end();

		const T* end() const;

		T* data();

		const T* data() const;

		template <class U>
		U data_cast();

		template <class U>
		const U data_cast() const;

		T& front();

		const T& front() const;

		T& back();

		const T& back() const;

		// set shape
		void set_shape(const dt_shape_st<size_type>& shape);

		void trs_shape_2d();

		template <class U, eDev Dev>
		void set_shape(const Vctr<U, Dev>& vctr, dt_bool bb_size = true);

	#ifdef __CUDACC__
 		FCNS_DEF_GPU_GRID_BLK_VCTR;
	#endif

	private:

		void set_picth();

		void set_capacity(size_type size_r);

		void set_shape_cstr(dt_shape_st<size_type>& shape);

		// reallocate and copy memory
		void allocate(dt_shape_st<size_type> shape, dt_bool bb_reserve=false);

		// destroy memory on the device
		void init();

		// destroy memory on the device
		void destroy();
	};
}


#include "../src/vctr_cpu.inl"