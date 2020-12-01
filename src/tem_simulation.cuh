/*
 * This file is part of MULTEM.
 * Copyright 2014 Ivan Lobato <Ivanlh20@gmail.com>
 *
 * MULTEM is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * MULTEM is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with MULTEM. If not, see <http:// www.gnu.org/licenses/>.
 */

#ifndef TEM_SIMULATION_H
#define TEM_SIMULATION_H

#include <fftw3.h>
#include "math.cuh"
#include "types.cuh"
#include "traits.cuh"
#include "input_multislice.cuh"
#include "output_multislice.hpp"
#include "cpu_fcns.hpp"
#include "gpu_fcns.cuh"
#include "cgpu_fcns.cuh"
#include "energy_loss.cuh"
#include "wave_function.cuh"
#include "timing.cuh"
#include "matlab_mex.cuh"
#include "projected_potential.cuh"

namespace mt
{
	template <class T, eDevice dev>
	class Multislice
	{
		public:
			using T_r = T;
			using T_c = complex<T>;

			static const eDevice device = dev;

			static bool ext_stop_sim;
			static int ext_niter;
			static int ext_iter;

			void set_input_data(Input_Multislice<T_r> *input_multislice_i, Stream<dev> *stream_i, FFT<T_r, dev> *fft2_i)
			{
				input_multislice = input_multislice_i;
				stream = stream_i;
				fft_2d = fft2_i;

				if(input_multislice->is_EELS_EFTEM())
				{
					energy_loss.set_input_data(input_multislice, stream, fft_2d);
					psi_thk.resize(input_multislice->grid_2d.nxy());
					if(input_multislice->eels_fr.is_Mixed_Channelling())
					{
						trans_thk.resize(input_multislice->grid_2d.nxy());
					}
				}
				if(input_multislice->is_EDX())
				{
					local_potential.set_input_data(input_multislice, stream, fft_2d);
					psi_thk.resize(input_multislice->grid_2d.nxy());
				}

				wave_function.set_input_data(input_multislice, stream, fft_2d);
			}

			template <class TOutput_multislice>
			void operator()(TOutput_multislice &output_multislice)
			{
				if(input_multislice->is_STEM_ISTEM())
				{
					STEM_ISTEM(output_multislice);
				}
				else if(input_multislice->is_CBED_CBEI())
				{
					CBED_CBEI(output_multislice);
				}
				else if(input_multislice->is_ED_HRTEM())
				{
					ED_HRTEM(output_multislice);
				}
				else if(input_multislice->is_PED_HCTEM())
				{
					PED_HCTEM(output_multislice);
				}
				else if(input_multislice->is_EWFS_EWRS())
				{
					EWFS_EWRS(output_multislice);
				}
				else if(input_multislice->is_EELS_EFTEM())
				{
					EELS_EFTEM(output_multislice);
				}
				else if(input_multislice->is_EDX())
				{
					EDX(output_multislice);
				}	
			}

		private:
			template <class TOutput_multislice>
			void STEM_ISTEM(TOutput_multislice &output_multislice)
			{
				ext_niter = input_multislice->scanning.size()*input_multislice->number_conf();
				ext_iter = 0;
				/*****************************************************************/

				T_r w_pr_0 = input_multislice->get_phonon_rot_weight();

				output_multislice.init();

				input_multislice->iscan.resize(1);
				input_multislice->beam_x.resize(1);
				input_multislice->beam_y.resize(1);

				if(input_multislice->is_STEM() && input_multislice->pn_coh_contrib)
				{
					for(auto iscan = 0; iscan < input_multislice->scanning.size(); iscan++)
					{	
						output_multislice.init_psi_coh();
						input_multislice->iscan[0] = iscan;
						input_multislice->set_iscan_beam_position();	
						for(auto iconf = input_multislice->fp_iconf_0; iconf <= input_multislice->pn_nconf; iconf++)
						{
							wave_function.move_atoms(iconf);
							wave_function.set_incident_wave(wave_function.psi_z);
							wave_function.psi(w_pr_0, wave_function.psi_z, output_multislice);

							ext_iter++;
							if(ext_stop_sim) break;
						}
						wave_function.set_m2psi_coh(output_multislice);

						if(ext_stop_sim) break;
					}
				}
				else
				{
					if(input_multislice->is_illu_mod_full_integration())
					{
						Q1<double, e_host> qt;
						Q2<double, e_host> qs;

						ext_niter *= qt.size();
						
						// Load quadratures
						cond_lens_temporal_spatial_quadratures(input_multislice->cond_lens, qt, qs);
						double c_10_0 = input_multislice->cond_lens.c_10;
						
						for(auto iconf = input_multislice->fp_iconf_0; iconf <= input_multislice->pn_nconf; iconf++)
						{
							wave_function.move_atoms(iconf);
							/*TODO: why no spatial incoherence loop as in CBED? a test run for probe HWHM 0.45 and 2 give results with neigilible difference ~ 10^-9. Maybe this is missed somehow*/
							/*TODO: the temporal incoherence gives NaN in a test run*/
							// temporal incoherence
							for(auto itemp = 0; itemp<qt.size(); itemp++)
							{
								auto c_10 = c_10_0 + qt.x[itemp];
								auto w = w_pr_0*qt.w[itemp];
								input_multislice->cond_lens.set_defocus(c_10); 
								
								for(auto iscan = 0; iscan < input_multislice->scanning.size(); iscan++)
								{
									input_multislice->iscan[0] = iscan;
									input_multislice->set_iscan_beam_position();
									wave_function.set_incident_wave(wave_function.psi_z);
									wave_function.psi(w, wave_function.psi_z, output_multislice);

									ext_iter++;
									if(ext_stop_sim) break;
								}
								if(ext_stop_sim) break;
							}
							if(ext_stop_sim) break;
						}

						wave_function.set_m2psi_coh(output_multislice);
						
						input_multislice->cond_lens.set_defocus(c_10_0);
					}
					else
					{
						for(auto iconf = input_multislice->fp_iconf_0; iconf <= input_multislice->pn_nconf; iconf++)
						{
							wave_function.move_atoms(iconf);	
							for(auto iscan = 0; iscan < input_multislice->scanning.size(); iscan++)
							{
								input_multislice->iscan[0] = iscan;
								input_multislice->set_iscan_beam_position();
								wave_function.set_incident_wave(wave_function.psi_z);
								wave_function.psi(w_pr_0, wave_function.psi_z, output_multislice);

								ext_iter++;
								if(ext_stop_sim) break;
							}
							if(ext_stop_sim) break;
						}

						wave_function.set_m2psi_coh(output_multislice);
					}
				}
			}

			template <class TOutput_multislice>
			void CBED_CBEI(TOutput_multislice &output_multislice)
			{
				output_multislice.init();

				if(input_multislice->is_illu_mod_full_integration())
				{
					Q1<double, e_host> qt;
					Q2<double, e_host> qs;

					// Load quadratures
					cond_lens_temporal_spatial_quadratures(input_multislice->cond_lens, qt, qs);

					/*****************************************************************/
					double w_pr_0 = input_multislice->get_phonon_rot_weight();
					double c_10_0 = input_multislice->cond_lens.c_10;
					const int nbeams = input_multislice->number_of_beams();

					Vector<T_r, e_host> beam_x(nbeams);
					Vector<T_r, e_host> beam_y(nbeams);

					ext_niter = qs.size()*qt.size()*input_multislice->number_conf();
					ext_iter = 0;

					for(auto iconf = input_multislice->fp_iconf_0; iconf <= input_multislice->pn_nconf; iconf++)
					{
						wave_function.move_atoms(iconf);		

						// spatial incoherence
						for(auto ispat = 0; ispat<qs.size(); ispat++)
						{
							for(auto ibeam = 0; ibeam<nbeams; ibeam++)
							{
								beam_x[ibeam] = input_multislice->iw_x[ibeam] + qs.x[ispat];
								beam_y[ibeam] = input_multislice->iw_y[ibeam] + qs.y[ispat];
							}

							// temporal incoherence
							for(auto itemp = 0; itemp<qt.size(); itemp++)
							{
								auto c_10 = c_10_0 + qt.x[itemp];
								auto w = w_pr_0*qs.w[ispat]*qt.w[itemp];

								input_multislice->cond_lens.set_defocus(c_10); 
								wave_function.set_incident_wave(wave_function.psi_z, beam_x, beam_y);
								wave_function.psi(w, wave_function.psi_z, output_multislice);

								ext_iter++;
								if(ext_stop_sim) break;
							}
							if(ext_stop_sim) break;
						}

						if(ext_stop_sim) break;
					}
					wave_function.set_m2psi_coh(output_multislice);

					input_multislice->cond_lens.set_defocus(c_10_0);
					input_multislice->set_beam_position(input_multislice->iw_x, input_multislice->iw_y);
				}
				else
				{
					EWFS_EWRS(output_multislice);
				}
			}

			template <class TOutput_multislice>
			void ED_HRTEM(TOutput_multislice &output_multislice)
			{
				EWFS_EWRS(output_multislice);
			}

			template <class TOutput_multislice>
			void PED_HCTEM(TOutput_multislice &output_multislice)
			{
				ext_niter = input_multislice->nrot*input_multislice->number_conf();
				ext_iter = 0;
				/*****************************************************************/

				T_r w = input_multislice->get_phonon_rot_weight();

				output_multislice.init();

				for(auto iconf = input_multislice->fp_iconf_0; iconf <= input_multislice->pn_nconf; iconf++)
				{
					wave_function.move_atoms(iconf);		
					for(auto irot = 0; irot < input_multislice->nrot; irot++)
					{
						input_multislice->set_phi(irot);
						wave_function.set_incident_wave(wave_function.psi_z);
						wave_function.psi(w, wave_function.psi_z, output_multislice);

						ext_iter++;
						if(ext_stop_sim) break;
					}
					if(ext_stop_sim) break;
				}

				wave_function.set_m2psi_coh(output_multislice);
			}

			template <class TOutput_multislice>
			void EWFS_EWRS(TOutput_multislice &output_multislice)
			{
				ext_niter = input_multislice->number_conf();
				ext_iter = 0;
				/*****************************************************************/

				T_r w = input_multislice->get_phonon_rot_weight();

				output_multislice.init();

				for(auto iconf = input_multislice->fp_iconf_0; iconf <= input_multislice->pn_nconf; iconf++)
				{
					wave_function.move_atoms(iconf);

					wave_function.set_incident_wave(wave_function.psi_z);

					wave_function.psi(w, wave_function.psi_z, output_multislice);

					ext_iter++;
					if(ext_stop_sim) break;
				}

				wave_function.set_m2psi_coh(output_multislice);
			}

			template <class TOutput_multislice>
			void EELS_EFTEM(TOutput_multislice &output_multislice)
			{
				ext_niter = wave_function.slicing.slice.size()*input_multislice->number_conf();
				ext_iter = 0;
				/* set the total number of  iteration ext_niter to number of slices times number of phonon configurations*/

				if(input_multislice->is_EELS())
				{
					ext_niter *= input_multislice->scanning.size();
				}
				/* if is STEM_EELS, then further times the number of scanning positions*/
				/*****************************************************************/

				T_r w = input_multislice->get_phonon_rot_weight();
				/* w is the weight equal to 1/number of phonon configurations times number of rotations. What is rotation in this context?*/
				// TF: I think this possibly refers to precession? It's not properly documented, but from context it may be to simulate precession experiments. See input_multislice.cuh line 524!
				auto psi = [&](T_r w, Vector<T_c, dev> &psi_z, TOutput_multislice &output_multislice) 
				{
					T_r gx_0 = input_multislice->gx_0();
					T_r gy_0 = input_multislice->gy_0();
					/*gx and gy is the projected incident scattering vector, similar to sin(theta)/lambda but with beam tilt within plane phi considered */ 

					for(auto islice = 0; islice < wave_function.slicing.slice.size(); islice++) //loop over slices
					{
						if(input_multislice->eels_fr.is_Mixed_Channelling())
						{
							wave_function.trans(islice, wave_function.slicing.slice.size()-1, trans_thk);
						}
						/* TODO: why mixed_channleing is different?*/			
						// TF: Mixed channeling changes the algorithm from the inelastic event onwards from multislice to a single slice. I guess here it is determining wether the inelastic event happened in that slice. But again... difficult to find out. 
						for(auto iatoms = wave_function.slicing.slice[islice].iatom_0; iatoms <= wave_function.slicing.slice[islice].iatom_e; iatoms++) //loop over atoms within one slice
						{
							if(wave_function.atoms.Z[iatoms] == input_multislice->eels_fr.Z) // check if the atom is the selected type in eels
							{
								input_multislice->set_eels_fr_atom(iatoms, wave_function.atoms);
								energy_loss.set_atom_type(input_multislice->eels_fr);
								/* set atom for eels calcualtions*/

								for(auto ikn = 0; ikn < energy_loss.kernel.size(); ikn++) //loop over energy kernel
								{
									mt::multiply(*stream, energy_loss.kernel[ikn], psi_z, wave_function.psi_z);
									wave_function.psi(islice, wave_function.slicing.slice.size()-1, w, trans_thk, output_multislice);
								}
								/* multiply the wave function with the energy kernel
								then update the wave function, presumbly the electron with energy loss.
								TODO: It looks does not allow mutiple energy loss scattering event. Once the electron lost its energy, it propogates to the exit surface to be collected by the detetor.
								*/
							}

							if(ext_stop_sim) break;
						}
						wave_function.psi_slice(gx_0, gy_0, islice, psi_z);
						/*update wave function (transmit and propogate), presmbly for electrons without energy loss */

						ext_iter++;
						if(ext_stop_sim) break;
					}
				};

				output_multislice.init();

				input_multislice->iscan.resize(1);
				input_multislice->beam_x.resize(1);
				input_multislice->beam_y.resize(1);
				/* TODO: resize the input scanning position to 1D vector. What is the temporty beam_x and y, the location of the probe?*/
				// I really don't understand what that does
				if(input_multislice->is_EELS())
				{
					for(auto iconf = input_multislice->fp_iconf_0; iconf <= input_multislice->pn_nconf; iconf++) //loop over the phonon configurations
					{
						wave_function.move_atoms(iconf); // atom displacement 		
						for(auto iscan = 0; iscan < input_multislice->scanning.size(); iscan++) //loop over the scanning positions
						{
							input_multislice->iscan[0] = iscan;
							input_multislice->set_iscan_beam_position();
							wave_function.set_incident_wave(psi_thk);
							psi(w, psi_thk, output_multislice);
							/* update wave function, how is this different from wave_function.psi and wave_function.psi_slice above*/
							// TF: This psi refers to the definition in line 347. "auto psi = [&] ..." is actually a function containing function calls to wave_function.psi and wave_function.psi_slice
							if(ext_stop_sim) break;
						}

						if(ext_stop_sim) break;
					}
				}
				else
				{
					for(auto iconf = input_multislice->fp_iconf_0; iconf <= input_multislice->pn_nconf; iconf++) //loop over phonon configurations for EFTEM mode
					{
						wave_function.move_atoms(iconf);		
						wave_function.set_incident_wave(psi_thk);
						psi(w, psi_thk, output_multislice);

						if(ext_stop_sim) break;
					}
				}
			}

			template <class TOutput_multislice>
			void EDX(TOutput_multislice &output_multislice)
			{
				ext_niter = input_multislice->scanning.size()*input_multislice->number_conf();
				ext_iter = 0;
				T_r w = input_multislice->get_phonon_rot_weight();

				output_multislice.init();

				input_multislice->iscan.resize(1);
				input_multislice->beam_x.resize(1);
				input_multislice->beam_y.resize(1);		

				for(auto iscan = 0; iscan < input_multislice->scanning.size(); iscan++) //loop over scanning position
				{
					output_multislice.init_psi_coh(); 
					input_multislice->iscan[0] = iscan;
					input_multislice->set_iscan_beam_position();	
					
					for(auto iconf = input_multislice->fp_iconf_0; iconf <= input_multislice->pn_nconf; iconf++)//loop over phonon 
					{	
						wave_function.move_atoms(iconf); //random displacement 
						wave_function.set_incident_wave(wave_function.psi_z);
						wave_function.incoherent_imaging(w, wave_function.psi_z, projected_potential.ionised_potential, output_multislice);
						
						ext_iter++;
						if(ext_stop_sim) break;
					}
					if(ext_stop_sim) break;
				}
			}

			Input_Multislice<T_r> *input_multislice;
			Stream<dev> *stream;
			FFT<T_r, dev> *fft_2d;

			Wave_Function<T_r, dev> wave_function;
			Energy_Loss<T_r, dev> energy_loss;
			Projected_Potential<T, dev> projected_potential;

			Vector<T_c, dev> psi_thk;
			Vector<T_c, dev> trans_thk;
	};

	template <class T, eDevice dev>
	bool Multislice<T, dev>::ext_stop_sim = false;

	template <class T, eDevice dev>
	int Multislice<T, dev>::ext_niter = 0;

	template <class T, eDevice dev>
	int Multislice<T, dev>::ext_iter = 0;

	Propagator<T_r, dev> propagator;
} // namespace mt

#endif
