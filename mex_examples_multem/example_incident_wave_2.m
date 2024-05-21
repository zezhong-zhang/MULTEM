% output_multislice = input_multem.ilc_multem perform TEM simulation
% Incident wave simulation
% All parameters of the input_multem structure are explained in ilm_dflt_input_multem()
% Copyright 2023 Ivan Lobato <Ivanlh20@gmail.com>

clear; clc;
addpath([fileparts(pwd) filesep 'mex_bin'])
addpath([fileparts(pwd) filesep 'crystalline_materials'])
addpath([fileparts(pwd) filesep 'matlab_functions'])

input_multem = ilm_dflt_input_multem();          % Load default values;
% input_multem = multem_input.parameters;          % Load default values;

input_multem.system_conf.precision = 1;                     % eP_Float = 1, eP_double = 2
input_multem.system_conf.device = 1;                        % eD_CPU = 1, eD_GPU = 2
input_multem.system_conf.cpu_nthread = 1; 
input_multem.system_conf.gpu_device = 0;

input_multem.E_0 = 200;                          % Acceleration Voltage (keV)
input_multem.theta = 0.0;
input_multem.phi = 0.0;

input_multem.spec_lx = 50;
input_multem.spec_ly = 50;

input_multem.nx = 1024; 
input_multem.ny = 1024;

%%%%%%%%%%%%%%%%%%%%%%%%%%% Incident wave %%%%%%%%%%%%%%%%%%%%%%%%%%
input_multem.iw_type = 3;   % 1: Plane_Wave, 2: Convergent_wave, 3:User_Define, 4: auto
input_multem.iw_psi = read_psi_0_multem(input_multem.nx, input_multem.ny);    % user define incident wave
input_multem.iw_x = 0.0;    % x position 
input_multem.iw_y = 0.0;    % y position

%%%%%%%%%%%%%%%%%%%%%%%% condenser lens %%%%%%%%%%%%%%%%%%%%%%%%
input_multem.cond_lens_m = 0;                  % Vortex momentum
input_multem.cond_lens_c_10 = -15.836;         % Defocus (�)
input_multem.cond_lens_c_30 = 1e-03;           % Third order spherical aberration (mm)
input_multem.cond_lens_c_50 = 0.00;            % Fifth order spherical aberration (mm)
input_multem.cond_lens_c_12 = 0.0;             % Twofold astigmatism (�)
input_multem.cond_lens_phi_12 = 0.0;           % Azimuthal angle of the twofold astigmatism (�)
input_multem.cond_lens_c_23 = 0.0;             % Threefold astigmatism (�)
input_multem.cond_lens_phi_23 = 0.0;           % Azimuthal angle of the threefold astigmatism (�)
input_multem.cond_lens_inner_aper_ang = 0.0;   % Inner aperture (mrad) 
input_multem.cond_lens_outer_aper_ang = 21.0;  % Outer aperture (mrad)
input_multem.cond_lens_ti_sigma = 32;                % standard deviation (�)
input_multem.cond_lens_ti_npts = 10;               % # of integration points. It will be only used if illumination_model=4
input_multem.cond_lens_si_sigma = 0.2;             % standard deviation: For parallel ilumination(�^-1); otherwise (�)
input_multem.cond_lens_si_rad_npts = 8;             % # of integration points. It will be only used if illumination_model=4

for x = (0.4:0.025:0.6)*input_multem.spec_lx
    for y = (0.4:0.025:0.6)*input_multem.spec_ly
        
        input_multem.iw_x = x;
        input_multem.iw_y = y;
        input_multem.iw_x = input_multem.spec_lx/2;
        input_multem.iw_y = input_multem.spec_ly/2;
        
        tic;
%         output_incident_wave = input_multem.ilc_incident_wave; 
%         output_incident_wave = ilc_incident_wave(input_multem.toStruct); 
        output_multislice = ilc_multem(input_multem.system_conf, input_multem);
        toc;
        psi_0 = flipud(output_incident_wave.psi_0);
        figure(2);
        subplot(1, 2, 1);
        imagesc(abs(psi_0).^2);
        title('intensity');
        axis image;
        colormap gray;

        subplot(1, 2, 2);
        imagesc(angle(psi_0));
        title('phase');
        axis image;
        colormap gray;
        pause(0.1);
    end
end