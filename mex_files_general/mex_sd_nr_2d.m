% Copyright 2021 Ivan Lobato <Ivanlh20@gmail.com>
clc; clear;
addpath(['..', filesep, 'matlab_functions'])

ilm_mex('release', 'ilc_sd_nr_2d.cu', '../src');