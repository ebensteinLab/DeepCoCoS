Pre-compiled files for the MATLAB FileExchange contribution "Fast Gaussian Point Spread Function Fitting (MEX)".


-- Pre-compiled files for Linux (compiled with Kubuntu 14.4) --

% LINUX NOTE:
Matlab in Linux comes with its own c++ standard library, which is usually too
old and not compatible with shared libraries used by ceres.
As Matlab loads its own STL before the system libraries (by
setting LD_LIBRARY_PATH to the MATLAB library path) this will result
in failures when the mex file (shared library) is called.
%
If you encounter invalid mex files while executing the program, or
runtime linking errors try setting the LD_PRELOAD environment variable 
before starting matlab to your system libraries (where libstdc++ and
libgfortran are located.
   
   
If you still encounter problems, consider installing the ceres dependencies by executing (works only for Ubuntu/Kubuntu etc.):
# google-glog + gflags
sudo apt-get install libgoogle-glog-dev
# BLAS & LAPACK
sudo apt-get install libatlas-base-dev
# Eigen3
sudo apt-get install libeigen3-dev
# SuiteSparse and CXSparse (optional)
# - If you want to build Ceres as a *static* library (the default)
#   you can use the SuiteSparse package in the main Ubuntu package
#   repository:
sudo apt-get install libsuitesparse-dev
# - However, if you want to build Ceres as a *shared* library, you must
#   add the following PPA:
sudo add-apt-repository ppa:bzindovic/suitesparse-bugfix-1319687
sudo apt-get update
sudo apt-get install libsuitesparse-dev
	
