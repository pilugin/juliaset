# juliaset
CUDA + Qt visualization of Julia set fractal
This is a Mandelbrot-like set. `Z(n+1) = Z(n)^2 + C`, where Z and C are complex values. 
Z is mapped to pixels and C is a constant

# Prerequisites
Install Qt
Install CUDA (apt-get install cuda-nvcc-10-2 or any other version)

# Build & Run
qmake && make && ./bin/juliaset-qt

# UI hints
* Drag with left mouse button pressed - navigate 
* Drag vertically with right mouse button pressed - zoom in/out
* Drag with middle mouse button pressed - modify C (vertical direction - adjust real part of C; horizontal direction - adjust imaginary part of C or ther otherway around)
