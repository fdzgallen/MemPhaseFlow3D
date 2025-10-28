# MemPhaseFlow3D
Code to simulate a biological membrane inside a fluid flow in 

 > Please cite as: A. F. Gallen, J. Muñoz-Biosca, M. Castro and A. Hernandez-Machado, 

This repository is a Work in Progress and will be updated over time.

Code to simulate the temporal evolution of a biological membrane inside a fluid flow in 3D.

This is the code used in the peer-review article "Vorticity-Stream vector formulation for 3-dimensional viscous flow of deformable bodies". It is uploaded here for transparency and to make the use or implementation of our model easier for any person interested.

Modifying the code as explained in the article you can simulate multiple flows: Poiseuille, Couette, temporal-dependent etc. Here we will provide a few versions of the code so you can simulate different systems without issue, but playing with the code to explore new possibilities is encouraged.

# Input


# Output

The simulation will create a folder with the name that it is given in the input file. Inside this folder the code stores images and data of the state of the simulation every a given interval of iterations.
It also creates a variety of folders to store the data of the different variables in the simulation, "phi", "stream", and "vorticity". In each of this folders the state of each of this fields is stored over time. 
In case there is interest on analysing more in depth the evolution of the system you can write a code that reads these output files and computes the desired values from the data without need of running the simulation all over again. This uses storage space but saves a lot of time if you want to compute a new value (for example the evolution of the center of masss of the cell (phi) over time) but you do not want to add the calculation to the code and run the simulation again.


# How does it work?

The details of the mathematical model can be read in the article.
