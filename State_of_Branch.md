At this commit we have a method that takes a structure and a dictionary (called pyprismatic kwargs) and runs the relevant simulation. This is achieved using 3 util functions.

Next stages include:

Taming the output of the simulation, currently we import the .mrc file (using a prismatic defined IO) to form a 3D numpy array. This does not align well with PyCrystEM as it currently stands.

Deciding what kind of file cleaning to do, for example should we destroy .XYZ and .mrc files after use? This could be a user choice, although most users with interesting filename needs would probably be better of accessing prismatic through the command line (or directly via a short docopted python script).

Deciding where/how to store meta, the class that contains all of the parameters of interest.

Implementing an exciting way of guessing what the multislice index should be, namely by looking at structure. This is probably best done via another util. 
