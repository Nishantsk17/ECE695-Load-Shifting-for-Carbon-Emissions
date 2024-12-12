# ECE695-Load-Shifting-for-Carbon-Emissions
In order to run the training and testing in the above model, just adjust the hyperparameters accordingly in the main function of loadShift.py and run it.
There are two dataLoader files in this repo, dataLoader_1 doesn't tale into account temporal training but dataLoader_2 does. The dataLoader_2 file trains on ~5 months of data tests on the 6th month. Change the import command with the right dataLoader file in the top of loadShift.py  
