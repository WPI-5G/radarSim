# import torch
# print(torch.cuda.is_available())

import h5py

# Open the HDF5 file
file = h5py.File('Training.jld2', 'r')

# Print the name of the file
print(file.filename)

# Get the data from the "data" dataset
data = file["inputs"]

# Print the data
print(len(data[:,0]))

# Close the file
file.close()
