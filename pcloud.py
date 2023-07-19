import numpy as np
import point_cloud_utils as pcu
import pyvista as pv

v, f, n = pcu.load_mesh_vfn('mesh.obj')
pc = pcu.sample_mesh_lloyd(v, f, 1000)
output_file = 'pcloud' + '.npy'
np.save(output_file, pc)

input_pc = np.load('pcloud.npy')
print(input_pc.shape)
cloud = pv.PolyData(input_pc)
cloud.plot()