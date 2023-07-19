import numpy as np
import point_cloud_utils as pcu
import pyvista as pv

v, f, n = pcu.load_mesh_vfn('mesh.obj')
'''
a = len(n)
b = len(n[0])
n_points = a + b
print(n_points)
'''
pc = pcu.sample_mesh_lloyd(v, f, 2048)
output_file = 'pcloud' + '.npy'
np.save(output_file, pc)

input_pc = np.load('pcloud.npy')
print(input_pc.shape)
cloud = pv.PolyData(input_pc)
cloud.plot()