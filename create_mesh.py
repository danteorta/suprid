import numpy as np
import pygmsh
from matplotlib import pyplot as plt

geom = pygmsh.built_in.Geometry()

# Bottom boundary

z_well = -1950
frac_x_spacing = 2.0
frac_z_spacing = 3.0
frac_in_cluster = 5
z_steps = 5

# Define the points for a generic cluster of fractures
# Base vector of x's
xs = np.linspace(0, frac_x_spacing*frac_in_cluster, num=frac_in_cluster)
# Base vector of z's
stt_y = z_well - (z_steps * frac_z_spacing / 2.0)
end_y = stt_y + z_steps * frac_z_spacing
zs = np.linspace( stt_y, end_y, num=z_steps)
# Just an array of the same thing (unused)
[x_mesh1, z_mesh1] = np.meshgrid(xs, zs)
[x_mesh2, z_mesh2] = np.meshgrid(xs + 50, zs)
# Left origin for each cluster
x_cluster_orig = [10, 50]
# Mesh quality parameter
lc =1
# plt.scatter(x_mesh1, z_mesh1)
# plt.scatter(x_mesh2, z_mesh2)
# plt.show(False)

# Create the fracture points and lines
for clust_orig in x_cluster_orig:
    for i_x in range(len(xs)):
        my_x = xs[i_x] + clust_orig
        for i_z in range(len(zs)):
            last_p = geom.add_point([my_x, 0, zs[i_z]],lc)
# Same Loop. Separate to get the geo code more tidy
for pt_cnt in range(int(last_p.id) + 1):
    if pt_cnt <= int(last_p.id) - z_steps and pt_cnt % int(z_steps) > 0:
        print(pt_cnt)
        last_line = geom.add_line(pt_cnt-1,pt_cnt)
        last_line = geom.add_line(pt_cnt, pt_cnt+z_steps)
        last_line = geom.add_line(pt_cnt+z_steps, pt_cnt+z_steps-1)
        last_line = geom.add_line(pt_cnt+z_steps-1, pt_cnt-1)
        lst_ln = int(last_line.id)
        # Add line loop in the fractures inner area
        last_ll = geom.add_line_loop(range(lst_ln -3, lst_ln))

#Define boundaries
top_frac = z_well + (z_steps * frac_z_spacing / 2.0)
bottom_frac = z_well - (z_steps * frac_z_spacing / 2.0)
x_bounds = [-50, 100]
z_bounds = [-2000, bottom_frac, top_frac, -1900]

for x_bnd in x_bounds:
    for z_bnd in z_bounds:
        geom.add_point([x_bnd, 0, z_bnd], 10 * lc)

# geom.add_point([left_b, 0, top_frac], 10*lc)
# geom.add_point([left_b, 0, bottom_frac], 10*lc)
# geom.add_point([right_b, 0, top_frac], 10*lc)
# geom.add_point([right_b, 0, bottom_frac], 10*lc)


code = geom.get_code()
folder_pth = 'C:/Users/dorta/Dropbox/Stanford/Research/workspace/new geometry/'
my_file = open(folder_pth + 'my_file.geo','w')
my_file.write(code)
my_file.close()