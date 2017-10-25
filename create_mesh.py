import numpy as np
import pygmsh
from matplotlib import pyplot as plt

geom = pygmsh.built_in.Geometry()

# Bottom boundary
z0 = -2000
z_well = z0 + 50
frac_x_spacing = 2.0
frac_z_spacing = 3.0
frac_in_cluster = 5
z_steps = 5.0

# Define the points for a generic cluster of fractures
# Base vector of xs
xs = np.linspace(0, frac_x_spacing*frac_in_cluster, num=frac_in_cluster)
# Base vector of ys
stt_y = z_well - (z_steps * frac_z_spacing / 2.0)
end_y = stt_y + z_steps * frac_z_spacing
zs = np.linspace( stt_y, end_y, num=z_steps)


[x_mesh1, z_mesh1] = np.meshgrid(xs, zs)
[x_mesh2, z_mesh2] = np.meshgrid(xs + 50, zs)

# plt.scatter(x_mesh1, z_mesh1)
# plt.scatter(x_mesh2, z_mesh2)
# plt.show(False)

# Left origin for each cluster
x_cluster_orig = [10, 50]
lc =1
pt_cnt = 0
for clust_orig in x_cluster_orig:
    for i_x in range(len(xs)):
        my_x = xs[i_x] + clust_orig
        for i_z in range(len(zs)):
            geom.add_point([my_x, 0, zs[i_z]],lc)
            if pt_cnt % int(z_steps) > 0:
                geom.add_line(pt_cnt -1, pt_cnt)
            pt_cnt += 1

code = geom.get_code()
folder_pth = 'C:/Users/dorta/Dropbox/Stanford/Research/workspace/new geometry/'
my_file = open(folder_pth + 'my_file.geo','w')
my_file.write(code)
my_file.close()

#Define boundaries
top_bo = -1900;
bottom_b = -2000;
left_b = -50;
right_b = 100;
top_frac = z_well + (z_steps * frac_z_spacing / 2.0)
bottom_frac = top_frac = z_well - (z_steps * frac_z_spacing / 2.0)

geom.add_point([left_b, 0, top_frac])
geom.add_point([left_b, 0, bottom_frac])
geom.add_point([right_b, 0, top_frac])
geom.add_point([right_b, 0, bottom_frac])
