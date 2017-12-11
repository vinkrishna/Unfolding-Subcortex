######################## Unfold the subcortical structures using non-linear dimensionality reduction
#
#
######################## Dr. Alessandro Crimi alessandro.crimi@usz.ch 


from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
from sklearn import manifold, datasets 
import numpy as np

# Load the STL files and add the vectors 
your_mesh = mesh.Mesh.from_file('Left_Thalamus.stl')

# Convert from groups of vertices (for each triangle) to list of vertices																																													
tri, points, dim = np.shape(your_mesh.vectors)
data_mesh = np.zeros((tri*points,dim))
for index_t in range(tri):
    for index_p in range(points):
        data_mesh[index_t + index_p + index_t*2,:] = your_mesh.vectors[index_t,index_p,:]

# Isomap
#X_r = manifold.Isomap(n_neighbors=10,n_components=2).fit_transform(data_mesh)
# LLE
X_r, err = manifold.locally_linear_embedding(data_mesh, n_neighbors=37,n_components=2 )
# T-sne
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0 )
#X_r = tsne.fit_transform(data_mesh) 


# Plot original vertices
figure = pyplot.figure()
ax = figure.add_subplot(211, projection= '3d')
ax.scatter(data_mesh[:, 0], data_mesh[:, 1], data_mesh[:, 2], c=data_mesh[:, 0], cmap=pyplot.cm.Spectral)
#pyplot.show()
ax.set_title("Original data")

# plor re-mesh
#print X_r
#figure2 = pyplot.figure()
ax = figure.add_subplot(212)
ax.scatter(X_r[:,0] ,X_r[:,1], c=data_mesh[:, 0], cmap=pyplot.cm.Spectral )
pyplot.show()


'''
# Plot original surface
# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors))
# Auto scale to the mesh size
scale = your_mesh.points.flatten(-1)
axes.auto_scale_xyz(scale, scale, scale)
# Show the plot to the screen
pyplot.show()
'''
