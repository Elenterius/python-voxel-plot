import numpy as np


def mesh_each_voxel_as_cube(volume: np.ndarray):
    assert volume.ndim == 3
    cubes = []

    for x in range(volume.shape[0]):
        for y in range(volume.shape[1]):
            for z in range(volume.shape[2]):
                v = volume[x, y, z]
                if v > 0:
                    vertices, faces = _create_cube_mesh(x, y, z)
                    cubes.append([vertices, faces, v])

    return cubes

# Python port of the greedy voxel mesher
# source: https://github.com/mikolalysenko/mikolalysenko.github.com/blob/8f23b8973939f2e29065044178ecddb8d9428fa7/MinecraftMeshes2/js/greedy_tri.js
# author: Mikola Lysenko 
# license: MIT
# copyright: Copyright (c) 2011 Jerome Etienne, http://jetienne.com
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
def mesh_greedy(volume: np.ndarray) -> [np.ndarray, np.ndarray]:
    assert volume.ndim == 3
    volume = volume.astype(dtype=int)
    dims = volume.shape
    mask = np.zeros(4096, dtype=int)
    vertices, faces = [], []

    for d in range(3):
        u, v = (d + 1) % 3, (d + 2) % 3
        xyz = [0, 0, 0]
        q = [0, 0, 0]
        q[d] = 1

        if len(mask) < dims[u] * dims[v]:
            mask = np.zeros(dims[u] * dims[v], dtype=int)

        xyz[d] = -1
        while xyz[d] < dims[d]:
            # Compute mask
            n = 0
            xyz[v] = 0
            while xyz[v] < dims[v]:
                xyz[u] = 0
                while xyz[u] < dims[u]:
                    a = volume[xyz[0], xyz[1], xyz[2]] if 0 <= xyz[d] else 0
                    b = volume[xyz[0] + q[0], xyz[1] + q[1], xyz[2] + q[2]] if xyz[d] < dims[d] - 1 else 0
                    if a == b:
                        mask[n] = 0
                    elif a != 0:
                        mask[n] = a
                    else:
                        mask[n] = -b
                    xyz[u] += 1
                    n += 1
                xyz[v] += 1

            xyz[d] += 1

            # generate mesh for mask using lexicographic ordering
            n = 0
            for j in range(dims[v]):
                i = 0
                while i < dims[u]:
                    c = mask[n]
                    if c != 0:
                        # compute width
                        width = 1
                        while c == mask[n + width] and i + width < dims[u]:
                            width += 1

                        # compute height
                        done = False
                        height = 1
                        while j + height < dims[v]:
                            for w in range(0, width):
                                if c != mask[n + w + height * dims[u]]:
                                    done = True
                                    break
                            if done:
                                break
                            height += 1

                        # add quad
                        xyz[u], xyz[v] = i, j
                        _add_quad(vertices, faces, xyz, width, height, u, v, c)

                        # zero-out mask
                        for h in range(0, height):
                            for w in range(0, width):
                                mask[n + w + h * dims[u]] = 0
                        i += width
                        n += width
                    else:
                        i += 1
                        n += 1
    return np.array(vertices), np.array(faces)


def _add_quad(vertices, faces, xyz, width, height, u, v, c):
    du = [0, 0, 0]
    dv = [0, 0, 0]
    if c > 0:
        dv[v] = height
        du[u] = width
    else:
        c = -c
        du[v] = height
        dv[u] = width

    vertex_count = len(vertices)
    vertices.append([xyz[0], xyz[1], xyz[2]])
    vertices.append([xyz[0] + du[0], xyz[1] + du[1], xyz[2] + du[2]])
    vertices.append([xyz[0] + du[0] + dv[0], xyz[1] + du[1] + dv[1], xyz[2] + du[2] + dv[2]])
    vertices.append([xyz[0] + dv[0], xyz[1] + dv[1], xyz[2] + dv[2]])
    faces.append([vertex_count, vertex_count + 1, vertex_count + 2, c])
    faces.append([vertex_count, vertex_count + 2, vertex_count + 3, c])


def _create_cube_mesh(x, y, z) -> [np.ndarray, np.ndarray]:
    vertices = []
    for i in range(8):
        vertices.append([x + i // 4, y + i // 2 % 2, z + i % 2])

    faces = np.array([
        [0, 1, 0, 1, 0, 6, 2, 7, 4, 7, 1, 7],
        [1, 2, 1, 4, 2, 2, 3, 3, 5, 5, 3, 3],
        [2, 3, 4, 5, 4, 4, 6, 6, 6, 6, 5, 5]
    ]).T

    return np.array(vertices), faces
