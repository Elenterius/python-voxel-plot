import numpy as np
import random
import plotly.graph_objects as go

import voxel_mesher


class PastelColorUtil:
    """
    Modified version of the random pastel color script by Andreas Dewes
	original source: https://gist.github.com/adewes/5884820
    """

    @staticmethod
    def random_color(pastel_factor=0.5):
        return [(x + pastel_factor) / (1.0 + pastel_factor) for x in [random.uniform(0, 1.0) for _ in [1, 2, 3]]]

    @staticmethod
    def _color_distance(c1, c2):
        return sum([abs(x[0] - x[1]) for x in zip(c1, c2)])

    @staticmethod
    def generate_color(existing_colors=None, pastel_factor=0.5, alpha: float = None):
        if existing_colors is None or len(existing_colors) == 0:
            color = PastelColorUtil.random_color(pastel_factor=pastel_factor)
            if alpha:
                color.append(alpha)
            return color

        max_distance = None
        best_color = None

        for i in range(0, 100):
            color = PastelColorUtil.random_color(pastel_factor=pastel_factor)
            best_distance = min([PastelColorUtil._color_distance(color, c) for c in existing_colors])
            if not max_distance or best_distance > max_distance:
                max_distance = best_distance
                best_color = color

        if alpha:
            best_color.append(alpha)
        return best_color

    @staticmethod
    def generate_colors(num_colors: int, pastel_factor=0.5, alpha: float = None):
        colors = []
        for i in range(0, num_colors):
            colors.append(PastelColorUtil.generate_color(colors, pastel_factor, alpha))
        return colors


def _create_voxel_mesh_figure(vertices: np.ndarray, faces: np.ndarray, face_colors: np.ndarray, name: str) -> go.Figure:
    x, y, z = vertices.T
    i, j, k = faces.T
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=1,
            facecolor=face_colors
        )
    ])
    camera = dict(
        up=dict(x=0, y=1, z=1),
        eye=dict(x=1.25, y=-1.25, z=1.25),
        center=dict(x=0, y=0, z=-0.2)
    )
    fig.update_layout(
        scene_camera=camera,
        title=name,
        autosize=False,
        width=500, height=500,
        margin=dict(l=20,r=20, b=20, t=50, pad=4),
    )
    return fig


def create_voxel_figure(volume: np.ndarray, name: str, colors=None) -> go.Figure:
    assert volume.ndim == 3

    vertices, faces_ = voxel_mesher.mesh_greedy(volume)
    face_colors = faces_.copy()
    faces = faces_[:, :3]

    ids = np.unique(faces_[:, 3])
    if colors is None:
        colors = (np.array(PastelColorUtil.generate_colors(len(ids), 0.8, 1.0)) * 255).astype(int)
    for _id, color in zip(ids, colors):
        face_colors[(faces_[:, 3] == _id)] = color

    return _create_voxel_mesh_figure(vertices, faces, face_colors, name)
