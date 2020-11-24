from math import cos, sin

import cv2
import numpy as np
from scipy.io import loadmat


class MorphabelModel(object):
    """docstring for  MorphabelModel
    model: nver: number of vertices. ntri: number of triangles. *: must have. ~: can generate ones array for place holder.
            'shapeMU': [3*nver, 1]. * mean shape
            'shapePC': [3*nver, n_shape_para]. * principal component
            'shapeEV': [n_shape_para, 1]. ~ eigen value
            'expMU': [3*nver, 1]. ~ mean expression
            'expPC': [3*nver, n_exp_para]. ~ principal component
            'expEV': [n_exp_para, 1]. ~ eigen value
            'texMU': [3*nver, 1]. ~ mean texture
            'texPC': [3*nver, n_tex_para]. ~ principal component
            'texEV': [n_tex_para, 1]. ~ eigen value
            'tri': [ntri, 3] (start from 1, should sub 1 in python and c++). *
            'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles). ~
            'kpt_ind': [68,] (start from 1). ~
    """

    def __init__(self, model_path, model_type='BFM'):
        super(MorphabelModel, self).__init__()
        if model_type == 'BFM':
            self.model = load_BFM(model_path)
        else:
            print('sorry, not support other 3DMM model now')
            exit()

        # fixed attributes
        self.nver = self.model['shapePC'].shape[0] / 3
        self.ntri = self.model['tri'].shape[0]
        self.n_shape_para = self.model['shapePC'].shape[1]
        self.n_exp_para = self.model['expPC'].shape[1]
        self.n_tex_para = self.model['texMU'].shape[1]

        kpt_ind = self.model['kpt_ind']
        kpt_ind = np.tile(kpt_ind[np.newaxis, :], [3, 1]) * 3
        kpt_ind[1, :] += 1
        kpt_ind[2, :] += 2
        self.kpt_ind_all = kpt_ind.flatten('F')

        self.triangles = self.model['tri']
        self.full_triangles = np.vstack(
            (self.model['tri'], self.model['tri_mouth']))

        self.shMU = self.model['shapeMU'][self.kpt_ind_all]
        self.shPC = self.model['shapePC'][self.kpt_ind_all]
        self.expMU = self.model['expMU'][self.kpt_ind_all]
        self.expPC = self.model['expPC'][self.kpt_ind_all]

    # ------------------------------------- shape: represented with mesh(vertices & triangles(fixed))
    def get_shape_para(self, type='random'):
        if type == 'zero':
            sp = np.random.zeros((self.n_shape_para, 1))
        elif type == 'random':
            sp = np.random.rand(self.n_shape_para, 1) * 1e04
        return sp

    def get_exp_para(self, type='random'):
        if type == 'zero':
            ep = np.zeros((self.n_exp_para, 1))
        elif type == 'random':
            ep = -1.5 + 3 * np.random.random([self.n_exp_para, 1])
            ep[6:, 0] = 0

        return ep

    def generate_vertices(self, shape_para, exp_para):
        '''
        Args:
            shape_para: (n_shape_para, 1)
            exp_para: (n_exp_para, 1)
        Returns:
            vertices: (nver, 3)
        '''
        vertices = self.model['shapeMU'] + self.model['shapePC'].dot(
            shape_para) + self.model['expPC'].dot(exp_para)
        vertices = np.reshape(vertices,
                              [int(3), int(len(vertices) / 3)], 'F').T

        return vertices

    def generate_landmark(self, shape_para, exp_para):
        '''
        Args:
            shape_para: (n_shape_para, 1)
            exp_para: (n_exp_para, 1)
        Returns:
            vertices: (nver, 3)
        '''
        vertices = self.shMU + self.shPC.dot(shape_para) + self.expPC.dot(exp_para)
        vertices = np.reshape(vertices, [int(3), int(len(vertices) / 3)], 'F').T
        vertices[:, 1] = -vertices[:, 1]

        return vertices

    # -------------------------------------- texture: here represented with rgb value(colors) in vertices.
    def get_tex_para(self, type='random'):
        if type == 'zero':
            tp = np.zeros((self.n_tex_para, 1))
        elif type == 'random':
            tp = np.random.rand(self.n_tex_para, 1)
        return tp

    def generate_colors(self, tex_para):
        '''
        Args:
            tex_para: (n_tex_para, 1)
        Returns:
            colors: (nver, 3)
        '''
        colors = self.model['texMU'] + self.model['texPC'].dot(
            tex_para * self.model['texEV'])
        colors = np.reshape(colors, [int(3), int(len(colors) / 3)],
                            'F').T / 255.

        return colors

    # ------------------------------------------- transformation
    # -------------  transform
    def rotate(self, vertices, angles):
        ''' rotate face
        Args:
            vertices: [nver, 3]
            angles: [3] x, y, z rotation angle(degree)
            x: pitch. positive for looking down
            y: yaw. positive for looking left
            z: roll. positive for tilting head right
        Returns:
            vertices: rotated vertices
        '''
        return rotate(vertices, angles)

    def transform(self, vertices, s, angles, t3d):
        R = angle2matrix(angles)
        return similarity_transform(vertices, s, R, t3d)

    def transform_3ddfa(self, vertices, s, angles,
                        t3d):  # only used for processing 300W_LP data
        R = angle2matrix_3ddfa(angles)
        return similarity_transform(vertices, s, R, t3d)


def load_BFM(model_path):
    ''' load BFM 3DMM model
    Args:
        model_path: path to BFM model. 
    Returns:
        model: (nver = 53215, ntri = 105840). nver: number of vertices. ntri: number of triangles.
            'shapeMU': [3*nver, 1]
            'shapePC': [3*nver, 199]
            'shapeEV': [199, 1]
            'expMU': [3*nver, 1]
            'expPC': [3*nver, 29]
            'expEV': [29, 1]
            'texMU': [3*nver, 1]
            'texPC': [3*nver, 199]
            'texEV': [199, 1]
            'tri': [ntri, 3] (start from 1, should sub 1 in python and c++)
            'tri_mouth': [114, 3] (start from 1, as a supplement to mouth triangles)
            'kpt_ind': [68,] (start from 1)
    PS:
        You can change codes according to your own saved data.
        Just make sure the model has corresponding attributes.
    '''
    C = loadmat(model_path)
    model = C['model']
    model = model[0, 0]

    # change dtype from double(np.float64) to np.float32,
    # since big matrix process(espetially matrix dot) is too slow in python.
    model['shapeMU'] = (model['shapeMU'] + model['expMU']).astype(np.float32)
    model['shapePC'] = model['shapePC'].astype(np.float32)
    model['shapeEV'] = model['shapeEV'].astype(np.float32)
    model['expEV'] = model['expEV'].astype(np.float32)
    model['expPC'] = model['expPC'].astype(np.float32)

    # matlab start with 1. change to 0 in python.
    model['tri'] = model['tri'].T.copy(order='C').astype(np.int32) - 1
    model['tri_mouth'] = model['tri_mouth'].T.copy(order='C').astype(np.int32) - 1

    # kpt ind
    model['kpt_ind'] = (np.squeeze(model['kpt_ind']) - 1).astype(np.int32)

    return model


def rotate(vertices, angles):
    ''' rotate vertices. 
    X_new = R.dot(X). X: 3 x 1   
    Args:
        vertices: [nver, 3]. 
        rx, ry, rz: degree angles
        rx: pitch. positive for looking down 
        ry: yaw. positive for looking left
        rz: roll. positive for tilting head right
    Returns:
        rotated vertices: [nver, 3]
    '''
    R = angle2matrix(angles)
    rotated_vertices = vertices.dot(R.T)

    return rotated_vertices


def angle2matrix(angles):
    ''' get rotation matrix from three rotation angles(degree). right-handed.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch. positive for looking down.
        y: yaw. positive for looking left. 
        z: roll. positive for tilting head right. 
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    # x
    Rx = np.array([[1,      0,       0],
                   [0, cos(x),  -sin(x)],
                   [0, sin(x),   cos(x)]])
    # y
    Ry = np.array([[cos(y), 0, sin(y)],
                   [0, 1,      0],
                   [-sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), -sin(z), 0],
                   [sin(z),  cos(z), 0],
                   [0,       0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R.astype(np.float32)


def angle2matrix_3ddfa(angles):
    ''' get rotation matrix from three rotation angles(radian). The same as in 3DDFA.
    Args:
        angles: [3,]. x, y, z angles
        x: pitch.
        y: yaw. 
        z: roll. 
    Returns:
        R: 3x3. rotation matrix.
    '''
    # x, y, z = np.deg2rad(angles[0]), np.deg2rad(angles[1]), np.deg2rad(angles[2])
    x, y, z = angles[0], angles[1], angles[2]

    # x
    Rx = np.array([[1,      0,       0],
                   [0, cos(x),  sin(x)],
                   [0, -sin(x),   cos(x)]])
    # y
    Ry = np.array([[cos(y), 0, -sin(y)],
                   [0, 1,      0],
                   [sin(y), 0, cos(y)]])
    # z
    Rz = np.array([[cos(z), sin(z), 0],
                   [-sin(z),  cos(z), 0],
                   [0,       0, 1]])
    R = Rx.dot(Ry).dot(Rz)
    return R.astype(np.float32)


def similarity_transform(vertices, s, R, t3d):
    ''' similarity transform. dof = 7.
    3D: s*R.dot(X) + t
    Homo: M = [[sR, t],[0^T, 1]].  M.dot(X)
    Args:(float32)
        vertices: [nver, 3]. 
        s: [1,]. scale factor.
        R: [3,3]. rotation matrix.
        t3d: [3,]. 3d translation vector.
    Returns:
        transformed vertices: [nver, 3]
    '''
    t3d = np.squeeze(np.array(t3d, dtype=np.float32))
    transformed_vertices = s * vertices.dot(R.T) + t3d[np.newaxis, :]

    return transformed_vertices


def transform_to_roibox(vertices, roi_box):
    vertices_2d = vertices[:, :2]
    bbox = *np.min(vertices_2d, 0), *np.max(vertices_2d, 0)
    sx, sy, ex, ey = bbox
    sxx, syy, exx, eyy = roi_box
    scale = ((exx - sxx) / (ex - sx) + (eyy - syy) / (ey - sy)) / 2

    # transform
    vertices = vertices - np.min(vertices, axis=0)
    vertices = vertices * scale * 0.8
    return vertices
