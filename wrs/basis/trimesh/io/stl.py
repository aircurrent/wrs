import numpy as np
import os

from ..util import is_binary_file

# define a numpy datatype for the STL file
_stl_dtype_header = np.dtype([('header', np.void, 80), ('face_count', np.uint32)])
_stl_dtype = np.dtype([
    ('normals', np.float32, (3,)),
    ('vertices', np.float32, (3, 3)),
    ('attr', np.uint16)
])


def load_stl(file_obj, file_type=None):
    # file_type 这个参数可以不用，但必须接
    start = file_obj.read(512)
    file_obj.seek(0)
    s = start.lstrip().lower()

    if s.startswith(b"solid") and (b"facet" in s):
        return load_stl_ascii(file_obj)
    return load_stl_binary(file_obj)


def load_stl_binary(file_obj, file_type=None):
    raw = file_obj.read(84)
    if len(raw) != 84:
        raise ValueError("STL header too short (need 84 bytes).")

    header = np.frombuffer(raw, dtype=_stl_dtype_header, count=1)[0]
    # force scalar
    header_fc = int(np.asarray(header['face_count']).reshape(-1)[0])

    data_start = file_obj.tell()  # 84
    file_obj.seek(0, os.SEEK_END)
    file_size = file_obj.tell()
    file_obj.seek(data_start)

    # binary STL: 50 bytes per triangle
    payload = file_size - 84
    if payload < 0:
        raise ValueError("Invalid STL size.")
    calc_fc = payload // 50

    # prefer header if exact; else fall back to file size
    if file_size == 84 + 50 * header_fc:
        face_count = header_fc
    elif file_size == 84 + 50 * calc_fc:
        face_count = int(calc_fc)
    else:
        raise ValueError(
            f"Not a valid binary STL layout: file_size={file_size}, "
            f"header_fc={header_fc} (expects {84+50*header_fc}), "
            f"calc_fc={calc_fc} (expects {84+50*calc_fc})."
        )

    blob = np.frombuffer(file_obj.read(face_count * _stl_dtype.itemsize),
                         dtype=_stl_dtype, count=face_count)

    faces = np.arange(face_count * 3, dtype=np.int32).reshape((-1, 3))
    return {
        'vertices': blob['vertices'].reshape((-1, 3)),
        'face_normals': blob['normals'].reshape((-1, 3)),
        'faces': faces
    }


def load_stl_ascii(file_obj):
    """
    :param file_obj:
    :return:
    author: updated by weiwei
    date: 20230811
    """
    header = file_obj.readline()
    text = file_obj.read()
    if hasattr(text, 'decode'):
        text = text.decode('utf-8')
    text = text.lower().split('endsolid')[0]
    blob = np.array(text.split())
    # there are 21 'words' in each face
    face_len = 21
    face_count = len(blob) / face_len
    if (len(blob) % face_len) != 0:
        raise ValueError('Incorrect number of values in STL file!')
    face_count = int(face_count)
    # this offset is to be added to a fixed set of indices that is tiled
    offset = face_len * np.arange(face_count).reshape((-1, 1))
    normal_index = np.tile([2, 3, 4], (face_count, 1)) + offset
    vertex_index = np.tile([8, 9, 10, 12, 13, 14, 16, 17, 18], (face_count, 1)) + offset
    # faces are groups of three sequential vertices, as vertices are not references
    faces = np.arange(face_count * 3).reshape((-1, 3))
    face_normals = blob[normal_index].astype(float)
    vertices = blob[vertex_index.reshape((-1, 3))].astype(float)
    return {'vertices': vertices,
            'faces': faces,
            'face_normals': face_normals}


def export_stl(mesh):
    '''
    Convert a Trimesh object into a binary STL file.
    Arguments
    ---------
    mesh: Trimesh object
    Returns
    ---------
    export: bytes, representing mesh in binary STL form
    '''
    header = np.zeros(1, dtype=_stl_dtype_header)
    header['face_count'] = len(mesh.faces)
    packed = np.zeros(len(mesh.faces), dtype=_stl_dtype)
    packed['normals'] = mesh.face_normals
    packed['vertices'] = mesh.triangles
    export = header.tostring()
    export += packed.tostring()
    return export


_stl_loaders = {'stl': load_stl}
