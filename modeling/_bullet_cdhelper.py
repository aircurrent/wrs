from panda3d.bullet import BulletRigidBodyNode, BulletPlaneShape
from panda3d.bullet import BulletTriangleMeshShape, BulletTriangleMesh
import numpy as np
import basis.data_adapter as da

SCALE_FOR_PRECISION = 10000.0


def gen_cdmesh_vvnf(vertices, vertex_normals, faces, name='auto'):
    geom = da.pandageom_from_vvnf(vertices * SCALE_FOR_PRECISION, vertex_normals, faces)
    geombullmesh = BulletTriangleMesh()
    geombullmesh.addGeom(geom)
    bullet_triangles_shape = BulletTriangleMeshShape(geombullmesh, dynamic=True)
    bullet_triangles_shape.setMargin(0)
    geombullnode = BulletRigidBodyNode(name=name)
    geombullnode.addShape(bullet_triangles_shape)
    return geombullnode


def gen_plane_cdmesh(updirection=np.array([0, 0, 1]), offset=0, name='autogen'):
    """
    generate a plane bulletrigidbody node
    :param updirection: the normal parameter of bulletplaneshape at panda3d
    :param offset: the d parameter of bulletplaneshape at panda3d
    :param name:
    :return: bulletrigidbody
    author: weiwei
    date: 20170202, tsukuba
    """
    bulletplnode = BulletRigidBodyNode(name)
    bulletplshape = BulletPlaneShape(da.npv3_to_pdv3(updirection), offset)
    bulletplshape.setMargin(0)
    bulletplnode.addShape(bulletplshape)
    return bulletplnode


def is_collided(objcm0, objcm1):
    """
    check if two objcm are collided after converting the specified cdmesh_type
    :param objcm0:
    :param objcm1:
    :return:
    author: weiwei
    date: 20210117
    """
    # avoid using objcm.cdmesh -> be compatible with other cdhelpers
    obj0 = gen_cdmesh_vvnf(*objcm0.extract_rotated_vvnf())
    obj1 = gen_cdmesh_vvnf(*objcm1.extract_rotated_vvnf())
    result = base.physicsworld.contactTestPair(obj0, obj1)
    contacts = result.getContacts()
    contact_points = [da.pdv3_to_npv3(ct.getManifoldPoint().getPositionWorldOnB()) / SCALE_FOR_PRECISION for ct in
                      contacts]
    return (True, contact_points) if len(contact_points) > 0 else (False, contact_points)


def rayhit_closet(pfrom, pto, objcm):
    """
    :param pfrom:
    :param pto:
    :param objcm:
    :return:
    author: weiwei
    date: 20190805, 20210118
    """
    # avoid using objcm.cdmesh -> be compatible with other cdhelpers
    tgt_cdmesh = gen_cdmesh_vvnf(*objcm.extract_rotated_vvnf())
    base.physicsworld.attach(tgt_cdmesh)
    result = base.physicsworld.rayTestClosest(da.npv3_to_pdv3(pfrom), da.npv3_to_pdv3(pto))
    base.physicsworld.removeRigidBody(tgt_cdmesh)
    if result.hasHit():
        return [da.pdv3_to_npv3(result.getHitPos()), da.pdv3_to_npv3(result.getHitNormal())]
    else:
        return [None, None]


def rayhit_all(pfrom, pto, objcm):
    """
    :param pfrom:
    :param pto:
    :param objcm:
    :return:
    author: weiwei
    date: 20190805, 20210118
    """
    # avoid using objcm.cdmesh -> be compatible with other cdhelpers
    tgt_cdmesh = gen_cdmesh_vvnf(*objcm.extract_rotated_vvnf())
    base.physicsworld.attach(tgt_cdmesh)
    result = base.physicsworld.rayTestAll(da.npv3_to_pdv3(pfrom), da.npv3_to_pdv3(pto))
    base.physicsworld.removeRigidBody(tgt_cdmesh)
    if result.hasHits():
        return [[da.pdv3_to_npv3(hit.getHitPos()), da.pdv3_to_npv3(-hit.getHitNormal())] for hit in result.getHits()]
    else:
        return []


if __name__ == '__main__':
    import os, math, basis
    import numpy as np
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm
    import modeling.collisionmodel as cm

    # wd.World(campos=[1.0, 1, .0, 1.0], lookatpos=[0, 0, 0])
    # objpath = os.path.join(basis.__path__[0], 'objects', 'yumifinger.stl')
    # objcm1 = cm.CollisionModel(objpath, cdmesh_type='triangles')
    # homomat = np.array([[-0.5, -0.82363909, 0.2676166, -0.00203699],
    #                     [-0.86602539, 0.47552824, -0.1545085, 0.01272306],
    #                     [0., -0.30901703, -0.95105648, 0.12604253],
    #                     [0., 0., 0., 1.]])
    # # homomat = np.array([[ 1.00000000e+00,  2.38935501e-16,  3.78436685e-17, -7.49999983e-03],
    # #                     [ 2.38935501e-16, -9.51056600e-01, -3.09017003e-01,  2.04893537e-02],
    # #                     [-3.78436685e-17,  3.09017003e-01, -9.51056600e-01,  1.22025304e-01],
    # #                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    # objcm1.set_homomat(homomat)
    # objcm1.set_rgba([1, 1, .3, .2])
    #
    # objpath = os.path.join(basis.__path__[0], 'objects', 'tubebig.stl')
    # objcm2 = cm.CollisionModel(objpath, cdmesh_type='triangles')
    # iscollided, contact_points = is_collided(objcm1, objcm2)
    # objcm1.show_cdmesh()
    # objcm2.show_cdmesh()
    # objcm1.attach_to(base)
    # objcm2.attach_to(base)
    # print(iscollided)
    # for ct_pnt in contact_points:
    #     gm.gen_sphere(ct_pnt, radius=.001).attach_to(base)
    # pfrom = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    # pto = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -0.9])
    # hitpos, hitnrml = rayhit_closet(pfrom=pfrom, pto=pto, objcm=objcm2)
    # gm.gen_sphere(hitpos, radius=.003, rgba=np.array([0, 1, 1, 1])).attach_to(base)
    # gm.gen_stick(spos=pfrom, epos=pto, thickness=.002).attach_to(base)
    # gm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, thickness=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    # base.run()

    wd.World(campos=[1.0, 1, .0, 1.0], lookatpos=[0, 0, 0])
    objpath = os.path.join(basis.__path__[0], 'objects', 'yumifinger.stl')
    objcm1 = cm.CollisionModel(objpath, cdprimit_type='polygons')
    # homomat = np.array([[-0.5, -0.82363909, 0.2676166, -0.00203699],
    #                     [-0.86602539, 0.47552824, -0.1545085, 0.01272306],
    #                     [0., -0.30901703, -0.95105648, 0.12604253],
    #                     [0., 0., 0., 1.]])
    homomat = np.array([[ 1.00000000e+00,  2.38935501e-16,  3.78436685e-17, -7.49999983e-03],
                        [ 2.38935501e-16, -9.51056600e-01, -3.09017003e-01,  2.04893537e-02],
                        [-3.78436685e-17,  3.09017003e-01, -9.51056600e-01,  1.22025304e-01],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
    objcm1.set_homomat(homomat)
    objcm1.set_rgba([1, 1, .3, .2])

    objpath = os.path.join(basis.__path__[0], 'objects', 'tubebig.stl')
    objcm2 = cm.CollisionModel(objpath, cdprimit_type='polygons')
    objcm2.set_rgba([1, 1, .3, .2])
    iscollided, contact_points = is_collided(objcm1, objcm2)
    objcm1.show_cdmesh()
    objcm2.show_cdmesh()
    objcm1.attach_to(base)
    objcm2.attach_to(base)
    print(iscollided)
    for ct_pnt in contact_points:
        gm.gen_sphere(ct_pnt, radius=.001).attach_to(base)
    pfrom = np.array([0, 0, 0]) + np.array([1.0, 1.0, 1.0])
    pto = np.array([0, 0, 0]) + np.array([-1.0, -1.0, -0.9])
    hitpos, hitnrml = rayhit_closet(pfrom=pfrom, pto=pto, objcm=objcm2)
    gm.gen_sphere(hitpos, radius=.003, rgba=np.array([0, 1, 1, 1])).attach_to(base)
    gm.gen_stick(spos=pfrom, epos=pto, thickness=.002).attach_to(base)
    gm.gen_arrow(spos=hitpos, epos=hitpos + hitnrml * .07, thickness=.002, rgba=np.array([0, 1, 0, 1])).attach_to(base)
    base.run()

