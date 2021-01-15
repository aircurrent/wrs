import math
import numpy as np
import basis.robotmath as rm
import grasping.annotated.utils as gutil

if __name__ == '__main__':

    import os
    import basis
    import robotsim.grippers.yumi_gripper.yumi_gripper as yg
    import modeling.collisionmodel as cm
    import visualization.panda.world as wd

    base = wd.World(campos=[.5, .5, .3], lookatpos=[0, 0, 0])
    gripper_instance = yg.YumiGripper(enable_cc=True, cdmesh_type='triangles')
    objcm = cm.CollisionModel('../objects/tubebig.stl')
    objcm.attach_to(base)
    objcm.show_localframe()
    # objcm.show_cdmesh()
    # base.run()
    grasp_info_list = []
    for height in [.085, .095]:
        for roll_angle in [math.pi*.1, math.pi*.2]:
            gl_hndz = rm.rotmat_from_axangle(np.array([1,0,0]), roll_angle).dot(np.array([0,0,-1]))
            grasp_info_list += gutil.define_grasp_with_rotation(gripper_instance,
                                                                objcm,
                                                                gl_jaw_center=np.array([0,0,height]),
                                                                gl_hndz=gl_hndz,
                                                                gl_hndx=np.array([1,0,0]),
                                                                jaw_width=.02,
                                                                rotation_ax=np.array([0,0,1]))
    for grasp_info in grasp_info_list:
        jaw_width, gl_jaw_center, pos, rotmat = grasp_info
        # gic = gripper_instance.copy()
        gripper_instance.fix_to(pos, rotmat)
        gripper_instance.jaw_to(jaw_width)
        gripper_instance.gen_meshmodel().attach_to(base)
    base.run()
