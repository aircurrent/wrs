import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.modelcollection as mc
import modeling.collisionmodel as cm
import robotsim._kinematics.jlchain as jl
import robotsim.manipulators.sia5.sia5 as sia
import robotsim.grippers.robotiq85.robotiq85 as rtq
from panda3d.core import CollisionNode, CollisionBox, Point3
import robotsim.robots.robot_interface as ri


class SDA5F(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='ur3dual', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # left side
        self.lft_base = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(1), name='lft_base_jl')
        self.lft_base.jnts[1]['loc_pos'] = np.array([0.045, 0, 0.7296])
        self.lft_base.jnts[2]['loc_pos'] = np.array([0.15, 0.101, 0.1704])
        self.lft_base.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(-math.pi / 2.0, -math.pi / 2.0, 0)
        self.lft_base.lnks[0]['name'] = "sda5f_lft_base"
        self.lft_base.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.lft_base.lnks[0]['collisionmodel'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "base_link.stl"),
            cdprimit_type="user_defined", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.lft_base.lnks[0]['rgba'] = [.7, .7, .7, 1.0]
        self.lft_base.lnks[1]['name'] = "sda5f_lft_torso"
        self.lft_base.lnks[1]['loc_pos'] = np.array([0, 0, 0])
        self.lft_base.lnks[1]['collisionmodel'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "torso_link.stl"),
            cdprimit_type="user_defined", expand_radius=.005,
            userdefined_cdprimitive_fn=self._torso_combined_cdnp)
        self.lft_base.lnks[1]['rgba'] = [.7, .7, .7, 1.0]
        self.lft_base.reinitialize()
        lft_arm_homeconf = np.zeros(7)
        # lft_arm_homeconf[0] = math.pi / 3.0
        # lft_arm_homeconf[1] = -math.pi * 1.0 / 3.0
        # lft_arm_homeconf[2] = -math.pi * 2.0 / 3.0
        # lft_arm_homeconf[3] = math.pi
        # lft_arm_homeconf[4] = -math.pi / 2.0
        self.lft_arm = sia.SIA5(pos=self.lft_base.jnts[-1]['gl_posq'],
                                rotmat=self.lft_base.jnts[-1]['gl_rotmatq'],
                                homeconf=lft_arm_homeconf,
                                enable_cc=False)
        # lft hand offset (if needed)
        self.lft_hnd_offset = np.zeros(3)
        lft_hnd_pos, lft_hnd_rotmat = self.lft_arm.cvt_loc_intcp_to_gl(loc_pos=self.lft_hnd_offset)
        self.lft_hnd = rtq.Robotiq85(pos=lft_hnd_pos,
                                     rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'],
                                     enable_cc=False)
        # right side
        self.rgt_base = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(1), name='rgt_base_jl')
        self.rgt_base.jnts[1]['loc_pos'] = np.array([0.045, 0, 0.7296])  # right from robot view
        self.rgt_base.jnts[2]['loc_pos'] = np.array([0.15, -0.101, 0.1704])
        self.rgt_base.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(math.pi / 2.0, -math.pi / 2.0, 0)
        self.rgt_base.lnks[0]['name'] = "sda5f_rgt_base"
        self.rgt_base.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_base.lnks[0]['meshfile'] = None
        self.rgt_base.lnks[1]['name'] = "sda5f_rgt_torso"
        self.rgt_base.lnks[1]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_base.lnks[1]['meshfile'] = None
        self.rgt_base.reinitialize()
        rgt_arm_homeconf = np.zeros(7)
        # rgt_arm_homeconf[0] = -math.pi * 1.0 / 3.0
        # rgt_arm_homeconf[1] = -math.pi * 2.0 / 3.0
        # rgt_arm_homeconf[2] = math.pi * 2.0 / 3.0
        # rgt_arm_homeconf[4] = math.pi / 2.0
        self.rgt_arm = sia.SIA5(pos=self.rgt_base.jnts[-1]['gl_posq'],
                                rotmat=self.rgt_base.jnts[-1]['gl_rotmatq'],
                                homeconf=rgt_arm_homeconf,
                                enable_cc=False)
        # rgt hand offset (if needed)
        self.rgt_hnd_offset = np.zeros(3)
        rgt_hnd_pos, rgt_hnd_rotmat = self.rgt_arm.cvt_loc_intcp_to_gl(loc_pos=self.rgt_hnd_offset)
        # TODO replace using copy
        self.rgt_hnd = rtq.Robotiq85(pos=rgt_hnd_pos,
                                     rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'],
                                     enable_cc=False)
        # tool center point
        # lft
        self.lft_arm.tcp_jntid = -1
        self.lft_arm.tcp_loc_pos = np.array([0, 0, .145])
        self.lft_arm.tcp_loc_rotmat = np.eye(3)
        # rgt
        self.rgt_arm.tcp_jntid = -1
        self.rgt_arm.tcp_loc_pos = np.array([0, 0, .145])
        self.rgt_arm.tcp_loc_rotmat = np.eye(3)
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.lft_oih_infos = []
        self.rgt_oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()

    @staticmethod
    def _base_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(.0, 0.0, 0.225),
                                              x=.14 + radius, y=.14 + radius, z=.225 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.031, 0.0, 0.73),
                                              x=.0855 + radius, y=.0855 + radius, z=.27 + radius)
        collision_node.addSolid(collision_primitive_c1)
        return collision_node

    @staticmethod
    def _torso_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c2 = CollisionBox(Point3(0.195, 0.0, 0.1704),
                                              x=.085 + radius, y=.101 + radius, z=.09 + radius)
        collision_node.addSolid(collision_primitive_c2)
        return collision_node

    def enable_cc(self):
        super().enable_cc()
        # raise NotImplementedError

    def move_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.lft_base.fix_to(self.pos, self.rotmat)
        self.lft_arm.fix_to(pos=self.lft_base.jnts[-1]['gl_posq'], rotmat=self.lft_base.jnts[-1]['gl_rotmatq'])
        lft_hnd_pos, lft_hnd_rotmat = self.lft_arm.get_worldpose(relpos=self.rgt_hnd_offset)
        self.lft_hnd.fix_to(pos=lft_hnd_pos, rotmat=lft_hnd_rotmat)
        self.rgt_base.fix_to(self.pos, self.rotmat)
        self.rgt_arm.fix_to(pos=self.rgt_base.jnts[-1]['gl_posq'], rotmat=self.rgt_base.jnts[-1]['gl_rotmatq'])
        rgt_hnd_pos, rgt_hnd_rotmat = self.rgt_arm.get_worldpose(relpos=self.rgt_hnd_offset)
        self.rgt_hnd.fix_to(pos=rgt_hnd_pos, rotmat=rgt_hnd_rotmat)

    def fk(self, general_jnt_values):
        """
        :param general_jnt_values: 3+7 or 3+7+1, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :return:
        author: weiwei
        date: 20201208toyonaka
        """
        self.pos = np.zeros(0)  # no translation in z
        self.pos[:2] = general_jnt_values[:2]
        self.rotmat = rm.rotmat_from_axangle([0, 0, 1], general_jnt_values[2])
        # left side
        self.lft_base.fix_to(self.pos, self.rotmat)
        self.lft_arm.fix_to(pos=self.lft_base.jnts[-1]['gl_posq'], rotmat=self.lft_base.jnts[-1]['gl_rotmatq'],
                            jnt_values=general_jnt_values[3:10])
        if len(general_jnt_values) != 10:  # gripper is also set
            general_jnt_values[10] = None
        lft_hnd_pos, lft_hnd_rotmat = self.lft_arm.get_worldpose(relpos=self.rgt_hnd_offset)
        self.lft_hnd.fix_to(pos=lft_hnd_pos, rotmat=lft_hnd_rotmat, jnt_values=general_jnt_values[10])
        # right side
        self.rgt_base.fix_to(self.pos, self.rotmat)
        self.rgt_arm.fix_to(pos=self.rgt_base.jnts[-1]['gl_posq'], rotmat=self.rgt_base.jnts[-1]['gl_rotmatq'],
                            jnt_values=general_jnt_values[3:10])
        if len(general_jnt_values) != 10:  # gripper is also set
            general_jnt_values[10] = None
        rgt_hnd_pos, rgt_hnd_rotmat = self.rgt_arm.get_worldpose(relpos=self.rgt_hnd_offset)
        self.rgt_hnd.fix_to(pos=rgt_hnd_pos, rotmat=rgt_hnd_rotmat, jnt_values=general_jnt_values[10])

    def gen_stickmodel(self, name='xarm7_shuidi_mobile'):
        stickmodel = mc.ModelCollection(name=name)
        self.lft_base.gen_stickmodel().attach_to(stickmodel)
        self.lft_arm.gen_stickmodel().attach_to(stickmodel)
        self.lft_hnd.gen_stickmodel().attach_to(stickmodel)
        self.rgt_base.gen_stickmodel().attach_to(stickmodel)
        self.rgt_arm.gen_stickmodel().attach_to(stickmodel)
        self.rgt_hnd.gen_stickmodel().attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self, name='xarm_gripper_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.lft_base.gen_meshmodel().attach_to(meshmodel)
        self.lft_arm.gen_meshmodel().attach_to(meshmodel)
        self.lft_hnd.gen_meshmodel().attach_to(meshmodel)
        self.rgt_base.gen_meshmodel().attach_to(meshmodel)
        self.rgt_arm.gen_meshmodel().attach_to(meshmodel)
        self.rgt_hnd.gen_meshmodel().attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(campos=[3, 0, 3], lookatpos=[0, 0, 1])
    gm.gen_frame().attach_to(base)
    sdarbt = SDA5F()
    sdarbt_meshmodel = sdarbt.gen_meshmodel()
    sdarbt_meshmodel.attach_to(base)
    # sdarbt_meshmodel.show_cdprimit()
    sdarbt.gen_stickmodel().attach_to(base)
    base.run()
