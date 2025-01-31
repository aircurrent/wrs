import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.modelcollection as mc
import modeling.collisionmodel as cm
import robotsim._kinematics.jlchain as jl
import robotsim.manipulators.ur3e.ur3e as ur
import robotsim.grippers.robotiqhe.robotiqhe as rtq
from panda3d.core import CollisionNode, CollisionBox, Point3
import robotsim.robots.robot_interface as ri


class UR3EDual(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='ur3edual', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # left side
        self.lft_base = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='lft_base_jl')
        self.lft_base.jnts[1]['loc_pos'] = np.array([.365, .345, 1.33])  # left from robot view
        self.lft_base.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(math.pi / 2.0, 0,
                                                                   math.pi / 2.0)  # left from robot view
        self.lft_base.lnks[0]['name'] = "ur3e_dual_base"
        self.lft_base.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.lft_base.lnks[0]['collisionmodel'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3e_dual_base.stl"),
            cdprimit_type="user_defined", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.lft_base.lnks[0]['rgba'] = [.3, .3, .3, 1.0]
        self.lft_base.reinitialize()
        lft_arm_homeconf = np.zeros(6)
        lft_arm_homeconf[0] = -math.pi * 2.0 / 3.0
        lft_arm_homeconf[1] = -math.pi * 2.0 / 3.0
        lft_arm_homeconf[2] = math.pi / 2.0
        lft_arm_homeconf[3] = math.pi
        lft_arm_homeconf[4] = -math.pi / 2.0
        self.lft_arm = ur.UR3E(pos=self.lft_base.jnts[-1]['gl_posq'],
                               rotmat=self.lft_base.jnts[-1]['gl_rotmatq'],
                               homeconf=lft_arm_homeconf,
                               enable_cc=False)
        # lft hand offset (if needed)
        self.lft_hnd_offset = np.zeros(3)
        lft_hnd_pos, lft_hnd_rotmat = self.lft_arm.cvt_loc_intcp_to_gl(loc_pos=self.lft_hnd_offset)
        self.lft_hnd = rtq.RobotiqHE(pos=lft_hnd_pos,
                                     rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'],
                                     enable_cc=False)
        # rigth side
        self.rgt_base = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='rgt_base_jl')
        self.rgt_base.jnts[1]['loc_pos'] = np.array([.365, -.345, 1.33])  # right from robot view
        self.rgt_base.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(math.pi / 2.0, 0,
                                                                   math.pi / 2.0)  # left from robot view
        self.rgt_base.lnks[0]['name'] = "ur3e_dual_base"
        self.rgt_base.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_base.lnks[0]['meshfile'] = None
        self.rgt_base.lnks[0]['rgba'] = [.3, .3, .3, 1.0]
        self.rgt_base.reinitialize()
        rgt_arm_homeconf = np.zeros(6)
        rgt_arm_homeconf[0] = math.pi * 2.0 / 3.0
        rgt_arm_homeconf[1] = -math.pi / 3.0
        rgt_arm_homeconf[2] = -math.pi / 2.0
        rgt_arm_homeconf[4] = math.pi / 2.0
        self.rgt_arm = ur.UR3E(pos=self.rgt_base.jnts[-1]['gl_posq'],
                               rotmat=self.rgt_base.jnts[-1]['gl_rotmatq'],
                               homeconf=rgt_arm_homeconf,
                               enable_cc=False)
        # rgt hand offset (if needed)
        self.rgt_hnd_offset = np.zeros(3)
        rgt_hnd_pos, rgt_hnd_rotmat = self.rgt_arm.cvt_loc_intcp_to_gl(loc_pos=self.rgt_hnd_offset)
        # TODO replace using copy
        self.rgt_hnd = rtq.RobotiqHE(pos=rgt_hnd_pos,
                                     rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'],
                                     enable_cc=False)
        # tool center point
        # lft
        self.lft_arm.tcp_jntid = -1
        self.lft_arm.tcp_loc_pos = np.array([0, 0, .07])
        self.lft_arm.tcp_loc_rotmat = np.eye(3)
        # rgt
        self.rgt_arm.tcp_jntid = -1
        self.rgt_arm.tcp_loc_pos = np.array([0, 0, .07])
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
        collision_primitive_c0 = CollisionBox(Point3(0.54, 0.0, 0.39),
                                              x=.54 + radius, y=.6 + radius, z=.39 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.06, 0.0, 0.9),
                                              x=.06 + radius, y=.375 + radius, z=.9 + radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(0.18, 0.0, 1.77),
                                              x=.18 + radius, y=.21 + radius, z=.03 + radius)
        collision_node.addSolid(collision_primitive_c2)
        collision_primitive_l0 = CollisionBox(Point3(0.2425, 0.345, 1.33),
                                              x=.1225 + radius, y=.06 + radius, z=.06 + radius)
        collision_node.addSolid(collision_primitive_l0)
        collision_primitive_r0 = CollisionBox(Point3(0.2425, -0.345, 1.33),
                                              x=.1225 + radius, y=.06 + radius, z=.06 + radius)
        collision_node.addSolid(collision_primitive_r0)
        collision_primitive_l1 = CollisionBox(Point3(0.21, 0.405, 1.07),
                                              x=.03 + radius, y=.06 + radius, z=.29 + radius)
        collision_node.addSolid(collision_primitive_l1)
        collision_primitive_r1 = CollisionBox(Point3(0.21, -0.405, 1.07),
                                              x=.03 + radius, y=.06 + radius, z=.29 + radius)
        collision_node.addSolid(collision_primitive_r1)
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

    def fk(self, component_name, jnt_values):
        """
        :param jnt_values: [nparray, nparray], 6+6, meter-radian
        :jlc_name 'lft_arm', 'rgt_arm', 'both_arm'
        :param component_name:
        :return:
        author: weiwei
        date: 20201208toyonaka
        """

        def update_oih(component_name='rgt_arm'):
            # inline function for update objects in hand
            if component_name == 'rgt_arm':
                oih_info_list = self.rgt_oih_infos
            elif component_name == 'lft_arm':
                oih_info_list = self.lft_oih_infos
            for obj_info in oih_info_list:
                gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat

        super().fk(component_name, jnt_values)
        # examine length
        if component_name == 'lft_arm' or component_name == 'rgt_arm':
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
                raise ValueError("An 1x6 npdarray must be specified to move a single arm!")
            self.manipulator_dict[component_name].fk(jnt_values=jnt_values)
            self.get_hnd_on_component(component_name).fix_to(
                pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'])
            update_oih(component_name=component_name)
        elif component_name == 'both_arm':
            if (not isinstance(jnt_values, list)
                    or jnt_values[0].size != 6
                    or jnt_values[1].size != 6):
                raise ValueError("A list of two 1x6 npdarrays must be specified to move both arm!")
            self.lft_arm.fk(jnt_values=jnt_values[0])
            self.lft_hnd.fix_to(pos=self.lft_arm.jnts[-1]['gl_posq'],
                                rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'])
            self.rgt_arm.fk(jnt_values=jnt_values[1])
            self.rgt_hnd.fix_to(pos=self.rgt_arm.jnts[-1]['gl_posq'],
                                rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'])
            update_oih(component_name='rgt_arm')
            update_oih(component_name='lft_arm')
        elif component_name == 'all':
            pass
        else:
            raise ValueError("The given component name is not available!")

    def gen_stickmodel(self, name='ur3e_dual'):
        stickmodel = mc.ModelCollection(name=name)
        self.lft_base.gen_stickmodel().attach_to(stickmodel)
        self.lft_arm.gen_stickmodel().attach_to(stickmodel)
        self.lft_hnd.gen_stickmodel().attach_to(stickmodel)
        self.rgt_base.gen_stickmodel().attach_to(stickmodel)
        self.rgt_arm.gen_stickmodel().attach_to(stickmodel)
        self.rgt_hnd.gen_stickmodel().attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self, name='ur3e_dual'):
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
    u3ed = UR3EDual()
    # u3ed.fk(.85)
    u3ed_meshmodel = u3ed.gen_meshmodel()
    u3ed_meshmodel.attach_to(base)
    # u3ed_meshmodel.show_cdprimit()
    u3ed.gen_stickmodel().attach_to(base)
    base.run()
