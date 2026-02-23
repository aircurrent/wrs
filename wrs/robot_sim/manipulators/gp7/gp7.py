import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm

class GP7(mi.ManipulatorInterface):
    """Yaskawa Motoman GP7 model for WRS.

    ```
    Notes
    - Joint/link kinematic parameters are taken from the provided MJCF (MuJoCo XML).
    - This implementation focuses on a *runnable* model (FK + joint motion + visualization).
    - Mesh loading is optional: if mesh files are not found, the model still works via stickmodel.
    """

    # enable_cc=False 碰撞检测
    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 ik_solver='a',
                 name='gp7',
                 enable_cc=False,
                 use_mesh=True):
        # 建立一个
        super().__init__(pos=pos, rotmat=rotmat, home_conf=np.zeros(6), name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)

        # --------
        # Optional meshes (place your
        # GP7 mesh files under: <this_dir>/meshes/
        # Suggested filenames (you may rename as needed, but keep mapping consistent):
        #   base.stl, s_axis.stl, l_axis.stl, u_axis.stl, r_axis.stl, b_axis.stl, t_axis.stl
        # If use_mesh=False or files are missing, the model remains runnable (stickmodel).
        # --------
        def _try_set_cmodel(link_obj, filename, rgba=None, loc_pos=None, loc_rotmat=None):
            fpath = os.path.join(current_file_dir, "meshes", filename)
            if not os.path.exists(fpath):
                return
            link_obj.cmodel = mcm.CollisionModel(initor=fpath, name=os.path.splitext(filename)[0])
            if rgba is not None:
                link_obj.cmodel.rgba = np.array(rgba)
            if loc_pos is not None:
                link_obj.loc_pos = np.array(loc_pos)
            if loc_rotmat is not None:
                link_obj.loc_rotmat = loc_rotmat

        # ----------------
        # Anchor (base link)
        # ----------------
        # ----------------
        # Kinematics (from GP7 xacro / mujoco xml)
        # coordinate: local, angle: radian
        # Joint order: S, L, U, R, B, T
        # ----------------

        # Joint S (about +Z)
        self.jlc.jnts[0].loc_pos = np.array([0.0, 0.0, 0.0])
        self.jlc.jnts[0].loc_motion_ax = np.array([0.0, 0.0, 1.0])
        self.jlc.jnts[0].motion_range = np.array([-2.96705973, 2.96705973])

        # Joint L (about +Y)
        self.jlc.jnts[1].loc_pos = np.array([0.040, 0.0, 0.330])
        self.jlc.jnts[1].loc_motion_ax = np.array([0.0, 1.0, 0.0])
        self.jlc.jnts[1].motion_range = np.array([-1.13446401, 2.53072742])

        # Joint U (about -Y)
        self.jlc.jnts[2].loc_pos = np.array([0.0, 0.0, 0.445])
        self.jlc.jnts[2].loc_motion_ax = np.array([0.0, -1.0, 0.0])
        self.jlc.jnts[2].motion_range = np.array([-2.02458193, 4.45058959])

        # Joint R (about -X)
        self.jlc.jnts[3].loc_pos = np.array([0.0, 0.0, 0.040])
        self.jlc.jnts[3].loc_motion_ax = np.array([-1.0, 0.0, 0.0])
        self.jlc.jnts[3].motion_range = np.array([-3.31612558, 3.31612558])

        # Joint B (about -Y)
        self.jlc.jnts[4].loc_pos = np.array([0.440, 0.0, 0.0])
        self.jlc.jnts[4].loc_motion_ax = np.array([0.0, -1.0, 0.0])
        self.jlc.jnts[4].motion_range = np.array([-2.35619449, 2.35619449])

        # Joint T (about -X)
        self.jlc.jnts[5].loc_pos = np.array([0.080, 0.0, 0.0])
        self.jlc.jnts[5].loc_motion_ax = np.array([-1.0, 0.0, 0.0])
        self.jlc.jnts[5].motion_range = np.array([-6.28318531, 6.28318531])

        # Finalize chain (build FK/IK structures)
        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)

        self._loc_tcp_pos = np.array([0.125751, -0.089503, 0.081999])
        Rx90 = np.array([[1, 0, 0],
                         [0, 0, -1],
                         [0, 1, 0]])
        self._loc_tcp_rotmat = Rx90
        self._is_gl_tcp_delayed = True

        # ----------------
        # Visual / collision meshes
        # Notes:
        # - In your URDF/xacro, there is a fixed joint:
        #     base_link -> vendor_base_link with origin xyz="0 0 0.330"
        #   If your base STL is defined in vendor_base_link frame, we must offset it by +Z 0.330 here.
        # ----------------

        # Base (vendor_base offset)
        _try_set_cmodel(
            # self.jlc.anchor,
            self.jlc.anchor.lnk_list[0],
            "gp7_base_axis.stl",
            rgba=[.5, .5, .5, 1.0],
            # loc_pos=[0.0, 0.0, 0.330]
        )
        # --- mesh coordinate fix per axis (optional) ---
        # Use 3x3 rotation matrices to compensate STL local frame mismatch.
        R_ID = np.eye(3)

        # 先给 L 轴留一个入口：你把这里改成你试出来正确的旋转即可
        R_L = rm.rotmat_from_axangle([0, 1, 0], np.pi / 4)

        axis_mesh_map = [
            # fname, rgba, loc_rotmat, loc_pos
            ("gp7_s_axis.stl", [.1, .3, .5, 1.0], R_ID, None),
            ("gp7_l_axis.stl", [.7, .7, .7, 1.0], R_ID, None),  # <- L轴补偿在这里
            ("gp7_u_axis.stl", [.35, .35, .35, 1.0], R_ID, None),
            ("gp7_r_axis.stl", [.7, .7, .7, 1.0], R_ID, None),
            ("gp7_b_axis.stl", [.1, .3, .5, 1.0], R_ID, None),
            ("gp7_t_axis.stl", [.5, .5, .5, 1.0], R_ID, None),
        ]

        for i, (fname, rgba, R, p) in enumerate(axis_mesh_map):
            _try_set_cmodel(
                self.jlc.jnts[i].lnk,
                fname,
                rgba=rgba,
                loc_rotmat=R if R is not None else None,
                loc_pos=p if p is not None else None
            )

        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        """Basic collision-pair setup (optional)."""
        lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        # Conservative pairs: wrist vs base/shoulder
        from_list = [l3, l4, l5]
        into_list = [lb, l0, l1]
        self.cc.set_cdpair_by_ids(from_list, into_list)


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd


    base = wd.World(cam_pos=[2.0, 0.0, 1.2],
                    lookat_pos=[0.0, 0.0, 0.4])
    mgm.gen_frame().attach_to(base)

    arm = GP7(enable_cc=False, use_mesh=True)

    # 一定要先设关节角
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    arm.goto_given_conf(q)

    # 画 stick（可选）
    arm_stick = arm.gen_stickmodel(
        toggle_flange_frame=True,
        toggle_jnt_frames=True
    )
    arm_stick.attach_to(base)

    # 画 mesh
    arm_mesh = arm.gen_meshmodel()
    arm_mesh.attach_to(base)

    base.run()

