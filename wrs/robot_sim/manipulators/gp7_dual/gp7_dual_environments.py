import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.robots.dual_arm_robot_interface as dari
from wrs.robot_sim.manipulators.gp7.gp7 import GP7
import wrs.modeling.constant as mconst
import wrs.visualization.panda.world as wd

_CUR_DIR = os.path.dirname(__file__)

DEFAULT_LFT_TOOL = os.path.join(_CUR_DIR, "meshes", "lft_hand.stl")
DEFAULT_RGT_TOOL = os.path.join(_CUR_DIR, "meshes", "rgt_hand.stl")

class GP7_Dual(dari.DualArmRobotInterface):
    """
    Minimal dual-arm GP7:
    - one anchor with 2 flanges
    - two GP7 instances mounted on the flanges
    - override each arm's end-link vmodel/cmodel with different grippers (route-1)
    - no collision checker setup (yet)
    """

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 name="gp7_dual",
                 enable_cc=False,
                 # 两个工具（STL 路径）
                 lft_gripper_stl=None,
                 rgt_gripper_stl=None,
                 # 工具相对末端link的局部位姿（如果你STL已在CAD对齐到flange原点，就保持None/单位阵）
                 lft_gripper_loc_pos=None,
                 lft_gripper_loc_rotmat=None,
                 rgt_gripper_loc_pos=None,
                 rgt_gripper_loc_rotmat=None):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)

        # ===== 默认工具兜底 =====
        if lft_gripper_stl is None:
            lft_gripper_stl = DEFAULT_LFT_TOOL
        if rgt_gripper_stl is None:
            rgt_gripper_stl = DEFAULT_RGT_TOOL

        # --- anchor: only provides two mounting flanges ---
        self._body = rkjlc.rkjl.Anchor(
            name=self.name + "_anchor",
            pos=self.pos,
            rotmat=self.rotmat,
            n_flange=2,
            n_lnk=1
        )
        self._body.lnk_list[0].name = self.name + "_base_link"

        # --- mount poses (先随便定一组) ---
        # base坐标系定在inserting robot的支座的底部，0.54是picking robot距离inserting robot的距离，
        self._body.loc_flange_pose_list[0] = [rm.vec(0.0, 0.0, 0.585), np.eye(3)]
        self._body.loc_flange_pose_list[1] = [rm.vec(0.023, +0.54, 0.535), np.eye(3)]
        # self._body.loc_flange_pose_list[1] = [rm.vec(0.0, 0.0, 0.585), rm.rotmat_from_axangle([0, 0, 1], np.pi)]

        # --- instantiate arms ---
        # 用 body 上的第 0 个 flange 位姿创建一个 GP7并命名为 “左臂”
        self._lft_arm = GP7(
            pos=self._body.gl_flange_pose_list[0][0],
            rotmat=self._body.gl_flange_pose_list[0][1],
        )

        # 用 body 上的第 1 个 flange 位姿创建一个 GP7并命名为 “右臂”
        self._rgt_arm = GP7(
            pos=self._body.gl_flange_pose_list[1][0],
            rotmat=self._body.gl_flange_pose_list[1][1],
        )

        # 左臂 TCP
        self._lft_loc_tcp_pos = np.array([125.751, -89.503, 81.999], dtype=float) * 0.001
        eye = np.eye(3)
        self._lft_loc_tcp_rotmat  = np.array([[1, 0, 0],
                          [0, 0, -1],
                          [0, 1, 0]], dtype=float)
        self._set_arm_tcp(
            self._lft_arm,
            loc_tcp_pos=self._lft_loc_tcp_pos,
            loc_tcp_rotmat=self._lft_loc_tcp_rotmat
        )

        # 右臂 TCP（示例）
        self._rgt_loc_tcp_pos = np.array([132.707, -60, 0], dtype=float) * 0.001
        eye = np.eye(3)
        # self._loc_rgt_tcp_rot = eye
        self._loc_rgt_tcp_rot = np.array([[0, 0, 1],
                          [0, 1, 0],
                          [-1, 0, 0]], dtype=float)
        self._set_arm_tcp(
            self._rgt_arm,
            loc_tcp_pos=self._rgt_loc_tcp_pos,
            loc_tcp_rotmat=self._loc_rgt_tcp_rot
        )

        # --- override end-link tool models (route-1) ---
        if lft_gripper_stl is not None:
            self._set_tool_on_end_link(
                arm=self._lft_arm,
                stl_path=lft_gripper_stl,
                rgba=[0.0, 0.6, 0.6, 1.0],
                loc_pos=lft_gripper_loc_pos,
                loc_rotmat=lft_gripper_loc_rotmat
            )
        if rgt_gripper_stl is not None:
            self._set_tool_on_end_link(
                arm=self._rgt_arm,
                stl_path=rgt_gripper_stl,
                rgba=[0.6, 0.6, 0.0, 1.0],
                loc_pos=rgt_gripper_loc_pos,
                loc_rotmat=rgt_gripper_loc_rotmat
            )

        self._hairpin_model = None
        # home + default delegator
        self.goto_home_conf()
        self.use_lft()

    @staticmethod
    def _set_tool_on_end_link(arm: GP7,
                              stl_path: str,
                              rgba=None,
                              loc_pos=None,
                              loc_rotmat=None):
        """
        Override arm end-link vmodel/cmodel to be the given tool STL.
        Assumes the tool should be attached to joint-5's link (6th axis link).
        """
        if not os.path.exists(stl_path):
            raise FileNotFoundError(stl_path)

        end_lnk = arm.jlc.jnts[5].lnk

        # visual
        vm = mgm.GeometricModel(initor=stl_path, name=os.path.splitext(os.path.basename(stl_path))[0] + "_tool_v")
        if rgba is not None:
            vm.rgba = np.array(rgba)

        # 尽量用 set_pos/set_rotmat（兼容不同WRS版本）
        if loc_rotmat is not None:
            if hasattr(vm, "set_rotmat"):
                vm.set_rotmat(loc_rotmat)
            else:
                vm.rotmat = loc_rotmat
        if loc_pos is not None:
            if hasattr(vm, "set_pos"):
                vm.set_pos(np.array(loc_pos))
            else:
                vm.pos = np.array(loc_pos)

        end_lnk.vmodel = vm

        # collision (先挂上，后面你启用cc再用)
        cm = mcm.CollisionModel(initor=stl_path, name=os.path.splitext(os.path.basename(stl_path))[0] + "_tool_c")
        if rgba is not None:
            cm.rgba = np.array(rgba)
        if loc_pos is not None:
            cm.loc_pos = np.array(loc_pos)
        if loc_rotmat is not None:
            if hasattr(cm, "set_rotmat"):
                cm.set_rotmat(loc_rotmat)
            else:
                cm.rotmat = loc_rotmat
        end_lnk.cmodel = cm

    def _set_arm_tcp(self, arm, loc_tcp_pos, loc_tcp_rotmat):
        arm._loc_tcp_pos = np.asarray(loc_tcp_pos, dtype=float).reshape(3, )
        arm._loc_tcp_rotmat = np.asarray(loc_tcp_rotmat, dtype=float).reshape(3, 3)
        # 有些版本需要置 True 才会在下次更新全局 TCP
        if hasattr(arm, "_is_gl_tcp_delayed"):
            arm._is_gl_tcp_delayed = True

    def load_environment_models(self, base):
        """
        存放不会移动的物体，对于会移动的物体则另说
        Load a, b, c models from ./objects folder.
        Assume STL local coordinates are already correct.
        """

        obj_dir = os.path.join(_CUR_DIR, "objects")

        paths = {
            "camera_frame": os.path.join(obj_dir, "camera_frame.stl"),
            "fin_support": os.path.join(obj_dir, "fin_support.stl"),
            "robot_support_rack": os.path.join(obj_dir, "robot_support_rack.stl"),
            "hairpin_container": os.path.join(obj_dir, "hairpin_container.stl"),
            "fins": os.path.join(obj_dir, "fins.stl"),
        }

        self._env_models = []

        color_map = {
            "camera_frame": [0.6, 0.6, 0.6, 0.6],
            "fin_support": [0.5, 0.5, 0.5, 1],
            "robot_support_rack": [0.3, 0.3, 0.3, 1],
            "hairpin_container": [0.5, 0.6, 0.2, 1],
            "fins": [0.5, 0.3, 0.7, 1],
        }

        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(path)

            rot_fix = rm.rotmat_from_axangle([1, 0, 0], np.pi / 2)

            model = mcm.CollisionModel(
                initor=path,
                name=name,
                cdprim_type=mconst.CDPrimType.AABB,
                cdmesh_type = mconst.CDMeshType.CONVEX_HULL
            )

            # 不调位置
            # model.pos = np.zeros(3)
            model.rotmat = rot_fix
            model.rgba = np.array(color_map[name])
            model.attach_to(base)
            self._env_models.append(model)

    def goto_given_conf(self, jnt_values, ee_values=None):
        """
        For GP7 manipulator, ignore ee_values (gripper is only visual here).
        """
        if self._delegator is None:
            # both arms mode: jnt_values should be concatenated [lft(6), rgt(6)]
            lftd = self._lft_arm.manipulator.n_dof
            self._lft_arm.goto_given_conf(jnt_values[:lftd])
            self._rgt_arm.goto_given_conf(jnt_values[lftd:lftd + self._rgt_arm.manipulator.n_dof])
        else:
            # single arm mode: just pass joints down, DO NOT pass ee_values
            self._delegator.goto_given_conf(jnt_values=jnt_values)

    def load_hairpin(self, base):
        """
        Spawn a single hairpin in the scene (world frame).
        Units: meters, radians.
        Assumption: STL local coordinates are already correct (or you provide rot_fix).
        """
        obj_dir = os.path.join(_CUR_DIR, "objects")
        hairpin_path = os.path.join(obj_dir, "hairpin.stl")  # 你把文件名改成真实的

        if not os.path.exists(hairpin_path):
            raise FileNotFoundError(hairpin_path)

        # -----------------------------
        # 1) 创建碰撞模型
        # -----------------------------
        hairpin = mcm.CollisionModel(
            initor=hairpin_path,
            name="hairpin",
            cdprim_type=mconst.CDPrimType.CAPSULE,         # hairpin 这种细长件用 CAPSULE 更合适
            cdmesh_type=mconst.CDMeshType.CONVEX_HULL      # 规划/碰撞更稳定
        )

        # -----------------------------
        # 2) 初始摆放位姿（你后面改成实际料盒位置）
        #    pos: [x,y,z] in meters
        # -----------------------------
        # 示例：放在桌面上方 2cm（你按你的桌面高度/容器位置改）
        # init_pos = np.array([0.48, 0.413, 0.563])  # TODO: 替换成你实际的位置（米）
        init_pos = np.array([0.480, -0.563, 0.417])  # TODO: 替换成你实际的位置（米）

        # 示例：让 hairpin 平躺，绕 world 的某轴旋转一下（按你的 stl 朝向修正）
        # rot_fix = rm.rotmat_from_axangle([0, 0, 1], -np.pi / 2)  # 和你环境一致的修正（必要时改）
        Rz = rm.rotmat_from_axangle([0, 0, 1], -np.pi / 2)
        Ry = rm.rotmat_from_axangle([0, 1, 0], -np.pi / 2)
        # 右边先执行
        init_rot = Ry @ Rz

        hairpin.pos = init_pos
        hairpin.rotmat = init_rot

        # -----------------------------
        # 3) 颜色
        # -----------------------------
        hairpin.rgba = np.array([0.95, 0.65, 0.10, 1.0])  # 橙黄，不透明（方便看）

        # -----------------------------
        # 4) attach 到 world
        # -----------------------------
        hairpin.attach_to(base)

        # 可选：显示碰撞 primitive，调试用
        # hairpin.show_cdprim()

        self._hairpin_model = hairpin
        return hairpin




if __name__ == "__main__":


    base = wd.World(cam_pos=[3, 3, 2], lookat_pos=[0, 0, 0.8])
    lens = base.cam.node().getLens()  # Panda3D lens
    lens.setNearFar(0.005, 5000.0)  # near 更小，far 更大

    mgm.gen_frame(ax_length=.5).attach_to(base)

    robot = GP7_Dual(
        enable_cc=False,
        lft_gripper_stl=DEFAULT_LFT_TOOL,
        rgt_gripper_stl=DEFAULT_RGT_TOOL,
        # 如果你CAD里已经让STL原点=法兰原点、方向=法兰方向，就都保持None
        lft_gripper_loc_pos=None,
        lft_gripper_loc_rotmat=None,
        rgt_gripper_loc_pos=None,
        rgt_gripper_loc_rotmat=None,
    )

    # print([k for k in dir(robot._lft_arm) if "jlc" in k.lower() or "chain" in k.lower() or "manip" in k.lower()])

    robot.gen_meshmodel().attach_to(base)
    robot.gen_stickmodel(toggle_flange_frame=True).attach_to(base)

    # 先让机器人到 home 或某个姿态
    robot.goto_home_conf()

    # 画左侧robot
    q_l = robot._lft_arm.get_jnt_values()
    tcp_pos_l, tcp_rot_l = robot._lft_arm.fk(jnt_values=q_l)
    # 画 TCP 坐标系
    mgm.gen_frame(pos=tcp_pos_l,
                  rotmat=tcp_rot_l,
                  ax_length=0.08).attach_to(base)

    # 画右侧robot
    q_r = robot._rgt_arm.get_jnt_values()
    tcp_pos_r, tcp_rot_r = robot._rgt_arm.fk(jnt_values=q_r)
    mgm.gen_frame(pos=tcp_pos_r,
                  rotmat=tcp_rot_r,
                  ax_length=0.08).attach_to(base)

    # 加载环境中的各个物体
    # robot.load_environment_models(base)
    # robot.load_hairpin(base)
    base.run()
