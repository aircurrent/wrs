import numpy as np
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
from wrs.robot_sim.manipulators.gp7_dual.gp7_dual_environments import GP7_Dual


# ========= 可调参数 =========
N_STEP = 60
SHOW_TCP_PATH = True
SHOW_TCP_FRAMES = True
AX_LEN = 0.12

# 起始关节角（要保证 IK 可解）
q0 = np.zeros(6)
q_lft_start = q0
q_rgt_start = q0

# TCP 位移（m）：世界坐标系下位移
dp_lft = np.array([0.02, 0.03, -0.03])
dp_rgt = np.array([0.02, 0.03, -0.03])
# ============================


def ensure_ik_ready(robot: GP7_Dual):
    """确保左右臂都初始化好 IK solver。

    注意：GP7_Dual 内部通过 delegator 切换左右臂。
    需要分别对两条链 finalize，否则调用 ik() 可能报 "IK solver undefined"。
    """
    # 左臂
    robot.use_lft()
    robot.delegator.jlc.finalize(ik_solver='n')
    # 右臂
    robot.use_rgt()
    robot.delegator.jlc.finalize(ik_solver='n')
    # 默认回到左臂
    robot.use_lft()


def _set_delegator_tcp_for_arm(robot_dual: GP7_Dual, arm: str) -> None:
    """确保 delegator 的 TCP 与指定 arm 一致。

    经验上：WRS 的某些 dual-arm wrapper 在切 arm 后，delegator 的 _loc_tcp_* 未必
    自动同步到该 arm 的 TCP 定义，导致 IK 的末端参考系混乱。
    这里显式把对应 arm 的 TCP 写入 delegator。
    """
    if arm not in ("lft", "rgt"):
        raise ValueError("arm must be 'lft' or 'rgt'")

    if arm == "rgt":
        tcp_pos = np.asarray(robot_dual._rgt_loc_tcp_pos, dtype=float)
        tcp_rot = np.asarray(getattr(robot_dual, "_rgt_loc_tcp_rotmat", np.eye(3)), dtype=float)
    else:
        tcp_pos = np.asarray(robot_dual._lft_loc_tcp_pos, dtype=float)
        tcp_rot = np.asarray(getattr(robot_dual, "_lft_loc_tcp_rotmat", np.eye(3)), dtype=float)

    # delegator 在不同版本里可能叫 delegator / _delegator
    d = getattr(robot_dual, "delegator", None)
    if d is None:
        d = getattr(robot_dual, "_delegator")

    # 兼容不同字段命名
    if hasattr(d, "_loc_tcp_pos"):
        d._loc_tcp_pos = tcp_pos.copy()
    if hasattr(d, "loc_tcp_pos"):
        d.loc_tcp_pos = tcp_pos.copy()
    if hasattr(d, "_loc_tcp_rotmat"):
        d._loc_tcp_rotmat = tcp_rot.copy()
    if hasattr(d, "loc_tcp_rotmat"):
        d.loc_tcp_rotmat = tcp_rot.copy()
    if hasattr(d, "_is_gl_tcp_delayed"):
        d._is_gl_tcp_delayed = True


def ik_tcp_via_flange(robot_dual: GP7_Dual,
                      arm: str,
                      tgt_pos_tcp: np.ndarray,
                      tgt_rot_tcp: np.ndarray,
                      seed_jnt_values: np.ndarray):
    """用“强制 flange-IK”的方式做 TCP IK（稳定版）。

    关键做法：
    - 先用 flange->tcp 的固定偏置把 TCP 目标换算成 flange 目标
    - 临时把 delegator 的 TCP 置零（让 IK/FK 的末端定义锁死为 flange）
    - 调用 robot_dual.ik 解 flange 目标
    - 恢复 TCP
    """
    if arm == "lft":
        robot_dual.use_lft()
    elif arm == "rgt":
        robot_dual.use_rgt()
    else:
        raise ValueError("arm must be 'lft' or 'rgt'")

    _set_delegator_tcp_for_arm(robot_dual, arm)

    d = getattr(robot_dual, "delegator", None)
    if d is None:
        d = getattr(robot_dual, "_delegator")

    p_ft = np.asarray(getattr(d, "_loc_tcp_pos", np.zeros(3)), dtype=float).copy()
    R_ft = np.asarray(getattr(d, "_loc_tcp_rotmat", np.eye(3)), dtype=float).copy()

    tgt_pos_tcp = np.asarray(tgt_pos_tcp, dtype=float)
    tgt_rot_tcp = np.asarray(tgt_rot_tcp, dtype=float)

    # flange 目标：T_base_flange = T_base_tcp * inv(T_flange_tcp)
    R_flg_tgt = tgt_rot_tcp @ R_ft.T
    p_flg_tgt = tgt_pos_tcp - R_flg_tgt @ p_ft

    # 临时置零 TCP：锁定 IK/FK 末端为 flange
    if hasattr(d, "_loc_tcp_pos"):
        d._loc_tcp_pos = np.zeros(3)
    if hasattr(d, "loc_tcp_pos"):
        d.loc_tcp_pos = np.zeros(3)
    if hasattr(d, "_loc_tcp_rotmat"):
        d._loc_tcp_rotmat = np.eye(3)
    if hasattr(d, "loc_tcp_rotmat"):
        d.loc_tcp_rotmat = np.eye(3)
    if hasattr(d, "_is_gl_tcp_delayed"):
        d._is_gl_tcp_delayed = True

    q = robot_dual.ik(
        tgt_pos=p_flg_tgt,
        tgt_rotmat=R_flg_tgt,
        seed_jnt_values=seed_jnt_values,
    )

    # 恢复 TCP
    if hasattr(d, "_loc_tcp_pos"):
        d._loc_tcp_pos = p_ft
    if hasattr(d, "loc_tcp_pos"):
        d.loc_tcp_pos = p_ft
    if hasattr(d, "_loc_tcp_rotmat"):
        d._loc_tcp_rotmat = R_ft
    if hasattr(d, "loc_tcp_rotmat"):
        d.loc_tcp_rotmat = R_ft
    if hasattr(d, "_is_gl_tcp_delayed"):
        d._is_gl_tcp_delayed = True

    return q


def build_cartesian_traj_for_arm(robot_dual: GP7_Dual,
                                 arm: str,
                                 q_start: np.ndarray,
                                 dp_world: np.ndarray,
                                 n_step: int):
    """
    为某一只臂生成：直线 TCP 轨迹(世界坐标系) -> IK -> q_list
    - arm: "lft" or "rgt"
    - q_start: 6-dof
    - dp_world: TCP displacement in WORLD frame (meters)
    """
    if arm == "lft":
        robot_dual.use_lft()
    elif arm == "rgt":
        robot_dual.use_rgt()
    else:
        raise ValueError("arm must be 'lft' or 'rgt'")

    # 非常重要：FK 取起点时也要保证 delegator 的 TCP 与当前 arm 一致。
    # 否则 p0/R0 可能是用“上一只臂的 TCP”算出来的，导致 step=0 也会 IK 失败。
    _set_delegator_tcp_for_arm(robot_dual, arm)

    # 1) 起点 FK：TCP 起点与姿态
    # p0是计算出的tcp相对于世界base坐标系的位置
    p0, R0 = robot_dual.fk(jnt_values=q_start)
    p0 = np.asarray(p0)
    R0 = np.asarray(R0)

    # 2) 直线 TCP 轨迹（姿态固定）
    s_list = np.linspace(0.0, 1.0, n_step)
    tcp_pos_list = [p0 + s * dp_world for s in s_list]

    # 3) IK 跟踪（上一帧作为 seed）
    q_list = []
    q_prev = q_start.copy()

    for i, p in enumerate(tcp_pos_list):
        # 注意：WRS 的某些 IK solver 默认按 flange 末端求解，且对 TCP 偏置不自洽。
        # 这里统一用“TCP目标 -> 换算成flange目标 -> 临时置零TCP -> IK -> 恢复TCP”的稳定路径。
        q = ik_tcp_via_flange(
            robot_dual=robot_dual,
            arm=arm,
            tgt_pos_tcp=p,
            tgt_rot_tcp=R0,
            seed_jnt_values=q_prev,
        )
        if q is None:
            raise RuntimeError(f"[IK FAILED] arm={arm}, step={i}, tgt_pos={p}")

        # FK 误差验证
        # 这里 fk() 输出 TCP（与 tgt_pos_tcp 同坐标系），应当与 p 对齐
        p_check, _ = robot_dual.fk(jnt_values=q)
        err = float(np.linalg.norm(np.asarray(p_check) - p))
        print(f"[{arm}] step {i}: FK error = {err:.6e} m")

        q_list.append(q)
        q_prev = q

    return tcp_pos_list, q_list


class DualGP7VizRunner:
    """
    统一管理：
    - 双臂 mesh 刷新
    - 双臂 TCP frame 刷新（跟着动）
    - sequence 播放（task 回调）
    """
    def __init__(self, robot: GP7_Dual, base, ax_length=0.12, show_tcp_frames=True):
        self.robot = robot
        self.base = base
        self.ax_length = ax_length
        self.show_tcp_frames = show_tcp_frames

        self.mesh_holder = {"lft": None, "rgt": None}
        self.tcp_frame_holder = {"lft": None, "rgt": None}

        self.sequence = []
        self.state = {"seg": 0, "i": 0}

    @staticmethod
    def _detach_if_exist(holder: dict, key: str):
        if holder.get(key, None) is not None:
            holder[key].detach()
            holder[key] = None

    def refresh_mesh(self):
        # lft
        self._detach_if_exist(self.mesh_holder, "lft")
        self.robot.use_lft()
        self.mesh_holder["lft"] = self.robot.gen_meshmodel()
        self.mesh_holder["lft"].attach_to(self.base)

        # rgt
        self._detach_if_exist(self.mesh_holder, "rgt")
        self.robot.use_rgt()
        self.mesh_holder["rgt"] = self.robot.gen_meshmodel()
        self.mesh_holder["rgt"].attach_to(self.base)

    def refresh_tcp_frames(self):
        if not self.show_tcp_frames:
            return

        # 左臂 TCP
        q_l = self.robot._lft_arm.get_jnt_values()
        tcp_pos_l, tcp_rot_l = self.robot._lft_arm.fk(jnt_values=q_l)

        self._detach_if_exist(self.tcp_frame_holder, "lft")
        self.tcp_frame_holder["lft"] = mgm.gen_frame(
            pos=np.asarray(tcp_pos_l),
            rotmat=np.asarray(tcp_rot_l),
            ax_length=self.ax_length
        )
        self.tcp_frame_holder["lft"].attach_to(self.base)

        # 右臂 TCP
        q_r = self.robot._rgt_arm.get_jnt_values()
        tcp_pos_r, tcp_rot_r = self.robot._rgt_arm.fk(jnt_values=q_r)

        self._detach_if_exist(self.tcp_frame_holder, "rgt")
        self.tcp_frame_holder["rgt"] = mgm.gen_frame(
            pos=np.asarray(tcp_pos_r),
            rotmat=np.asarray(tcp_rot_r),
            ax_length=self.ax_length
        )
        self.tcp_frame_holder["rgt"].attach_to(self.base)

    def refresh_all(self):
        self.refresh_mesh()
        self.refresh_tcp_frames()

    def set_sequence(self, sequence):
        """sequence: [(arm_str, traj_q_list), ...]"""
        self.sequence = sequence
        self.state = {"seg": 0, "i": 0}

    def step(self, task):
        if self.state["seg"] >= len(self.sequence):
            return task.done

        arm, traj = self.sequence[self.state["seg"]]

        if self.state["i"] >= len(traj):
            self.state["seg"] += 1
            self.state["i"] = 0
            return task.again

        if arm == "lft":
            self.robot.use_lft()
        else:
            self.robot.use_rgt()

        self.robot.goto_given_conf(traj[self.state["i"]])
        self.state["i"] += 1

        self.refresh_all()
        return task.again




def main():
    # 1) 创建一个3D仿真世界对象，相机在世界坐标系中的位置，摄像机看向的点。
    base = wd.World(cam_pos=[2.2, 0.2, 1.3], lookat_pos=[0.0, 0.0, 0.5])
    # 生成一个 坐标系模型（XYZ三轴），并加到世界里。
    mgm.gen_frame(ax_length=.5).attach_to(base)

    # 2) Robot
    robot = GP7_Dual(enable_cc=False)
    ensure_ik_ready(robot)

    # # 3) 环境（可选）
    # robot.load_environment_models(base)
    # robot.load_hairpin(base)

    # 4) 生成左右臂 Cartesian IK 轨迹
    tcp_lft, qtraj_lft = build_cartesian_traj_for_arm(
        robot_dual=robot, arm="lft",
        q_start=q_lft_start, dp_world=dp_lft, n_step=N_STEP
    )
    print("=== Left arm Cartesian IK finished ===")

    tcp_rgt, qtraj_rgt = build_cartesian_traj_for_arm(
        robot_dual=robot, arm="rgt",
        q_start=q_rgt_start, dp_world=dp_rgt, n_step=N_STEP
    )
    print("=== Right arm Cartesian IK finished ===")

    # 5) 可视化：TCP 轨迹点（球）
    if SHOW_TCP_PATH:
        for p in tcp_lft:
            mgm.gen_sphere(pos=p, radius=0.004).attach_to(base)
        for p in tcp_rgt:
            mgm.gen_sphere(pos=p, radius=0.004).attach_to(base)

    # 6) 初始化到起点
    robot.use_lft()
    robot.goto_given_conf(qtraj_lft[0])
    robot.use_rgt()
    robot.goto_given_conf(qtraj_rgt[0])

    # # 7) Runner：负责刷新 mesh + TCP frame + 播放
    runner = DualGP7VizRunner(robot, base, ax_length=AX_LEN, show_tcp_frames=SHOW_TCP_FRAMES)

    sequence = [
        ("lft", qtraj_lft),
        ("rgt", qtraj_rgt),
    ]
    runner.set_sequence(sequence)

    runner.refresh_all()  # 开场先画一次
    base.taskMgr.doMethodLater(0.01, runner.step, "dual_cartesian_seq_update")
    print("rgt tcp offset norm =", np.linalg.norm(robot._rgt_arm._loc_tcp_pos))
    base.run()


if __name__ == "__main__":
    main()
