import numpy as np

from wrs.robot_sim.manipulators.gp7.gp7 import GP7
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
from wrs.visualization.utils.motion_player import MotionPlayer


# ========= 可调参数 =========
N_STEP = 60
ENABLE_GHOST = False       # True: 残影, False: 单一模型
SHOW_TCP_PATH = True       # 是否画 TCP 轨迹点

# 起始关节角（你已知 IK 可解的姿态）
q_start = np.array([0.5, -0.5, 0.8, 0.0, 1.2, 0.0])

# TCP 位移（m）：例如沿 Z 方向向下 50mm
dp = np.array([0.0, 0.0, -0.15])
# ============================


def main():
    # 1) World
    base = wd.World(cam_pos=[2.2, 0.2, 1.3],
                    lookat_pos=[0.0, 0.0, 0.5])
    mgm.gen_frame().attach_to(base)

    # 2) Robot
    robot = GP7(enable_cc=False, use_mesh=True)
    robot.jlc.finalize(ik_solver='n')
    print("IK solver:", robot.jlc._ik_solver)

    # 3) 起点 FK：得到 TCP 起点与姿态
    p0, R0 = robot.fk(jnt_values=q_start, update=False)
    p0 = np.asarray(p0)
    R0 = np.asarray(R0)


    # 4) Cartesian 直线轨迹（位置插值，姿态固定）
    s_list = np.linspace(0.0, 1.0, N_STEP)
    tcp_pos_list = [p0 + s * dp for s in s_list]

    # 5) IK 跟踪（上一帧解作为 seed）
    q_list = []
    q_prev = q_start.copy()

    for i, p in enumerate(tcp_pos_list):
        q = robot.ik(
            tgt_pos=p,
            tgt_rotmat=R0,
            seed_jnt_values=q_prev
        )
        if q is None:
            print(f"[IK FAILED] step={i}, tgt_pos={p}")
            return

        # 可选：数值验证 FK
        p_check, _ = robot.fk(jnt_values=q, update=False)
        err = np.linalg.norm(p_check - p)
        print(f"step {i}: FK error = {err:.6e} m")

        q_list.append(q)
        q_prev = q

    print("=== Cartesian motion IK finished successfully ===")

    # 6) 可视化初始化
    holder = {"mesh": None, "ghost": []}

    robot.goto_given_conf(q_list[0])
    holder["mesh"] = robot.gen_meshmodel()
    holder["mesh"].attach_to(base)

    # 7) 画 TCP 轨迹点
    if SHOW_TCP_PATH:
        for p in tcp_pos_list:
            mgm.gen_sphere(pos=p, radius=0.005).attach_to(base)

    state = {"i": 0}

    player = MotionPlayer(
        base=base,
        robot=robot,
        traj=q_list,
        enable_ghost=True,
        max_ghost=300
    )
    player.play()

    base.run()


if __name__ == "__main__":
    main()
