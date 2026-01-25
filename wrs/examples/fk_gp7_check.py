import os
import sys
import numpy as np
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
from wrs.robot_sim.manipulators.gp7.gp7 import GP7

# --- 仅在你没有 pip install -e . 时需要 ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

def main():

    # 2) 实例化
    robot = GP7(enable_cc=False, use_mesh=True)  # enable_cc 是否开启碰撞检测看你需要

    # 3) 设定一组关节角（弧度）
    q = np.array([0.1, -0.5, 0.8, 0.0, 1.2, 0.0], dtype=float)

    # 4) 做 FK
    # 不同版本可能是 fk(jnt_values=...) 或 fk(...)
    tcp_pos, tcp_rot = robot.fk(jnt_values=q, update=True)


    # ========= 可视化部分：看这个姿态在仿真里长什么样 =========
    base = wd.World(cam_pos=[2.0, 0.0, 1.2],
                    lookat_pos=[0.0, 0.0, 0.4])

    # 世界坐标系三轴
    mgm.gen_frame().attach_to(base)

    # ========= 1) 姿态A：旋转前（通常用零位） =========
    robot0 = GP7(enable_cc=False, use_mesh=False)  # 先用 stick，清晰
    q0 = np.zeros(6)
    robot0.goto_given_conf(q0)

    stick0 = robot0.gen_stickmodel(toggle_flange_frame=True,
                                   toggle_jnt_frames=True)
    stick0.attach_to(base)

    # ========= 2) 姿态B：旋转后（你给定的 q） =========
    robot1 = GP7(enable_cc=False, use_mesh=True)  # 用 mesh 更直观
    robot1.goto_given_conf(q)

    mesh1 = robot1.gen_meshmodel()
    mesh1.attach_to(base)

    stick1 = robot1.gen_stickmodel(toggle_flange_frame=True,
                                   toggle_jnt_frames=False)  # 可选：避免太乱
    stick1.attach_to(base)

    # 打印两者的 TCP（用于数值对照）
    p0, R0 = robot0.fk(jnt_values=q0, update=False)
    p1, R1 = robot1.fk(jnt_values=q, update=False)
    print("=== BEFORE q0 ===")
    print("TCP pos =", p0)
    print("=== AFTER  q ===")
    print("TCP pos =", p1)

    # 打开窗口（会阻塞，直到你关掉窗口）
    base.run()

if __name__ == "__main__":
    main()
