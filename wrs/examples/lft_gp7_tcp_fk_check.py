import os
import sys
import numpy as np
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
from wrs.robot_sim.manipulators.gp7.gp7 import GP7
# from wrs.robot_con.yaskawa_gp7.gp7_encoder import GP7Encoder

# --- 仅在你没有 pip install -e . 时需要 ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

def ik_tcp_via_flange(robot, p_tcp_base, R_tcp_base, seed):
    # 0) 保存真实 flange->tcp
    p_ft = robot._loc_tcp_pos.copy()
    R_ft = robot._loc_tcp_rotmat.copy()

    # 1) 计算 flange 目标（用“真实TCP”做换算）
    R_flg_base = R_tcp_base @ R_ft.T
    p_flg_base = p_tcp_base - R_flg_base @ p_ft

    # 2) 临时把 TCP 置零：强制 IK/FK 都以 flange 为末端
    robot._loc_tcp_pos = np.zeros(3)
    robot._loc_tcp_rotmat = np.eye(3)
    robot._is_gl_tcp_delayed = True

    # 3) IK（现在 tgt_pos/tgt_rotmat 明确就是 flange 目标）
    q = robot.ik(tgt_pos=p_flg_base, tgt_rotmat=R_flg_base, seed_jnt_values=seed)

    # 4) 恢复真实 TCP
    robot._loc_tcp_pos = p_ft
    robot._loc_tcp_rotmat = R_ft
    robot._is_gl_tcp_delayed = True

    return q

def main():


    # ========= 可视化部分：看这个姿态在仿真里长什么样 =========
    base = wd.World(cam_pos=[2.0, 0.0, 1.2],
                    lookat_pos=[0.0, 0.0, 0.4])

    # # 世界坐标系三轴



    robot0 = GP7(enable_cc=False, use_mesh=True, pos=np.array([0.0, 0.0, 0.33]), ik_solver='n')  # 用 mesh 更直观
    q0 = np.zeros(6)
    robot0.goto_given_conf(q0)
    mesh1 = robot0.gen_meshmodel()
    mesh1.attach_to(base)
    stick1 = robot0.gen_stickmodel(toggle_flange_frame=True,
                                   toggle_jnt_frames=False)  # 可选：避免太乱
    stick1.attach_to(base)

    # 1) 先拿到你要对齐的 TCP 目标（你现在就是这么拿的）
    p_tcp_tgt, R_tcp_tgt = robot0.fk(jnt_values=q0, update=True)

    frame = mgm.gen_frame(
        pos=p_tcp_tgt,
        rotmat=R_tcp_tgt,
        ax_length=0.08,  # 坐标轴长度（自己调）
    )
    frame.attach_to(base)

    q_re = ik_tcp_via_flange(robot0, p_tcp_tgt, R_tcp_tgt, seed=q0)

    p_re, R_re = robot0.fk(jnt_values=q_re, update=True)
    print(p_tcp_tgt)
    print(p_re)
    print("pos_err_tcp =", np.linalg.norm(p_re - p_tcp_tgt))

    frame = mgm.gen_frame(
        pos=p_re,
        rotmat=R_re,
        ax_length=0.1,  # 坐标轴长度（自己调）
    )

    frame.attach_to(base)
    ###########################################

    # # print("=== BEFORE q0 ===")
    # print("TCP pos =", p0)
    # print("TCP rot =", R0)
    # print("=== AFTER  q ===")
    # print("TCP pos =", p1*1000)

    # 打开窗口（会阻塞，直到你关掉窗口）
    base.run()

if __name__ == "__main__":
    main()
