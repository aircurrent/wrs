import numpy as np
from wrs.robot_sim.manipulators.gp7.gp7 import GP7
import wrs.visualization.panda.world as wd
import wrs.modeling.geometric_model as mgm
from wrs.visualization.utils.motion_player import MotionPlayer

# ========= 只改这一行 True/False 就能切换 =========
ENABLE_GHOST = False   # True: 残影模式, False: 不残影
# 可选：限制残影数量（避免太多卡）
MAX_GHOST = 60
# ===================================================

def main():
    base = wd.World(cam_pos=[2.2, 0.2, 1.3],
                    lookat_pos=[0.0, 0.0, 0.5])
    mgm.gen_frame().attach_to(base)

    robot = GP7(enable_cc=False, use_mesh=True)

    q0 = np.zeros(6)
    q1 = np.array([0.5, -0.5, 0.8, 0.0, 1.2, 0.0])

    n_step = 120
    traj = [q0 + s*(q1-q0) for s in np.linspace(0, 1, n_step)]

    state = {"i": 0}
    holder = {"mesh": None, "ghost": []}

    # 初始姿态
    robot.goto_given_conf(q0)
    holder["mesh"] = robot.gen_meshmodel()
    holder["mesh"].attach_to(base)

    player = MotionPlayer(
        base=base,
        robot=robot,
        traj=traj,
        enable_ghost=True,
        max_ghost=30
    )
    player.play()

    base.run()

if __name__ == "__main__":
    main()
