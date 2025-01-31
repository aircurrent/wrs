import os
import time
import math
import basis
import numpy as np
import modeling.geometricmodel as gm
import modeling.collisionmodel as cm
import robotsim.robots.xarm7_shuidi_mobile.xarm7_shuidi_mobile as xav


if __name__ == '__main__':
    import visualization.panda.world as wd
    import motion.probabilistic.rrt_connect as rrtc
    import visualization.panda.rpc.rviz_client as rv_client

    # # local code
    global_frame = gm.gen_frame()
    # define robot and robot anime info
    robot_instance = xav.XArm7YunjiMobile()
    robot_meshmodel_parameters = [None,  # tcp_jntid
                                  None,  # tcp_loc_pos
                                  None,  # tcp_loc_rotmat
                                  False,  # toggle_tcpcs
                                  False,  # toggle_jntscs
                                  [0, .7, 0, .3],  # rgba
                                  'auto']  # name
    # define object and object anime info
    objfile = os.path.join(basis.__path__[0], 'objects', 'bunnysim.stl')
    obj = cm.CollisionModel(objfile)
    obj_parameters = [[.3, .2, .1, 1]]  # rgba
    obj_path = [[np.array([.85, 0, .17]), np.eye(3)]]  # [pos, rotmat]
    obj.set_pos(np.array([.85, 0, .17]))
    obj.set_rgba([.5, .7, .3, .1])

    # remote code
    rvc = rv_client.RVizClient(host="localhost:182001")
    rvc.reset()
    rvc.load_common_definition(__file__, line_ids = range(1,8))
    rvc.change_campos_and_lookatpos(np.array([5,0,2]), np.array([0,0,.5]))
    # copy to remote
    rmt_global_frame = rvc.showmodel_to_remote(global_frame)
    rmt_bunny = rvc.showmodel_to_remote(obj)
    rmt_robot_instance = rvc.copy_to_remote(robot_instance)
    # rvc.show_stationary_obj(rmt_obj)
    robot_jlc_name = 'arm'
    robot_instance.fk(np.array([0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0]), component_name=robot_jlc_name)
    rrtc_planner = rrtc.RRTConnect(robot_instance)
    path = rrtc_planner.plan(start_conf=np.array([0, math.pi * 2 / 3, 0, math.pi, 0, -math.pi / 6, 0]),
                             goal_conf=np.array([math.pi / 3, math.pi * 1 / 3, 0, math.pi / 2, 0, math.pi / 6, 0]),
                             obstacle_list=[obj],
                             ext_dist=.1,
                             rand_rate=70,
                             maxtime=300,
                             component_name=robot_jlc_name)
    import copy
    rmt_anime_robotinfo = rvc.add_anime_robot(rmt_robot_instance=rmt_robot_instance,
                                              loc_robot_jlc_name=robot_jlc_name,
                                              loc_robot_meshmodel_parameters=robot_meshmodel_parameters,
                                              loc_robot_motion_path=path)
    # rmt_robot_meshmodel = rvc.add_stationary_robot(rmt_robot_instance=rmt_robot_instance,
    #                                                loc_robot_instance=robot_instance)
    time.sleep(1)
    # draw sequence, problem: cannot work together with anime? (lost poses) -> cannot use the same remote instance
    rmt_robot_mesh_list = []
    newpath = copy.deepcopy(path)
    rmt_robot_instance2 = rvc.copy_to_remote(robot_instance)
    for pose in newpath:
        robot_instance.fk(pose, component_name='arm')
        # rmt_robot_mesh_list.append(rvc.showmodel_to_remote(robot_instance.gen_meshmodel()))
        rmt_robot_mesh_list.append(rvc.add_stationary_robot(rmt_robot_instance2, robot_instance))
        # time.sleep(.1)
    # rvc.delete_anime_robot(rmt_anime_robotinfo)
    # rvc.delete_stationary_robot(rmt_robot_meshmodel)
    # robot_instance.fk(path[-1], jlc_name=robot_jlc_name)
    # rmt_robot_meshmodel = rvc.add_stationary_robot(rmt_robot_instance='robot_instance', loc_robot_instance=robot_instance)
    # obj.set_pos(obj.get_pos()+np.array([0,.1,0]))
    # obj.set_rgba([1,0,0,1])
    # rvc.update_remote(rmt_bunny, obj)