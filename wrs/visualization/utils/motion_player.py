class MotionPlayer:
    def __init__(self,
                 base,
                 robot,
                 traj,
                 enable_ghost=False,
                 max_ghost=None):
        """
        base        : wd.World
        robot       : GP7 / ManipulatorInterface
        traj        : list[np.ndarray]  # joint trajectory
        enable_ghost: bool
        max_ghost   : int or None
        """
        self.base = base
        self.robot = robot
        self.traj = traj
        self.enable_ghost = enable_ghost
        self.max_ghost = max_ghost

        self._i = 0
        self._mesh = None
        self._ghost = []

    def _update(self, task):
        if self._i >= len(self.traj):
            return task.done

        self.robot.goto_given_conf(self.traj[self._i])

        if self.enable_ghost:
            m = self.robot.gen_meshmodel()
            m.attach_to(self.base)
            self._ghost.append(m)

            if self.max_ghost is not None and len(self._ghost) > self.max_ghost:
                old = self._ghost.pop(0)
                old.detach()
        else:
            if self._mesh is not None:
                self._mesh.detach()
            self._mesh = self.robot.gen_meshmodel()
            self._mesh.attach_to(self.base)

        self._i += 1
        return task.cont

    def play(self, task_name="motion_player"):
        self.base.taskMgr.add(self._update, task_name)
