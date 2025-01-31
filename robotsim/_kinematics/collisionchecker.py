import basis.data_adapter as da
import modeling.modelcollection as mc
from panda3d.core import NodePath, CollisionTraverser, CollisionHandlerQueue, BitMask32


class CollisionChecker(object):
    """
    A fast collision checker that allows maximum 32 collision pairs
    author: weiwei
    date: 20201214osaka
    """

    def __init__(self, name="auto"):
        self.ctrav = CollisionTraverser()
        self.chan = CollisionHandlerQueue()
        self.np = NodePath(name)
        self.is_nprendered = False
        self.nbitmask = 0 # capacity 1-30
        self._bitmask_ext = BitMask32(2**31) # 31 is prepared for cd with external objects
        self.all_cdelements = [] # a list of cdlnks or cdobjs for quick accessing the cd elements (cdlnks/cdobjs)

    def add_cdlnks(self, jlcobj, lnk_idlist):
        """
        The collision node of the given links will be attached to self.np, but their collision bitmask will be cleared
        When the a robot is treated as an obstacle by another robot, the IntoCollideMask of its all_cdelements will be
        set to BitMask32(2**31), so that the other robot can compare its active_cdelements with the all_cdelements.
        :param jlcobj:
        :param lnk_idlist:
        :return:
        author: weiwei
        date: 20201216toyonaka
        """
        for id in lnk_idlist:
            if jlcobj.lnks[id]['cdprimit_childid'] == -1:  # first time add
                cdnp = jlcobj.lnks[id]['collisionmodel'].copy_cdnp_to(self.np, clearmask=True)
                self.ctrav.addCollider(cdnp, self.chan)
                self.all_cdelements.append(jlcobj.lnks[id])
                jlcobj.lnks[id]['cdprimit_childid'] = len(self.all_cdelements)-1
            else:
                raise ValueError("The link is already added!")

    def set_active_cdlnks(self, activelist):
        """
        The specified collision links will be used for collision detection with external obstacles
        :param activelist: essentially a from list like [jlchain.lnk0, jlchain.lnk1...]
                           the correspondent tolist will be set online in cd functions
                           TODO use all elements in self.all_cdnlks if None
        :return:
        author: weiwei
        date: 20201216toyonaka
        """
        for cdlnk in activelist:
            if cdlnk['cdprimit_childid'] == -1:
                raise ValueError("The link needs to be added to collider using the add_cdlnks function first!")
            cdnp = self.np.getChild(cdlnk['cdprimit_childid'])
            cdnp.node().setFromCollideMask(self._bitmask_ext)

    def set_cdpair(self, fromlist, intolist):
        """
        The given collision pair will be used for self collision detection
        :param fromlist: [[bool, cdprimit_cache], ...]
        :param intolist: [[bool, cdprimit_cache], ...]
        :return:
        author: weiwei
        date: 20201215
        """
        if self.nbitmask >= 30:
            raise ValueError("Too many collision pairs! Maximum: 29")
        cdmask = BitMask32(2 ** self.nbitmask)
        for cdlnk in fromlist:
            if cdlnk['cdprimit_childid'] == -1:
                raise ValueError("The link needs to be added to collider using the addjlcobj function first!")
            cdnp = self.np.getChild(cdlnk['cdprimit_childid'])
            current_from_cdmask = cdnp.node().getFromCollideMask()
            new_from_cdmask = current_from_cdmask | cdmask
            cdnp.node().setFromCollideMask(new_from_cdmask)
        for cdlnk in intolist:
            if cdlnk['cdprimit_childid'] == -1:
                raise ValueError("The link needs to be added to collider using the addjlcobj function first!")
            cdnp = self.np.getChild(cdlnk['cdprimit_childid'])
            current_into_cdmask = cdnp.node().getIntoCollideMask()
            new_into_cdmask = current_into_cdmask | cdmask
            cdnp.node().setIntoCollideMask(new_into_cdmask)
        self.nbitmask += 1

    def add_cdobj(self, objcm, rel_pos, rel_rotmat, intolist):
        """
        :return: cdobj_info, a dictionary that mimics a joint link; Besides that, there is an additional 'intolist'
                 key to hold intolist to easily toggle off the bitmasks.
        """
        cdobj_info = {}
        cdobj_info['collisionmodel'] = objcm # for reversed lookup
        cdobj_info['gl_pos'] = objcm.get_pos()
        cdobj_info['gl_rotmat'] = objcm.get_rotmat()
        cdobj_info['rel_pos'] = rel_pos
        cdobj_info['rel_rotmat'] = rel_rotmat
        cdobj_info['intolist'] = intolist
        cdnp = objcm.copy_cdnp_to(self.np, clearmask=True)
        cdnp.node().setFromCollideMask(self._bitmask_ext) # set active
        self.ctrav.addCollider(cdnp, self.chan)
        self.all_cdelements.append(cdobj_info)
        cdobj_info['cdprimit_childid'] = len(self.all_cdelements)-1
        self.set_cdpair([cdobj_info], intolist)
        return cdobj_info

    def delete_cdobj(self, cdobj_info):
        """
        :param cdobj_info: an lnk-like object generated by self.add_objinhnd
        :param objcm:
        :return:
        """
        self.all_cdelements.remove(cdobj_info)
        cdnp = self.np.getChild(cdobj_info['cdprimit_childid'])
        self.ctrav.removeCollider(cdnp)
        for cdlnk in cdobj_info['intolist']:
            cdnp = self.np.getChild(cdlnk['cdprimit_childid'])
            current_into_cdmask = cdnp.node().getIntoCollideMask()
            new_into_cdmask = current_into_cdmask & ~cdnp.node().getFromCollideMask()
            cdnp.node().setIntoCollideMask(new_into_cdmask)

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        """
        :param obstacle_list: staticgeometricmodel
        :param otherrobot_list:
        :return:
        """
        for cdelement in self.all_cdelements:
            pos = cdelement['gl_pos']
            rotmat = cdelement['gl_rotmat']
            cdnp = self.np.getChild(cdelement['cdprimit_childid'])
            cdnp.setMat(da.npv3mat3_to_pdmat4(pos, rotmat))
            # print(da.npv3mat3_to_pdmat4(pos, rotmat))
            # print("From", cdnp.node().getFromCollideMask())
            # print("Into", cdnp.node().getIntoCollideMask())
        # print("xxxx colliders xxxx")
        # for collider in self.ctrav.getColliders():
        #     print(collider.getMat())
        #     print("From", collider.node().getFromCollideMask())
        #     print("Into", collider.node().getIntoCollideMask())
        # attach obstacles
        for obstacle in obstacle_list:
            obstacle.objpdnp.reparentTo(self.np)
        # attach other robots
        for robot in otherrobot_list:
            for cdnp in robot.cc.np.getChildren():
                current_into_cdmask = cdnp.node().getIntoCollideMask()
                new_into_cdmask = current_into_cdmask | self._bitmask_ext
                cdnp.node().setIntoCollideMask(new_into_cdmask)
            robot.cc.np.reparentTo(self.np)
        # collision check
        self.ctrav.traverse(self.np)
        # clear obstacles
        for obstacle in obstacle_list:
            obstacle.objpdnp.detachNode()
        # clear other robots
        for robot in otherrobot_list:
            for cdnp in robot.cc.np.getChildren():
                current_into_cdmask = cdnp.node().getIntoCollideMask()
                new_into_cdmask = current_into_cdmask & ~self._bitmask_ext
                cdnp.node().setIntoCollideMask(new_into_cdmask)
            if robot.cc.is_nprendered:
                robot.cc.np.reparentTo(base.render)
            else:
                robot.cc.np.detachNode()
        if self.chan.getNumEntries() > 0:
            return True
        else:
            return False

    def show_cdprimit(self):
        # print("call show_cdprimit")
        self.np.reparentTo(base.render)
        self.is_nprendered = True # TODO possible bug for copy
        for cdelement in self.all_cdelements:
            pos = cdelement['gl_pos']
            rotmat = cdelement['gl_rotmat']
            cdnp = self.np.getChild(cdelement['cdprimit_childid'])
            cdnp.setMat(da.npv3mat3_to_pdmat4(pos, rotmat))
            # print(cdnp.getMat())
            cdnp.show()

    def unshow_cdprimit(self):
        for child in self.np.getChildren():
            child.hide()
        self.np.detachNode()
        self.is_nprendered = False

    def disable(self):
        """
        clear pairs and nodepath
        :return:
        """
        for cdelement in self.all_cdelements:
            cdelement['cdprimit_childid'] = -1
        self.all_cdelements = []
        for child in self.np.getChildren():
            child.removeNode()
        self.nbitmask = 0