"""
Created on Jun 01, 2024
@author: Heng-Sheng (Hanson) Chang
"""

from elastica import CosseratRod
from elastica.callback_functions import CallBackBaseClass


class BasicCallBackBaseClass(CallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        CallBackBaseClass.__init__(self)
        self.every = step_skip
        self.callback_params = callback_params

    def make_callback(self, system, time, current_step: int):
        if current_step % self.every == 0:
            self.save_params(system, time)

    def save_params(
        self,
    ):
        return NotImplementedError


class RodCallBack(BasicCallBackBaseClass):
    def __init__(self, step_skip: int, callback_params: dict):
        BasicCallBackBaseClass.__init__(self, step_skip, callback_params)

    def save_params(self, system: CosseratRod, time: float):
        self.callback_params["time"].append(time)
        self.callback_params["radius"].append(system.radius.copy())
        self.callback_params["dilatation"].append(system.dilatation.copy())
        self.callback_params["voronoi_dilatation"].append(
            system.voronoi_dilatation.copy()
        )
        self.callback_params["position"].append(
            system.position_collection.copy()
        )
        self.callback_params["director"].append(
            system.director_collection.copy()
        )
        self.callback_params["velocity"].append(
            system.velocity_collection.copy()
        )
        self.callback_params["omega"].append(system.omega_collection.copy())
        self.callback_params["sigma"].append(system.sigma.copy())
        self.callback_params["kappa"].append(system.kappa.copy())
