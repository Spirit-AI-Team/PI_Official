from typing import Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy
import pdb
import time
import pyarrow as pa
class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """
    default_action = [ 0.31771099,  0.00442588, -0.39217242,  0.47255166, -0.12761308,
       -2.20674003, -0.96154621, -0.30637348, -0.01802089, -0.44487334,
        0.14385187,  0.19578579, -1.27791037, -1.05855188]

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int, frequency: int):
        self._policy = policy

        self._action_horizon = action_horizon
        self._frequency = frequency
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None
        self.time_counter = 0
        self.time_step = 0

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006

        if obs['fake_action']:
            fake_results = {'actions': np.array(ActionChunkBroker.default_action)}
            return fake_results
        
        if self._last_results is None or obs['flush_buffer']:
            # pdb.set_trace()
            tic = time.time()
            self._last_results = self._policy.infer(obs)
            # pdb.set_trace()
            self._last_results = self.postprocess(self._last_results)
            toc = time.time()
            self.time_counter += toc - tic
            self.time_step += 1
            # pdb.set_trace() 
            self._cur_step = 0

        results = tree.map_structure(lambda x: x[self._cur_step, ...], self._last_results)
        self._cur_step += 1
        if self._cur_step >= self._frequency:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0

    def postprocess(self, raw_results) -> dict:
        # import pdb; pdb.set_trace() 
        raw_actions = raw_results['actions']
        actions = raw_actions[:self._action_horizon, ...]
        _, act_dim = actions.shape # ah, 14
        new_actions = np.zeros((act_dim, self._frequency))#14, freq

        for i in range(act_dim): 
            new_actions[i] = np.interp(np.linspace(0, 1, self._frequency), np.linspace(0,1,self._action_horizon), actions.T[i])
            # if i == 6 or i==13:
            #     new_actions[i] -= 1
        new_results = {'actions':new_actions.T}
        return new_results
    
class ActionChunkBroker2(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int, frequency: int):
        self._policy = policy

        self._action_horizon = action_horizon
        self._frequency = frequency
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None
        self.time_counter = 0
        self.time_step = 0

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006


        self._last_results = self._policy.infer(obs)
        # pdb.set_trace()
        self._last_results = self.postprocess(self._last_results)
        results = self._last_results
        # results = tree.map_structure(lambda x: x[self._cur_step, ...], self._last_results)


        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0

    def postprocess(self, raw_results) -> dict:

        raw_actions = raw_results['actions']
        actions = raw_actions[:self._action_horizon, ...]
        _, act_dim = actions.shape # ah, 14
        new_actions = np.zeros((act_dim, self._frequency))#14, freq

        for i in range(act_dim): 
            new_actions[i] = np.interp(np.linspace(0, 1, self._frequency), np.linspace(0,1,self._action_horizon), actions.T[i])
            if i == 6 or i == 13:
                new_actions[i] -= 1
        # arrow_actions = pa.array(new_actions.T)
        new_results = {'actions': new_actions.T}
        return new_results