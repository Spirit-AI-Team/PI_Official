import logging
import threading
import time

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber
import numpy as np
import pyarrow as pa
from dora import Node
import tree

class Runtime:
    """The core module orchestrating interactions between key components of the system."""

    def __init__(
        self,
        environment: _environment.Environment,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
    ) -> None:
        self._environment = environment
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._in_episode = False
        self._episode_steps = 0
        self.cur_step = 0

    def run(self) -> None:
        """Runs the runtime loop continuously until stop() is called or the environment is done."""
        for _ in range(self._num_episodes):
            self._run_episode()

    def run_in_new_thread(self) -> threading.Thread:
        """Runs the runtime loop in a new thread."""
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread

    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False

    def _run_episode(self) -> None:
        """Runs a single episode."""
        logging.info("Starting episode...")
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()
        pa.array([]) 
        node = Node()
        first_run = True
        for event in node:
            event_type = event["type"]

            if event_type == "INPUT":
                event_id = event["id"]

                if event_id == "actions":
                    actions = event["value"]
                    np_actions = actions['actions'].to_numpy()
                    new_actions =  {'actions': np_actions}
                    if first_run:
                        self.cur_step = 0
                    else:
                        self.cur_step = 2
                    for _ in range(20):
                        action = tree.map_structure(lambda x: x[self.cur_step, ...], new_actions)
                        self.cur_step+=1
                        self._environment.apply_action(action)
                        self._episode_steps += 1

                        # Sleep to maintain the desired frame rate
                        now = time.time()
                        dt = now - last_step_time
                        if dt < step_time:
                            time.sleep(step_time - dt)
                            last_step_time = time.time()
                        else:
                            last_step_time = now


                if self._environment.is_episode_complete() or (
                    self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
                ):
                    self.mark_episode_complete()
                logging.info("Episode completed.")


