import threading
import time
import logging
import pyarrow as pa
import tree
from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber
import queue
import multiprocessing
import asyncio
class Runtime:
    """The core module orchestrating interactions between key components of the system."""

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps
        self._space_step = 0
        self._in_episode = True
        self._episode_steps = 0
        self.first = True
        self.action_event = threading.Event()
        self.inference_event = threading.Event()
        self.action_queue = queue.Queue()
        self.cur_step = 0
            

    def start_threads(self) -> None:
        """Starts both inference and execution in separate threads."""
        # self._stop_event.clear()
        self._environment.reset()
        self._agent.reset()
        self.observation = self._environment.get_observation()
        thread = threading.Thread(target=self.run_exec_loop, daemon=True)
        thread2 = threading.Thread(target=self.run_inference_loop, daemon=True)
        thread.start()
        thread2.start()


    def run_inference_loop(self) -> None:
        """Runs inference in a loop every 0.2s in a separate thread."""
        # logging.info("Starting inference thread...")
        self._run_inference()
        self.action_event.set()
        while True:
            self.inference_event.wait()
            self.inference_event.clear()
            # start_time = time.time() 
            self.action_queue.join()
            self.action_event.set()
            self._run_inference()
            end_time = time.time()
            # inference_time = end_time - start_time
            # print(inference_time)


    def run_exec_loop(self) -> None:
        """Runs execution in a loop in a separate thread, only executing when actions are updated."""
        logging.info("Starting execution thread...")
        while True:  
                self.action_event.wait()
                
                self._run_exec()
                self.action_event.clear()

    def _run_inference(self) -> None:
        """Runs a single inference cycle."""
        actions = self._agent.get_action(self.observation)  # 更新 actions
        self.action_queue.put(actions)

        

    def _run_exec(self) -> None:
        """Executes actions continuously only when actions are updated."""
        # logging.info("Executing actions...")
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()
        self.cur_step = self._space_step
        actions = self.action_queue.get()
        for _ in range(self._max_hz-self._space_step):
            if self.first is True:
                action = tree.map_structure(lambda x: x[self.cur_step-self._space_step, ...], actions)
            else:
                action = tree.map_structure(lambda x: x[self.cur_step, ...], actions)
            
                # self.cur_step = 0
            # print(action)
            
            self._environment.apply_action(action)

            # Sleep to maintain the desired frame rate
            now = time.time()
            dt = now - last_step_time
            # print(dt)
            if dt < step_time:
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now
            
            self.cur_step += 1
            if self.cur_step==self._max_hz-self._space_step:
                self.observation = self._environment.get_observation()
                self.inference_event.set()
        self.first = False
        self.action_queue.task_done()

        
    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False