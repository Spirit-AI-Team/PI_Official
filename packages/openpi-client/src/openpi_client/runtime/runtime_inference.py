import threading
import time
import logging
import pyarrow as pa
import tree
from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber
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

        self._in_episode = True
        self._episode_steps = 0
        self.update = False
        self._inference_thread = None
        self._exec_thread = None
        self.observation = None
        self.old_actions = None
        # self._stop_event = threading.Event()  # 控制线程停止
        # self._action_updated = threading.Event()  # 控制 `actions` 何时更新
        self.cur_step = 0
        self.actions = None  # Placeholder for actions

    def run_inference_loop(self) -> None:
        """Runs inference in a loop every 0.2s in a separate thread."""
        # logging.info("Starting inference thread...")
        self._run_inference()
        self.update = True
        while True:
            # logging.info("Running inference...")
            if self.cur_step == 8:
                start_time = time.time() 
                self._run_inference()
                end_time = time.time()
                inference_time = end_time - start_time
                print(inference_time)
            if self.cur_step == 10:
                # time.sleep(0.2-inference_time)
                self.update = True
                
            # self._action_updated.set()  # 触发执行线程更新
            if self._in_episode == False:
                # logging.info("Episode execution completed.")
                break
              # 每 0.2s 运行一次推理

    def run_exec_loop(self) -> None:
        """Runs execution in a loop in a separate thread, only executing when actions are updated."""
        logging.info("Starting execution thread...")
        while True:  
                if self.update:
                    self.update = False
                    self._run_exec()
                    

    def start_threads(self) -> None:
        """Starts both inference and execution in separate threads."""
        # self._stop_event.clear()
        self._environment.reset()
        self._agent.reset()
        self.observation = self._environment.get_observation()
        thread = threading.Thread(target=self.run_exec_loop, daemon=True)
        thread.start()

        self.run_inference_loop()


    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False

    def _run_inference(self) -> None:
        """Runs a single inference cycle."""
        # logging.info("Starting inference...")
        self._in_episode = True
        self._episode_steps = 0
        self.actions = self._agent.get_action(self.observation)  # 更新 actions
        

    def _run_exec(self) -> None:
        """Executes actions continuously only when actions are updated."""
        # logging.info("Executing actions...")
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()
        self.cur_step = 3

        for _ in range(7):
            if self.old_actions is None:
                action = tree.map_structure(lambda x: x[self.cur_step-2, ...], self.actions)
            else:
                action = tree.map_structure(lambda x: x[self.cur_step, ...], self.actions)
            
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
            self.observation = self._environment.get_observation()
            self.cur_step += 1
            if self.update or self.cur_step==10:
                print(f"step is {self.cur_step}")
                break
        self.cur_step = 0
        self.old_actions = 0
        # self._environment.apply_action(action)
        # self._episode_steps += 1
        # if self._environment.is_episode_complete() or (
        #     self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        # ):
        #     self.mark_episode_complete()
        
