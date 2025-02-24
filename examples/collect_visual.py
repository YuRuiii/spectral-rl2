import os
import torch
import hydra
import numpy as np
from dm_env import specs
from omegaconf import DictConfig, OmegaConf
from tqdm import trange

from spectralrl.algo.visual import DiffSR_DrQv2, DrQv2, MuLVRep_DrQv2
from spectralrl.buffer.visual import VisualReplayBuffer
# from spectralrl.utils.logger import TensorboardLogger
from UtilsRL.logger import CompositeLogger
from spectralrl.utils.utils import set_device, set_seed_everywhere
from spectralrl.utils.video import VideoRecorder

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'


class Collector:
    def __init__(self, cfg):
        self.cfg = cfg

        # self.logger = TensorboardLogger(
        #     "/".join([cfg.log_dir, cfg.algo.cls, cfg.name, cfg.task]),
        #     "_".join(["seed"+str(cfg.seed), cfg.name]),
        #     activate=not cfg.debug
        # )
        
        # setup logger, seed and device
        self.logger = CompositeLogger(
            log_dir="/".join([cfg.log_dir, cfg.algo.cls, cfg.name, cfg.task]),
            name="_".join(["seed"+str(cfg.seed), cfg.name]),
            logger_config={
                "TensorboardLogger": {}, 
            }, 
            activate=not cfg.debug
        )
        
        self.recorder = VideoRecorder(
            self.logger.output_dir if cfg.save_video else None
        )
        self.seed = set_seed_everywhere(cfg.seed)
        self.device = set_device(cfg.device)

        self.domain, self.task = cfg.task.split("_")
        if self.domain == "metaworld":
            import spectralrl.env.metaworld_env as env
        elif self.domain == "dmc":
            import spectralrl.env.dmc_env as env
        else:
            raise NotImplementedError(f"Unrecognized domain: {self.domain}.")
        self.train_env = env.make(self.task, cfg.frame_stack, cfg.action_repeat, cfg.env_seed)
        self.eval_env = env.make(self.task, cfg.frame_stack, cfg.action_repeat, cfg.env_seed)

        # create buffer
        data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1, ), np.float32, "reward"),
            specs.Array((1, ), np.float32, "discount")
        )
        self.replay_buffer = VisualReplayBuffer(
            buffer_size=cfg.buffer_size,
            batch_size=cfg.batch_size,
            nstep=cfg.nstep,
            discount=cfg.discount,
            frame_stack=cfg.frame_stack,
            data_specs=data_specs
        )

        algo_cls = {
            "drqv2": DrQv2,
            "diffsr_drqv2": DiffSR_DrQv2,
            "mulvrep_drqv2": MuLVRep_DrQv2
        }.get(cfg.algo.cls)
        self.agent = algo_cls(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            cfg.algo,
            self.device
        )
        
        # load pretrained model
        if algo_cls == DrQv2:
            path = cfg.algo.pretrained_path
            self.agent.actor.load_state_dict(torch.load(f"{path}/actor.pt"))
            self.agent.critic.load_state_dict(torch.load(f"{path}/critic.pt"))
            self.agent.encoder.load_state_dict(torch.load(f"{path}/encoder.pt"))
        else:
            raise NotImplementedError
        

        self.global_step = 0
        self.global_episode = 0

        self.best_return = 0
        self.best_success = 0

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def collect(self):
        eval_metrics = self.evaluate()
        self.logger.log_scalars("eval", eval_metrics, step=self.global_frame)
        self.logger.info(eval_metrics)
        
    def save(
            self, 
            observation_list,
            action_list,
            reward_list,
            is_terminal_list,
            is_success_list,
            is_first_list,
            is_last_list,
            eval_episode
        ):
        is_success = 1 in is_success_list
        path = f"expert_{self.cfg.algo.cls}_{self.cfg.task}/{eval_episode}_success{is_success}_length{len(observation_list)}.npz"
        
        # turn list to numpy array
        np.savez(
            path,
            observation=np.array(observation_list, dtype=np.uint8),
            action=np.array(action_list, dtype=np.float32),
            reward=np.array(reward_list, dtype=np.float32),
            is_terminal=np.array(is_terminal_list, dtype=bool),
            is_success=np.array(is_success_list, dtype=bool),
            is_first=np.array(is_first_list, dtype=bool),
            is_last=np.array(is_last_list, dtype=bool)
        )
        
    def evaluate(self):
        self.agent.train(False)
        all_lengths = []
        all_returns = []
        all_success = []
        for i_episode in range(self.cfg.eval_episode):
            time_step = self.eval_env.reset()
            length = ret = success = 0
            self.recorder.init(self.eval_env, enabled=(i_episode==0))
            
            observation_list = []
            action_list = []
            reward_list = []
            is_terminal_list = []
            is_success_list = []
            is_first_list = []
            is_last_list = []
            
            while not time_step.last():
                action = self.agent.select_action(time_step.observation, self.global_step, deterministic=True)
                time_step = self.eval_env.step(action)
                self.recorder.record(self.eval_env)
                ret += time_step.reward
                length += 1
                if hasattr(time_step, "success"):
                    success += float(time_step.success)
                    
                # save to buffer
                observation_list.append(time_step.observation)
                action_list.append(action)
                reward_list.append(time_step.reward)
                is_terminal_list.append(time_step.last() or time_step.success)
                is_success_list.append(time_step.success)
                is_first_list.append(len(observation_list) == 1)
                is_last_list.append(time_step.last())
                
                if time_step.last():
                    self.save(
                        observation_list,
                        action_list,
                        reward_list,
                        is_terminal_list,
                        is_success_list,
                        is_first_list,
                        is_last_list,
                        self.cfg.eval_episode
                    )
                
                
            self.recorder.save(f"eval_{self.global_frame}.mp4")
            all_lengths.append(length)
            all_returns.append(ret)
            all_success.append(float(success>=1.0))
        all_lengths = np.asarray(all_lengths)
        all_returns = np.asarray(all_returns)
        all_success = np.asarray(all_success, dtype=np.float32)
        metrics = {
            "return_mean": all_returns.mean(),
            "return_std": all_returns.std(),
            "length_mean": all_lengths.mean(),
            "success_mean": all_success.mean()
        }

        # agent evaluate if needed
        if self.global_frame != 0: # make sure there is sample
            agent_metrics, reconstruction = self.agent.evaluate(self.replay_buffer)
            metrics.update(agent_metrics)
            if reconstruction is not None:
                self.logger.log_image("info/reconstruction", reconstruction, step=self.global_frame)
        self.agent.train(True)
        return metrics

@hydra.main(version_base=None, config_path="./config/visual", config_name="config")
def main(cfg: DictConfig) -> None:
    c = Collector(cfg)
    c.collect()

if __name__ == "__main__":
    main()
