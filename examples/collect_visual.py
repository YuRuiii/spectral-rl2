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
            self.logger.output_dir if (cfg.save_video and not cfg.debug) else None
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
        
        self.load_model()
        self.global_step = 0
        self.global_episode = 0

        self.best_return = 0
        self.best_success = 0
        
    def load_model(self):
        # Load pretrained model
        dir_name = os.listdir(f"/data2/wangyc/spectral-rl2/log/{self.cfg.algo.cls}/debug/{self.cfg.task}")[0]
        path = f"/data2/wangyc/spectral-rl2/log/{self.cfg.algo.cls}/debug/{self.cfg.task}/{dir_name}/best_return"

        # Load actor state_dict and check for missing/unexpected keys
        actor_state_dict = torch.load(f"{path}/actor.pt")
        actor_load_result = self.agent.actor.load_state_dict(actor_state_dict)
        if actor_load_result.missing_keys or actor_load_result.unexpected_keys:
            raise RuntimeError(f"Failed to load actor state_dict: {actor_load_result}")
        print("Actor state_dict loaded successfully.")

        # Load critic state_dict and check for missing/unexpected keys
        critic_state_dict = torch.load(f"{path}/critic.pt")
        critic_load_result = self.agent.critic.load_state_dict(critic_state_dict)
        if critic_load_result.missing_keys or critic_load_result.unexpected_keys:
            raise RuntimeError(f"Failed to load critic state_dict: {critic_load_result}")
        print("Critic state_dict loaded successfully.")

        # Load encoder or vae based on algorithm class
        if self.cfg.algo.cls == "drqv2":
            encoder_state_dict = torch.load(f"{path}/encoder.pt")
            encoder_load_result = self.agent.encoder.load_state_dict(encoder_state_dict)
            if encoder_load_result.missing_keys or encoder_load_result.unexpected_keys:
                raise RuntimeError(f"Failed to load encoder state_dict: {encoder_load_result}")
            print("Encoder state_dict loaded successfully.")

        elif self.cfg.algo.cls == "diffsr_drqv2":
            vae_state_dict = torch.load(f"{path}/vae.pt")
            vae_load_result = self.agent.vae.load_state_dict(vae_state_dict)
            if vae_load_result.missing_keys or vae_load_result.unexpected_keys:
                raise RuntimeError(f"Failed to load vae state_dict: {vae_load_result}")
            print("Vae state_dict loaded successfully.")

        else:
            raise NotImplementedError("The algorithm class is not supported.")

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
        os.makedirs('/data2/wangyc/spectral-rl2/data', exist_ok=True)
        os.makedirs(f'/data2/wangyc/spectral-rl2/data/{self.cfg.algo.cls}_{self.cfg.task}_expert', exist_ok=True)
        path = f"/data2/wangyc/spectral-rl2/data/{self.cfg.algo.cls}_{self.cfg.task}_expert/{eval_episode}_success{int(sum(is_success_list))}.npz"
        
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
        for i_episode in trange(self.cfg.eval_episode):
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
                # if hasattr(time_step, "success"):
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
                        i_episode
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
