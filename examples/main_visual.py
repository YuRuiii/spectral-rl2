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


class Trainer:
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
        
        self.agent = self.get_agent()

        self.global_step = 0
        self.global_episode = 0

        self.best_return = 0
        self.best_success = 0
        
    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def train(self):
        cfg = self.cfg

        ep_step, ep_return, ep_succ = 0, 0, 0
        time_step = self.train_env.reset()
        self.replay_buffer.add(time_step)
        for i_frame in trange(cfg.train_frames // cfg.action_repeat + 1, desc="main"):
            if time_step.last():
                self.global_episode += 1
                ep_frame = ep_step * cfg.action_repeat
                self.logger.log_scalars("", {
                    "rollout/return": ep_return,
                    "rollout/success": (ep_succ >= 1.0)*1.0 if self.domain == "metaworld" else 0.0,
                    "rollout/episode_frame": ep_frame
                }, step=self.global_frame)

                time_step = self.train_env.reset()
                self.replay_buffer.add(time_step)
                ep_step = ep_return = ep_succ = 0

            if self.global_frame < cfg.random_frames:
                sample = self.train_env.action_spec().generate_value()
                action = np.random.uniform(low=-1, high=1, size=sample.shape)
                action = action.astype(sample.dtype)
                train_metrics = {}
            else:
                if cfg.pretrain_steps > 0 and self.global_frame == cfg.random_frames:
                    for i_pretrain in trange(cfg.pretrain_steps, desc="pretrain"):
                        pretrain_metrics = self.agent.pretrain_step(self.replay_buffer, step=i_pretrain)
                action = self.agent.select_action(time_step.observation, self.global_step, deterministic=False)
                for i_update in range(cfg.utd):
                    train_metrics = self.agent.train_step(self.replay_buffer, self.global_step)

            if self.global_frame % cfg.log_frames == 0:
                self.logger.log_scalars("", train_metrics, step=self.global_frame)
                

            if self.global_frame % cfg.eval_frames == 0:
                eval_metrics = self.evaluate_training_model()
                self.logger.log_scalars("eval", eval_metrics, step=self.global_frame)
                self.logger.info(eval_metrics)

                self.save_model("last", eval_metrics)
                
                saved_eval_metrics = self.evaluate_pretrained_model("last")
                self.logger.log_scalars("eval_saved", saved_eval_metrics, step=self.global_frame)

                # save the best model 
                if eval_metrics["return_mean"] > self.best_return and eval_metrics["success_mean"] >= self.best_success:
                    self.best_return = eval_metrics["return_mean"]
                    self.best_success = eval_metrics["success_mean"]
                    self.save_model("best_return", eval_metrics)
                    
                    if self.best_success == 1:
                        break
                
            time_step = self.train_env.step(action)
            ep_return += time_step.reward
            ep_step += 1
            if self.domain == "metaworld":
                ep_succ += time_step.success
            self.replay_buffer.add(time_step)
            self.global_step += 1
            
    def evaluate(self, agent):
        agent.train(False)
        all_lengths = []
        all_returns = []
        all_success = []
        for i_episode in range(self.cfg.eval_episode):
            time_step = self.eval_env.reset()
            length = ret = success = 0
            self.recorder.init(self.eval_env, enabled=(i_episode==0))
            
            if self.cfg.mode == "collect":
                observation_list = []
                action_list = []
                reward_list = []
                is_terminal_list = []
                is_success_list = []
                is_first_list = []
                is_last_list = []
            
            while not time_step.last():
                action = agent.select_action(time_step.observation, self.global_step, deterministic=True)
                time_step = self.eval_env.step(action)
                self.recorder.record(self.eval_env)
                ret += time_step.reward
                length += 1
                # if hasattr(time_step, "success"):
                success += float(time_step.success)
                
                # save to buffer
                if self.cfg.mode == "collect":
                    observation_list.append(time_step.observation)
                    action_list.append(action)
                    reward_list.append(time_step.reward)
                    is_terminal_list.append(time_step.last() or time_step.success)
                    is_success_list.append(time_step.success)
                    is_first_list.append(len(observation_list) == 1)
                    is_last_list.append(time_step.last())
                
            if self.cfg.mode == "collect":
                self.save_data(
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
            agent_metrics, reconstruction = agent.evaluate(self.replay_buffer)
            metrics.update(agent_metrics)
            if reconstruction is not None:
                self.logger.log_image("info/reconstruction", reconstruction, step=self.global_frame)
        agent.train(True)
        return metrics
    
    def evaluate_training_model(self):
        return self.evaluate(self.agent)

    def evaluate_pretrained_model(self, name):
        agent = self.get_agent()
        agent = self.load_model(agent, name)
        return self.evaluate(agent)
    
    def get_agent(self):
        algo_cls = {
            "drqv2": DrQv2,
            "diffsr_drqv2": DiffSR_DrQv2,
            "mulvrep_drqv2": MuLVRep_DrQv2
        }.get(self.cfg.algo.cls)
        agent = algo_cls(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            self.cfg.algo,
            self.device
        )
        return agent
        
    def load_model(self, agent, name):
        # Load pretrained model
        if self.cfg.model_dir is None:
            print("model_dir is not set, using default path")
            self.cfg.model_dir = self.logger.log_dir
        path = f"{self.cfg.model_dir}/{name}"

        agent.actor.load_state_dict(torch.load(f"{path}/actor.pt"))
        agent.critic.load_state_dict(torch.load(f"{path}/critic.pt"))
        if self.cfg.algo.cls == "drqv2":
            agent.encoder.load_state_dict(torch.load(f"{path}/encoder.pt"))
        elif self.cfg.algo.cls == "diffsr_drqv2":
            agent.vae.load_state_dict(torch.load(f"{path}/vae.pt"))
        else:
            raise NotImplementedError
        
        return agent


    def save_model(self, name, eval_metrics):
        dir = os.path.join(self.logger.log_dir, name)
        os.makedirs(dir, exist_ok=True)
                
        if self.cfg.algo.cls == "drqv2":
            torch.save(obj=self.agent.encoder.state_dict(), f=os.path.join(dir, "encoder.pt"))
        elif self.cfg.algo.cls == "diffsr_drqv2":
            torch.save(obj=self.agent.vae.state_dict(), f=os.path.join(dir, "vae.pt"))        
        torch.save(obj=self.agent.actor.state_dict(), f=os.path.join(dir, "actor.pt"))
        torch.save(obj=self.agent.critic.state_dict(), f=os.path.join(dir, "critic.pt"))
        
        # if self.cfg.algo.cls == "diffsr_drqv2":
        #     self.logger.log_object(name="vae.pt", object=self.agent.vae.state_dict(), path=dir)
        # elif self.cfg.algo.cls == "drqv2":
        #     self.logger.log_object(name="encoder.pt", object=self.agent.encoder.state_dict(), path=dir)
        # else:
        #     raise NotImplementedError
        # self.logger.log_object(name="actor.pt", object=self.agent.actor.state_dict(), path=dir)
        # self.logger.log_object(name="critic.pt", object=self.agent.critic.state_dict(), path=dir)
        
        if eval_metrics is not None:
            info = f"{self.global_episode}, save {name}, return {eval_metrics['return_mean']}, success {eval_metrics['success_mean']}"
            print(info)
        
            with open(f"{dir}/example.txt", "a") as file:
                file.write(f"{info}\n") 
            
    def save_data(
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
        os.makedirs('/home/amax/yr/spectral-rl2/data', exist_ok=True)
        os.makedirs(f'/home/amax/yr/spectral-rl2/data/{self.cfg.algo.cls}_{self.cfg.task}_expert', exist_ok=True)
        path = f"/home/amax/yr/spectral-rl2/data/{self.cfg.algo.cls}_{self.cfg.task}_expert/{eval_episode}_success{int(sum(is_success_list))}.npz"
        
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

@hydra.main(version_base=None, config_path="./config/visual", config_name="config")
def main(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    if cfg.mode == "train":
        trainer.train()
    elif cfg.mode == "collect":
        trainer.evaluate_pretrained_model("last")
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
