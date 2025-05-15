import os
import torch
import shutil

from torch.utils.data import Sampler
from torch import nn
from transformers import Trainer
from transformers.trainer import is_sagemaker_mp_enabled, get_parameter_names, has_length, ALL_LAYERNORM_LAYERS, logger

from typing import List, Optional

from friday.util import maybe_zero_3
from .sampling import LengthGroupedSampler

class FridayTrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.vision_adapter_lr is not None:
                projector_parameters = [name for name, _ in opt_model.get_vision_adapter().parameters() if "mm_projector" in name or "vision_tower" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.vision_adapter_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.vision_adapter_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if
                            (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped / 2 ** 20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2 ** 20}M params")

        return self.optimizer
    

    def _save_checkpoint(self, model, trial, metrics=None):
        super()._save_checkpoint(model, trial)

        if self.args.local_rank in (0, -1):
            if "wandb" in getattr(self.args, "report_to", []):
                # 1. setup directories
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)

                # 2. delete the full weight checkpoint
                full_wts_ckpt = os.path.join(output_dir, f"global_step{self.state.global_step}", "mp_rank_00_model_states.pt")
                if os.path.exists(full_wts_ckpt):
                    os.remove(full_wts_ckpt)

                # 3. zip the checkpoint folder
                zip_path = shutil.make_archive(output_dir, 'zip', output_dir)

                # 2. upload the zip file to wandb
                import wandb
                artifact = wandb.Artifact(
                    name="checkpoint",
                    type="model",
                    description="MM-projector only, frozen backbone",
                    metadata={
                        "global_step": self.state.global_step,
                        "epoch": self.state.epoch,
                        **self.state.log_history[-1],
                    },
                )
                artifact.add_file(zip_path)
                wandb.log_artifact(artifact)


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if self.args.save_only_vision_adapter:
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                
                mm_projector_state = {}
                for name, p in self.model.get_vision_adapter().named_parameters():
                    mm_projector_state[name] = maybe_zero_3(p, ignore_status=True, name=name)

                torch.save(mm_projector_state, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(FridayTrainer, self)._save(output_dir, state_dict)


    