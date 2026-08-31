# put in _base_palm/mem_probe_hook.py, add to custom_imports + custom_hooks
import torch
from mmengine.hooks import Hook
from mmdet.registry import HOOKS

@HOOKS.register_module()
class MemProbeHook(Hook):
    def _log(self, runner, tag):
        if torch.cuda.is_available():
            a = torch.cuda.memory_allocated() / 1024**3
            r = torch.cuda.memory_reserved() / 1024**3
            runner.logger.info(f'[MEM/{tag}] allocated={a:.2f} GB reserved={r:.2f} GB')
    def before_val(self, runner):  self._log(runner, 'before_val')
    def after_val(self, runner):
        self._log(runner, 'after_val_pre_empty')
        torch.cuda.empty_cache()
        self._log(runner, 'after_val_post_empty')