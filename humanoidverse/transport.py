import sys
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import torch
import onnxruntime
import numpy as np
from humanoidverse.utils.inference_helpers import export_policy_as_onnx
from hydra.utils import instantiate
@hydra.main(config_path="config", config_name="base_eval")
def main(override_config):
    # 合并config
    if override_config.checkpoint is not None:
        checkpoint = Path(override_config.checkpoint)
        config_path = checkpoint.parent / "config.yaml"
        if not config_path.exists():
            config_path = checkpoint.parent.parent / "config.yaml"
        if config_path.exists():
            train_config = OmegaConf.load(config_path)
            if train_config.get("eval_overrides", None) is not None:
                train_config = OmegaConf.merge(train_config, train_config.eval_overrides)
            config = OmegaConf.merge(train_config, override_config)
        else:
            config = override_config
    else:
        if override_config.get("eval_overrides", None) is not None:
            config = OmegaConf.merge(override_config, override_config.eval_overrides)
        else:
            config = override_config
    if config.get("device", None):
        device = config.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    from humanoidverse.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
    from humanoidverse.utils.helpers import pre_process_config
    import torch
    from humanoidverse.utils.inference_helpers import export_policy_as_jit, export_policy_as_onnx, export_policy_and_estimator_as_onnx

    pre_process_config(config)
    env: BaseEnv = instantiate(config=config.env, device=device)
    algo: BaseAlgo = instantiate(config.algo, env=env, device=device, log_dir=None)
    algo.setup()
    algo.load(config.checkpoint)

    # 2. 构造观测样本
    example_obs_dict = algo.get_example_obs()  # 如果没有这个方法，手动构造一个观测样本

    # 3. 设置导出路径和文件名
    exported_policy_path = "./exported"
    os.makedirs(exported_policy_path, exist_ok=True)
    exported_onnx_name = "policy.onnx"

    # 4. 自动导出ONNX
    export_policy_as_onnx(
        algo.inference_model,
        exported_policy_path,
        exported_onnx_name,
        example_obs_dict
    )

    print(f"导出成功，已保存为 {os.path.join(exported_policy_path, exported_onnx_name)}")

if __name__ == "__main__":
    main()