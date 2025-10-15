# cotnav/eval/cli.py
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from torch.utils.data import DataLoader
import torch

from cotnav.builders import build_dataset, build_model
from cotnav.eval.runners import load_eval_cfg, EpisodicRunner

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

@hydra.main(config_path='../../configs', config_name='eval/default', version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    assert all(k in cfg for k in ['dataset', 'model']), "Config must specify dataset, model."
    
    # Dataset: pass either a short name ("generic_nav") or a dotted path
    ds_name = cfg.dataset.get("name") or cfg.dataset.get("class_path")
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    ds = build_dataset(**dataset_cfg)

    # Optional: model if your pivot needs it
    model = None
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    model = build_model(**model_cfg)

    dl = DataLoader(
        ds,
        batch_size=cfg.loader.batch_size,
        shuffle=cfg.loader.shuffle,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        drop_last=cfg.loader.drop_last,
        collate_fn=getattr(ds, "collate_fn", None),
    )
    eval_cfg = OmegaConf.to_container(cfg.eval, resolve=True)
    runner = EpisodicRunner(
        dataloader=dl,
        model=model,
        eval_cfg=load_eval_cfg(eval_cfg),  # accepts dict or path
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
        device=str(device)
    )
    runner.run(max_batches=cfg.get("limit", None))

if __name__ == "__main__":
    main()
