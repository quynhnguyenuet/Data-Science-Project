import hydra
import torch
import torch.nn as nn
from hydra.utils import instantiate
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from tsa import AutoEncForecast, train, evaluate
from tsa.utils import load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@hydra.main(config_path="./", config_name="config", version_base=None)

def run(cfg):
    ts = instantiate(cfg.data)
    train_iter, test_iter, nb_features = ts.get_loaders()

    model = AutoEncForecast(cfg.training, input_size=nb_features).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    if cfg.general.do_eval and cfg.general.get("ckpt", False):
        model, _, loss, epoch = load_checkpoint(cfg.general.ckpt, model, optimizer, device)
        evaluate(test_iter, loss, model, cfg, ts)
    elif cfg.general.do_train:
        train(train_iter, test_iter, model, criterion, optimizer, cfg, ts)


if __name__ == "__main__":
    run()
