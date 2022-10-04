from tqdm.auto import tqdm, trange
import torch
import wandb
import numpy as np
from pprint import pprint
import random

from src.model import Dvoryaninormer
from src.data import CoilDataset, collate_graphs, load_coil_graphs


DEFAULT_PARAMS = {
    "device": "cuda",
    "adamw": {"lr": 1e-3, "weight_decay": 1e-4},
    "epochs": 300,
    "scheduler": {"max_lr": 1e-3, "anneal_strategy": "linear", "pct_start": 0.1},
    "run_evaluation": True,
    "log_steps": 20,
    "eval_steps": 1000,
    "max_grad_norm": 10.0,
    "model": {},
    "random_state": 42,
    "batch_size": 128,
    "wandb": {},
}


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def dict_to_device(d, device="cuda"):
    for key in d:
        d[key] = d[key].to(device)
    return d


def eval_model(model, dataloader, criterion, device="cuda:0"):
    model.eval()

    eval_loss = 0.0
    eval_guesses = 0
    eval_num = 0
    with torch.inference_mode():
        for batch in dataloader:
            y = batch["target"].to(device)
            del batch["target"]
            output = model(**dict_to_device(batch, device))
            loss = criterion(output, y)
            eval_loss += loss.item()
            eval_num += y.size()[0]

            _, y_pred = torch.max(output, 1)
            eval_guesses += (y_pred == y).sum().item()

    return eval_loss / eval_num, eval_guesses / eval_num


def train_dvoryanin(D_train, y_train, D_test, y_test, params=None):
    config = DEFAULT_PARAMS.copy()
    unknown_params = set(params.keys()) - set(config.keys())
    if len(unknown_params) > 0:
        raise ValueError(f"Unknown parameters {unknown_params}")
    config.update(params)

    set_random_seed(config["random_state"])

    train_dataset = CoilDataset(D_train, y_train)
    test_dataset = CoilDataset(D_test, y_test)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_graphs
    )
    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["batch_size"], collate_fn=collate_graphs)

    config["total_steps"] = config["epochs"] * len(train_loader)

    model = Dvoryaninormer(**config["model"]).to(config["device"])
    optimizer = torch.optim.AdamW(model.parameters(), **config["adamw"])
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    config["scheduler"]["total_steps"] = config["total_steps"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **config["scheduler"])

    wandb.init(
        **params["wandb"],
        config=config,
    )
    wandb.watch(model)

    steps = 0
    train_loss = 0.0
    train_num = 0

    eval_loss, eval_acc = eval_model(model, eval_loader, criterion, config["device"])
    wandb.log({"eval_loss": eval_loss, "eval_accuracy": eval_acc}, step=steps)

    try:
        with tqdm(total=config["total_steps"]) as tbar:
            for epoch in range(config["epochs"]):
                for batch in train_loader:
                    model.train()
                    y = batch["target"].to(config["device"])
                    del batch["target"]
                    output = model(**dict_to_device(batch, config["device"]))

                    loss = criterion(output, y)
                    train_loss += loss.item()
                    train_num += y.size()[0]

                    loss.backward()

                    if config["max_grad_norm"] is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])

                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    tbar.update(1)
                    steps += 1

                    wandb.log({"adamw_learning_rate": optimizer.param_groups[0]["lr"], "epoch": epoch}, step=steps)

                    if steps % config["log_steps"] == 0:
                        wandb.log({"train_loss": train_loss / train_num}, step=steps)

                        train_loss = 0.0
                        train_num = 0

                    if steps % config["eval_steps"] == 0:
                        eval_loss, eval_acc = eval_model(model, eval_loader, criterion, config["device"])
                        wandb.log({"eval_loss": eval_loss, "eval_accuracy": eval_acc}, step=steps)

    except KeyboardInterrupt:
        print("TRAINING STOPPED!")

    if config["run_evaluation"]:
        print("TRAINING FINISHED. EVALUATING.")

        train_loss, train_acc = eval_model(model, train_loader, criterion, config["device"])
        eval_loss, eval_acc = eval_model(model, eval_loader, criterion, config["device"])

        metrics = {
            "final_train_loss": train_loss,
            "final_eval_loss": eval_loss,
            "final_train_accuracy": train_acc,
            "final_eval_accuracy": eval_acc,
        }

        wandb.log(metrics)
        pprint(metrics)

    wandb.finish()
    return model
