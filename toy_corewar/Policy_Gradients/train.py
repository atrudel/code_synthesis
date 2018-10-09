


def train_PG(
    reward_func,
    episodes,
    # Neural network parameters
    lr,
    gamma,
    # verbose and log
    log_dir=None
):
    env = Env(reward_func)
    
    if log_dir is not None:
        log_file = os.path.join(log_dir, "logs")
        with open(log_file, "w") as f:
            print("Starting training for {} episodes...".format(episodes), file=f)
            print("Algorithm: Policy Gradients", file=f)
            print("Reward function:\n\n{}\n\n\n".format(inspect.getsource(reward_func)), file=f)
        model_dir = os.path.join(log_dir, "models")
        os.makedirs(model_dir)