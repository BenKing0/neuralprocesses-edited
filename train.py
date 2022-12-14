import argparse
import os
import sys
from functools import partial
import lab as B
import numpy as np
import torch
import wbml.out as out
from wbml.experiment import WorkingDirectory
import experiment as exp
import neuralprocesses.torch as nps

__all__ = ["main"]


def train(state, model, opt, objective, gen, *, epoch):
    """Train for an epoch."""
    vals = []
    for batch in gen.epoch():
        state, obj = objective(
            state,
            model,
            batch["contexts"],
            batch["xt"],
            batch["yt"],
            epoch=epoch,
        )
        vals.append(B.to_numpy(obj))
        # Be sure to negate the output of `objective`.
        val = -B.mean(obj)
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()

    out.kv("Loglik (T)", exp.with_err(B.concat(*vals)))
    return state, B.mean(B.concat(*vals))


def eval(state, model, objective, gen):
    """Perform evaluation."""
    with torch.no_grad():
        vals, kls, kls_diag = [], [], []
        for batch in gen.epoch():
            state, obj = objective(
                state,
                model,
                batch["contexts"],
                batch["xt"],
                batch["yt"],
            )

            # Save numbers.
            n = nps.num_data(batch["xt"], batch["yt"])
            vals.append(B.to_numpy(obj))
            if "pred_logpdf" in batch:
                kls.append(B.to_numpy(batch["pred_logpdf"] / n - obj))
            if "pred_logpdf_diag" in batch:
                kls_diag.append(B.to_numpy(batch["pred_logpdf_diag"] / n - obj))

        # Report numbers.
        out.kv("Loglik (V)", exp.with_err(B.concat(*vals)))
        if kls:
            out.kv("KL (full)", exp.with_err(B.concat(*kls)))
        if kls_diag:
            out.kv("KL (diag)", exp.with_err(B.concat(*kls_diag)))

        return state, B.mean(B.concat(*vals))


def main(**kw_args):

    # Setup arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, nargs="*", default=["_experiments"])
    parser.add_argument("--subdir", type=str, nargs="*")
    parser.add_argument("--dim-x", type=int, default=1)
    parser.add_argument("--dim-y", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--model",
        choices=[
            "cnp",
            "gnp",
            "np",
            "acnp",
            "agnp",
            "anp",
            "convcnp",
            "convgnp",
            "convnp",
            "fullconvgnp",
            "newconvmodel_lowrank",
        ],
        default="convcnp",
    )
    parser.add_argument("--arch", choices=["unet", "dws"], default="unet")
    parser.add_argument(
        "--data",
        choices=exp.data,   ## why can exp.data not register 't-distribution' and 'bi-modal'?
        default="eq",
    )
    parser.add_argument("--objective", choices=["loglik", "elbo"], default="loglik")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--resume-at-epoch", type=int)
    parser.add_argument("--train-test", action="store_true")
    parser.add_argument("--check-completed", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--evaluate-last", action="store_true")
    parser.add_argument("--evaluate-fast", action="store_true")
    parser.add_argument("--evaluate-plot-num-samples", type=int, default=10)
    parser.add_argument(
        "--evaluate-objective",
        choices=["loglik", "elbo"],
        default="loglik",
    )
    parser.add_argument("--evaluate-num-samples", type=int, default=4096)
    parser.add_argument("--evaluate-batch-size", type=int, default=16)
    parser.add_argument("--no-action", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--ar", action="store_true")
    if kw_args:
        # Load the argument from the keyword arguments passed to the function.
        # Carefully convert these to command line arguments.
        args = parser.parse_args(
            sum(
                [
                    [f"--{k}"] + ([str(v)] if v is not True else [])
                    for k, v in kw_args.items()
                ],
                [],
            )
        )
    else:
        args = parser.parse_args()

    # Remove the architecture argument if a model doesn't use it.
    if args.model not in {
        "convcnp",
        "convgnp",
        "convnp",
        "fullconvgnp",
        "newconvmodel_lowrank",
    }:
        del args.arch

    # Remove the dimensionality specification if the experiment doesn't need it.
    if not exp.data[args.data]["requires_dim_x"]:
        del args.dim_x
    if not exp.data[args.data]["requires_dim_y"]:
        del args.dim_y

    # Determine the mode of the script.
    if args.check_completed or args.no_action:
        # Don't add any mode suffix.
        mode = ""
    elif args.evaluate:
        mode = "_evaluate"
        if args.ar:
            mode += "_ar"
    else:
        # The default is training.
        mode = "_train"

    # Setup script.
    out.report_time = True
    B.epsilon = 1e-8
    wd = WorkingDirectory(
        *args.root,
        *(args.subdir or ()),
        args.data,
        *((f"x{args.dim_x}_y{args.dim_y}",) if hasattr(args, "dim_x") else ()),
        args.model,
        *((args.arch,) if hasattr(args, "arch") else ()),
        args.objective,
        log=f"log{mode}.txt",
        diff=f"diff{mode}.txt",
    )

    # Check if a run has completed.
    if args.check_completed:
        if os.path.exists(wd.file("model-last.torch")):
            d = torch.load(wd.file("model-last.torch"), map_location="cpu")
            if d["epoch"] == args.epochs:
                out.out("Completed!")
                sys.exit(0)
        out.out("Not completed.")
        sys.exit(1)

    # Use a GPU if one is available.
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    B.set_global_device(device)
    # Maintain an explicit random state through the execution.
    state = B.create_random_state(torch.float32, seed=0)

    # General architecture choices:
    config = {
        "width": 256,
        "dim_embedding": 256,
        "num_heads": 8,
        "num_layers": 6,
        "unet_channels": (64, 64, 64, 128, 128, 128, 256),
        "dws_channels": 128,
        "num_basis_functions": 512,     ## now should be 64
    }

    gen_train, gen_cv, gens_eval = exp.data[args.data]["setup"](
        args,
        config,
        num_tasks_train=2**9 if args.train_test else 2**14,
        num_tasks_cv=2**9 if args.train_test else 2**12,
        num_tasks_eval=2**9 if args.evaluate_fast else 2**14,
        device=device,
    )

    # Construct the model.
    if args.model == "cnp":
        model = nps.construct_gnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            likelihood="het",
            transform=config["transform"],
        )
    elif args.model == "gnp":
        model = nps.construct_gnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            likelihood="lowrank",
            num_basis_functions=config["num_basis_functions"],
            transform=config["transform"],
        )
    elif args.model == "np":
        model = nps.construct_gnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            likelihood="het",
            dim_lv=config["dim_embedding"],
            transform=config["transform"],
        )
    elif args.model == "acnp":
        model = nps.construct_agnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            num_heads=config["num_heads"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            likelihood="het",
            transform=config["transform"],
        )
    elif args.model == "agnp":
        model = nps.construct_agnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            num_heads=config["num_heads"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            likelihood="lowrank",
            num_basis_functions=config["num_basis_functions"],
            transform=config["transform"],
        )
    elif args.model == "anp":
        model = nps.construct_agnp(
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            dim_embedding=config["dim_embedding"],
            num_heads=config["num_heads"],
            num_dec_layers=config["num_layers"],
            width=config["width"],
            likelihood="het",
            dim_lv=config["dim_embedding"],
            transform=config["transform"],
        )
    elif args.model == "convcnp":
        model = nps.construct_convgnp(
            points_per_unit=config["points_per_unit"],
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            likelihood="het",
            conv_arch=args.arch,
            unet_channels=config["unet_channels"],
            margin=config["margin"],
            transform=config["transform"],
        )
    elif args.model == "convgnp":
        model = nps.construct_convgnp(
            points_per_unit=config["points_per_unit"],
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            likelihood="lowrank",
            conv_arch=args.arch,
            unet_channels=config["unet_channels"],
            num_basis_functions=config["num_basis_functions"],
            margin=config["margin"],
            transform=config["transform"],
        )
    elif args.model == "convnp":
        model = nps.construct_convgnp(
            points_per_unit=config["points_per_unit"],
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            likelihood="het",
            conv_arch=args.arch,
            unet_channels=config["unet_channels"],
            dim_lv=16,
            margin=config["margin"],
            transform=config["transform"],
        )
    elif args.model == "fullconvgnp":
        model = nps.construct_fullconvgnp(
            points_per_unit=config["points_per_unit"],
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            conv_arch=args.arch,
            unet_channels=config["unet_channels"],
            margin=config["margin"],
            transform=config["transform"],
        )
    elif args.model == "newconvmodel_lowrank":
        model = nps.construct_convgnp(
            points_per_unit=config["points_per_unit"],
            dim_x=config["dim_x"],
            dim_yc=(1,) * config["dim_y"],
            dim_yt=config["dim_y"],
            likelihood="het",
            lv_likelihood="lowrank",
            conv_arch=args.arch,
            unet_channels=config["unet_channels"],
            dim_lv=16,
            margin=config["margin"],
            transform=config["transform"],
        )
    else:
        raise ValueError(f'Invalid model "{args.model}".')

    # Settings specific model the model:
    if args.model in {
        "cnp",
        "gnp",
        "acnp",
        "agnp",
        "convcnp",
        "convgnp",
        "fullconvgnp",
    }:
        config["fix_noise"] = False
    elif args.model in {
        "np", 
        "anp", 
        "convnp",
        "newconvmodel_lowrank",
        "newconvmodel_dense",
        }:
        config["fix_noise"] = True
    else:
        raise ValueError(f'Invalid model "{args.model}".')

    # Ensure that the model is on the GPU and print some statistics.
    model = model.to(device)
    out.kv("Number of parameters", nps.num_params(model))
    out.kv("Config", config)

    # Setup training objective.
    if args.objective == "loglik":
        objective = partial(
            nps.loglik,
            num_samples=args.num_samples,
            normalise=True,
        )
        objective_cv = partial(
            nps.loglik,
            num_samples=args.num_samples,
            normalise=True,
        )
        objectives_eval = [
            (
                "Loglik",
                partial(
                    nps.loglik,
                    num_samples=args.evaluate_num_samples,
                    batch_size=args.evaluate_batch_size,
                    normalise=True,
                ),
            )
        ]
    elif args.objective == "elbo":
        objective = partial(
            nps.elbo,
            num_samples=args.num_samples,
            subsume_context=True,
            normalise=True,
        )
        objective_cv = partial(
            nps.elbo,
            num_samples=args.num_samples,
            subsume_context=False,  # Lower bound the right quantity.
            normalise=True,
        )
        objectives_eval = [
            (
                "ELBO",
                partial(
                    nps.elbo,
                    # Don't need a high number of samples, because it is unbiased.
                    num_samples=5,
                    subsume_context=False,  # Lower bound the right quantity.
                    normalise=True,
                ),
            ),
            (
                "Loglik",
                partial(
                    nps.loglik,
                    num_samples=args.evaluate_num_samples,
                    batch_size=args.evaluate_batch_size,
                    normalise=True,
                ),
            ),
        ]
    else:
        raise RuntimeError(f'Invalid objective "{args.objective}".')

    # See if the point was to just load everything.
    if args.load:
        return {
            "wd": wd,
            "gen_train": gen_train,
            "gen_cv": gen_cv,
            "gens_eval": gens_eval,
            "model": model,
        }

    # The user can just want to see some statistics about the model.
    if args.no_action:
        exit()

    if args.evaluate:
        # Perform evaluation.
        if args.evaluate_last:
            name = "model-last.torch"
        else:
            name = "model-best.torch"
        model.load_state_dict(torch.load(wd.file(name)["weights"], map_location=device))

        if not args.ar:
            # Make some plots.
            for i in range(args.evaluate_plot_num_samples):
                exp.visualise(
                    model,
                    gen_cv,
                    path=wd.file(f"evaluate-{i + 1:03d}.pdf"),
                    config=config,
                )

            # For every objective and evaluation generator, do the evaluation.
            for objecive_name, objective_eval in objectives_eval:
                with out.Section(objecive_name):
                    for gen_name, gen in gens_eval():
                        with out.Section(gen_name.capitalize()):
                            state, _ = eval(state, model, objective_eval, gen)

        # Do AR evaluation, but only for the conditional models.
        if args.model in {"cnp", "acnp", "convcnp"}:
            # Make some plots.
            for i in range(args.evaluate_plot_num_samples):
                exp.visualise(
                    model,
                    gen_cv,
                    path=wd.file(f"evaluate-ar-{i + 1:03d}.pdf"),
                    config=config,
                    predict=nps.ar_predict,
                )

            # For both random and left-to-right ordering, do AR testing.
            for order in ["random", "left-to-right"]:
                with out.Section(order.capitalize()):
                    for name, gen in gens_eval():
                        with out.Section(name.capitalize()):
                            state, _ = eval(
                                state,
                                model,
                                partial(nps.ar_loglik, order=order, normalise=True),
                                gen,
                            )
    else:
        # Perform training. First, check if we want to resume training.
        start = 0
        if args.resume_at_epoch:
            start = args.resume_at_epoch - 1
            model.load_state_dict(
                torch.load(wd.file("model-last.torch"), map_location=device)["weights"]
            )

        # Setup training loop.
        opt = torch.optim.Adam(model.parameters(), args.rate)
        best_eval_lik = -np.inf

        # Set regularisation high for the first epochs.
        original_epsilon = B.epsilon
        B.epsilon = 1e-2

        for i in range(start, args.epochs):
            with out.Section(f"Epoch {i + 1}"):
                # Set regularisation to normal after the first epoch.
                if i > 0:
                    B.epsilon = original_epsilon

                # Perform an epoch.
                state, _ = train(
                    state,
                    model,
                    opt,
                    objective,
                    gen_train,
                    epoch=i if config["fix_noise"] else None,
                )

                # The epoch is done. Now evaluate.
                ## As this train.py is in neuralprocesses (not venv2), the data generator is in neuralprocesses.experiment.data (not venv2)
                # state, val = eval(state, model, objective_cv, gen_cv())
                state, val = eval(state, model, objective_cv, gen_cv)

                # Save current model.
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "objective": val,
                        "epoch": i + 1,
                    },
                    wd.file(f"model-last.torch"),
                )

                # Check if the model is the new best. If so, save it.
                if val > best_eval_lik:
                    out.out("New best model!")
                    best_eval_lik = val
                    torch.save(
                        {
                            "weights": model.state_dict(),
                            "objective": val,
                            "epoch": i + 1,
                        },
                        wd.file(f"model-best.torch"),
                    )

                # Visualise a prediction by the model.
                if args.data == 'bimodal':
                    exp.visualise(
                        model,
                        # gen_cv(),
                        gen_cv,
                        path=wd.file(f"train-epoch-{i + 1:03d}.pdf"),
                        config=config,
                        inc_ground_truth=True,
                    )
                else:  
                    exp.visualise(
                        model,
                        # gen_cv(),
                        gen_cv,
                        path=wd.file(f"train-epoch-{i + 1:03d}.pdf"),
                        config=config,
                    )


if __name__ == "__main__":
    main()
