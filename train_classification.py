import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import numpy as np
import sys
import imp
import molgrid
import argparse
import os
import time
# import wandb
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from datetime import datetime
import os
import json as json


best_plot = {"auc": float("-inf"), "labels": None, "probs": None, "iter": None}

class CsvLogger:
    HEADERS = ["time_iso","iter","phase","train_loss","train_accuracy","test_loss","test_accuracy","test_auc","learning_rate","trial_number"]
    def __init__(self, path):
        self.path = path
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        if not os.path.exists(path):
            with open(self.path, "w") as f:
                f.write(",".join(self.HEADERS) + "\n")
    @staticmethod
    def now(): return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    def log(self, **row):
        vals = []
        for h in self.HEADERS:
            v = row.get(h, "")
            if isinstance(v, float): v = f"{v:.6f}"
            vals.append(str(v))
        with open(self.path, "a") as f:
            f.write(",".join(vals) + "\n")

torch.backends.cudnn.benchmark = True

def parse_args(argv=None):
    '''Return argument namespace and commandline'''
    parser = argparse.ArgumentParser(description='Train neural net on .types data.')

    # optuna
    parser.add_argument('--optuna', action='store_true', help='Run Optuna hyperparameter search')
    parser.add_argument('--n_trials', type=int, default=25, help='Number of Optuna trials')
    parser.add_argument('--csv_log', type=str, default=None, help='CSV file for metrics')
    parser.add_argument('--metrics_json', type=str, default=None, help='Final metrics JSON')

    parser.add_argument('-m', '--model', type=str, required=True, help="Model template python file")
    parser.add_argument('--train_types', type=str, required=True, help="training types file")
    parser.add_argument('--test_types', type=str, required=True, help="test types file")
    parser.add_argument('-i', '--iterations', type=int, default=10000, help="Number of iterations to run, default 10,000")
    parser.add_argument('-d', '--data_dir', type=str, default="", help="Root directory of data")
    parser.add_argument('--train_recmolcache', type=str, default="", help="path to receptor molcache")
    parser.add_argument('--test_recmolcache', type=str, default="", help="path to receptor molcache")
    parser.add_argument('-b', '--batch_size', type=int, default=50, help="Batch size for training, default 50")
    parser.add_argument('-s', '--seed', type=int, default=0, help="Random seed, default 0")
    parser.add_argument('-t', '--test_interval', type=int, default=1000, help="How frequently to test (iterations)")
    parser.add_argument('-r', '--run_name', type=str, required=False, help="name for run")
    parser.add_argument('-o', '--outprefix', type=str, required=True, help="Prefix for output files")
    parser.add_argument('--percent_reduced', type=float, default=100,
                        help='Create a reduced set on the fly based on types file, using the given percentage: 10 means 10%%.')
    parser.add_argument('--checkpoint', type=str, required=False, help="file to continue training from")

    parser.add_argument('--solver', type=str, default='SGD', help="Solver type: SGD, Nesterov or Adam")
    parser.add_argument('--step_reduce', type=float, default=0.1, help="Reduce LR by this factor w/ ReduceLROnPlateau")
    parser.add_argument('--step_end_cnt', type=float, default=3, help='Terminate after this many LR reductions')
    parser.add_argument('--step_when', type=int, default=15, help="No-improve evals before scheduler patience triggers")
    parser.add_argument('--base_lr', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help="Momentum (SGD/Nesterov)")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="Weight decay")
    parser.add_argument('--clip_gradients', type=float, default=10.0, help="Clip gradients threshold")

    args = parser.parse_args(argv)

    argdict = vars(args)
    line = ''
    for (name, val) in list(argdict.items()):
        if val != parser.get_default(name):
            line += ' --%s=%s' % (name, val)

    return (args, line)

def plot_roc_curve(labels, probs, outprefix, iteration):
    fpr, tpr, _ = roc_curve(labels, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC={roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel('FP rate'); plt.ylabel('TP rate')
    plt.legend(loc="lower right"); plt.grid(True)
    plt.savefig(f'{outprefix}_roc_curve_iter_{iteration}.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_learning_curves(history, outprefix):
    """loss and accuracy vs iterations"""
    iters = history["iter"]
    plt.figure()
    if history["train_loss"]:
        plt.plot(iters, history["train_loss"], label="Train loss", linewidth=2)
    if history["test_loss"]:
        plt.plot(iters, history["test_loss"], label="Test loss", linewidth=2)
    plt.xlabel("Iteration"); plt.ylabel("Loss"); plt.title("Learning curve (loss)")
    plt.grid(True); plt.legend()
    plt.savefig(f"{outprefix}_learning_loss.png", dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure()
    if history["train_acc"]:
        plt.plot(iters, history["train_acc"], label="Train accuracy", linewidth=2)
    if history["test_acc"]:
        plt.plot(iters, history["test_acc"], label="Test accuracy", linewidth=2)
    plt.xlabel("Iteration"); plt.ylabel("Accuracy"); plt.title("Learning curve (accuracy)")
    plt.grid(True); plt.legend()
    plt.savefig(f"{outprefix}_learning_accuracy.png", dpi=150, bbox_inches='tight')
    plt.close()

def initialize_model(model, args):
    model.cuda()
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        pretrained_path = 'first_model_fold1_best_test_auc_85001.pth.tar'
        if os.path.isfile(pretrained_path):
            print(f"Loading pretrained weights from {pretrained_path}")
            model.load_pretrained_weights(pretrained_path)
        else:
            print("No pretrained model found, initializing from scratch.")
    return model

def get_model_gmaker_eproviders(args):
    # train provider
    eptrain = molgrid.ExampleProvider(shuffle=True, stratify_receptor=True, labelpos=0, balanced=True,
                                      data_root=args.data_dir, recmolcache=args.train_recmolcache)
    eptrain.populate(args.train_types)
    # test providers
    eptest_large = molgrid.ExampleProvider(shuffle=False, stratify_receptor=False, labelpos=0, balanced=False,
        data_root=args.data_dir, iteration_scheme=molgrid.IterationScheme.LargeEpoch,
        default_batch_size=args.batch_size, recmolcache=args.test_recmolcache)
    eptest_large.populate(args.test_types)
    eptest_small = molgrid.ExampleProvider(shuffle=True, stratify_receptor=True, labelpos=0, balanced=True,
        data_root=args.data_dir, iteration_scheme=molgrid.IterationScheme.SmallEpoch,
        default_batch_size=args.batch_size, recmolcache=args.test_recmolcache)
    eptest_small.populate(args.test_types)

    gmaker = molgrid.GridMaker()
    _ = gmaker.grid_dimensions(eptrain.num_types()) 

    model_file = imp.load_source("model", args.model)
    torch.manual_seed(args.seed)
    model = model_file.Model()

    return model, gmaker, eptrain, eptest_large, eptest_small

def train_and_test(args, model, eptrain, eptest_large, eptest_small, gmaker):
    # CSV logger (per-trial)
    base = getattr(args, 'outprefix', None) or getattr(args, 'run_name', None) or "run"
    csv_path = getattr(args, 'csv_log', None) or f"{base}_metrics.csv"
    logger = CsvLogger(csv_path)

    def test_model(model, ep, gmaker, percent_reduced, batch_size):
        all_losses, all_accuracy, all_labels, all_probs = [], [], [], []

        class_weights = torch.tensor([0.7, 0.3], dtype=torch.float32).cuda()
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

        input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda', requires_grad=False)
        float_labels = torch.zeros((batch_size, 4), dtype=torch.float32, device='cuda')

        model.eval()
        with torch.no_grad():
            for batch in ep:
                batch.extract_labels(float_labels)
                centers = float_labels[:, 1:]
                labels = float_labels[:, 0].long().to('cuda')

                for b in range(batch_size):
                    center = molgrid.float3(float(centers[b][0]), float(centers[b][1]), float(centers[b][2]))
                    gmaker.forward(center, batch[b].coord_sets[0], input_tensor[b])

                output = model(input_tensor[:, :14])
                loss = criterion(output, labels)
                predicted = torch.argmax(output, dim=1)
                accuracy = labels.eq(predicted).sum().float() / batch_size

                all_losses.append(loss.detach())
                all_accuracy.append(accuracy)
                all_labels.append(labels.cpu())
                all_probs.append(F.softmax(output, dim=1).detach().cpu())

        all_labels = torch.flatten(torch.stack(all_labels)).cpu().numpy()
        all_probs = torch.flatten(torch.stack(all_probs), start_dim=0, end_dim=1).cpu().numpy()
        total_test_loss_mean = torch.mean(torch.stack(all_losses)).cpu()
        total_test_accuracy_mean = torch.mean(torch.stack(all_accuracy)).cpu()
        auc_val = roc_auc_score(all_labels, all_probs[:, 1])
        return total_test_loss_mean, total_test_accuracy_mean, auc_val, all_labels, all_probs

    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
    initialize_model(model, args)

    iterations     = args.iterations
    test_interval  = args.test_interval
    batch_size     = args.batch_size
    percent_reduced = args.percent_reduced
    outprefix      = args.outprefix

    prev_total_loss_snap = ''
    prev_total_accuracy_snap = ''
    prev_total_auc_snap = ''
    prev_snap = ''

    initial = 0
    if args.checkpoint:
        initial = checkpoint['Iteration']
    last_test = 0

    # optimizer for unfrozen parameters only
    if 'SGD' in args.solver:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif 'Nesterov' in args.solver:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif 'Adam' in args.solver:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=args.base_lr, weight_decay=args.weight_decay)
    else:
        print("No valid solver argument passed (SGD, Adam, Nesterov)")
        sys.exit(1)
    if args.checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',
                                                           factor=args.step_reduce,
                                                           patience=args.step_when,
                                                           verbose=True)
    if args.checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    

    Bests = {
        'train_iteration': 0,
        'test_loss': torch.from_numpy(np.asarray(np.inf)),
        'test_accuracy': torch.from_numpy(np.asarray([0])),
        'test_auc': torch.from_numpy(np.asarray([0])),
    }
    if args.checkpoint:
        Bests = checkpoint['Bests']

    # capture train metrics at bests
    Bests.setdefault('iter_at_best_auc', None)
    Bests.setdefault('train_loss_at_best_auc', None)
    Bests.setdefault('train_acc_at_best_auc', None)

    Bests.setdefault('iter_at_best_loss', None)
    Bests.setdefault('train_loss_at_best_loss', None)
    Bests.setdefault('train_acc_at_best_loss', None)

    Bests.setdefault('iter_at_best_acc', None)
    Bests.setdefault('train_loss_at_best_acc', None)
    Bests.setdefault('train_acc_at_best_acc', None)

    dims = gmaker.grid_dimensions(eptrain.num_types())
    tensor_shape = (batch_size,) + dims

    model.cuda()
    input_tensor = torch.zeros(tensor_shape, dtype=torch.float32, device='cuda', requires_grad=False)
    float_labels = torch.zeros((batch_size, 4), dtype=torch.float32, device='cuda')
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    last_train_loss = None
    last_train_acc  = None

    # learning-curve history
    history = {
        "iter": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for i in range(initial, iterations):
        batch = eptrain.next_batch(batch_size)
        batch.extract_labels(float_labels)
        centers = float_labels[:, 1:]
        labels = float_labels[:, 0].long().to('cuda')

        for b in range(batch_size):
            center = molgrid.float3(float(centers[b][0]), float(centers[b][1]), float(centers[b][2]))
            transformer = molgrid.Transform(center, 0, True)  # random rotation
            transformer.forward(batch[b], batch[b])
            gmaker.forward(center, batch[b].coord_sets[0], input_tensor[b])

        optimizer.zero_grad()
        output = model(input_tensor[:, :14])
        loss = criterion(output, labels)
        loss.backward()
        predicted = torch.argmax(output, dim=1)
        accuracy = labels.eq(predicted).sum().float() / batch_size
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradients)
        optimizer.step()

        last_train_loss = float(loss.item())
        last_train_acc  = float(accuracy.item())

        # periodic evaluation
        if i % test_interval == 0 and i != 0:
            model.eval()

            test_loss, test_accuracy, _, _, _ = test_model(model, eptest_small, gmaker, percent_reduced, batch_size)
            _, _, test_auc, test_labels, test_probs = test_model(model, eptest_large, gmaker, percent_reduced, batch_size)

            scheduler.step(test_auc)

            current_lr = float(optimizer.param_groups[0]['lr'])
            logger.log(
                time_iso=CsvLogger.now(),
                iter=i + 1,
                phase="test",
                train_loss=last_train_loss if last_train_loss is not None else "",
                train_accuracy=last_train_acc if last_train_acc is not None else "",
                test_loss=float(test_loss),
                test_accuracy=float(test_accuracy),
                test_auc=float(test_auc),
                learning_rate=current_lr,
                trial_number = getattr(args, "trial_number", "")
            )

            history["iter"].append(i + 1)
            history["train_loss"].append(last_train_loss if last_train_loss is not None else np.nan)
            history["train_acc"].append(last_train_acc if last_train_acc is not None else np.nan)
            history["test_loss"].append(float(test_loss))
            history["test_acc"].append(float(test_accuracy))

            if test_loss < Bests['test_loss']:
                Bests['test_loss'] = test_loss
                Bests['iter_at_best_loss'] = i + 1
                Bests['train_loss_at_best_loss'] = last_train_loss
                Bests['train_acc_at_best_loss']  = last_train_acc
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'Bests': Bests,
                            'Iteration': i + 1}, outprefix + f'_best_test_loss_{i+1}.pth.tar')
                if prev_total_loss_snap:
                    os.remove(prev_total_loss_snap)
                prev_total_loss_snap = outprefix + f'_best_test_loss_{i+1}.pth.tar'

            if test_accuracy > Bests['test_accuracy']:
                Bests['test_accuracy'] = test_accuracy
                Bests['iter_at_best_acc'] = i + 1
                Bests['train_loss_at_best_acc'] = last_train_loss
                Bests['train_acc_at_best_acc']  = last_train_acc
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'Bests': Bests,
                            'Iteration': i + 1}, outprefix + f'_best_test_accuracy_{i+1}.pth.tar')
                if prev_total_accuracy_snap:
                    os.remove(prev_total_accuracy_snap)
                prev_total_accuracy_snap = outprefix + f'_best_test_accuracy_{i+1}.pth.tar'

            if test_auc > Bests['test_auc']:
                Bests['test_auc'] = test_auc
                Bests['iter_at_best_auc'] = i + 1
                Bests['train_loss_at_best_auc'] = last_train_loss
                Bests['train_acc_at_best_auc']  = last_train_acc

                best_plot["auc"] = float(test_auc)
                best_plot["labels"] = test_labels
                best_plot["probs"]  = test_probs
                best_plot["iter"]   = i + 1

                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict(),
                            'Bests': Bests,
                            'Iteration': i + 1}, outprefix + f'_best_test_auc_{i+1}.pth.tar')
                if prev_total_auc_snap:
                    os.remove(prev_total_auc_snap)
                prev_total_auc_snap = outprefix + f'_best_test_auc_{i+1}.pth.tar'
                Bests['train_iteration'] = i

            # trigger early stop
            if (i - Bests['train_iteration'] >= args.step_when and
                optimizer.param_groups[0]['lr'] <= ((args.step_reduce) ** args.step_end_cnt) * args.base_lr):
                last_test = 1

            print("Iteration {}, train_loss: {:.3f}, train_acc: {:.3f}, "
                  "test_loss: {:.3f}, test_acc: {:.3f}, test_auc: {:.3f}, "
                  "Best_test_loss: {:.3f}, Best_test_accuracy: {:.3f}, Best_test_auc: {:.3f}, LR: {:.7f}".format(
                    i + 1, last_train_loss, last_train_acc,
                    float(test_loss), float(test_accuracy), float(test_auc),
                    float(Bests['test_loss']), float(Bests['test_accuracy']), float(Bests['test_auc']),
                    optimizer.param_groups[0]['lr']))

            
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'Bests': Bests,
                        'Iteration': i + 1}, outprefix + f'_{i+1}.pth.tar')
            if prev_snap:
                os.remove(prev_snap)
            prev_snap = outprefix + f'_{i+1}.pth.tar'

            model.train()

        if last_test:
            try:
                plot_learning_curves(history, outprefix)
            except Exception as _e:
                print(f"could not plot learning curves: {_e}", flush=True)

            if best_plot["labels"] is not None:
                plot_roc_curve(best_plot["labels"], best_plot["probs"], outprefix, best_plot["iter"])

            for k in ("test_loss", "test_accuracy", "test_auc"):
                try:
                    Bests[k] = float(Bests[k])
                except Exception:
                    try:
                        Bests[k] = float(Bests[k].item())
                    except Exception:
                        import numpy as _np
                        Bests[k] = float(_np.asarray(Bests[k]).astype(float))
            return Bests

    try:
        plot_learning_curves(history, outprefix)
    except Exception as e:
        print(f"could not plot learning curves: {e}", flush=True)

    if best_plot["labels"] is not None:
        plot_roc_curve(best_plot["labels"], best_plot["probs"], outprefix, best_plot["iter"])

    for k in ("test_loss", "test_accuracy", "test_auc"):
        try:
            Bests[k] = float(Bests[k])
        except Exception:
            try:
                Bests[k] = float(Bests[k].item())
            except Exception:
                import numpy as _np
                Bests[k] = float(_np.asarray(Bests[k]).astype(float))
    return Bests

def run_one(args):
    
    base = getattr(args, 'outprefix', None) or getattr(args, 'run_name', None) or "run"
    csv_path = getattr(args, 'csv_log', None) or f"{base}_metrics.csv"
    json_path = getattr(args, 'metrics_json', None) or f"{base}_summary.json"

    logger = CsvLogger(csv_path)

    model, gmaker, eptrain, eptest_large, eptest_small = get_model_gmaker_eproviders(args)
    Bests = train_and_test(args, model, eptrain, eptest_large, eptest_small, gmaker)
    if Bests is None:
        Bests = {"test_auc": 0.0, "test_accuracy": 0.0, "test_loss": float("inf")}

    logger.log(
        time_iso=CsvLogger.now(),
        iter=getattr(args, 'iterations', ''),
        phase='final',
        train_loss='',
        train_accuracy='',
        test_loss=float(Bests.get('test_loss', '')) if 'test_loss' in Bests else '',
        test_accuracy=float(Bests.get('test_accuracy', '')) if 'test_accuracy' in Bests else '',
        test_auc=float(Bests.get('test_auc', '')) if 'test_auc' in Bests else '',
        learning_rate=''
    )


    try:
        import json as _json
        payload = dict(Bests)
        payload["trial_number"] = getattr(args, "trial_number", None)
        with open(json_path, "w") as f:
            _json.dump(payload, f, indent=2)
        print(f"wrote {json_path}", flush=True)
    except Exception as _e:
        print(f"could not write metrics_json: {_e}", flush=True)
    return Bests

if __name__ == '__main__':
    (args, cmdline) = parse_args()

    if getattr(args, 'optuna', False):
        import copy, optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
        from optuna.trial import TrialState
        import os as os

        storage_url = os.environ.get("OPTUNA_STORAGE", None)
        study_name  = os.environ.get("OPTUNA_STUDY", "transfer-learning-hparam-search")

        def suggest_params(trial):
            p = {}
            p['iterations']   = trial.suggest_int('iterations', 10000, 100000, step=10000)
            p['batch_size']   = trial.suggest_categorical('batch_size', [8, 16, 24, 32])
            p['lr']           = trial.suggest_categorical('lr', [1e-3, 1e-4, 1e-5])
            p['weight_decay'] = trial.suggest_categorical('weight_decay', [0.0, 1e-5, 1e-4, 1e-3, 1e-2])
            p['dropout']      = trial.suggest_categorical('dropout', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            p['optimizer']    = trial.suggest_categorical('optimizer', ['adam','adamw','sgd'])
            p['lr_sched']     = trial.suggest_categorical('lr_sched', ['none','step','cosine'])
            if p['lr_sched'] == 'step':
                p['gamma']     = trial.suggest_float('gamma', 0.5, 0.99)
                p['step_size'] = trial.suggest_int('step_size', 1, 10)
            else:
                p['gamma'], p['step_size'] = 1.0, 0
            p['grad_accum']   = trial.suggest_categorical('grad_accum', [1, 2, 4])
            return p

        def objective(trial):
            a = copy.deepcopy(args)
            for k, v in suggest_params(trial).items():
                setattr(a, k, v)
            base = getattr(args, 'outprefix', None) or getattr(args, 'run_name', 'run')
            a.outprefix = f"{base}_trial{trial.number:03d}"
            a.trial_number = trial.number
            Bests = run_one(a)
            val = Bests.get('test_auc', None)
            if val is None:
                val = Bests.get('test_accuracy', 0.0)
            try:
                return float(val)
            except Exception:
                try:
                    return float(val.item())
                except Exception:
                    return float(np.asarray(val).astype(float))

        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=TPESampler(seed=42, multivariate=True, group=True),
            pruner=MedianPruner(n_startup_trials=8, n_warmup_steps=0),
            storage=storage_url,
            load_if_exists=True if storage_url else False,
        )
        study.optimize(objective,
                       n_trials=int(getattr(args, 'n_trials', 25)),
                       gc_after_trial=True,
                       show_progress_bar=True)

        completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
        if completed:
            print("\nBest Trial", flush=True)
            print(f"Value: {study.best_value:.5f}", flush=True)
            print("Params:", flush=True)
            for k, v in study.best_trial.params.items():
                print(f"  {k}: {v}", flush=True)
            with open("optuna_best.json", "w") as f:
                json.dump({
                    "best_value": study.best_value,
                    "best_params": study.best_trial.params,
                    "best_trial_number": study.best_trial.number
                }, f, indent=2)
            print("wrote optuna_best.json", flush=True)
            with open("optuna_trials.csv", "w") as f:
                f.write("number,state,value,params\n")
                for t in study.trials:
                    f.write(f"{t.number},{t.state.name},{t.value},{t.params}\n")
            import copy as _copy
            best_args = _copy.deepcopy(args)
            for k, v in study.best_trial.params.items():
                setattr(best_args, k, v)
            base = getattr(args, 'outprefix', None) or getattr(args, 'run_name', 'run')
            best_args.outprefix = f"{base}_best"
            best_args.trial_number = study.best_trial.number
            final_bests = run_one(best_args)
            print(f"\nbest config metrics: {final_bests}", flush=True)
        else:
            print("\nNo completed trials", flush=True)
    else:
        Bests = run_one(args)
        print(f"Bests: {Bests}", flush=True)