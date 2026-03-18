import torch
import sys

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        # result should be either a scalar (e.g., MRR) or a (valid, test) pair
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, f=sys.stdout, last_best=False):
        if run is not None:
            # ---- per-run ----
            if len(self.results[run]) == 0:
                print('No results recorded for this run.', file=f)
                return

            result_tensor = torch.tensor(self.results[run])

            # (valid, test) pairs
            if result_tensor.ndim == 2 and result_tensor.size(1) >= 2:
                result = 100 * result_tensor
                if last_best:
                    argmax = result.size(0) - result[:, 0].flip(dims=[0]).argmax().item() - 1
                else:
                    argmax = result[:, 0].argmax().item()
                print(f'Run {run + 1:02d}:', file=f)
                print(f'Highest Valid: {result[:, 0].max():.2f}', file=f)
                print(f'Highest Eval Point: {argmax + 1}', file=f)
                print(f'   Final Test: {result[argmax, 1]:.2f}', file=f)
                return

            # scalar metrics (e.g., a single MRR value per eval)
            if result_tensor.ndim == 1 and result_tensor.numel() > 0:
                result = 100 * result_tensor
                print(f'Run {run + 1:02d}:', file=f)
                print(f'Highest Score: {result.max().item():.2f}', file=f)
                print(f'   Final Score: {result[-1].item():.2f}', file=f)
                return

            # anything else -> skip
            print('Results have unexpected shape; skipping.', file=f)
            return

        # ---- all runs summary ----
        pair_best = []   # (valid, test)
        scalar_best = [] # single numbers

        any_data = False
        for res_list in self.results:
            if len(res_list) == 0:
                continue
            any_data = True
            t = torch.tensor(res_list)

            if t.ndim == 2 and t.size(1) >= 2:
                r = 100 * t
                if last_best:
                    argmax = r.size(0) - r[:, 0].flip(dims=[0]).argmax().item() - 1
                else:
                    argmax = r[:, 0].argmax().item()
                pair_best.append((r[:, 0].max().item(), r[argmax, 1].item()))
            elif t.ndim == 1 and t.numel() > 0:
                r = 100 * t
                scalar_best.append(r.max().item())

        if not any_data:
            print('No results to report.', file=f)
            return

        print('All runs:', file=f)
        if pair_best:
            pb = torch.tensor(pair_best)
            v = pb[:, 0]; te = pb[:, 1]
            print(f'Highest Valid: {v.mean():.2f} ± {v.std(unbiased=False):.2f}', file=f)
            print(f'   Final Test: {te.mean():.2f} ± {te.std(unbiased=False):.2f}', file=f)

        if scalar_best:
            sb = torch.tensor(scalar_best)
            print(f'Highest Score: {sb.mean():.2f} ± {sb.std(unbiased=False):.2f}', file=f)
