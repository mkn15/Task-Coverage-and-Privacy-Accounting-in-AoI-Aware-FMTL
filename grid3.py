import numpy as np, json
from multiprocessing import Pool
from fmtl import run, n
import acct2
R = 150; DELTA = 1e-5; KS = [5, 10, 20, 30, 40, 60, 80, 95]; EPS0 = [32.0, 16.0, 8.0]; SEEDS = range(5)
SIG = {(e, k): acct2.sigma_visible(k, n, R, e, DELTA) for e in EPS0 for k in KS}
def job(a):
    e, k, s = a; return (e, k, s, run(k, SIG[(e, k)], R, s))
if __name__ == "__main__":
    jobs = [(e, k, s) for e in EPS0 for k in KS for s in SEEDS]
    with Pool() as p: res = p.map(job, jobs)
    cur = {}
    for e, k, s, c in res: cur.setdefault((e, k), []).append(c)
    json.dump(dict(R=R, delta=DELTA, ks=KS, model="visible", eps0=[str(e) for e in EPS0],
        sigma={f"{e}_{k}": v for (e, k), v in SIG.items()},
        mean={f"{e}_{k}": np.mean(v, 0).tolist() for (e, k), v in cur.items()},
        std={f"{e}_{k}": np.std(v, 0).tolist() for (e, k), v in cur.items()}), open("grid3.json", "w"))
    for e in EPS0: print(e, [(k, round(float(np.mean(cur[(e, k)], 0)[-1]), 3)) for k in KS])
