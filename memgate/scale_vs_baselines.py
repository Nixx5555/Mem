"""Held-out learned scorer (PR #12 recipe, seed 0) against RAG and P1 at the
paper's budgets, from results/scale/replays.jsonl, with clustered 95% CIs.
The numbers in docs/SCALE_TEST.md section 2."""
import json, sys, numpy as np

from train_transfer import cluster_bootstrap
rows=[json.loads(l) for l in open('results/scale/replays.jsonl') if l.strip()]
by={}
for r in rows:
    if r['seed']==0: by.setdefault((r['corpus'],r['S'],r['setup']),{})[r['unit']]=r
main={'locomo':4096,'realtalk':4096,'beam':26000,'beam500':26000,'beam1m':26000,'beam10m':26000,'mb1':2048}
for m in ('answer','soft'):
    for c,S in main.items():
        out=[]
        for a_s,b_s in (('pr12','rag'),('pr12','heuristic')):
            A=by[(c,S,a_s)]; B=by[(c,S,b_s)]; units=sorted(A)
            a=np.array([np.nan if x is None else float(x) for u in units for x in A[u][m]])
            b=np.array([np.nan if x is None else float(x) for u in units for x in B[u][m]])
            conv=np.array([i for i,u in enumerate(units) for _ in A[u][m]])
            o,lo,hi=cluster_bootstrap(a,b,conv)
            out.append(f"vs {b_s}: {o:+.1f} [{lo:+.1f}, {hi:+.1f}]{' SIG' if lo>0 or hi<0 else ''}")
        print(m,c,S,' | '.join(out))
