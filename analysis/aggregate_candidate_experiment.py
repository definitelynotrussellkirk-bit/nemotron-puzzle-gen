#!/usr/bin/env python3
"""
aggregate_candidate_experiment.py — reproduce the BIT candidate-generation
dissociation numbers from the released per-row evidence file
(candidate_experiment_rows.jsonl).

This is the row-level evidence behind the headline result: on two fixed 64-item
BIT evals the model GENERATES an 8-candidate slate from the examples, then picks.
We report parse rate, gold-recall@8, and conditional selection (pick | gold in slate).

Run: python3 analysis/aggregate_candidate_experiment.py
"""
import json, os, collections

HERE = os.path.dirname(__file__)
ROWS = os.path.join(HERE, "candidate_experiment_rows.jsonl")

def main():
    rows = [json.loads(l) for l in open(ROWS)]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["condition"]].append(r)
    for cond, rs in by.items():
        n = len(rs)
        parse = sum(r["parse_ok"] for r in rs)
        recall = sum(r["gold_in_candidate_set"] for r in rs)
        present = [r for r in rs if r["gold_in_candidate_set"]]
        sel_given = sum(r["selection_correct"] for r in present)
        e2e = sum(r["selection_correct"] for r in rs)
        print(f"== {cond}  (n={n}) ==")
        print(f"  parse_ok                         : {parse}/{n}")
        print(f"  gold_in_candidate_set (recall@8) : {recall}/{n}")
        print(f"  selection_correct | gold present : {sel_given}/{len(present)}")
        print(f"  end-to-end exact                 : {e2e}/{n}")
        cnts = collections.Counter(r["candidate_count"] for r in rs)
        print(f"  candidate_count distribution     : {dict(cnts)}")
    # assertions: the published headline numbers
    m = {c: rs for c, rs in by.items()}
    def agg(c):
        rs = m[c]; pres = [r for r in rs if r["gold_in_candidate_set"]]
        return (sum(r["parse_ok"] for r in rs), len(rs),
                sum(r["gold_in_candidate_set"] for r in rs),
                sum(r["selection_correct"] for r in pres), len(pres))
    if "mask_qset_k8" in m:
        p,n,rec,sg,pr = agg("mask_qset_k8")
        assert (p,n)==(64,64), (p,n); assert rec==50, rec; assert (sg,pr)==(50,50), (sg,pr)
    if "qset_only_k8" in m:
        p,n,rec,sg,pr = agg("qset_only_k8")
        assert (p,n)==(64,64), (p,n); assert rec==51, rec; assert (sg,pr)==(51,51), (sg,pr)
    print("\nassertions passed: published parse/recall/selection counts reproduce.")

if __name__ == "__main__":
    main()
