#!/usr/bin/env python3
"""
filter_miniskills.py — produce the RELEASED mini-skill sample/gallery from raw
generator output, applying explicit, documented filters. We deliberately do NOT
call this "semantic coherence"; it is answer/format-consistency filtering:

  1. reach-the-boxed-answer: drop rows whose reasoning doesn't reach their boxed
     answer (done upstream when the raw dump was taken).
  2. exclude known-ambiguous skills: `bit_scan_single` (boxes an answer the
     reasoning never derives) and the two transformation-style skills
     `trans_style_audit` / `trans_style_pick`, whose competing style
     interpretations can render the SAME displayed value (so the "which style?"
     question is not always uniquely answerable). Until the generator enforces
     "exactly one interpretation matches", these are excluded from the release.
  3. de-duplicate: drop exact-duplicate (user,assistant) message pairs.

Run: python3 analysis/filter_miniskills.py   (rewrites the sample + gallery blob)
"""
import json, re, os, collections

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
SAMPLE = os.path.join(ROOT, "miniskills_sample.jsonl")
GALLERY = os.path.join(ROOT, "miniskills.html")
EXCLUDE = {"bit_scan_single", "trans_style_audit", "trans_style_pick"}

def filter_sample():
    rows = [json.loads(l) for l in open(SAMPLE)]
    seen = set(); out = []; dropped_skill = dropped_dup = 0
    for r in rows:
        if r.get("skill_name") in EXCLUDE:
            dropped_skill += 1; continue
        k = tuple((m.get("role"), m.get("content")) for m in r.get("messages", []))
        if k in seen:
            dropped_dup += 1; continue
        seen.add(k); out.append(r)
    with open(SAMPLE, "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    skills = {r.get("skill_name") for r in out}
    print(f"sample: kept {len(out)} (dropped {dropped_skill} excluded-skill, "
          f"{dropped_dup} duplicate); {len(skills)} distinct skills")
    return len(out), len(skills)

def filter_gallery():
    html = open(GALLERY).read()
    m = re.search(r'(<script[^>]*id="data"[^>]*>)(.*?)(</script>)', html, re.S)
    data = json.loads(m.group(2))
    kept = [d for d in data if d.get("skill") not in EXCLUDE]
    # de-dup on (skill, user, trace)
    seen = set(); dd = []
    for d in kept:
        k = (d.get("skill"), d.get("u"), d.get("t"))
        if k in seen: continue
        seen.add(k); dd.append(d)
    new = m.group(1) + json.dumps(dd, ensure_ascii=False) + m.group(3)
    open(GALLERY, "w").write(html[:m.start()] + new + html[m.end():])
    skills = {d.get("skill") for d in dd}
    print(f"gallery: kept {len(dd)} cards; {len(skills)} distinct skills")
    return len(skills)

if __name__ == "__main__":
    filter_sample(); filter_gallery()
