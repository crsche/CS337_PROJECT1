import re
import json
import sys
import spacy




def _strip_handles(s):
    out = []
    for w in s.split():
        w = w.strip()
        
        if not w:
            continue
        if w.lower() == "rt":
            continue
        if w[0] in {"@", "#"}:
            continue
        out.append(w)
    return " ".join(out)

def _denoise(s):
    for bad in ("rt", "golden", "globes"):
        s = s.replace(bad, "")
    return s

def _person_pairs(doc):
    names = []
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        txt = ent.text.strip()
        if not txt:
            continue
        parts = txt.split()
        if len(parts) < 2:
            continue
        cand = " ".join(parts[:2])
        if cand.replace(" ", "").isalpha():
            names.append(cand)
    return names



def _score_list(d):
    rows = []
    for k, (pos, neg) in d.items():
        tot = pos + neg
        if tot == 0:
            continue
        pol = abs(pos / tot - 0.5)
        rows.append([k, pos, neg, pol])
    rows = [r for r in rows if not (r[1] < 5 and r[2] < 5)]
    return rows

def _top_n(seq, key, n=5, reverse=True):
    seq.sort(key=key, reverse=reverse)
    head = [r[0] for r in seq[:n]]
    return head, (seq[0][1], seq[0][2]) if seq else (0, 0)

def run_redcarpet(year):
    nlp = spacy.load("en_core_web_sm")
    with open(f"gg{year}.json", encoding="utf-8") as f:
        data = json.load(f)

    bt, wt = [], []
    for it in data:
        t = _strip_handles(it.get("text", "").lower())
        if not t:
            continue
        has_best = "best dressed" in t
        has_worst = "worst dressed" in t
        if has_best and has_worst:
            continue
        if has_best:
            bt.append(_denoise(t))
        elif has_worst:
            wt.append(_denoise(t))

    tally = {}
    for tw in wt:
        for name in _person_pairs(nlp(tw)):
            if name not in tally:
                tally[name] = [0, 0]
            tally[name][1] += 1
    for tw in bt:
        for name in _person_pairs(nlp(tw)):
            if name not in tally:
                tally[name] = [0, 0]
            tally[name][0] += 1

    table = _score_list(tally)

    best_list, best_votes = _top_n(table[:], key=lambda r: r[1], n=5, reverse=True)
    worst_list, worst_votes = _top_n(table[:], key=lambda r: r[2], n=5, reverse=True)
    controversial_list, controversy = _top_n(table[:], key=lambda r: r[3], n=5, reverse=False)

    y = str(year)
    print(f"The five best dressed of the {y} Golden Globes were:")
    for i, name in enumerate(best_list[:5], 1):
        print(f"{i}. {name}")
    if best_list:
        print(f"The single best dressed red carpeter was {best_list[0]} with {best_votes[0]} votes for best dressed.")
    else:
        print("The single best dressed red carpeter was NA with 0 votes for best dressed.")
    print("")

    print(f"The five worst dressed of the {y} Golden Globes were:")
    for i, name in enumerate(worst_list[:5], 1):
        print(f"{i}. {name}")
    if worst_list:
        print(f"The single worst dressed red carpeter was {worst_list[0]} with {worst_votes[1]} votes for worst dressed.")
    else:
        print("The single worst dressed red carpeter was NA with 0 votes for worst dressed.")
    print("")

    print(f"The five most controversial red carpeters of the {y} Golden Globes were:")
    for i, name in enumerate(controversial_list[:5], 1):
        print(f"{i}. {name}")
    if controversial_list:
        print(f"The most controversial red carpeter was {controversial_list[0]} with {controversy[0]} votes for best dressed and {controversy[1]} votes for worst dressed.")
    else:
        print("The most controversial red carpeter was NA with 0 votes for best dressed and 0 votes for worst dressed.")
    print("")

if __name__ == "__main__":
    run_redcarpet(sys.argv[1] if len(sys.argv) > 1 else "2013")
