"""Complaint taxonomy for 1 and 2 star review text (Portuguese keyword rules, no model).

Each rule was checked by reading 25 random reviews it matched, plus 50 bad reviews that matched no rule
(analysis/iter2/taxonomy_check.py; verdicts in analysis/iter2/out/taxonomy_check.csv). Every tag below scored 80%
precision or better, the bar for showing it in the app. The rules are conservative: 38 of the 50 unmatched reviews did
describe one of these problems in other words, so every share computed from the tags is a lower bound.

After the check, two word boundary bugs were fixed ("peso veio" matched "so veio"; "uma qualidade" matched
"ma qualidade"), the generic "não veio a/o" rule was dropped ("não veio a cor" is a wrong item), and a partial delivery
is no longer also tagged "Not received". These changes only remove matches, one or more of them wrong, so precision can only rise
so the measured figures in PRECISION are conservative."""
import pandas as pd

TAGS = {
    "Not received": r"n[aã]o (?:recebi|chegou|foi entregue|entreg)|ainda n[aã]o (?:recebi|chegou)|nunca chegou|n[aã]o recebemos",
    "Late or slow delivery": r"atras|demor|fora do prazo|passou do prazo|ap[oó]s o prazo|depois do prazo"
                             r"|prazo.{0,25}(?:venc|expir|esgot|ultrapass|estour)|n[aã]o.{0,30}(?:no|dentro do) prazo",
    "Missing item": r"falt(?:ou|ando|a) (?:um|uma|o |a |\d|item|produto|pe[cç]a)|\bapenas (?:um|uma|1)\b|\bs[oó] (?:um|uma|1|veio|recebi)\b"
                    r"|\bsomente (?:um|uma|1)\b|incomplet|\bmetade\b|outra parte|\bum dos\b|\bveio s[oó]\b|\brecebi s[oó]\b",
    "Wrong item or not as described": r"errad|diferente|outro produto|trocad|n[aã]o (?:era|é) o",
    "Defect or quality": r"defeit|quebrad|danific|n[aã]o funciona|parou de funcionar|estragad|rasgad|amassad|falsific|vazamento|vazand"
                         r"|riscad|\b(?:m[aá]|baixa|p[eé]ssima|ruim|sem|nenhuma) qualidade|qualidade (?:ruim|baixa|p[eé]ssima|inferior|horr)",
    "Refund or return": r"devol|reembols|estorn|dinheiro de volta",
}
PRECISION = {"Not received": 0.84, "Late or slow delivery": 0.96, "Missing item": 0.80, "Wrong item or not as described": 0.92,
             "Defect or quality": 0.92, "Refund or return": 1.00}
# What the customer complains about (refund or return is what they ask for, so it is not a cause).
CAUSE_TAGS = ["Not received", "Late or slow delivery", "Missing item", "Wrong item or not as described", "Defect or quality"]
SELLER_SIDE_TAGS = ["Missing item", "Wrong item or not as described", "Defect or quality"]
TAG_COLS = {"Not received": "t_not_received", "Late or slow delivery": "t_late", "Missing item": "t_missing",
            "Wrong item or not as described": "t_wrong", "Defect or quality": "t_defect", "Refund or return": "t_refund"}


def tag_reviews(text: pd.Series) -> pd.DataFrame:
    """Boolean column per tag; rows without text are all False."""
    t = text.fillna("").str.lower()
    out = pd.DataFrame({k: t.str.contains(p, regex=True) for k, p in TAGS.items()}, index=text.index)
    out["Not received"] &= ~out["Missing item"]          # "only one of the two arrived" is a missing item, not a lost order
    return out
