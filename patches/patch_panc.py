#!/usr/bin/env python
"""Point BCP's Pancreas loader at this machine's data lists.

`code/pancreas/dataloaders.py` hardcodes the original author's path for the CoraNet split lists:

    '/home/ubuntu/byh/code/CoraNet-master/data_lists'

Patched to read PANCREAS_LISTS, defaulting to the original string so the file still describes
upstream's behaviour. Same treatment as D9 (the hardcoded LA paths in test_3d_patch.py); no
logic change, the same three files are read in the same order.

Note on the split: CoraNet publishes a single pancreas split (12 labeled / 50 unlabeled /
18 test, i.e. the 20% setting). It does not ship a 10% split, so only `20percent` exists here
and the 10% row of BCP's table cannot be reproduced without inventing a split, which would
break comparability with every published number.
"""
import sys

P = "/dcs/large/u5535699/bcp-official/code/pancreas/dataloaders.py"
s = open(P).read()
if "_PANC_LISTS" in s:
    print("already patched")
    sys.exit(0)

old = ("    return ['/'.join(['/home/ubuntu/byh/code/CoraNet-master/data_lists', "
       "dataset, labelp, f]) for f in files]")
new = ("    return ['/'.join([_PANC_LISTS, dataset, labelp, f]) for f in files]")
assert old in s, "hardcoded data_lists path not found verbatim"
s = s.replace(old, new)

lines = s.split("\n")
i = max(j for j, l in enumerate(lines[:40]) if l.startswith(("import ", "from ")))
lines.insert(i + 1,
             '\nimport os as _os\n'
             '_PANC_LISTS = _os.environ.get(\n'
             '    "PANCREAS_LISTS", "/home/ubuntu/byh/code/CoraNet-master/data_lists")')
open(P, "w").write("\n".join(lines))
print("patched pancreas/dataloaders.py -> PANCREAS_LISTS")
