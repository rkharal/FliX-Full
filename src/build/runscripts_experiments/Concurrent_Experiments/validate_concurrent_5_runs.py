#!/usr/bin/env python3
"""Validate a five-run mixed-batch benchmark's CSV and output log."""
import csv, math, re, sys
from pathlib import Path

csv_path, log_path, variant, ins, dele, query, build, rounds = sys.argv[1:]
ins, dele, query, build, rounds = map(int, (ins, dele, query, build, rounds))
indexes = {'FliX-ST':'cg_rtx_index_updates_3', 'FliX-TL':'cg_rtx_index_updates_5',
           'GPU_BTREE':'tree_awad_full', 'SLABHASH':'hashtable_slab'}
with open(csv_path, newline='') as f:
    rows = list(csv.DictReader(f, skipinitialspace=True))
assert rows, 'Empty results CSV'
latencies, totals = {}, {}
for row in rows:
    assert row['index'] == indexes[variant] and row['distribution'] == 'uniform'
    assert int(row['build_size']) == build and int(row['batch_size']) == 100_000_000
    assert tuple(int(row[k]) for k in ('Insert_mix','Del_mix','Query_mix')) == (ins,dele,query)
    assert row['validated'].lower() in ('1','true') and row['rebuild_enabled'].lower() in ('0','false')
    run=int(row['run']); assert run in range(5)
    if row['record']=='round' and row['DESCRIPTION']=='mixed_kernel_ms':
        rnd=int(row['round']); key=(run,rnd); assert rnd in range(1,rounds+1) and key not in latencies
        insert_count=1_000_000*ins; delete_count=1_000_000*dele; query_count=1_000_000*query
        hits=min(delete_count//2,build//2) if build==8_388_608 and rnd==1 else delete_count//2
        first=min(delete_count//2,build//2) if build==8_388_608 else delete_count//2
        expected=(insert_count,hits,delete_count-hits,query_count,build+rnd*insert_count-first-(rnd-1)*(delete_count//2))
        assert tuple(int(row[k]) for k in ('insert_count','delete_hit_count','delete_miss_count','query_count','live_keys_after'))==expected
        ms=float(row['VALUE']); assert math.isfinite(ms) and ms>0
        latencies[key]=ms
    if row['record']=='summary' and row['DESCRIPTION']=='total_mixed_kernel_ms':
        assert run not in totals and int(row['rounds'])==rounds
        totals[run]=float(row['VALUE'])
assert set(latencies)=={(run,rnd) for run in range(5) for rnd in range(1,rounds+1)}
assert set(totals)==set(range(5))
for run in range(5):
    assert math.isclose(totals[run],sum(latencies[run,r] for r in range(1,rounds+1)),rel_tol=1e-4,abs_tol=0.02)
log=Path(log_path).read_text(errors='replace')
markers=re.findall(r'\| run=(\d+)/5 \| round=(\d+)/(\d+) .*?operation_results=PASS \| post_state=PASS',log)
assert set((int(run),int(rnd),int(total)) for run,rnd,total in markers)=={(run,rnd,rounds) for run in range(1,6) for rnd in range(1,rounds+1)}
assert len(markers)==5*rounds and log.count(f'all {rounds} rounds validated.')==5
assert 'FliX mixed reuse:' not in log
print(f'PASS {variant} {ins}/{dele}/{query}: five runs, {rounds} rounds each validated')
