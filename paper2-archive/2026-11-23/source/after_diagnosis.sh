#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
py=/opt/conda/bin/python
exec 9>logs/follow_up.lock
flock -n 9 || exit 0
until "$py" -c 'import json,pathlib; p=pathlib.Path("diagnostics/C0/summary.json"); raise SystemExit(0 if p.is_file() and json.loads(p.read_text()).get("complete") else 1)'; do
    if ! kill -0 "$(cat logs/after_C0.pid)" 2>/dev/null; then
        echo "C0 diagnosis worker stopped before completion"; exit 2
    fi
    sleep 30
done
"$py" -Bu choose_from_val.py
arm=$("$py" -c 'import json; print(json.load(open("decision.json"))["arm"])')
if "$py" -Bu evaluate_test.py >> logs/C0.test.log 2>&1; then
    "$py" -Bu keep_results.py C0 test || echo "Test backup needs retry"
else
    echo "C0 Test evaluation needs inspection; the independently selected Val arm can proceed"
fi
if [[ "$arm" == NMS075 ]]; then
    "$py" -Bu nms_control.py >> logs/NMS075.log 2>&1
    "$py" -Bu keep_results.py NMS075 val
else
    if [[ ! -f records/"$arm"/trained.json ]]; then "$py" -Bu run.py train "$arm" >> "logs/$arm.train.log" 2>&1; fi
    "$py" -Bu run.py evaluate "$arm" >> "logs/$arm.val.log" 2>&1
    "$py" -Bu keep_results.py "$arm" val || echo "Candidate Val backup needs retry"
    "$py" -Bu diagnose.py --arm "$arm" --phase base >> "logs/$arm.diagnosis.log" 2>&1
    "$py" -Bu keep_results.py "$arm" diagnosis
fi
