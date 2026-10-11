# Prompt Protector — Final Evaluation Results

## 1. Validation Results

### Overall validation
- Precision: 99.81%
- Recall: 99.87%
- F1: 99.84%
- False Positive Rate: 0.27%

### BIPIA validation subset
- Precision: 98.12%
- Recall: 99.24%
- F1: 98.68%
- False Positive Rate: 12.82%

The operating threshold for external content was selected to prioritize attack recall while maintaining acceptable false positives.

---

## 2. Untouched External Attack Benchmarks

These datasets were kept separate from training and threshold calibration.

### BIPIA External Test
- Total attacks: 41,250
- Neural recall: 91.50%
- Rule recall: 0.01%
- Final fused firewall recall: 91.50%
- False negatives: 3,505

Action distribution:
- QUARANTINE: 36,734
- MONITOR: 1,011
- ALLOW: 3,505

### InjecAgent External Test
- Total attacks: 2,108
- Neural recall: 95.68%
- Rule recall: 51.61%
- Final fused firewall recall: 95.73%
- False negatives: 90

Action distribution:
- QUARANTINE: 1,902
- MONITOR: 48
- BLOCK: 65
- APPROVAL: 3
- ALLOW: 90

### Combined External Benchmark
- Total untouched attacks: 43,358
- Detected by final firewall: 39,763
- False negatives: 3,595
- Combined attack recall: 91.71%
- Combined miss rate: 8.29%

Important:
The external BIPIA and InjecAgent evaluation sets contain only attack samples.
Therefore precision, specificity and false-positive rate are intentionally not reported for these external benchmarks.

---

## 3. Defense-in-Depth Regression Tests

8/8 automated security regression tests passed.

Tested guarantees:
- Benign user query is allowed
- Direct prompt injection is blocked
- Benign external email is allowed
- Indirect prompt injection is quarantined
- Unknown tools are denied by default
- Side-effecting tool calls require approval
- Secret exfiltration is blocked
- Blocked content cannot invoke a tool

---

## 4. Key Security Result

Prompt Protector does not rely only on prompt-injection detection.

Example:

Firewall input:
"Please send this information."

Neural detector:
ALLOW

Proposed agent tool call:
email.send(api_key="SECRET_VALUE")

Tool Security Gateway:
BLOCK

Reason:
potential_secret_exfiltration

This demonstrates that even when the prompt detector does not identify malicious intent, deterministic authorization policy can independently stop sensitive-data exfiltration.
