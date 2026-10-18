# LogSentinel Model Card

## Model overview

LogSentinel combines a calibrated anomaly detector with a root-cause classifier for short AI Infra log windows.

- Anomaly detector: TF-IDF/SVD text features + numeric features + HistGradientBoosting, followed by Platt scaling.
- Root-cause classifier: TF-IDF/SVD + numeric features + HistGradientBoosting over five fault classes.
- Input: one normalized log window plus latency, retry/restart counts, CPU/memory/disk utilization, error count, request count, and hour.
- Output: anomaly probability, binary decision, root-cause label, and root-cause confidence.

## Intended use

- Reproducible ML/MLOps portfolio project.
- Baseline for AI Infra log triage experiments.
- Teaching example for class imbalance, threshold selection, probability calibration, and FastAPI model serving.

Not intended for unattended production incident response. A real deployment needs validation on organization-specific logs, drift monitoring, access control, and human review.

## Training data

The repository generates 6,000 synthetic log windows with 18% anomalies. The data includes:

- five fault classes: OOM, image pull failure, network timeout, crash loop, disk pressure;
- hard normal negatives from batch jobs and maintenance windows;
- handled ERROR logs that should remain normal;
- about 38% of anomaly windows without an explicit fault line.

The synthetic generator is deterministic under a fixed seed. It is not a substitute for production data.

## Evaluation protocol

- Split: 60% train / 20% validation / 20% test, stratified by class.
- Threshold: selected only on the validation split by maximizing F1.
- Test labels are never used for model, feature, or threshold selection.

## Test metrics

| Metric | Baseline | Main model |
|---|---:|---:|
| Precision | 0.949 | 0.959 |
| Recall | 0.949 | 0.968 |
| F1 | 0.949 | 0.963 |
| PR-AUC | 0.989 | 0.994 |
| ROC-AUC | 0.997 | 0.999 |

Root-cause classification: macro-F1 = 0.981; top-2 accuracy = 1.000.

## Limitations

- Synthetic distributions cannot represent all real log formats and failure modes.
- The root-cause label set is closed; unknown faults need open-set handling.
- The current model does not model long-range temporal dependencies across windows.
- The reported numbers are benchmark results, not production SLAs.

## Next steps

- Add Drain3 template mining and OpenTelemetry/Kubernetes ingestion.
- Evaluate temporal models and change-point detection.
- Add drift monitoring, ONNX export, and experiment tracking.
