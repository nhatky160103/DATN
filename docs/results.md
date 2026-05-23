# Experimental Results

This page summarizes the research results used to guide model selection and threshold design.

## Recognition Accuracy

| Training setup | CFP-FP | CPLFW | AgeDB | CALFW | LFW |
| --- | ---: | ---: | ---: | ---: | ---: |
| MS1MV3, R100, ArcFace | 98.79 | 93.21 | 98.23 | 96.02 | 99.83 |
| IBUG500K, R100, ArcFace | 98.87 | 93.43 | 98.38 | 96.10 | 99.83 |
| MS1MV3, R100, CDML | 98.94 | 94.08 | 97.75 | 96.05 | 99.85 |
| MS1MV3, lightweight, CDML + distillation | 97.98 | 92.48 | 97.03 | 95.55 | 99.76 |

## Lightweight Backbone Comparison

| Dataset | r50_lite | r100_lite |
| --- | ---: | ---: |
| LFW | 99.47 ± 0.37 | 99.67 ± 0.27 |
| CFP-FP | 92.87 ± 1.41 | 92.83 ± 1.92 |
| CFP-FF | 99.57 ± 0.34 | 99.63 ± 0.31 |
| CALFW | 95.32 ± 1.02 | 95.10 ± 1.27 |
| CPLFW | 88.83 ± 1.63 | 89.08 ± 1.85 |
| AgeDB-30 | 96.35 ± 0.94 | 95.95 ± 0.91 |

## IJB-B and IJB-C

| Model | Dataset | 1e-6 | 1e-5 | 1e-4 | 0.001 | 0.01 | 0.1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| r100_lite | IJB-B | 36.85 | 83.57 | 91.15 | 94.30 | 96.87 | 98.33 |
| r100_lite | IJB-C | 83.06 | 89.50 | 93.13 | 95.56 | 97.59 | 98.82 |
| r50_lite | IJB-B | 36.11 | 83.61 | 90.83 | 94.35 | 96.81 | 98.49 |
| r50_lite | IJB-C | 83.78 | 89.42 | 93.15 | 95.64 | 97.65 | 98.88 |

## Inference Cost

| Backbone | Parameters | Size | CPU latency / image | GFLOPs |
| --- | ---: | ---: | ---: | ---: |
| R18 | 24.03M | 91.65 MB | 46.40 ms | 2.63 |
| R34 | 34.14M | 130.20 MB | 74.92 ms | 4.48 |
| R50 | 43.59M | 166.28 MB | 108.02 ms | 6.33 |
| R100 | 65.16M | 248.55 MB | 194.91 ms | 12.13 |
| R18_lite | 9.22M | 35.70 MB | 16.82 ms | 0.67 |
| R50_lite | 14.12M | 53.87 MB | 39.39 ms | 1.60 |
| R100_lite | 19.52M | 74.47 MB | 79.10 ms | 3.05 |
| R_lightweight | 5.09M | 19.43 MB | 8.71 ms | 0.098 |

## Pipeline Model Latency

| Model | Parameters | CPU latency / image | Size |
| --- | ---: | ---: | ---: |
| MTCNN | 495,850 | 289.60 ms | 446.21 MB |
| FASNet | 868,146 | 35.93 ms | 211.59 MB |
| LightQNet | 130,915 | 11.17 ms | 444.84 MB |

## Threshold Findings

The research dataset contained 1,131 Vietnamese celebrity identities and more than 18,000 images.

Observed behavior:

- Accuracy and TAR peaked near a cosine-distance threshold around `0.70`.
- FAR increased when the threshold became too permissive.
- FRR decreased when the system accepted more pairs.
- Precision stayed high while recall changed with the expected trade-off.

Production note:

The current runtime uses Qdrant cosine similarity, where higher score is better. Calibrate `qdrant.match_threshold` from the active production embeddings instead of directly copying a distance threshold from offline experiments.

## Error Analysis

Common false reject causes:

- large pose change;
- blur or low quality;
- age difference;
- expression change;
- color/black-white domain difference.

Common false accept causes:

- similar facial structure;
- similar hairstyle or makeup;
- similar pose and expression;
- permissive threshold.

Operational mitigation:

- enroll multiple images per identity;
- keep quality filtering enabled;
- use track-level aggregation;
- keep `UNKNOWN` as a valid decision;
- calibrate threshold on deployment data.
