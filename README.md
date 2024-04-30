# Real-time Super Resolution

The goal of this project is to provide SISR models for realtime video supre-resolution.

## Benchmark
| Model       | Type              |  Input           | Output       | Device              | Latency (ms) |
|-------------|-------------------|------------------|--------------|---------------------|--------------|
| anime v1 3x | Keras SavedModel  | 640x360          | 1920x1080    | Apple M1 Max        |         8.01 |
| anime v1 3x | Keras SavedModel  | 1280x720         | 3840x2160    | Apple M1 Max        |        32.31 |
| anime v1 3x | TfLite fp16       | 640x360          | 1920x1080    | SD 8Gen3 Adreno 750 |        20.05 |

## Example
![example](examples/suzume.png)
![example](examples/ff7_rebirth.png)
model: anime_v1_3x, no over-sharpening, no over-smoothing.

