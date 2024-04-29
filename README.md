# Real-time Super Resolution

The goal of this project is to provide SISR models for realtime video supre-resolution.

## Example
![example](examples/example.png)
model: anime_v1_3x, no over-sharpening, no over-smoothing.

## Benchmark
| Model      | Type              |  Input           | Output       | Device              | Latency (ms) |
|------------|-------------------|------------------|--------------|---------------------|--------------|
| anime v0.1 | Keras Saved Model | 640x360          | 1920x1080    | Apple M1 Max        |         8.47 |
| anime v0.1 | TfLite fp16       | 640x360          | 1920x1080    | SD 8Gen3 Adreno 750 |        27.80 |
