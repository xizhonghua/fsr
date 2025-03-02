# Real-time Super Resolution

The goal of this project is to provide SISR models for realtime video supre-resolution.

## Benchmark
| Model       | Type              |  Input           | Output       | Device              | Latency (ms) |
|-------------|-------------------|------------------|--------------|---------------------|--------------|
| anime v1 3x | Keras SavedModel  | 640x360          | 1920x1080    | Apple M1 Max        |         8.01 |
| anime v1 3x | Keras SavedModel  | 1280x720         | 3840x2160    | Apple M1 Max        |        32.31 |
| anime v1 3x | TfLite fp16       | 640x360          | 1920x1080    | SD 8Gen3 Adreno 750 |        20.05 |

## Demo Video
<a href="https://www.youtube.com/embed/L2p6j3Epypg?si=mdG_No4Dyhc-bQqp"><img src="https://i3.ytimg.com/vi/L2p6j3Epypg/maxresdefault.jpg"></a>

## Example
<img src="examples/suzume_lr.png" width="384" /><img src="examples/suzume_sr.png" width="384" />

<img src="examples/ff7_lr.png" width="384" /><img src="examples/ff7_sr.png" width="384" />

model: anime_v1_3x, no over-sharpening, no over-smoothing.

