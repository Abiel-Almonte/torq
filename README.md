```python
import torq as tq

# Compose a multi-camera fusion pipeline
system = tq.Sequential(
    tq.Concurrent(
        tq.Sequential(rgb_camera, rgb_preprocess, rgb_model),
        tq.Sequential(depth_camera, depth_preprocess, depth_model)
    ),
    fusion_model,
    console_writer
)

# Build DAG, optimize lazily
system = tq.compile(system)
system.run()
```