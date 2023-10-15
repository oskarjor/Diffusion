# Diffusion

## Different version
- V1: baseline. Linear scheduler, 200 timesteps. Trained for up to 300 epochs
- V2: similar to V1, but with a cosine scheduler. Trained for up to 400 epochs.
- V3: experimenting with fewer timesteps to increase training speed while maintaining quality. Reduced timesteps to 50. Also reduced batch size to see if training speed is further improved.