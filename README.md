# Flames
A mostly straightforward implementation of the Fractal Flames algorithm from https://flam3.com/flame_draves.pdf.

## Notable Extensions
 * Instead of iterating a single point, 262,144 points are iterated in parallel on the GPU. 
 * Arbitrary variations can be created using GLSL with support for float and image parameters.
 * In addition to static function probabilities, the probability of each function may depend on the previous function forming a Markov Chain.

## Setup
Install `uv` (see https://docs.astral.sh/uv/getting-started/installation/).

(Optional) `ffmpeg` is required for rendering animations.

## Example Usage
```bash
# Edit a flame
uv run editor examples/flames/triangle.json

# Re-render all example flames
uv run render --destination examples/renders/ -- examples/flames/*.json

# Render the example animation (requires ffmpeg)
uv run animation --frames 50 examples/flames/dragon.json examples/rotate_animation.py
```

## TODO
 * Improve `shuffler` backend performance. I think it might be able to improve on the performance of `basic` by avoiding predicate masks.
 * Get rid of multiple variations within a single function. It just overcomplicates things and doesn't generally produce good results when used.
 * Write a less janky editor.

## Examples
![](/examples/renders/image.png)
![](/examples/renders/spike.png)
![](/examples/renders/swirl_variation.png)