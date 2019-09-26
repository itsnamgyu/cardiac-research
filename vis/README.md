## Changes made from 1eec6f (commit 193)

By Namgyu Ho

- When `layer_idx` of `visualize_cam` points to a `Sequential` model, an error occurs
  while trying to get the output of this layer. To solve this, we added custom logic
	in `vis.losses.ActivationMaximization.build_loss` to handle `Sequential`-based layers.
