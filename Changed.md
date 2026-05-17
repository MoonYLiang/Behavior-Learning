````markdown
# Changelog

## v0.1.1

### New Features

- Added support for configurable numbers of U/C/T blocks:

  ```python
  num_u
  num_c
  num_t
````

This enables more flexible BL architectures and modular compositions.

* Added component-specific lambda initialization:

  ```python
  init_lambda_u
  init_lambda_c
  init_lambda_t
  ```

  allowing separate initialization of U-, C-, and T-block coefficients.

### Changes

* Changed the default recommended U-block activation from `tanh` to `"none"`.

  The `"none"` activation often improves optimization stability in deep BL architectures and behaves more similarly to identity-like mappings.

### Internal Refactoring

* Refactored `BLDeep` input validation and helper logic.
* Moved validation and preprocessing helper functions into `utils.py`.
* Simplified discrete task handling.

### Breaking Changes

* For discrete tasks, `score(x, y)` now only accepts `1D` class-index labels.

  One-hot labels are no longer accepted directly.

  Convert one-hot labels before calling the model:

  ```python
  y_idx = y_onehot.argmax(dim=1)
  ```