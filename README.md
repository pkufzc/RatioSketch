RatioSketch - Experiment code for learned sketch correction

Overview

This repository contains code for experiments combining classical
streaming sketches (e.g. Count-Min Sketch) with a learned decoder that
predicts multiplicative correction coefficients. The goal is to improve
frequency estimation accuracy under memory constraints.

Key components

- SourceCode/RatioSketch.py: Wrapper combining base sketch, decoder and loss.
- SourceCode/ModelModule/: Decoder, loss and hashing helpers.
- SourceCode/Sketches/: Implementations for CMS, TCMS, LegoSketch, MetaSketch.
- SourceCode/TaskModule/: Task generation (Zipf sampling) and test-task
  preparation (multiprocess loading).
- Dataset/: Helpers and expected .npz data files containing 'items' and 'freqs'.



Data preparation

The code expects dataset files named like `<dataset>_freqs_train.npz`
and `<dataset>_freqs_test.npz` with arrays 'items' and 'freqs'. By
default `Dataset/DataLoader.py` contains a `base_dir` pointing to the
original author's path. Update that variable or set up your dataset
files under a directory and modify `load_dataset_cache` accordingly.

Quick start

```powershell
python Ratio_seed0.py
```

The code currently only runs tests on real datasets. To run successfully,
please download the appropriate datasets and prepare them under the
`Dataset` folder. Testing on synthetic ZIPF datasets is not currently
supported.

Evaluation notes

- If you want to run evaluations that use pretrained decoders, the code
  includes example (commented) lines in `SourceCode/Logger.py` showing
  how to load a saved state_dict for a given model (these lines are
  intentionally left commented out because paths and model names will
  vary by experiment).
- Configure which base sketch is used for the RatioSketch experiment in
  the config (e.g. in `Ratio_seed0.py`) by setting
  `factory_config["MemoryModule_class"]` to `"CMS"` or `"TCMS"`.
- Only one model should be trained or evaluated per Python process. Do
  not enable both `CMS+RS` and `Tower+RS` in the same run as they will
  conflict; choose the appropriate eval metric(s) in
  `logger_config["eval_metrics"]`.

Dependencies and detected versions

The project requires Python and several scientific / ML libraries. Below
are the versions detected from the environment you provided. Use these
as a starting point for creating a reproducible environment.

Detected versions (from your environment):

- Python: 3.10.12
- torch: 2.8.0+cu128
- numpy: 2.2.6
- scipy: 1.15.3
- matplotlib: 3.10.6
- pandas: 2.3.2
- xxhash: 3.5.0
