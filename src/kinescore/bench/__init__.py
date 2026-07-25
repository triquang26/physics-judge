"""Benchmark harness: manifest -> run -> store -> stats.

* :mod:`kinescore.bench.manifest` -- enumerate clips to score (pluggable
  discovery) and verify paired gt/pred clips are comparable.
* :mod:`kinescore.bench.store` -- the one canonical result-record schema,
  and the one flattener for it.
* :mod:`kinescore.bench.runner` -- iterate a manifest, score, stream results.
* :mod:`kinescore.bench.stats` -- paired episode-level statistics.

Stays import-light on purpose: ``pandas``/``pyarrow``/``scipy`` (the ``bench``
extra) are imported lazily inside the functions that need them, so
``import kinescore.bench`` and its submodules succeed in the numpy-only CPU
test tier that doesn't install that extra -- only functions that actually
touch parquet or run a statistical test require it, and they raise a normal
``ImportError`` naming the missing package if it's absent.
"""
