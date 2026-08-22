"""Head-only training: cache the frozen backbone, then fit the keypoint head.

The backbone never trains, so its output for a given clip is fixed for the
whole run: :mod:`.cache` computes patch tokens once per episode and writes
them to a self-describing file, and :mod:`.trainer` reads those thousands of
times instead of re-running the frozen network per minibatch. :mod:`.splits`
partitions episodes by scene, so a validation number measures generalisation
rather than memorisation.

Nothing here imports torch at package level -- import the submodule you need.
"""
