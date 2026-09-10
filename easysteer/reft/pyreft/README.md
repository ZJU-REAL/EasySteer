# Bundled ReFT implementation

This directory contains EasySteer's adaptation of
[PyReFT](https://github.com/stanfordnlp/pyreft), including its
[PyVene](https://github.com/stanfordnlp/pyvene) intervention machinery.
It is installed as part of `easysteer`; it is not a separate PyReFT distribution.

Keep the `easysteer.reft.pyreft` module paths stable: saved intervention configs
refer to their class names. EasySteer's training helper lives in `../train.py`,
and its checkpoint-to-inference adapter lives in `../../vectors.py`.
Runnable examples belong in the repository's `examples/` and `replications/`
directories; CPU checks live under `tests/cpu/test_reft_*`.
