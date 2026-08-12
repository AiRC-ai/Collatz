# Reproducibility

The supported clean-clone path requires CMake 3.20 or newer, a C++20 compiler,
and Python 3:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The CPU CI workflow runs this same path. Small fixtures under
`fixtures/evidence_small/` validate the public evidence contract without access
to the large generated datasets.

The reproducibility path checks:

- canonical evidence validation
- generated evidence snapshot rendering
- source-alignment unmatched taxonomy
- confidence-gate behavior
- public-safety scanning

The public evidence summary records the hashes of the exact ML input files used
for the large experiments. Those generated inputs and the 1.2-billion-row audit
file are not committed, so the fixture suite verifies the code and evidence
contract—not a full rerun of every GPU experiment.
