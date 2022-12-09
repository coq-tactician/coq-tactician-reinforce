# A Graph -and Text-based machine learning interface for Tactician

## Prerequisites

This repository has an OCaml component that should be installed through the Opam package manager and a Python
component that should be installed through the Pip package manager. Additionally, some extra dependencies are needed:
- Opam 2.1.x
- Capt'n Proto >= 0.8
- XXHash >= 0.8
- Graphviz
- A reasonable set of development packages like git, bash, gmp, c/c++ compiler toolchains that should be installed
  on most systems.

The simplest and most reliable way to install these packages is through Conda. This repository provides a
`environment.yml` file with the required Conda dependencies. To set it up, follow these commands:
```
git clone --recurse-submodules git@github.com:coq-tactician/coq-tactician-reinforce.git # Clone this repo
cd coq-tactician-reinforce
conda env create -f environment.yml
```

If you don't want to use Conda, you'll have to install the dependencies listed above through your distributions
package manager. On Ubuntu 22.04 or newer, you can get the required packages as follows (older versions of Ubuntu
have to fall back to the Conda solutions because the bundled software is out of date)
```
sudo apt-get --yes install graphviz capnproto libcapnp-dev pkg-config libev-dev libxxhash-dev
```

After installing the prerequisites, you'll need a Python virtualenv and an Opam switch to install the software.
To create the virtualenv, run
```
python -m venv <desired-location-of-virtualenv>`
```
To activate the virtualenv run `source <location-of-virtualenv>`.

For the OCaml side, if you've never run Opam before, initialize it by running `opam init`. Then, create a switch
with the appropriate software repositories:
```
opam switch create tactician --empty --repos=custom-archive=https://github.com/LasseBlaauwbroek/custom-archive.git,coq-extra-dev=https://coq.inria.fr/opam/extra-dev,coq-core-dev=https://coq.inria.fr/opam/core-dev,coq-released=https://coq.inria.fr/opam/released,default
```
Make sure to follow any printed instructions regarding `eval $(opam env)` to activate the switch.

## Installation
To install the Python component of this repository, make sure that you have a virtualenv enabled then run the
command `pip install .` from the root of this repository.

To install the OCaml component of this repository, make sure that you have the appropriate switch activated and
run the command `opam install . --yes` from the root of this repository.

If you want maximum performance, it is recommended that you use an OCaml version with `flambda` enabled. On newer versions of Opam you can achieve this by installing `ocaml-option-flambda`.

## Usage of the Python software
The Python software provides both a software library to work with the graph based datasets extracted from Coq and
a number of executables. Available executables are as follows (use the `--help` flag for each executable to learn
about all the options).

- `pytact-check <dataset-root>`: Run sanity checks on a dataset and print some statistics
- `pytact-visualize <dataset-root>`: Start an interactive server that visualizes a dataset
- `pytact-server [--tcp <port>] [graph | text]`: A dummy example server that provides tactic predictions to
  Tactician's `synth` tactic inside of Coq. To learn how to interface Coq and Tactician with this server, see
  the sections below.
- `pytact-prover`: A dummy example client that interfaces with Coq and Tactician for reinforcement-learning-style
  communication. To learn how to interface Coq and Tactician with this client, see the sections below.

## Usage of the Coq plugin

### Available Commands

These commands will create a graph of some object, and write it to `graph.pdf` (if `graphviz` is available).

The following commands are always available:
```
[Shared] Graph [Depth <n>] Ident identifier.
[Shared] Graph [Depth <n>] Term term.
```
The normal commands print a fully transitive graph. Adding `Depth i` limits the traversal to visiting at most `i`
nested definitions.

Additionally, in proof mode, these commands are available:
```
[Shared] Graph [Depth <n>] Proof.
```

Options that modify the graphs generated by the commands above are
```
[Set | Unset] Tactician Neural Visualize Ordered.
[Set | Unset] Tactician Neural Visualize Labels.
[Set | Unset] Tactician Neural Visualize Hashes.
```

### Interaction with `synth`
In order to connect Tactician's `synth` tactic to a external tactic prediction server like the dummy
`pytact-server` described above, the plugin makes a number of commands and settings available in Coq.
The following settings govern the data that Coq will send to the server:
- `Set Tactician Truncate` determines wether the bodies of definitions will get truncated or not (on by default).
- `Set Tactician Textmode` determines wether Coq is communicating with a graph-based server or a text-based server (graph-based by default).
To let Coq take care of starting and stopping the server, use the command
```
Set Tactician Neural Executable "external-server-executable --argument1 --argument2".
```
If you have a prediction server already running somewhere over TCP, you can make Coq connect to it using
```
Set Tactician Neural Server "<address>:<port>".
```
At this point, you have the following commands available which will interact with the server:
- `Check Neural Alignment` will ask the which tactics and definitions currently in scope are unknown to it.
  This is meant as a sanity check.
- `Suggest` and `Debug Suggest` will ask the server for predictions for the current proof state.
- `synth` and `debug synth` will perform a proof search by repeatedly asking the server for predictions.

## Reinforcement interaction

Finally, the command `Reinforce.` will initiate a reinforcement session. An example of this is available in
[tests/ReinforceTest.v](theories/ReinforceTest.v).
To do this, you need to have a python client running. An example is available in the `pytact-prover` executable.
To see how it works, run
```
pytact-prover --pdfsequence --pdfname test
```
This will execute a dummy proof through the reinforcement learning interface. Visualizations of each proof state
are available in `test<n>.pdf`.
optionally `--file` option to point to a source Coq `.v` file.
Also with `--interactive` option the innteractive shell appears where you can
manually interact with the environment. Whenever a tactic is executed,
the resulting proof state if visualized in the file
`python_graph.pdf`.

## Generating a dataset
To generate a dataset, you currently have to install a slightly different version of the Coq plugin that resides
in the `generate-dataset` branch. The procedure to generate the dataset is as follows.

1. Create your switch
```
opam switch create tacgen --empty
```
2. Install coq-tactician-reinforce generate-dataset
```
git clone -b generate-dataset --recurse-submodules git@github.com:coq-tactician/coq-tactician-reinforce.git
cd coq-tactician-reinforce
opam install .
tactician inject # you can answer 'no' to recompiling
opam install coq-tactician-stdlib --keep-build-dir # make sure that you have the coq-extra-dev repo enabled
```

3. For your Coq dataset, e.g. `propositional`
```
cd ../propositional
tactician exec dune build
```
4. With opam build of `coq-package` do
```
opam install coq-package --keep-build-dir
```
and you find the `*.bin` in the directory `<switch>/.opam-switch/build`. The recorded
dependency paths are relative to `<switch>/.opam-switch/build`.

## CI
To verify the build and test locally by specification in `Dockerfile` you run

```
sudo docker build -t test .
```
The `Dockerfile` contains project build instruction and the set of tests.

Our plan for Github Actions CI to always reuse and refer to the same
`Dockerfile`.

In this way we can be sure that local CI is identical to GitHub
Actions CI, and that we can move easily to another platform if
necessary.

## CI caching
The `Dockefile` builds on top of the base layer `Dockerfile_base`
derived from canonical coq-community
`coqorg/coq:8.11.2-ocaml-4.11.2-flambda` that is based on
Debian.10/opam 2.0.9/coq 8.11.2/ocaml-variants-4.11.2+flambda.

The layer defined by `Dockerfile_base` adds `conda/python 3.9`,
`capnp` library and all opam package dependencies requested by the
coq-tactician-reinforce (including the opam package defined in git
submodule `coq-tactician`).

The image defined by `Dockerfile_base` can be updated by maintainers (currently Vasily) by
```
sudo sh ci-update-base.sh
```
This caching update is necessary only periodically and only
for optimisation of the speed of CI, but it is not strictly necessary for CI to perform correctly
(opam is supposed to reinstall packages if dependencies are changed -- to be confirmed by practice).
