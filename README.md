# Reinforcement learning for Tactician

## Assumptions

The recommended `opam` version is `>= 2.1.0`. Other versions might work as well, but you may have to install some dependencies manually.

Notice: the installation depends on ocaml version 4.11.2 that is in conflict with glibc version >= 2.34
and therefore fails on Ubuntu 21.10.

On Ubuntu 20.04 the packages installed by the following are required:

```
sudo apt-get --yes install graphviz capnproto libcapnp-dev pkg-config libev-dev
```

## Installation

Notice: with current limitation of pin-depends and pinned relative path it is strictly necessary to execute
`opam install ./coq-tactician-reinforce.opam --yes` in the below script from the directory of the opam file.

```
opam switch create tact --empty &&  eval $(opam env --switch=tact)
opam repo add coq-released https://coq.inria.fr/opam/released     # packages for officially released versions of Coq library
opam repo add coq-core-dev https://coq.inria.fr/opam/core-dev     # packages for development versions of Coq
opam repo add coq-extra-dev https://coq.inria.fr/opam/extra-dev   # packages for development versions of Coq libraries and Coq extensions
opam repo add custom-archive https://github.com/LasseBlaauwbroek/custom-archive.git # for Lasse's bugfixes of Coq
git clone --recurse-submodules git@github.com:coq-tactician/coq-tactician-reinforce.git
cd coq-tactician-reinforce
opam install ./coq-tactician-reinforce.opam --yes
```

## Available Commands

These commands will create a graph of some object, and write it to `graph.pdf` (if `graphviz` is available).

The following commands are always available:
```
[Full] Graph Ident identifier.
[Full] Graph Term term.
```
Normally, the commands print a non-transitive graph. The `[Full]` modifier changes this so that the full transitive graph of definitions is added.

Additionally, in proof mode, these commands are available:
```
[Full] Graph Proof.
```

Options that modify the graphs generated by the commands above are
```
[Set | Unset] Tactician Reinforce Visualize Ordered.
[Set | Unset] Tactician Reinforce Visualize Labels.
```

## Reinforcement

Finally, the command `Reinforce.` will initiate a reinforcement session. An example of this is available in
[tests/ReinforceTest.v](theories/ReinforceTest.v).
To do this, you need to have a python client running. An example is available in pip package `pytact` that you can install
with
```
pip install git+ssh://git@github.com/coq-tactician/coq-tactician-reinforce.git
```

To see how it works, run
```
pytact-test
```
optionally `--file` option to point to a source Coq `.v` file.
Also with `--interactive` option the innteractive shell appears where you can
manually interact with the environment. Whenever a tactic is executed,
the resulting proof state if visualized in the file
`python_graph.pdf`.

## Python package
We package the python package `pytact` in `setup.py` with python source code in `pytact`. To conveniently work on python code from the directory of `setup.py` run
```
pip install -e .
```
This installs python package in developer mode, so that python code updates propagate immediately
to the installed package.


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

## (EXPERIMENTAL), subject to changes

The procedure to generate the dataset is the following.

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
