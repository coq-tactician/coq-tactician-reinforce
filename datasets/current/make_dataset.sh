set -o nounset
set -o errexit

scriptpath="$(dirname "$0")"

if [ $# -lt 1 ]
then
    echo "Usage: make_dataset.sh dataset-name"
    exit 1
fi

datasetname=${1}; shift

if [ -d $datasetname ] || [ -f $datasetname ]; then
    echo "$datasetname already exists"
    exit 1
fi
if [ -f "$datasetname".squ ]; then
    echo "$datasetname.squ already exists"
    exit 1
fi

echo $(opam var prefix)
echo $datasetname

echo "Populating meta directory"

mkdir -p $datasetname/meta

cat > $datasetname/meta/LICENSE <<EOF
MIT License

Copyright (c) 2022 Tactician

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

cp $(opam var prefix)/share/coq-tactician-reinforce/graph_api.capnp $datasetname/meta/graph_api.capnp

for f in $scriptpath/stage*.sh; do
    base=$(basename $f)
    base=${base%.sh}
    opamfreeze="opam-$base"
    if [ ! -f "$opamfreeze" ]; then
        echo "Missing opam export file $opamfreeze"
        exit 1
    fi
    cp $f $datasetname/meta/
    cp $opamfreeze $datasetname/meta/
done

mkdir -p $datasetname/dataset

echo "Retrieving dataset"
cp -r $(opam var prefix)/.opam-switch/build/* $datasetname/dataset

echo "Removing files other than .bin and .v (if a corresponding .bin file exists)"
files=$(find $datasetname/dataset -not -type d)
for f in $files; do
    if [[ ! "$f" == *.bin ]]; then
        if [[ "$f" == *.v ]]; then
            fbin=${f%.*}.bin
            if [ ! -f "$fbin" ]; then
                rm -f "$f"
            fi
        else
            rm -f "$f"
        fi
    fi
done

echo "Deleting empty directories"
find $datasetname/dataset -type d -empty -delete

echo "Writing licensing information"
cat > $datasetname/LICENSE << EOF
This dataset is derived from a multitude of Coq developments. It includes both
the source files of these developments and compiled binary files generated by
the Coq compiler combined with the Tactician plugin. As such, the source files
and derived files in this dataset are subject to the same licsensing as their
original source distribution.

The licensing of this dataset is organized as follows:
- Files in the 'meta/' directory are licensed under the MIT license. See
  meta/LICENSE for more information.
- Each subdirectory of the 'dataset/' directory corresponds to a Coq
  development, with it's own license. A LICENSE file is present in each
  directory summarizing the origin of the development and corresponding license
  information. This information is obtained from the Opam package manager. This
  license information is not double-checked. Some licenses may be non-free.
  Please make sure that the license for each development in this dataset is
  compatible with your use-case.

Below, we include a summary for each development included in this dataset.
EOF

for package in $datasetname/dataset/*; do
    package=$(basename $package)
    echo "Writing licensing information for $package"
    license=$(opam show -f license $package)
    license=${license//\"}
    homepage=$(opam show -f license $package)
    homepage=${homepage//\"}
    summary=$(opam show -f synopsis $package)
    authors=$(opam show -f authors $package)
    if curl --head --silent --fail "https://spdx.org/licenses/$license" > /dev/null; then
        spdx="https://spdx.org/licenses/$license"
    else
        spdx="unknown"
    fi
    info=$(cat <<EOF
Package : $package
Summary : $summary
Homepage: $homepage
Authors : $authors
License : $license
SPDX URL: $spdx
EOF
        )
    printf "\n$info\n" >> $datasetname/LICENSE
    echo "$info" > $datasetname/dataset/$package/LICENSE
done

echo "Creating SquashFS archive"
mksquashfs "$datasetname" "$datasetname.squ" -comp gzip
