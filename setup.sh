#!/bin/bash
#
# One-time setup: install emp-toolkit dependencies at pinned commits.
#
set -e

# Pinned commits
EMP_TOOL_REV=802b5d4fb7cc7fcaadd411cd6aa5e72ed4dd57fd
EMP_OT_REV=a603ca0c77fcda37b9d088bd692111f67a4bef96
EMP_ZK_REV=73ab193a923d4c122be5a4f6bc1fe4f617966b02

# System dependencies
if command -v apt-get >/dev/null; then
    sudo apt-get install -y cmake git build-essential libssl-dev clang
elif command -v yum >/dev/null; then
    sudo yum install -y cmake git gcc make gcc-c++ openssl-devel clang
fi

mkdir -p setup && cd setup

for pkg in emp-tool:$EMP_TOOL_REV emp-ot:$EMP_OT_REV emp-zk:$EMP_ZK_REV; do
    name="${pkg%%:*}"
    rev="${pkg##*:}"
    if [ ! -d "$name" ]; then
        git clone "https://github.com/emp-toolkit/$name.git"
    fi
    cd "$name"
    git checkout "$rev"
    cmake .
    make -j4
    sudo make install
    cd ..
done
