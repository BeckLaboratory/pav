#!/usr/bin/env bash

set -euo pipefail

# Build binary dependencies for PAV


PREFIX=/opt/pav

HTSLIB_VERSION=1.23

SAMTOOLS_VERSION=${HTSLIB_VERSION}

MINIMAP_VERSION=2.30


### Init ###

apt-get update

### Libraries ###

apt-get install -y \
  libbz2-dev \
  libncurses-dev \
  libssh-dev \
  libssl-dev \
  libcurl4-openssl-dev \
  libdeflate-dev \
  liblzma-dev \
  wget \
  bzip2 \
  gcc \
  make \
  build-essential \
  gnupg \
  curl


### htslib ###

wget https://github.com/samtools/htslib/releases/download/${HTSLIB_VERSION}/htslib-${HTSLIB_VERSION}.tar.bz2

tar -xjf htslib-${HTSLIB_VERSION}.tar.bz2

pushd htslib-${HTSLIB_VERSION}

./configure --prefix=${PREFIX} CPPFLAGS="-I${PREFIX}/include" LDFLAGS="-L${PREFIX}/lib -Wl,-rpath,${PREFIX}/lib"

make -j $(nproc) && make install

popd

rm -r htslib-${HTSLIB_VERSION} htslib-${HTSLIB_VERSION}.tar.bz2

rm bin/htsfile


### samtools ###

wget https://github.com/samtools/samtools/releases/download/${SAMTOOLS_VERSION}/samtools-${SAMTOOLS_VERSION}.tar.bz2

tar -xjf samtools-${SAMTOOLS_VERSION}.tar.bz2

pushd samtools-${SAMTOOLS_VERSION}

./configure --prefix=${PREFIX} CPPFLAGS="-I${PREFIX}/include" LDFLAGS="-L${PREFIX}/lib -Wl,-rpath,${PREFIX}/lib"

make -j $(nproc) && make install

popd

rm -r samtools-${SAMTOOLS_VERSION} samtools-${SAMTOOLS_VERSION}.tar.bz2

rm bin/{ace2sam,wgsim} bin/{plot-,maq2sam}* bin/*.pl


### minimap2 ###

wget https://github.com/lh3/minimap2/releases/download/v${MINIMAP_VERSION}/minimap2-${MINIMAP_VERSION}.tar.bz2

tar -xjf minimap2-${MINIMAP_VERSION}.tar.bz2

make -j $(nproc) -C minimap2-${MINIMAP_VERSION}

install minimap2-${MINIMAP_VERSION}/minimap2 ${PREFIX}/bin/

rm -r minimap2-${MINIMAP_VERSION} minimap2-${MINIMAP_VERSION}.tar.bz2


### UCSC ###

curl http://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/bedToBigBed --output bin/bedToBigBed
chmod ugo+x bin/bedToBigBed


### Cleanup ###

rm -r share

rm lib/libhts.a
