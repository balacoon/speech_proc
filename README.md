# Speech Processing

A toolking for processing speech, extracting features.
Supports extraction of:
* `semantic_tokens` - hubert codes representing content
* `acoustic_tokens` - hierarchical quantized representations that work with VQ vocoder
* `pitch` - different representations of fundamental frequency

## Installation

Regular python installation works in general:
```bash
pip install -r requirements.txt
pip install -e .
```

`pitch` extraction requires `SPTK`:
```
apt-get install -y csh
wget -O SPTK-3.11.tar.gz http://downloads.sourceforge.net/sp-tk/SPTK-3.11.tar.gz
tar -xzvf SPTK-3.11.tar.gz
cd SPTK-3.11 && ./configure && make -j4 && make install
cd .. && rm SPTK-3.11.tar.gz
export PATH=$PWD/SPTK-3.11/bin/pitch/:$PATH
```