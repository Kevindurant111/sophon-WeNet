```bash
# install armadillo
sudo apt-get install -y liblapack-dev libblas-dev libopenblas-dev libarmadillo-dev libsndfile1-dev

# install kenlm
git clone https://github.com/kpu/kenlm.git
cd kenlm
mkdir - p build
cd build
cmake ..
make
sudo make install

# install openfst
wget http://www.openfst.org/twiki/pub/FST/FstDownload/openfst-1.6.3.tar.gz
tar -xzvf openfst-1.6.3.tar.gz
cd openfst-1.6.3
./configure --enable-far=true
make
sudo make install
echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib" >> ~/.bashrc
source ~/.bashrc
```

