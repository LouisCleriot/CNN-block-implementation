#check if the directory exists
if [ ! -d "fashion-mnist" ]; then
  mkdir fashion-mnist
fi

#download the dataset into the directory
wget -c http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz -P fashion-mnist 
wget -c http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz -P fashion-mnist 
wget -c http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz -P fashion-mnist 
wget -c http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz -P fashion-mnist 

#unzip the files
gunzip fashion-mnist/train-images-idx3-ubyte.gz
gunzip fashion-mnist/train-labels-idx1-ubyte.gz
gunzip fashion-mnist/t10k-images-idx3-ubyte.gz
gunzip fashion-mnist/t10k-labels-idx1-ubyte.gz

# remove the .gz files
rm fashion-mnist/*.gz

