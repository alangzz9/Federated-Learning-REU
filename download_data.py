from six.moves import urllib
import os
import scipy.io as sio
from sklearn.datasets import load_svmlight_file


if not os.path.exists('./data'):
    os.makedirs('./data')

urllib.request.urlretrieve("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a",'./data/a9a')
urllib.request.urlretrieve("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a",'./data/w8a')
urllib.request.urlretrieve("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna",'./data/cod-rna')
urllib.request.urlretrieve("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2",'./data/covtype.libsvm.binary.scale.bz2')
urllib.request.urlretrieve("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2",'./data/ijcnn1.bz2')
urllib.request.urlretrieve("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/rcv1_train.binary.bz2",'./data/rcv1_train.binary.bz2')
urllib.request.urlretrieve("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2",'./data/real-sim.bz2')

X,Y = load_svmlight_file("./data/a9a")
sio.savemat("./data/a9a.mat", {'labels' : Y, 'features' : X})

X,Y = load_svmlight_file("./data/w8a")
sio.savemat("./data/w8a.mat", {'labels' : Y, 'features' : X})

X,Y = load_svmlight_file("./data/cod-rna")
sio.savemat("./data/cod_rna.mat", {'labels' : Y, 'features' : X})


X,Y = load_svmlight_file("./data/ijcnn1.bz2")
sio.savemat("./data/ijcnn1.mat", {'labels' : Y, 'features' : X})

X,Y = load_svmlight_file("./data/covtype.libsvm.binary.scale.bz2")
sio.savemat("./data/covtype.mat", {'labels' : Y, 'features' : X})

X,Y = load_svmlight_file("./data/rcv1_train.binary.bz2")
sio.savemat("./data/rcv1.mat", {'labels' : Y, 'features' : X})

X,Y = load_svmlight_file("./data/real-sim.bz2")
sio.savemat("./data/real_sim.mat", {'labels' : Y, 'features' : X})





