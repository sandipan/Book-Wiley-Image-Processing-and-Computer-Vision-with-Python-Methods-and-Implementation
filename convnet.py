# ============================================================
# numpy_nn_fcnn_convnet.py
# Part 1 : Imports + Dataset + Activations + Base Layers
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import struct
from pathlib import Path

# ============================================================
# DATASET
# ============================================================

class IDXDataset:

    IMAGE_MAGIC=2051
    LABEL_MAGIC=2049

    def __init__(self,folder="images/mnist"):
        f=Path(folder)
        self.X_train=self._img(self._find(f,"train-images.idx3-ubyte"))
        self.y_train=self._lbl(self._find(f,"train-labels.idx1-ubyte"))
        self.X_test=self._img(self._find(f,"t10k-images.idx3-ubyte"))
        self.y_test=self._lbl(self._find(f,"t10k-labels.idx1-ubyte"))

    def subset_train(self,n):
        self.X_train,self.y_train=self.X_train[:n],self.y_train[:n]
        return self

    @staticmethod
    def _find(folder,name):
        for p in [folder/name,folder/name.replace(".", "-",1),folder/name.replace("-idx",".idx",1)]:
            if p.exists(): return p
        raise FileNotFoundError(name)

    @classmethod
    def _img(cls,path):
        with open(path,"rb") as f:
            m,n,r,c=struct.unpack(">IIII",f.read(16))
            assert m==cls.IMAGE_MAGIC
            x=np.frombuffer(f.read(),dtype=np.uint8).reshape(n,r*c)
        return x.astype(np.float32)/255.

    @classmethod
    def _lbl(cls,path):
        with open(path,"rb") as f:
            m,n=struct.unpack(">II",f.read(8))
            assert m==cls.LABEL_MAGIC
            y=np.frombuffer(f.read(),dtype=np.uint8)
        return y.astype(np.int64)

# ============================================================
# ACTIVATIONS
# ============================================================

class Activation:
    def forward(self,x): raise NotImplementedError
    def backward(self,g): raise NotImplementedError

class Sigmoid(Activation):

    def forward(self,x):
        self.y=1/(1+np.exp(-np.clip(x,-40,40)))
        return self.y

    def backward(self,g):
        return g*self.y*(1-self.y)

class ReLU(Activation):

    def forward(self,x):
        self.m=x>0
        return np.maximum(x,0)

    def backward(self,g):
        return g*self.m

class Softmax(Activation):

    def forward(self,x):
        x=x-x.max(1,keepdims=True)
        e=np.exp(x)
        self.y=e/e.sum(1,keepdims=True)
        return self.y

    def backward(self,g):
        return g

# ============================================================
# BASE LAYER
# ============================================================

class Layer:

    def forward(self,x):
        raise NotImplementedError

    def backward(self,g):
        raise NotImplementedError

    def params(self):
        return []

    def grads(self):
        return []

# ============================================================
# INITIALIZERS
# ============================================================

def xavier(nin,nout):
    return np.random.uniform(-np.sqrt(6/(nin+nout)),np.sqrt(6/(nin+nout)),(nout,nin+1))

def he(shape):
    return np.random.randn(*shape)*np.sqrt(2/np.prod(shape[:-1]))

# ============================================================
# UTILITIES
# ============================================================

def one_hot(y,k):
    Y=np.zeros((len(y),k),dtype=np.float32)
    Y[np.arange(len(y)),y]=1
    return Y

def accuracy(y,p):
    return 100*np.mean(y==p)

def im2batch(X):
    return X.reshape(-1,28,28,1)

def batch2vec(X):
    return X.reshape(len(X),-1)

# ============================================================
# DENSE
# ============================================================

class Dense(Layer):

    def __init__(self,nin,nout,act=Sigmoid()):
        self.W=xavier(nin,nout)
        self.act=act

    def forward(self,x):
        self.x=np.c_[np.ones(len(x)),x]
        self.z=self.x@self.W.T
        self.y=self.act.forward(self.z)
        return self.y

    def backward(self,g):
        g=self.act.backward(g)
        self.dW=g.T@self.x/len(self.x)
        return g@self.W[:,1:]

    def params(self):
        return [self.W]

    def grads(self):
        return [self.dW]

# ============================================================
# CONVOLUTION
# ============================================================

class Conv2D(Layer):

    def __init__(self,nin,nf,f=3,pad=1,stride=1,act=ReLU()):
        self.f,self.pad,self.stride,self.act=f,pad,stride,act
        self.W=he((f,f,nin,nf))
        self.b=np.zeros(nf)

    def forward(self,x):

        self.x=x
        m,h,w,c=x.shape
        f,p,s,nf=self.f,self.pad,self.stride,self.W.shape[-1]

        self.xp=np.pad(x,((0,0),(p,p),(p,p),(0,0)))
        oh=(h-f+2*p)//s+1
        ow=(w-f+2*p)//s+1

        z=np.zeros((m,oh,ow,nf))

        for i in range(m):
            for y in range(oh):
                ys=y*s
                for x0 in range(ow):
                    xs=x0*s
                    r=self.xp[i,ys:ys+f,xs:xs+f]
                    for k in range(nf):
                        z[i,y,x0,k]=np.sum(r*self.W[:,:,:,k])+self.b[k]

        self.z=z
        self.y=self.act.forward(z)
        return self.y

    def backward(self,g):

        g=self.act.backward(g)

        m,h,w,c=self.x.shape
        f,p,s,nf=self.f,self.pad,self.stride,self.W.shape[-1]

        self.dW=np.zeros_like(self.W)
        self.db=np.sum(g,(0,1,2))
        dx=np.zeros_like(self.xp)

        oh,ow=g.shape[1:3]

        for i in range(m):
            for y in range(oh):
                ys=y*s
                for x0 in range(ow):
                    xs=x0*s
                    r=self.xp[i,ys:ys+f,xs:xs+f]
                    for k in range(nf):
                        self.dW[:,:,:,k]+=r*g[i,y,x0,k]
                        dx[i,ys:ys+f,xs:xs+f]+=self.W[:,:,:,k]*g[i,y,x0,k]

        self.dW/=m
        self.db/=m
        return dx[:,p:p+h,p:p+w]

    def params(self):
        return [self.W,self.b]

    def grads(self):
        return [self.dW,self.db]

# ============================================================
# MAXPOOL
# ============================================================

class MaxPool2D(Layer):

    def __init__(self,f=2,stride=2):
        self.f,self.stride=f,stride

    def forward(self,x):

        self.x=x
        m,h,w,c=x.shape
        f,s=self.f,self.stride

        oh=(h-f)//s+1
        ow=(w-f)//s+1

        y=np.zeros((m,oh,ow,c))
        self.mask=np.zeros_like(x,dtype=bool)

        for i in range(m):
            for yy in range(oh):
                ys=yy*s
                for xx in range(ow):
                    xs=xx*s
                    for k in range(c):
                        r=x[i,ys:ys+f,xs:xs+f,k]
                        j=np.argmax(r)
                        y[i,yy,xx,k]=r.flat[j]
                        a,b=np.unravel_index(j,(f,f))
                        self.mask[i,ys+a,xs+b,k]=1

        return y

    def backward(self,g):

        dx=np.zeros_like(self.x)
        f,s=self.f,self.stride
        oh,ow=g.shape[1:3]

        for i in range(len(g)):
            for yy in range(oh):
                ys=yy*s
                for xx in range(ow):
                    xs=xx*s
                    for k in range(g.shape[-1]):
                        dx[i,ys:ys+f,xs:xs+f,k]+=self.mask[i,ys:ys+f,xs:xs+f,k]*g[i,yy,xx,k]

        return dx

# ============================================================
# FLATTEN
# ============================================================

class Flatten(Layer):

    def forward(self,x):
        self.shape=x.shape
        return x.reshape(len(x),-1)

    def backward(self,g):
        return g.reshape(self.shape)

# ============================================================
# LOSS
# ============================================================

class CrossEntropy:

    def forward(self,p,y):
        self.p,self.y=p,y
        return -np.mean(np.log(p[np.arange(len(y)),y]+1e-9))

    def backward(self):
        g=self.p.copy()
        g[np.arange(len(self.y)),self.y]-=1
        return g/len(self.y)

# ============================================================
# OPTIMIZER
# ============================================================

class SGD:

    def __init__(self,lr=.01,momentum=.9,decay=.98):
        self.lr,self.m,self.decay=lr,momentum,decay
        self.v={}

    def step(self,layers):

        k=0

        for L in layers:

            P,G=L.params(),L.grads()

            for p,g in zip(P,G):

                if k not in self.v:self.v[k]=np.zeros_like(p)
                self.v[k]=self.m*self.v[k]-self.lr*g
                p+=self.v[k]
                k+=1

    def epoch(self):
        self.lr*=self.decay

# ============================================================
# SEQUENTIAL NETWORK
# ============================================================

class Sequential:

    def __init__(self):
        self.layers=[]

    def add(self,l):
        self.layers.append(l)

    def forward(self,x):
        for l in self.layers:x=l.forward(x)
        return x

    def backward(self,g):
        for l in self.layers[::-1]:g=l.backward(g)

    def predict(self,x):
        return np.argmax(self.forward(x),1)

    def fit(self,X,y,loss,opt,epochs=20,batch=128,verbose=True):

        hist=[]

        for e in range(epochs):

            idx=np.random.permutation(len(X))
            X,y=X[idx],y[idx]
            L=0
            nb=0

            for i in range(0,len(X),batch):

                xb,yb=X[i:i+batch],y[i:i+batch]

                p=self.forward(xb)
                L+=loss.forward(p,yb)
                self.backward(loss.backward())
                opt.step(self.layers)

                nb+=1

            opt.epoch()
            hist.append(L/nb)

            if verbose:
                print(f"Epoch {e+1:02d} Loss:{hist[-1]:.4f}")

        return hist

# ============================================================
# NETWORK BUILDERS
# ============================================================

def FCNN(sizes):

    net=Sequential()

    for i in range(len(sizes)-2):
        net.add(Dense(sizes[i],sizes[i+1],ReLU()))

    net.add(Dense(sizes[-2],sizes[-1],Softmax()))

    return net


def ConvNet():

    net=Sequential()

    net.add(Conv2D(1,32,3,1,1,ReLU()))
    net.add(MaxPool2D())

    net.add(Conv2D(32,64,3,1,1,ReLU()))
    net.add(MaxPool2D())

    net.add(Flatten())
    net.add(Dense(7*7*64,128,ReLU()))
    net.add(Dense(128,10,Softmax()))

    return net

# ============================================================
# VISUALIZATION
# ============================================================

class Visualizer:

    @staticmethod
    def show_images(X,n=100,title="MNIST",shape=(28,28)):
        plt.figure(figsize=(8,8))
        for i in range(min(n,len(X))):
            plt.subplot(10,10,i+1)
            plt.imshow(X[i].reshape(shape),cmap="gray")
            plt.axis("off")
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_curve(hist,title="Training Curve"):
        plt.figure(figsize=(6,4))
        plt.plot(hist,lw=2)
        plt.grid()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.show()

    @staticmethod
    def show_predictions(X,p,y,n=20,shape=(28,28)):
        plt.figure(figsize=(10,5))
        for i in range(min(n,len(X))):
            plt.subplot(4,5,i+1)
            plt.imshow(X[i].reshape(shape),cmap="gray")
            plt.title(f"P:{p[i]} T:{y[i]}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_heatmap(W,title):
        plt.figure(figsize=(10,4))
        plt.imshow(W,cmap="coolwarm",aspect="auto")
        plt.colorbar()
        plt.title(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_dense_features(W,title="Hidden Features",shape=(28,28),n=100):
        if W.ndim!=2:return
        if W.shape[1]==shape[0]*shape[1]+1: W=W[:,1:]
        plt.figure(figsize=(8,8))
        for i,w in enumerate(W[:min(n,len(W))]):
            plt.subplot(10,10,i+1)
            plt.imshow(w.reshape(shape),cmap="gray")
            plt.axis("off")
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_conv_filters(W,title="Conv Filters"):
        f,_,_,k=W.shape
        cols=int(np.ceil(np.sqrt(k)))
        rows=int(np.ceil(k/cols))
        plt.figure(figsize=(2*cols,2*rows))
        for i in range(k):
            plt.subplot(rows,cols,i+1)
            plt.imshow(W[:,:,0,i],cmap="gray")
            plt.axis("off")
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def show_feature_maps(net,img,max_maps=32):

        x=img.reshape(1,28,28,1)

        for L in net.layers:

            x=L.forward(x)

            if isinstance(L,Conv2D):

                m=min(x.shape[-1],max_maps)
                c=int(np.ceil(np.sqrt(m)))
                r=int(np.ceil(m/c))

                plt.figure(figsize=(2*c,2*r))

                for i in range(m):
                    plt.subplot(r,c,i+1)
                    plt.imshow(x[0,:,:,i],cmap="gray")
                    plt.axis("off")

                plt.suptitle("Feature Maps")
                plt.tight_layout()
                plt.show()

                return

    @staticmethod
    def dashboard(net,X,y,p,history,is_conv=False):

        Visualizer.show_curve(history)
        Visualizer.show_predictions(X,p,y)

        if is_conv:
            for L in net.layers:
                if isinstance(L,Conv2D):
                    Visualizer.show_conv_filters(L.W,"Learned Convolution Filters")
                    Visualizer.show_heatmap(L.W.reshape(-1,L.W.shape[-1]),"Conv Weight Heatmap")
                    Visualizer.show_feature_maps(net,X[0])
                    break
        else:
            for L in net.layers:
                if isinstance(L,Dense):
                    if L.W.shape[1]-1==784:
                        Visualizer.show_dense_features(L.W,"Hidden Layer Features")
                        Visualizer.show_heatmap(L.W,"Dense Weight Heatmap")
                        break

# ============================================================
# MAIN
# ============================================================

if __name__=="__main__":

    # np.random.seed(0)

    data=IDXDataset("images/mnist")

    print(data.X_train.shape,data.y_train.shape)
    print(data.X_test.shape,data.y_test.shape)

    Visualizer.show_images(data.X_train,title="MNIST Samples")

    # ============================================================
    # FCNN
    # ============================================================

    print("\n"+"="*60)
    print("FULLY CONNECTED NEURAL NETWORK")
    print("="*60)

    data.subset_train(50000)

    net=FCNN([784,256,128,10])

    history=net.fit(
        data.X_train,
        data.y_train,
        CrossEntropy(),
        SGD(lr=.10,momentum=.9,decay=.98),
        epochs=40,
        batch=256
    )

    pred=net.predict(data.X_test)

    print("\nFCNN Test Accuracy :",accuracy(data.y_test,pred),"%")

    Visualizer.dashboard(
        net,
        data.X_test,
        data.y_test,
        pred,
        history,
        is_conv=False
    )

    # ============================================================
    # CONVNET
    # ============================================================

    print("\n"+"="*60)
    print("CONVOLUTIONAL NEURAL NETWORK")
    print("="*60)

    data=IDXDataset("images/mnist")

    Xtr=im2batch(data.X_train)
    Xte=im2batch(data.X_test)

    conv=ConvNet()

    history=conv.fit(
        Xtr,
        data.y_train,
        CrossEntropy(),
        SGD(lr=.01,momentum=.90,decay=.995),
        epochs=20,
        batch=128
    )

    pred=conv.predict(Xte)

    print("\nConvNet Test Accuracy :",accuracy(data.y_test,pred),"%")

    Visualizer.dashboard(
        conv,
        Xte.reshape(-1,784),
        data.y_test,
        pred,
        history,
        is_conv=True
    )

    # ============================================================
    # COMPARISON
    # ============================================================

    plt.figure(figsize=(6,4))
    plt.bar(["FCNN","ConvNet"],[
        accuracy(data.y_test,net.predict(batch2vec(Xte))),
        accuracy(data.y_test,pred)
    ])
    plt.ylabel("Test Accuracy (%)")
    plt.title("MNIST Performance Comparison")
    plt.grid(axis="y")
    plt.show()

    print("\nDone.")