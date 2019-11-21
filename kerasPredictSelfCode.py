import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist



#load du lieu tu MINST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

'''
  X_train[50000:60000,:] co nghia la gi ?
  Giai thich:  X_train la 1 tuple chua  mang numpy darray gom 60000 phan tu, moi phan tu la 1 array cos dimension la 28*28
  50000:60000 o day tuc la lay ra 10000 phan tu tu 50000->5999999
  60000 o day chinh la so luong buc anh 28 * 28 chinh la so pixel cua moi buc anh, 1 npdarry se dai dien cho 1 tam anh
  dau":" o day co nghia la lay tat ca 28*28 phan trong 1 array con cua 10000 phan tu vua roi
  y_train[50000:60000] o day y la 1 numpydarray, chinh xac hon thi la array chua 60000 phan tu
  nen khi in ra man hinh y_train.shape ta duoc (60000, 1)
'''
X_val, y_val = X_train[50000:60000,:], y_train[50000:60000]  
X_train, y_trainn = X_train[:50000, :], y_train[:50000]


#3 Reshape lai du lieu cho dung kich thuoc input cua keras yeu cau
'''
 Dau vao cua dataset keras la 1 mang numpy darray 4 chieu vi the ta can resize lai
 do x_train la mang npdarray 3 chieu
 agurment thu nhat la so hang, chinh la shape[0]
 voi  2D->3D thi ta co the su dung tupple nhu kieu (28,28)
 nhung voi mang 3D->4D, ta chi viet cach nhu ben duoi
'''
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


#4 One hot coding  label (Y)

'''
	nputils.to_category() nhan dau vao lla 3 tham so 
	tham so thu nhat la 1 vector, chinh la 1 mang npdarray 1 chieu
	tham so thu 2 la tong so lop muon ma hoa
	tham so thu 3 la kieu du lieu nhu float, int..
'''
y_train = np_utils.to_categorical(y_train, 10)
y_val = np_utils.to_categorical(y_val, 10)
y_test = np_utils.to_categorical(y_test, 10)


#5 Dinh nghia model
'''
	Tao 1 ra doi tuong squential noi ma ta co the bo sung cac layer theo thu tu lan luot
	Sequential ban chat la 1 linear stack  cac layer do nguoi push vao
	nhan vao 3 tham so
	tham so thu nhat la self
	tham so thu 2 la so 1 list cac layer them vao
	tham so thu 3 la ten cua model
	khi moi khoi tao chua co layer nen layers mac dinh bang None
	name do nguoi dung tu dat
	ta hoan toan co the viet nhu nay:
	model = Sequential(layers = None, name = 'Sequential')
	thong thuong ta co the chi can khoi tao don gian
	modle - Sequential()
'''
model = Sequential()

# Thêm Convolutional layer với 32 kernel, kích thước kernel 3*3
# dùng hàm sigmoid làm activation và chỉ rõ input_shape cho layer đầu tiên

'''

	phuong thuc add() nhan 1 so tham so dau vao can thiet, do la:
	Tham so thu nhat la 1 2D layer nhu Conv2D, Dense ...
	ham Conv2D hoat dong tot voi du lieu 4D
	Tham so thu 2 la input_shape, input shape o day chinh la 1 tensor 3 chieu
	ma ta can chi ro cho layer  dau tien
	tensor nay phai co so chieu giong nhu so chieu cua du lieu dua vao
	Tham so thu 3 la ham activation nhu sigmod,relu...
	Ngoai ra con 1 so tham so khac, nhu batch_size, dim ...
	input_shape thuc ra co 4 argument, nhung keras mac dinh bo qua tham so dau tien la batch_size, so du lieu ta co trong tap train
	Vi sao keras yeu cau 1 mang npdarray 4 dau vao? 
	Do la do 1 argument cho so luong tam anh
	3 dau vao con lai chinh la dung de bieu dien cho 1 tam anh mau
	1 tam anh mau duoc bieu dien duoi ma tran 2 cheu, 1 phan tu la 1 tuple chua 3 gia tri RGB
	de tien cho viec tinh toan ta tach thanh 3 ma tran moi ma tran co kich thuoc bang nhau bang m*n
	voi m, n la kich thuoc cua 1 buc anh
	khi ta chong 3 buc anh do 3 ma tran channel tao ra ta duoc buc ah ban dau
'''

model.add(Conv2D(32, (3, 3), activation='sigmoid', input_shape=(28, 28,1)))

# Them convolution layer
model.add(Conv2D(32, (3,3), activation='sigmoid'))

# Them poolingmax
'''
	MaxPooling2D nhan vao 1 so doi so nhu pool_size, stride, padding,...
	Mac dinh pool_size = (2,2) va stride = None
'''
model.add(MaxPooling2D(pool_size=(2, 2), strides=2))

# Flatten layer chuyen tu tensor sang vector
model.add(Flatten())

# Them Fully Connected layer voi 128 nodes va dung ham sigmod
'''
	Dense la 2D layer
	Phuong thuc nhan 1 so tham so dau vao nhu so nodes hay unit,hamm activation, bias,...
'''
model.add(Dense(128, activation='sigmod'))

# Output layer voi 10 node va dung ham softmax  function de chuyen sang xac suat
model.add(Dense(10, activation='softmax'))

#6. complie mode;;, chi ro loss function nao duoc su dung, phuong thuc dung de toi uu hoa ham lossfunction

model.compile(loss='categorical_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])

#7Thuc hien train model voi data
'''
	Phuong thuc fit() dung de train model
	fit() nhan 1 so tham so dau vao nhu tap du lieu vao, tap du lieu dich, 1 tuple valiadatio_data
	batch-size, epoch, verbose, callback,...
	Khi so luong qua lon ta can chia nho tap du lieu de dua vao mang, con  goi la lo, bacth
	batch_size = N, tuc la dua N du lieu vao 1 lan 
	epoch la chu trinh khi ta dua duoc het tat ca du lieu vao mang,gom ca 2 buoc forward va backforward cho viec cap nhat lai trong so
	voi 1 epoch ham loss function van con lon, ta chua toi uu hoa duoc loss function, hay noi cach khac mo hinh mang chua hoc duoc gi nhieu
	do do ta can nhieu epoch hon, thuong lay la 10, 100, 1000...
	Verbose = 1 se bieu thi 1 thanh tien trinh cho viec train mo hinh
	verbose= 0 khong bieu thi gi het
	Verbose = 2 chi de cap de so luong verbose hien tai
'''
H = model.fit(X_train, y_train, validation_data=(X_val, y_val),
		   batch_size=32, epoch=10, verbose=1)


#8. Ve do thi loss, accrucary cua trainning set va validation set
fig = plt.figure()
numOfEpoch = 10
plt.plot(np.arange(0, numOfEpoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, numOfEpoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, numOfEpoch), H.history['acc'], label='accuracy')
plt.plot(np.arange(0, numOfEpoch), H.history['val_acc'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()

#9. Danh gia model voi du lieu test set
score = model.evaluate(X_test, y_test, verbose=0)
print(score)

#10. Du doanh anh
plt.imshow(X_test[0].reshape(28, 28), cmap='gray')

'''
	Can reshape ve (1,28, 28) la vi mo hinh nhan dau vao la tensor 3 chieu
	(1,28,28) dai dien cho 1 buc anh co kich thuoc la 28*28
'''
y_predict = model.predict(X_test[0].reshape(1, 28, 28))
print('Gia tri du doan:', np.argmax(y_predict))