import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files


ds = pydicom.dcmread("example0.dcm")
print("ds type".center(80, "-"))

print(type(ds))
print("ds content".center(80, "-"))

print(ds)
print(type(ds.pixel_array))

#print("{:-^80".format("shape"))
print("shape".center(80, "-"))
print(ds.pixel_array.shape)

print("ds array".center(80, "-"))
print(ds.pixel_array)

ds2 = pydicom.dcmread("example1.dcm")

fig = plt.figure()
ax = fig.add_subplot(2, 1, 1);
ax.set_title("example0.dcm")
ax.imshow(ds.pixel_array, cmap=plt.cm.gray)
ax = fig.add_subplot(2, 1, 2);
ax.set_title("example1.dcm")
ax.imshow(ds2.pixel_array, cmap=plt.cm.gray)

plt.tight_layout()
plt.show()
