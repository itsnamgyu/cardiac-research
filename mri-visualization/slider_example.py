import matplotlib.pyplot as plt
import pydicom

dicoms = []
dicoms.append(pydicom.dcmread("example0.dcm"))
dicoms.append(pydicom.dcmread("example1.dcm"))

index = 0

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

TITLE = "Short Axis"

def update_plot():
    global index, fig, ax

    fig.clf()
    ax = fig.add_subplot(1,1,1)
    plt.title(TITLE)
    ax.imshow(dicoms[index].pixel_array, cmap=plt.cm.gray)
    fig.canvas.draw()


def on_key_press(event):
    global index, fig, ax

    fig.canvas.draw()
    
    if event.key == 'left':
        index -= 1
    if event.key == 'right':
        index += 1
    index = index % 2
    update_plot()


def on_button_press(event):
    global index, fig, ax, TITLE

    fig.canvas.draw()
    
    if event.inaxes == ax:
        TITLE = "Short Axis (Clicked)"
    else:
        TITLE = "Short Axis"

    update_plot()


cid = fig.canvas.mpl_connect('key_press_event', on_key_press)
cid = fig.canvas.mpl_connect('button_press_event', on_button_press)

update_plot()
plt.tight_layout()
plt.show()
