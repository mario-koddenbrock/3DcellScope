
import os
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
import cv2


def imagesc3D(volume, mask=None, title=None, color=None, colorbar=True, initZ=None):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax.set_title('click to build line segments')
    #line, = ax.plot([0], [0])  # empty line
    #linebuilder = LineBuilder(line)

    #fig, ax = plt.subplots()
    if title is None:
        title=''
    #ax.userdata=(volume, mask, title, color, colorbar)
    #ax.volume = volume
    if initZ is not None:
        ax.index = initZ
        #ax.imgs=ax.imshow(volume[ax.index])
    else:
        ax.index = volume.shape[0] // 2
        #ax.imgs=ax.imshow(np.max(volume, axis=0))

    vMin = np.min(volume)
    vMax = np.max(volume)
    print('vMax = ',vMax)
    ax.imgs = ax.imshow(volume[ax.index])
    
    
    ax.userdata=(volume, mask, title, color, colorbar, vMin, vMax)
    update_slice(ax)
    colorbar=False # do not show several colorbar
    ax.userdata=(volume, mask, title, color, colorbar,  vMin, vMax)
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()


def imagesc3D_2(volume_list, mask_list=None, title_list=None,
                n_col = None, color=None,
                colorbar=True, initZ=None):

    if mask_list is not None:
        assert len(volume_list) == len(mask_list)

    fig = plt.figure()
    N = len(volume_list)
    cols = n_col if n_col is not None else N // 2
    rows = int(math.ceil(N / cols))
    # print('rows = {} cols = {}'.format(rows, cols))
    gs = gridspec.GridSpec(rows, cols)
    axes = []
    for n, volume in enumerate(volume_list):
        # print('n = {}'.format(n))
        # print('gs[n] = {}'.format(gs[n]))
        axes.append(fig.add_subplot(gs[n]))
        
    for ax in axes:
        mask = mask_list[n] if mask_list is not None else None
        title = title_list[n] if title_list is not None else ''
        ax.index = initZ if initZ is not None else volume.shape[0] // 2
        ax.imgs = ax.imshow(volume[ax.index])
        vMin=np.min(volume)
        vMax=np.max(volume)
        # print('vMax = ',vMax)
        ax.userdata=(volume, mask, title, color, colorbar, vMin, vMax)
        update_slice(ax)
        colorbar=False # do not show several colorbar
        ax.userdata=(volume, mask, title, color, colorbar,  vMin, vMax)
        
    fig.canvas.mpl_connect('key_press_event', process_key)
    fig.tight_layout()
    
    plt.show()


def process_key(event):
    fig = event.canvas.figure
    axes = fig.axes
    # for ax in axes:
    ax = fig.axes[0]
    # print('event.key='+event.key)
    if event.key == 'left':
        previous_slice(ax)
    elif event.key == 'right':
        next_slice(ax)
    fig.canvas.draw()


def previous_slice(ax):
    """Go to the previous slice."""
    #volume = ax.volume
    ax.index = (ax.index - 1) % ax.userdata[0].shape[0]  # wrap around using %
    update_slice(ax)
    #ax.title=str(ax.index)

def update_slice(ax):
    (volume, mask, title, color, colorbar, vMin, vMax)=ax.userdata
    im=np.copy(volume[ax.index])
    if mask is not None:
        if len(mask.shape)>2:
            mask=mask[ax.index % mask.shape[0]]
    (title, typeStr)=getImageDescription(volume, title+ " " +"Z="+str(ax.index)+"/"+str(volume.shape[0]))
    im=mergeImageWithMask(im, mask, color)
    #ax.images[0].set_array(im)
    ax.imgs.set_array(im)
    ax.set_title(title)
    if colorbar & (typeStr!="bool_"):
        cb=plt.colorbar(ax.images[0])
        cb.vmin=vMin
        cb.vmax=vMax


def next_slice(ax):
    """Go to the next slice."""
    #volume = ax.volume

    ax.index = (ax.index + 1) % ax.userdata[0].shape[0]
    update_slice(ax)

    #plt.imshow(im)
    #if typeStr != "bool_":
    #    plt.colorbar()
    #plt.title(title)
    #plt.show()

    #ax.images[0].set_array(im)
    #ax.set_title(title)

class LineBuilder:
    def __init__(self, line):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        print('click', event)
        if event.inaxes!=self.line.axes: return
        self.xs.append(event.xdata)
        self.ys.append(event.ydata)
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()


def mergeImageWithMask(im, mask=None, color=None):
    from skimage.morphology import erosion, dilation, ball
    if color is None:
        color=[1, 1, 1]
    if mask is not None:
        if np.prod(mask.shape) >0:
            # maskD=cv2.dilate(np.uint8(mask), np.ones((3,3)))
            maskD = dilation(np.uint8(mask), ball(3))
            maskD -= np.uint8(mask)
            maxVal = np.max(im)
            if len(im.shape)==3:
                for z in range(0, im.shape[2]):
                    im[maskD > 0] = maxVal * color
            else:
                im[maskD>0]=maxVal*color[0]
            # for r in range(0, im.shape[0]):
            #     for c in range(0, im.shape[1]):
            #         if len(im.shape)==3:
            #             for z in range(0, im.shape[2]):
            #                 if mask[r,c]>0 and np.min(mask[max(0,r-1):min(r+2, mask.shape[0]-1),max(0, c-1):min(c+2, mask.shape[1]-1)])==0:
            #                     im[r,c,z]=maxVal*color[z]
            #         else:
            #             if mask[r,c]>0 and np.min(mask[max(0,r-1):min(r+2, mask.shape[0]-1),max(0, c-1):min(c+2, mask.shape[1]-1)])==0:
            #                 im[r,c]=maxVal*color[0]
    return im

def getImageDescription(im, title=None):
    typeStr=str(im.dtype.type).split("'")[1].split('.')[1]
    if title is None:
        title=""

    title+= " " + typeStr + ' ['
    for c in range(0, len(im.shape)-1):
        title+=str(im.shape[c]) + ', '
    title+=str(im.shape[-1]) + ']'
    return title, typeStr

def imagesc(im, mask=None, title=None, color=None, colorbar=True):
    (title, typeStr)=getImageDescription(im, title)
    im=mergeImageWithMask(im, mask, color)

    plt.imshow(im)
    if (colorbar) & (typeStr != "bool_"):
        plt.colorbar()
    plt.title(title)
    plt.show()

def random_image(shape=(128, 128)):
    from scipy.ndimage.filters import gaussian_filter
    from skimage.measure import label
    img = gaussian_filter(np.random.normal(size=shape), min(shape) / 20)
    img = img > np.percentile(img, 80)
    img = label(img)
    img[img > 255] = img[img > 255] % 254 + 1
    return img


if __name__ == '__main__':
    # Small test
    data_shape = (64, 512, 512)  # must be (z, y, x) for 3D
    random_volume = random_image(data_shape)
    imagesc3D(random_volume)
