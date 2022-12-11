
import imageio
import os
import glob
from skimage import io, transform
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import color
import numpy as np

starts = {'bold': [3330, 3753, 4175, 4602, 5026],
          'vaso': [996, 1420, 1842, 2269, 2693]}

SF = 2400/1000




template = f'/Users/sebastiandresbach/Desktop/neurovascularCouplingFigures/OHBM/fig1-2.png'
image = io.imread(template)

for modality in ['bold']:

    tmpStarts = starts[modality]

    for stimDur in [1,2,4,12,24]:
    # for stimDur in [12,24]:
        # Cropping images
        files = sorted(glob.glob(f'/Users/sebastiandresbach/Desktop/items/{modality}_{stimDur}/Scree*.png'))
        # files = sorted(glob.glob(dumpFolder + '/*.png'))
        print(f'Cropping {len(files)} images')


        for i, file in enumerate(files):
            base = file.split('.')[0]
            image = io.imread(file)
            xStart = 1200
            xEnd = 1770
            yStart = 1100
            yEnd = 2500

            cropped = image[xStart:xEnd,yStart:yEnd]

            imageio.imwrite(f'/Users/sebastiandresbach/Desktop/items/{modality}_{stimDur}/frame{i:02d}.png', cropped)



    # stim = '/Users/sebastiandresbach/github/neurovascularCouplingVASO/code/stimulation/visual_0.png'
    # stim = io.imread(stim,as_gray=False)
    # stim = stim[::10,::10]
    # stim = color.gray2rgb(stim)
    #
    # maxFrames = len(glob.glob(f'/Users/sebastiandresbach/Desktop/items/{modality}_24/Scree*.png'))
    #
    # yStarts = np.asarray(tmpStarts)
    # yEnds = yStarts + 332

# tmp = io.imread(f'/Users/sebastiandresbach/Desktop/{modality}_{stimDur}/frame{frame:02d}.png')
# # tmp.shape
# # tmp = tmp[-333:,-1174:-175,:]
# # tmp.shape
# # plt.imshow(tmp)




from imageio import imsave


for frame in range(maxFrames):
# for frame in range(2):
    print(f'Making frame #{frame}')
    image = io.imread(template)

    for modality in ['bold', 'vaso']:
        tmpStarts = starts[modality]


        for i, stimDur in enumerate([1,2,4,12,24]):

            # Get max number of frames for stim duration
            maxFramesStim = len(glob.glob(f'/Users/sebastiandresbach/Desktop/items/{modality}_{stimDur}/Scree*.png'))-1

            if frame < maxFramesStim:
                tmp = io.imread(f'/Users/sebastiandresbach/Desktop/items/{modality}_{stimDur}/frame{frame:02d}.png')

            else:
                tmp = io.imread(f'/Users/sebastiandresbach/Desktop/items/{modality}_{stimDur}/frame{maxFramesStim:02d}.png')

            tmp = tmp[-332:,-1174:-175,:]
            plt.imshow(tmp)

            rottmp = transform.rotate(tmp, 90, resize=True)
            plt.imshow(rottmp)
            rottmp.shape


            image[tmpStarts[i]:tmpStarts[i]+332,1296:2295,:] = tmp

            if frame*0.785 < stimDur:
                if frame % 2 == 0:
                    stim = '/Users/sebastiandresbach/github/neurovascularCouplingVASO/code/stimulation/visual_0.png'
                if frame % 2 != 0:
                    stim = '/Users/sebastiandresbach/github/neurovascularCouplingVASO/code/stimulation/visual_1.png'

                stim = io.imread(stim)
                stim = stim[::5,::5]
                stim = color.gray2rgb(stim)

                image[tmpStarts[i]+332-154:tmpStarts[i]+332, 2090:2295,:3] = stim

        # imageio.imwrite(f'/Users/sebastiandresbach/Desktop/gifFrames/{modality}_{frame:02d}_large.png', image)

    # image = image[::3,::3,:]
    imageio.imwrite(f'/Users/sebastiandresbach/Desktop/gifFrames/{frame:02d}_small.png', image)


from skimage.transform import rescale, resize, downscale_local_mean
import PIL
from PIL import Image
from imageio import imsave




# compress images
import subprocess
images = sorted(glob.glob(f'/Users/sebastiandresbach/Desktop/gifFrames/*small.png'))

for image in images:
    print(f'working on {image}')

    filedata = imageio.imread(image)
    image_rescaled = rescale(filedata, 1/2.4, anti_aliasing=False, multichannel=True)

    imageio.imwrite(f'/Users/sebastiandresbach/Desktop/gifFrames/{frame:02d}_small_rescaled.png', image_rescaled)

    img = Image.open(f'/Users/sebastiandresbach/Desktop/gifFrames/{frame:02d}_small_rescaled.png')
    # img = io.imread(image)
    baseName = image.split('.')[0]
    # img = img.astype(np.uint8)
    # imsave(f'{baseName}_test.png', (color.convert_colorspace(img, 'HSV', 'RGB')*255).astype(np.uint8))

    rgb_im = img.convert('RGB')

    baseName = image.split('.')[0]
    rgb_im.save(baseName+'compressed.jpg')

files = sorted(glob.glob(f'/Users/sebastiandresbach/Desktop/gifFrames/*.jpg'))
for file in files:
    baseName = file.split('.')[0].split('/')[-1]

    subprocess.run(f'cp {file} /Users/sebastiandresbach/Desktop/gifFrames/jpg/{baseName}.jpg', shell=True)



# Assemble gif
# for size in ['large', 'small']:
for size in ['small']:
    files = sorted(glob.glob(f'/Users/sebastiandresbach/Desktop/gifFrames/*.jpg'))
    # files = sorted(glob.glob(dumpFolder + '/*.png'))
    print(f'Creating gif from {len(files)} images')
    print('Collected files')
    images = []
    for file in files:
        # print(f'Adding {file}')
        filedata = imageio.imread(file)

        images.append(filedata)


    print('Assembling gif')
    imageio.mimsave(f'/Users/sebastiandresbach/Desktop/gifFrames/movie_{size}.gif', images, duration = 1/4)
    # print('Deleting dump directory')
    # os.system(f'rm -r {dumpFolder}')





# =============================================================================
# rotated version
# =============================================================================


startsVertical = {'bold': 2625,
                  'vaso': 1575}

startsHorizontal = [207, 605, 1004, 1403, 1802]

template = f'/Users/sebastiandresbach/Desktop/neurovascularCouplingFigures/OHBM/fig1-1.png'
image = io.imread(template)

for frame in range(maxFrames):
# for frame in range(2):
    print(f'Making frame #{frame}')
    image = io.imread(template)

    for modality in ['bold', 'vaso']:
        startVertical = startsVertical[modality]


        for i, stimDur in enumerate([1,2,4,12,24]):

            # Get max number of frames for stim duration
            maxFramesStim = len(glob.glob(f'/Users/sebastiandresbach/Desktop/items/{modality}_{stimDur}/Scree*.png'))-1

            if frame < maxFramesStim:
                tmp = io.imread(f'/Users/sebastiandresbach/Desktop/items/{modality}_{stimDur}/frame{frame:02d}.png')

            else:
                tmp = io.imread(f'/Users/sebastiandresbach/Desktop/items/{modality}_{stimDur}/frame{maxFramesStim:02d}.png')

            tmp = tmp[-332:,-1174:-175,:]
            plt.imshow(tmp)

            rottmp = transform.rotate(tmp, 90, resize=True, preserve_range=True)

            # plt.imshow(rottmp)
            # rottmp.shape


            image[startVertical:startVertical+999,startsHorizontal[i]:startsHorizontal[i]+332,:] = rottmp


            if frame*0.785 < stimDur:
                if frame % 2 == 0:
                    stim = '/Users/sebastiandresbach/github/neurovascularCouplingVASO/code/stimulation/visual_0.png'
                if frame % 2 != 0:
                    stim = '/Users/sebastiandresbach/github/neurovascularCouplingVASO/code/stimulation/visual_1.png'

                stim = io.imread(stim)
                stim = stim[::5,::5]
                stim = color.gray2rgb(stim)

                image[startVertical+999-154:startVertical+999,startsHorizontal[i]+332-205:startsHorizontal[i]+332,:3] = stim

        # imageio.imwrite(f'/Users/sebastiandresbach/Desktop/gifFrames/{modality}_{frame:02d}_large.png', image)

    # image = image[::3,::3,:]
    imageio.imwrite(f'/Users/sebastiandresbach/Desktop/gifFrames/{frame:02d}_small.png', image)

import subprocess
images = sorted(glob.glob(f'/Users/sebastiandresbach/Desktop/gifFrames/*small.png'))

for image in images:
    print(f'working on {image}')

    filedata = imageio.imread(image)
    image_rescaled = rescale(filedata, 1/2.4, anti_aliasing=False, multichannel=True)

    imageio.imwrite(f'/Users/sebastiandresbach/Desktop/gifFrames/{frame:02d}_small_rescaled.png', image_rescaled)

    img = Image.open(f'/Users/sebastiandresbach/Desktop/gifFrames/{frame:02d}_small_rescaled.png')
    # img = io.imread(image)
    baseName = image.split('.')[0]
    # img = img.astype(np.uint8)
    # imsave(f'{baseName}_test.png', (color.convert_colorspace(img, 'HSV', 'RGB')*255).astype(np.uint8))

    rgb_im = img.convert('RGB')

    baseName = image.split('.')[0]
    rgb_im.save(baseName+'compressed.jpg')
