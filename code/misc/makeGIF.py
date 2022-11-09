
import imageio
import os
import glob
from skimage import io
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import color
import numpy as np
for modality in ['bold']:
    for stimDur in [1,2,4,12,24]:
    # for stimDur in [12,24]:
        # Cropping images
        files = sorted(glob.glob(f'/Users/sebastiandresbach/Desktop/{modality}_{stimDur}/Scree*.png'))
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

            imageio.imwrite(f'/Users/sebastiandresbach/Desktop/{modality}_{stimDur}/frame{i:02d}.png', cropped)


    template = f'/Users/sebastiandresbach/Desktop/neurovascularCouplingFigures/fig-{modality}ActivationSquare.png'
    image = io.imread(template)
    stim = '/Users/sebastiandresbach/github/neurovascularCouplingVASO/code/stimulation/visual_0.png'
    stim = io.imread(stim,as_gray=False)
    stim = stim[::10,::10]
    stim = color.gray2rgb(stim)

    maxFrames = len(glob.glob(f'/Users/sebastiandresbach/Desktop/{modality}_24/Scree*.png'))

    yStarts = np.asarray([148, 572, 994, 1421, 1845])
    yEnds = yStarts + 332

# tmp = io.imread(f'/Users/sebastiandresbach/Desktop/{modality}_{stimDur}/frame{frame:02d}.png')
# # tmp.shape
# # tmp = tmp[-333:,-1174:-175,:]
# # tmp.shape
# # plt.imshow(tmp)

    for frame in range(maxFrames):
    # for frame in range(2):
        print(f'Making frame #{frame}')
        image = io.imread(template)

        for i, stimDur in enumerate([1,2,4,12,24]):

            # Get max number of frames for stim duration
            maxFramesStim = len(glob.glob(f'/Users/sebastiandresbach/Desktop/{modality}_{stimDur}/Scree*.png'))-1

            if frame < maxFramesStim:
                tmp = io.imread(f'/Users/sebastiandresbach/Desktop/{modality}_{stimDur}/frame{frame:02d}.png')

            else:
                tmp = io.imread(f'/Users/sebastiandresbach/Desktop/{modality}_{stimDur}/frame{maxFramesStim:02d}.png')

            tmp = tmp[-332:,-1174:-175,:]
            image[yStarts[i]:yEnds[i],1296:2295,:] = tmp

            if frame*0.785 < stimDur:
                if frame % 2 == 0:
                    stim = '/Users/sebastiandresbach/github/neurovascularCouplingVASO/code/stimulation/visual_0.png'
                if frame % 2 != 0:
                    stim = '/Users/sebastiandresbach/github/neurovascularCouplingVASO/code/stimulation/visual_1.png'
                stim = io.imread(stim)
                stim = stim[::5,::5]
                stim = color.gray2rgb(stim)
                stim.shape

                image[yEnds[i]-154:yEnds[i], 2090:2295,:3] = stim

        # imageio.imwrite(f'/Users/sebastiandresbach/Desktop/gifFrames/{modality}_{frame:02d}_large.png', image)

        image = image[::3,::3,:]
        imageio.imwrite(f'/Users/sebastiandresbach/Desktop/gifFrames/{modality}_{frame:02d}_small.png', image)

    # Assemble gif
    # for size in ['large', 'small']:
    for size in ['small']:
        files = sorted(glob.glob(f'/Users/sebastiandresbach/Desktop/gifFrames/{modality}*{size}.png'))
        # files = sorted(glob.glob(dumpFolder + '/*.png'))
        print(f'Creating gif from {len(files)} images')

        images = []
        for file in files:
            # print(f'Adding {file}')
            filedata = imageio.imread(file)
            images.append(filedata)

        print('Collected files')
        print('Assembling gif')
        imageio.mimsave(f'/Users/sebastiandresbach/Desktop/gifFrames/{modality}_movie_{size}.gif', images, duration = 1/3)
        # print('Deleting dump directory')
        # os.system(f'rm -r {dumpFolder}')



    print('Done.')
