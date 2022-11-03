
import imageio
import os
import glob
from skimage import io
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import color

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

            cropped = image[xStart:xEnd:3,yStart:yEnd:3]

            imageio.imwrite(f'/Users/sebastiandresbach/Desktop/{modality}_{stimDur}/frame{i:02d}.png', cropped)


    template = f'/Users/sebastiandresbach/Desktop/neurovascularCouplingFigures/fig-{modality}Activation.png'
    image = io.imread(template)
    stim = '/Users/sebastiandresbach/github/neurovascularCouplingVASO/code/stimulation/visual_0.png'
    stim = io.imread(stim,as_gray=False)
    stim = stim[::10,::10]
    stim = color.gray2rgb(stim)

    maxFrames = len(glob.glob(f'/Users/sebastiandresbach/Desktop/{modality}_24/Scree*.png'))

    for frame in range(maxFrames):
        print(f'Making frame #{frame}')
        image = io.imread(template)

        for i, stimDur in enumerate([1,2,4,12,24]):

            # Get max number of frames for stim duration
            maxFramesStim = len(glob.glob(f'/Users/sebastiandresbach/Desktop/{modality}_{stimDur}/Scree*.png'))-1

            if frame < maxFramesStim:
                tmp = io.imread(f'/Users/sebastiandresbach/Desktop/{modality}_{stimDur}/frame{frame:02d}.png')

            else:
                tmp = io.imread(f'/Users/sebastiandresbach/Desktop/{modality}_{stimDur}/frame{maxFramesStim:02d}.png')

            tmp = tmp[-148:,-448:,:]
            image[316:464,xStarts[i]:xEnds[i],:] = tmp

            if frame*0.785 < stimDur:
                if frame % 2 == 0:
                    stim = '/Users/sebastiandresbach/github/neurovascularCouplingVASO/code/stimulation/visual_0.png'
                if frame % 2 != 0:
                    stim = '/Users/sebastiandresbach/github/neurovascularCouplingVASO/code/stimulation/visual_1.png'
                stim = io.imread(stim)
                stim = stim[::10,::10]
                stim = color.gray2rgb(stim)
                stim.shape

                image[387:464,xEnds[i]-103:xEnds[i],:3] = stim

        imageio.imwrite(f'/Users/sebastiandresbach/Desktop/gifFrames/{modality}_{frame:02d}.png', image)

    # Assemble gif
    files = sorted(glob.glob(f'/Users/sebastiandresbach/Desktop/gifFrames/{modality}*.png'))
    # files = sorted(glob.glob(dumpFolder + '/*.png'))
    print(f'Creating gif from {len(files)} images')

    images = []
    for file in files:
        # print(f'Adding {file}')
        filedata = imageio.imread(file)
        images.append(filedata)

    print('Collected files')
    print('Assembling gif')
    imageio.mimsave(f'/Users/sebastiandresbach/Desktop/gifFrames/{modality}_movie.gif', images, duration = 1/4)
    # print('Deleting dump directory')
    # os.system(f'rm -r {dumpFolder}')
    print('Done.')
