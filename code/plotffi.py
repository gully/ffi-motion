from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
def plot_ffi(filename, **kwargs):
    ffi = fits.open(filename)
    n=10
    fig, ax = plt.subplots(n, n, figsize=(30, 30), sharey=True, sharex=True)
    for i in range(n):
        for j in range(n):
            ax[i,j].axis('off')
            ax.set_facecolor('black')
    plt.xticks([])
    plt.yticks([])
    plt.subplots_adjust(hspace=0.02, wspace=0.02)


    #4, 3, 1, 2
    ax[0, 2].imshow(ffi[4].data[::-1, :], origin='top left', **kwargs)
    ax[0, 3].imshow(ffi[3].data[::-1, ::-1], origin='top right', **kwargs)
    ax[1, 2].imshow(ffi[1].data, origin='bottom left',**kwargs)
    ax[1, 3].imshow(ffi[2].data[:, ::-1], origin='bottom right', **kwargs)

    ax[2, 2].imshow(ffi[20].data[::-1, :], origin='top left', **kwargs)
    ax[2, 3].imshow(ffi[19].data[::-1, ::-1], origin='top right', **kwargs)
    ax[3, 2].imshow(ffi[17].data, origin='bottom left',**kwargs)
    ax[3, 3].imshow(ffi[18].data[:, ::-1], origin='bottom right', **kwargs)

    ax[0, 4].imshow(ffi[8].data[::-1, :], origin='top left', **kwargs)
    ax[0, 5].imshow(ffi[7].data[::-1, ::-1], origin='top right', **kwargs)
    ax[1, 4].imshow(ffi[5].data, origin='bottom left',**kwargs)
    ax[1, 5].imshow(ffi[6].data[:, ::-1], origin='bottom right', **kwargs)

    ax[2, 4].imshow(ffi[24].data[::-1, :], origin='top left', **kwargs)
    ax[2, 5].imshow(ffi[23].data[::-1, ::-1], origin='top right', **kwargs)
    ax[3, 4].imshow(ffi[21].data, origin='bottom left',**kwargs)
    ax[3, 5].imshow(ffi[22].data[:, ::-1], origin='bottom right', **kwargs)

    ax[0, 6].imshow(ffi[12].data[::-1, :], origin='top left', **kwargs)
    ax[0, 7].imshow(ffi[11].data[::-1, ::-1], origin='top right', **kwargs)
    ax[1, 6].imshow(ffi[9].data, origin='bottom left',**kwargs)
    ax[1, 7].imshow(ffi[10].data[:, ::-1], origin='bottom right', **kwargs)


    # 3, 2, 4, 1
    ax[2, 0].imshow(ffi[15].data[::-1, :].T[::-1, ::-1], **kwargs)
    ax[2, 1].imshow(ffi[14].data[::-1, ::-1].T, **kwargs)
    ax[3, 0].imshow(ffi[16].data.T, **kwargs)
    ax[3, 1].imshow(ffi[13].data[:, ::-1].T[::-1, ::-1], **kwargs)

    ax[4, 0].imshow(ffi[35].data[::-1, :].T[::-1, ::-1], **kwargs)
    ax[4, 1].imshow(ffi[34].data[::-1, ::-1].T, **kwargs)
    ax[5, 0].imshow(ffi[36].data.T, **kwargs)
    ax[5, 1].imshow(ffi[33].data[:, ::-1].T[::-1, ::-1], **kwargs)

    ax[6, 0].imshow(ffi[55].data[::-1, :].T[::-1, ::-1], **kwargs)
    ax[6, 1].imshow(ffi[54].data[::-1, ::-1].T, **kwargs)
    ax[7, 0].imshow(ffi[56].data.T, **kwargs)
    ax[7, 1].imshow(ffi[53].data[:, ::-1].T[::-1, ::-1], **kwargs)

    ax[4, 2].imshow(ffi[39].data[::-1, :].T[::-1, ::-1], **kwargs)
    ax[4, 3].imshow(ffi[38].data[::-1, ::-1].T, **kwargs)
    ax[5, 2].imshow(ffi[40].data.T, **kwargs)
    ax[5, 3].imshow(ffi[37].data[:, ::-1].T[::-1, ::-1], **kwargs)

    ax[6, 2].imshow(ffi[59].data[::-1, :].T[::-1, ::-1], **kwargs)
    ax[6, 3].imshow(ffi[58].data[::-1, ::-1].T, **kwargs)
    ax[7, 2].imshow(ffi[60].data.T, **kwargs)
    ax[7, 3].imshow(ffi[57].data[:, ::-1].T[::-1, ::-1], **kwargs)

    ax[4, 4].imshow(ffi[43].data[::-1, :].T[::-1, ::-1], **kwargs)
    ax[4, 5].imshow(ffi[42].data[::-1, ::-1].T, **kwargs)
    ax[5, 4].imshow(ffi[44].data.T, **kwargs)
    ax[5, 5].imshow(ffi[41].data[:, ::-1].T[::-1, ::-1], **kwargs)


    #1, 4, 2, 3
    ax[2, 6].imshow(ffi[25].data[::-1, :].T[::-1, ::-1], origin='top left', **kwargs)
    ax[2, 7].imshow(ffi[28].data[::-1, ::-1].T, origin='top right', **kwargs)
    ax[3, 6].imshow(ffi[26].data.T, origin='bottom left',**kwargs)
    ax[3, 7].imshow(ffi[27].data[:, ::-1].T[::-1, ::-1], origin='bottom right', **kwargs)

    ax[2, 8].imshow(ffi[29].data[::-1, :].T[::-1, ::-1], origin='top left', **kwargs)
    ax[2, 9].imshow(ffi[32].data[::-1, ::-1].T, origin='top right', **kwargs)
    ax[3, 8].imshow(ffi[30].data.T, origin='bottom left',**kwargs)
    ax[3, 9].imshow(ffi[31].data[:, ::-1].T[::-1, ::-1], origin='bottom right', **kwargs)

    ax[4, 6].imshow(ffi[45].data[::-1, :].T[::-1, ::-1], origin='top left', **kwargs)
    ax[4, 7].imshow(ffi[48].data[::-1, ::-1].T, origin='top right', **kwargs)
    ax[5, 6].imshow(ffi[46].data.T, origin='bottom left',**kwargs)
    ax[5, 7].imshow(ffi[47].data[:, ::-1].T[::-1, ::-1], origin='bottom right', **kwargs)

    ax[4, 8].imshow(ffi[49].data[::-1, :].T[::-1, ::-1], origin='top left', **kwargs)
    ax[4, 9].imshow(ffi[52].data[::-1, ::-1].T, origin='top right', **kwargs)
    ax[5, 8].imshow(ffi[50].data.T, origin='bottom left',**kwargs)
    ax[5, 9].imshow(ffi[51].data[:, ::-1].T[::-1, ::-1], origin='bottom right', **kwargs)

    ax[6, 8].imshow(ffi[69].data[::-1, :].T[::-1, ::-1], origin='top left', **kwargs)
    ax[6, 9].imshow(ffi[72].data[::-1, ::-1].T, origin='top right', **kwargs)
    ax[7, 8].imshow(ffi[70].data.T, origin='bottom left',**kwargs)
    ax[7, 9].imshow(ffi[71].data[:, ::-1].T[::-1, ::-1], origin='bottom right', **kwargs)


    #2, 1, 3, 4
    ax[6, 4].imshow(ffi[62].data[::-1, :], origin='top left', **kwargs)
    ax[6, 5].imshow(ffi[61].data[::-1, ::-1], origin='top right', **kwargs)
    ax[7, 4].imshow(ffi[63].data, origin='bottom left',**kwargs)
    ax[7, 5].imshow(ffi[64].data[:, ::-1], origin='bottom right', **kwargs)

    ax[6, 6].imshow(ffi[66].data[::-1, :], origin='top left', **kwargs)
    ax[6, 7].imshow(ffi[65].data[::-1, ::-1], origin='top right', **kwargs)
    ax[7, 6].imshow(ffi[67].data, origin='bottom left',**kwargs)
    ax[7, 7].imshow(ffi[68].data[:, ::-1], origin='bottom right', **kwargs)

    ax[8, 2].imshow(ffi[74].data[::-1, :], origin='top left', **kwargs)
    ax[8, 3].imshow(ffi[73].data[::-1, ::-1], origin='top right', **kwargs)
    ax[9, 2].imshow(ffi[75].data, origin='bottom left',**kwargs)
    ax[9, 3].imshow(ffi[76].data[:, ::-1], origin='bottom right', **kwargs)

    ax[8, 4].imshow(ffi[78].data[::-1, :], origin='top left', **kwargs)
    ax[8, 5].imshow(ffi[77].data[::-1, ::-1], origin='top right', **kwargs)
    ax[9, 4].imshow(ffi[79].data, origin='bottom left',**kwargs)
    ax[9, 5].imshow(ffi[80].data[:, ::-1], origin='bottom right', **kwargs)

    ax[8, 6].imshow(ffi[82].data[::-1, :], origin='top left', **kwargs)
    ax[8, 7].imshow(ffi[81].data[::-1, ::-1], origin='top right', **kwargs)
    ax[9, 6].imshow(ffi[83].data, origin='bottom left',**kwargs)
    ax[9, 7].imshow(ffi[84].data[:, ::-1], origin='bottom right', **kwargs)
    return fig
