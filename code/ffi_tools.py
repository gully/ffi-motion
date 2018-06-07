import numpy as np
#import fitsio
import matplotlib.pyplot as plt
import sep


def preprocess_dFFI(raw_data):
    """Preprocess the driftscan FFI

    Currently performs these pre processing steps:
    - Trims the edges to remove dark rows and collateral rows.
      TODO: currently hard coded to 50 and 60 pixel edges
    - Must use numpy C order to comply with sep's C functions

    Args:
        raw_data (numpy.ndarray): 2D numpy of one FFI channel

    Returns:
        trimmed_data (numpy.ndarray): A trimmed version of the input
    """
    raw_data = raw_data.copy(order='C')
    return raw_data[50:-50, 60:-60]



def quick_plot(data):
    """Make a quick plot of the FFI with sane screen stretch

    Returns:
        None
    """
    m, s = np.mean(data), np.std(data)
    plt.imshow(data, interpolation='nearest', cmap='gray', vmin=m-s, vmax=m+s, origin='lower')
    plt.colorbar();


def background_subtract(data, plot_bkg=False, return_bkg=False):
    """Background subtract the driftscan FFI

    Performs these steps:
    - Computes a mask based on the top 95th percentile of values
    - Estimate background with sep

    Args:
        data (numpy.ndarray): 2D numpy of one FFI channel
        plot_bkg (bool): Flag for whether to plot the background
                         default = False
        return_bkg (bool): Flag for whether to return the background
                         default = False

    Returns:
        background-subtracted FFI or optionally
        tuple of background-subtracted FFI, and background estimate
    """
    data = data.copy(order='C')
    mask = data > np.percentile(data, 95)
    bkg = sep.Background(data, mask=mask)

    if plot_bkg:
        bkg_image = bkg.back()
        # show the background
        plt.imshow(bkg_image, interpolation='nearest', cmap='gray', origin='lower')
        plt.colorbar();

    if return_bkg:
        return (data - bkg, bkg)
    else:
        return data - bkg


def get_kernel(read_saved=True, estimate_from=None, xy=[250,310,810,985]):
    """Get an estimate for the Kernel for this channel

    Defaults to reading a local kernel.

    Args:
        read_saved (bool): If True, return locally saved kernel
        estimate_from (numpy.ndarray): 2D numpy of one FFI channel
            The data to estimate from if read_saved=False
        xy (list): Limits of window around robust driftscan segment

    Returns:
        background-subtracted FFI or optionally
        tuple of background-subtracted FFI, and background estimate
    """

    if read_saved:
        return np.load('../data/ch41_FFI1_structured_kernel.npy')

    # Otherwise, compute everything from scratch...
    data = estimate_from

    kernel_raw = data[xy[0]:xy[1], xy[2]:xy[3]]
    kernel_raw = kernel_raw / np.percentile(kernel_raw, 95)
    ny, nx = kernel_raw.shape
    kernel_x = kernel_raw.mean(axis=0)
    kernel_y = kernel_raw.mean(axis=1)

    # Draw a line defined by the marginals
    x0, x1 = np.where(kernel_x/np.max(kernel_x) >0.5)[0][[0, -1]]
    y0, y1 = np.where(kernel_y/np.max(kernel_y) >0.5)[0][[0, -1]]
    line_vec_x, line_vec_y = np.arange(x0, x1), np.linspace(y0, y1, x1-x0)

    kernel_mask = kernel_raw*0.0

    for i,j in zip(line_vec_x, line_vec_y):
        kernel_mask[np.int(j), np.int(i)] = 1

    #Convolve the line with a 2 pixel Gaussian blur.
    from scipy.ndimage import gaussian_filter
    # Smooth hotdog-like kernel:
    final_kernel = gaussian_filter(kernel_mask, 2)
    # option for structured, Gaussian-tapered kernel:
    #final_kernel = np.abs(final_kernel*kernel_raw)
    return final_kernel

def get_aperture_mask(kernel, boxcar_size):
    """Get an estimate for the Kernel for this channel

    Defaults to reading a local kernel.

    Args:
        read_saved (bool): If True, return locally saved kernel
        estimate_from (numpy.ndarray): 2D numpy of one FFI channel
            The data to estimate from if read_saved=False
        xy (list): Limits of window around robust driftscan segment

    Returns:
        background-subtracted FFI or optionally
        tuple of background-subtracted FFI, and background estimate
    """

    #aper_mask = kernel
    from scipy.ndimage import gaussian_filter
    from scipy.signal import convolve
    aper_mask = gaussian_filter(kernel, 1)
    aper_mask = aper_mask / np.max(aper_mask)

    boxcar = np.ones((boxcar_size,boxcar_size))
    aper_mask = convolve(aper_mask,boxcar, mode='same')
    aper_mask = aper_mask >0.01
    return aper_mask


def estimate_driftscan_line_trace(data, xc, yc, aper_mask, show_plot=False):
    """Estimate the line trace of a driftscan line segment

    Args:
        data (numpy.ndarray): Background subtracted driftscan FFI
        xc (int): x index position of center of star trail
        yc (int): y index position of center of star trail
        show_plot (bool): Whether to show the FFI window-aperture overplot

    Returns:
        line_trace, flag: aperture photometry of line trace, and flag if it fails
    """
    ny, nx = aper_mask.shape
    x0 = np.int(xc - nx / 2)
    y0 = np.int(yc - ny / 2)
    ffi_cutout = data[y0:y0+ny, x0:x0+nx]

    # Plot every 8th trace
    if show_plot:
        m, s = np.mean(ffi_cutout), np.std(ffi_cutout)
        plt.imshow(aper_mask, interpolation='nearest', cmap='BuGn', vmin=0, vmax=1, origin='lower')
        plt.imshow(ffi_cutout, interpolation='nearest', cmap='gray', vmin=m-s, vmax=m+s, origin='lower', alpha=0.3)
        plt.show()

    try:
        output = np.sum(ffi_cutout * aper_mask, axis=0)
        flag = True
    except:
        output = np.zeros(nx)
        flag = False
    return (output, flag)


def iterate_line_traces(data, xcs, ycs, aper_mask, show_every_n_plots=np.NaN):
    """Estimate the line traces for many input x, y coordinates

    Assumes the same aperture mask for all traces

    Args:
        data (numpy.ndarray): Background subtracted driftscan FFI
        xcs (array-like): x indices of center of star trail
        ycs (array-like): y indices of center of star trail
        show_every_n_plots (int): Show every Nth plot

    Returns:
        line_traces: array of aperture photometry of line traces
    """
    n_sources = len(xcs)
    traces = []
    flags = []
    for i in range(n_sources):
        show_plot = (i % show_every_n_plots) == 0
        trace, flag = estimate_driftscan_line_trace(data, xcs[i], ycs[i], aper_mask, show_plot=show_plot)
        traces.append(trace)
        flags.append(flag)

    return (np.array(traces), np.array(flags))


def get_delta_x_offsets(traces, template_trace):
    """Get delta x offsets from traces

    Cross-correlates with a high signal-to-noise trace.

    Args:
        traces (numpy.ndarray): 2D array (N_traces x N_x)
            of star trail traces
        template_trace (numpy.array): 1D array (N_x)
            of high signal-to-noise ratio trace


    Returns:
        delta_xs (numpy.ndarray): the offsets in x from traces to target
    """
    from scipy.signal import correlate

    n_traces, n_x = traces.shape
    non_zero = template_trace > 0
    xcor = correlate(template_trace[non_zero], template_trace[non_zero], 'same')
    default_center = np.argmax(xcor)
    delta_xs = []
    for i in range(n_traces):
        xcor = correlate(template_trace[non_zero], traces[i, non_zero], 'same')
        center = np.argmax(xcor) - default_center
        delta_xs.append(center)

    return np.array(delta_xs)



def plot_kernel(kernel, aper_mask=None):
    """Make a quick plot of the kernel with sane screen stretch

    Args:
        aper_mask (np.ndarray): Optionally overplot an aperture mask

    Returns:
        None
    """
    # show the image
    plt.imshow(kernel, interpolation='nearest', cmap='BuGn', vmin=0, vmax=np.max(kernel), origin='lower')
    if aper_mask is not None:
        plt.imshow(aper_mask, interpolation='nearest', cmap='BuGn', vmin=0, vmax=1, origin='lower', alpha=0.5)
    plt.colorbar();
