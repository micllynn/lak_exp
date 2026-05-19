import tifffile

import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import xml.etree.ElementTree as ElementTree


def stitch_tiffs_memmap(directory, chan=0,
                        delete_after=False):
    """
    Stitch multiple TIFF files into a single compiled TIFF using memory mapping.

    Memory-efficient method for concatenating large TIFF files by using memmap
    to avoid loading entire files into memory.

    Parameters
    ----------
    directory : str
        Path to directory containing TIFF files starting with 'file' prefix.
    chan : int, optional
        Channel number for output filename, by default 0.
    delete_after : bool, optional
        Whether to delete original files after stitching (not implemented),
        by default False.

    Returns
    -------
    None
        Creates compiled_Ch{chan}.tif in the specified directory.
    """
    file_list_raw = os.listdir(directory)
    file_list = []

    for f in file_list_raw:
        if f.startswith('file') and f.endswith('.tif'):
            file_list.append(os.path.join(directory, f))

    output_fname = f'compiled_Ch{chan}.tif'

    with tifffile.TiffWriter(os.path.join(directory, output_fname),
                             bigtiff=True, append=True) as writer:
        for ind_f, f in enumerate(file_list[0:]):
            print(f'appending file {ind_f+1}/{len(file_list)}', end='\r')
            _tiff = tifffile.memmap(f)

            for t in range(_tiff.shape[0]):
                writer.write(_tiff[t], compression=None, contiguous=True)

            del _tiff

    return


def stitch_tiffs(directory, chan=0,
                 delete_after=False):
    """
    Stitch multiple TIFF files by loading into memory and concatenating.

    Loads all TIFF files into memory and concatenates them. Less memory-efficient
    than stitch_tiffs_memmap but simpler implementation.

    Parameters
    ----------
    directory : str
        Path to directory containing TIFF files (non-hidden files ending in .tif).
    chan : int, optional
        Channel number for output filename, by default 0.
    delete_after : bool, optional
        Whether to delete original files after stitching (not implemented),
        by default False.

    Returns
    -------
    None
        Creates compiled_Ch{chan}.tif in the specified directory.
    """
    os.chdir(directory)
    file_list_raw = os.listdir(directory)
    file_list = []

    for f in file_list_raw:
        if not f.startswith('.') and f.endswith('.tif'):
            file_list.append(f)

    tiff_concat = tifffile.imread(file_list[0])

    for ind_f, f in enumerate(file_list[1:]):
        print(f'appending file {ind_f}/{len(file_list)}', end='\r')
        tiff_concat = np.append(tiff_concat,
                                tifffile.imread(f), axis=0)

    with open(f'compiled_Ch{chan}.tif', 'wb') as f:
        tifffile.imwrite(f, tiff_concat)

    return


def stitch_and_move_all_tiffs(directory):
    """
    Takes an imaging folder with two channels that has been
    registered using suite2p.

    Stitches all tiffs from both channels in the suite2p folder,
    and moves them back to the main imaging directory.

    Parameters
    ---------
    directory: string
        Path to the main imaging folder (t-00x). Must
        have a suite2p folder inside
    """

    os.chdir(directory)

    print('Ch2....\n------------')
    stitch_tiffs_memmap('suite2p/plane0/reg_tif', chan=2)
    print('moving compiled_Ch2.tif....')
    os.rename('suite2p/plane0/reg_tif/compiled_Ch2.tif',
              'compiled_Ch2.tif')

    print('Ch1....\n------------')
    stitch_tiffs_memmap('suite2p/plane0/reg_tif_chan2', chan=1)
    print('moving compiled_Ch1.tif....')
    os.rename('suite2p/plane0/reg_tif_chan2/compiled_Ch1.tif',
              'compiled_Ch1.tif')

    return


def remove_frames_from_tiff_start(tiff_path, n_frames=500):
    """
    Remove a specified number of frames from the beginning of a TIFF file.

    Parameters
    ----------
    tiff_path : str
        Path to the input TIFF file.
    n_frames : int, optional
        Number of frames to remove from the start, by default 500.

    Returns
    -------
    None
        Creates a new TIFF file with prefix 'rmframes_' in the same directory.
    """
    path_parts = os.path.split(tiff_path)
    os.chdir(path_parts[0])
    new_tiff_name = 'rmframes_' + path_parts[1]

    tiff = tifffile.imread(tiff_path)
    tiff_reduced = np.delete(tiff, np.arange(n_frames), axis=0)

    with open(new_tiff_name, 'wb') as f:
        tifffile.imwrite(f, tiff_reduced)

    return


def calc_dff(trace, baseline_frames):
    """
    Calculate delta F over F (dF/F) for fluorescence trace.

    Parameters
    ----------
    trace : np.ndarray
        Fluorescence trace over time.
    baseline_frames : int
        Number of initial frames to use for baseline calculation.

    Returns
    -------
    dff : np.ndarray
        Delta F over F normalized fluorescence trace.
    """
    f0 = np.mean(trace[0:baseline_frames])
    dff = (trace-f0)/f0
    return dff


def clearmem():
    """
    Clear matplotlib figures and force garbage collection.

    Closes all matplotlib figures and runs garbage collection to free memory.

    Returns
    -------
    None
    """
    plt.close('all')
    plt.clf()
    gc.collect()
    return


class XMLParser(object):
    """
    Parse XML backup files from two-photon imaging systems.

    Parameters
    ----------
    path_to_backup_xml : str
        Path to the XML backup file from imaging session.

    Attributes
    ----------
    tree : ElementTree
        Parsed XML tree.
    root : Element
        Root element of the XML tree.
    """
    def __init__(self, path_to_backup_xml):
        self.tree = ElementTree.parse(path_to_backup_xml)
        self.root = self.tree.getroot()

    def print_children(self):
        """
        Print all child elements and their attributes from XML root.

        Returns
        -------
        None
        """
        for child in self.root:
            print(child.tag, child.attrib)

    def get_framerate(self):
        """
        Extract framerate from XML file based on frame timestamps.

        Returns
        -------
        framerate : float
            Imaging framerate in Hz.
        """
        # framerate = 1/float(self.root[2][1][3][0].attrib['value'])
        t_frame1 = float(self.root[2][2].attrib['absoluteTime'])
        t_frame2 = float(self.root[2][3].attrib['absoluteTime'])
        framerate = 1 / (t_frame2 - t_frame1)
        return framerate


def paq_read(file_path=None, plot=False, save_path=None):
    """
    Read PAQ file (from PackIO) into python
    Lloyd Russell 2015
    Parameters
    ==========
    file_path : str, optional
        full path to file to read in. if none is supplied a load file dialog
        is opened, buggy on mac osx - Tk/matplotlib. Default: None.
    plot : bool, optional
        plot the data after reading? Default: False.
    Returns
    =======
    data : ndarray
        the data as a m-by-n array where m is the number of channels and n is
        the number of datapoints
    chan_names : list of str
        the names of the channels provided in PackIO
    hw_chans : list of str
        the hardware lines corresponding to each channel
    units : list of str
        the units of measurement for each channel
    rate : int
        the acquisition sample rate, in Hz
    """

    # file load gui
    if file_path is None:
        print('No file path')
        import Tkinter
        import tkFileDialog
        root = Tkinter.Tk()
        root.withdraw()
        file_path = tkFileDialog.askopenfilename()
        root.destroy()

    # open file
    fid = open(file_path, 'rb')
    # get sample rate
    rate = int(np.fromfile(fid, dtype='>f', count=1))
    # get number of channels
    num_chans = int(np.fromfile(fid, dtype='>f', count=1))
    # get channel names
    chan_names = []
    for i in range(num_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        chan_name = ''
        for j in range(num_chars):
            chan_name = chan_name + chr(int(
                np.fromfile(fid, dtype='>f', count=1)))
        chan_names.append(chan_name)

    # get channel hardware lines
    hw_chans = []
    for i in range(num_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        hw_chan = ''
        for j in range(num_chars):
            hw_chan = hw_chan + chr(int(np.fromfile(fid, dtype='>f', count=1)))
        hw_chans.append(hw_chan)

    # get acquisition units
    units = []
    for i in range(num_chans):
        num_chars = int(np.fromfile(fid, dtype='>f', count=1))
        unit = ''
        for j in range(num_chars):
            unit = unit + chr(int(np.fromfile(fid, dtype='>f', count=1)))
        units.append(unit)

    # get data
    temp_data = np.fromfile(fid, dtype='>f', count=-1)
    num_datapoints = int(len(temp_data)/num_chans)
    data = np.reshape(temp_data, [num_datapoints, num_chans]).transpose()

    # close file
    fid.close()

    # plot
    if plot:
        import matplotlib
        import os
        matplotlib.use('Agg')
        import matplotlib.pylab as plt
        f, axes = plt.subplots(num_chans, 1, sharex=True)
        for idx, ax in enumerate(axes):
            ax.plot(data[idx])
            ax.set_xlim([0, num_datapoints-1])
            ax.set_ylabel(units[idx])
            ax.set_title(chan_names[idx])
        if save_path is not None:
            plt.savefig(os.path.join(
                save_path, 'paqRaw.png'), transparent=False)
        f.clear()
        plt.close(f)

    return {"data": data,
            "chan_names": chan_names,
            "hw_chans": hw_chans,
            "units": units,
            "rate": rate}
