""" Utilities for viewing volumes.  """

import pathlib
from typing import Sequence
from typing import Union

import matplotlib.animation
import matplotlib.colors
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np

from janelia_core.visualization.custom_color_maps import MultiParamCMap
from janelia_core.visualization.custom_color_maps import visualize_two_param_hsv_map


def make_rgb_z_plane_movie(z_imgs: Sequence[np.ndarray], save_path: str,
                           title: str = None, fps: int = 10,
                           cmap: MultiParamCMap = None, param_strs: Sequence[str] = None,
                           figsize: Sequence[int] = [7, 12],
                           facecolor: Union[float] = (0, 0, 0),
                           text_color: Union[float] = (1.0, 1.0, 1.0),
                           bitrate=-1):
    """ Generates a sequence of movie of z-plane images already in RGB space.

    Args:

        z_imgs: A sequende of numpy arrays of the images for each plane.  Each array should be of the shape [dx, dy, 3]

        save_path: The path for the saved video, including extension

        title: If not None, the title to display over the video

        fps: The frames per second of the generated movie.  Each plane is a different plane.

        figsize: The size of the figure to make the video as [width, height] in inches

        facecolor: The background color of the figure to plot into

        text_color: The color to use when plotting the title and colorbar ticks and labels

        bitrate: The bitrate to use when saving the video

    """

    # Define a helper function
    def update_image(z, images, im_h, title_h, title_s):
        im_h.set_data(images[z])
        title_h.set_text(title_s + 'z = ' + str(z))
        return im_h,

    # Get the first frame of the video
    frame0 = z_imgs[40]
    n_z_planes = len(z_imgs)

    # Setup the basic figure for plotting, showing the first frame
    fig = plt.figure(figsize=figsize, facecolor=facecolor)
    if cmap is None:
        z_im = plt.imshow(frame0)
    else:
        z_im = plt.subplot2grid([10, 10], [0, 0], 9, 10)
        z_im = z_im.imshow(frame0)

    # Setup the title
    if title is not None:
        title_str = title + ', '
    else:
        title_str = ''
    title_h = plt.title(title_str, color=text_color)

    # Show the color map if we are suppose to
    if cmap is not None:
        cmap_im = plt.subplot2grid([10, 10], [9, 8], 2, 2)
        visualize_two_param_hsv_map(cmap=cmap, plot_ax=cmap_im)
        cmap_im.axes.get_yaxis().set_tick_params(color=text_color, labelcolor=text_color)
        cmap_im.axes.get_xaxis().set_tick_params(color=text_color, labelcolor=text_color)
        if param_strs is not None:
            plt.xlabel(param_strs[1], color=text_color)
            plt.ylabel(param_strs[0], color=text_color)

    z_im.axes.get_xaxis().set_visible(False)
    z_im.axes.get_yaxis().set_visible(False)

    # Generate the movie
    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=bitrate, codec='libx264', extra_args=['-pix_fmt', 'yuv420p',
                                                                          '-crf', '18'])

    plane_animation = matplotlib.animation.FuncAnimation(fig=fig, func=update_image, frames=n_z_planes,
                                                         fargs=(z_imgs, z_im, title_h, title_str), interval=1,
                                                         blit=False, repeat=False)

    # Save the movie
    plane_animation.save(save_path, writer=writer, savefig_kwargs={'facecolor':facecolor})

    # Close the figure
    plt.close(fig)


def make_z_plane_movie(volume: np.ndarray, save_path: str,
                       clim: Sequence[float] = None,
                       cmap: Union[str, matplotlib.colors.Colormap] = 'hot',
                       title: str = None,
                       cbar_label: str = None,
                       fps: int = 10,
                       figsize: Sequence[int] = [7, 12],
                       facecolor: Union[float] = (0, 0,0),
                       text_color: Union[float] = (1.0, 1.0, 1.0),
                       bitrate=-1,
                       clim_percs: Sequence[float] = [.1, 99.9]):
    """
    Generates a movie of all the z-planes in a volume.

    Args:
        volume: The volume to visualize, with the z-dimension indexed by the first dimension.

        save_path: The path for the saved video, including extension

        clim: If not none, the color limits to apply to all planes.  If None, color limits will be automtically
        calculated by 1) taking a user assigned lower and upper percentile values, 2) calculating the maximum
        of the absolute value of these values and then 3) setting the limits to be +/- the maximum of the
        absolute value calculated in (2).

        cmap: The colormap to use for all planes

        title: If not None, the title to display over the video

        cbar_label: If not None, the label for the colorbar

        fps: The frames per second of the generated movie.  Each plane is a different plane.

        figsize: The size of the figure to make the video as [width, height] in inches

        facecolor: The background color of the figure to plot into

        text_color: The color to use when plotting the title and colorbar ticks and labels

        bitrate: The bitrate to use when saving the video

        clims_percs: The percentile limits to use if automatically setting clim between 0 and 100.
        The value of c_perc_limits[0] is the lower limit and the value of c_perc_limits[1] is the upper
        limit.

    """

    # Define a helper function
    def update_image(z, image, im_h, title_h, title_s):
        im_h.set_data(np.squeeze(image[z, :, :]))
        title_h.set_text(title_s + 'z = ' + str(z))
        return im_h,

    # If clim is None, calculate color limits
    if clim is None:
        p_lower = np.percentile(volume, clim_percs[0])
        p_upper = np.percentile(volume, clim_percs[1])
        c_range = np.max(np.abs([p_lower, p_upper]))
        clim = (-c_range, c_range)

    # Get the first frame of the video
    frame0 = np.squeeze(volume[0, :, :])
    n_z_planes = volume.shape[0]

    # Setup the basic figure for plotting, showing the first frame
    fig = plt.figure(figsize=figsize, facecolor=facecolor)
    z_im = plt.imshow(frame0, cmap=cmap, clim=clim)
    z_im.axes.get_xaxis().set_visible(False)
    z_im.axes.get_yaxis().set_visible(False)

    # Setup the color bar
    cbar = plt.colorbar()
    cbar.ax.yaxis.set_tick_params(color=text_color, labelcolor=text_color)

    if cbar_label is not None:
        cbar.set_label(cbar_label, color=text_color)

    # Setup the title
    if title is not None:
        title_str = title + ', '
    else:
        title_str = ''
    title_h = plt.title(title_str, color=text_color)

    # Generate the movie
    Writer = matplotlib.animation.writers['ffmpeg']
    writer = Writer(fps=fps, bitrate=bitrate, codec='libx264', extra_args=['-pix_fmt', 'yuv420p',
                                                                           '-crf', '18'])

    plane_animation = matplotlib.animation.FuncAnimation(fig=fig, func=update_image, frames=n_z_planes,
                                                        fargs=(volume, z_im, title_h, title_str), interval=1,
                                                        blit=False, repeat=False)

    # Save the movie
    plane_animation.save(save_path, writer=writer, savefig_kwargs={'facecolor':facecolor})

    # Close the figure
    plt.close(fig)


