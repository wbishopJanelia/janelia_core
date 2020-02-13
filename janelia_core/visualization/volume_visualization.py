""" Utilities for viewing volumes.  """

import pathlib
from typing import Sequence
from typing import Union

import matplotlib.animation
import matplotlib.colors
import matplotlib.cm
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from janelia_core.visualization.custom_color_maps import MultiParamCMap
from janelia_core.visualization.custom_color_maps import visualize_two_param_hsv_map
from janelia_core.visualization.image_generation import rgb_3d_max_project


def make_rgb_z_plane_movie(z_imgs: Sequence[np.ndarray], save_path: str,
                           title: str = None, fps: int = 10,
                           cmap: MultiParamCMap = None, cmap_param_strs: Sequence[str] = None,
                           cmap_param_vls: Sequence[Sequence] = None,
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

        cmap: An optional color map used to generate the images.  If provided, this colormap will be plotted in the
        video.

        cmap_param_strs: String labels for parameter 0 and 1 in cmap.  Only used if cmap is provided.

        cmap_param_vls: If not None, then cmap_param_vls[i] should contain a list of values for parameter i to plot
        the color map at.  cmap_param_vls[i] can also be none, which means the range of values plotted will be the
        range of values between the colormap saturation limits. Only used if cmap is provided.

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
        if cmap_param_vls is not None:
            cmap_p0_vls = cmap_param_vls[0]
            cmap_p1_vls = cmap_param_vls[1]
        else:
            cmap_p0_vls = None
            cmap_p1_vls = None
        visualize_two_param_hsv_map(cmap=cmap, plot_ax=cmap_im, p0_vls=cmap_p0_vls, p1_vls=cmap_p1_vls)
        cmap_im.axes.get_yaxis().set_tick_params(color=text_color, labelcolor=text_color)
        cmap_im.axes.get_xaxis().set_tick_params(color=text_color, labelcolor=text_color)
        if cmap_param_strs is not None:
            plt.xlabel(cmap_param_strs[1], color=text_color)
            plt.ylabel(cmap_param_strs[0], color=text_color)

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


def visualize_rgb_max_project(vol: np.ndarray, cmap_im: np.ndarray = None,
                              cmap_extent: Sequence[float] = None, cmap_xlabel: str = None, cmap_ylabel: str = None,
                              title: str = None, f: matplotlib.figure.Figure = None, buffer=.6,
                              facecolor: Sequence[float] = (0, 0, 0), textcolor: Sequence[float] = (1, 1, 1)):
    """ Generates a figure of max-projections of rgb volumes.

    Will generate a figure with the following layout:

            y_dim            z_dim
        --------------     ---------
        |             |    |        |
        |             |    | y-proj |    ^
        |             |    |        |    |
        |  z-proj     |    |        |   x_dim
        |             |    |        |    |
        |             |    |        |
        |             |    |        |
        ------------       ---------

        --------------
      ^ |             |     -------
      | | x-proj      |    | cmap  |
  z_dim |             |    |       |
      | |             |     -------
        ---------------

    Args:
        vol: The volume to generate the max projection of. Should be 4 dimensional, with the last dimension containing
        RGB values.  Dimensions are assumed to be in the convention [x, y, z],

        cmap_im: An optional image of an colormap to include

        cmap_extent: Values to associate the the image of the colormap in the form of [left, right, bottom, top]

        cmap_xlabel: An optional label for the x-axis of the colormap

        cmap_ylabel: An optional label for the y-axis of the colormap

        title: A string to use as the title for the figure. If None, no title will be created.

        f: Figure to plot into.  If not provided, one will be created

        buffer: The amount of space to put around plots in inches.

        facecolor: The color of the figure background, if we are creating a figure

        textcolor: The color to plot text in

    Raises:
        ValueError: If vol is not 4 dimensional.
    """

    if vol.ndim != 4:
        raise(ValueError('vol must be 4 dimensional.'))

    # Get volume dimensions
    d_x, d_y, d_z, _ = vol.shape

    # Form projections
    x_proj = rgb_3d_max_project(vol=vol, axis=0)
    y_proj = rgb_3d_max_project(vol=vol, axis=1)
    z_proj = rgb_3d_max_project(vol=vol, axis=2)

    # Create the figure if we need to
    if f is None:
        tgt_h = 8.0 # inches
        tgt_w = tgt_h*(d_y+d_z)/(d_x+d_z) + 3*buffer
        f = plt.figure(figsize=(tgt_w, tgt_h), facecolor=facecolor)

    # Get current figure size
    f_w, f_h = f.get_size_inches() # (width, height)

    # Determine how much usable space there is in the figure
    usable_w = f_w - 3*buffer
    usable_h = f_h - 3*buffer

    # Determine the total height we can use for plotting, considering the aspect ratio of the figure
    req_height_im = float(d_x + d_z) # Height we need for plotting, in number of image pixels
    req_width_im = float(d_y + d_z) # Height we need for plotting, in number of image pizels

    usable_h_w_ratio = usable_h/usable_w

    plot_h_w_ratio = req_height_im/req_width_im

    if usable_h_w_ratio < plot_h_w_ratio:
        # Figure is "wider" than what we need to plot, so we can scale up vertically as much as possible
        plottable_h = usable_h
    else:
        # Figure is not wide enough for the plot, so we can only scale up vertically until we run out of width
        plottable_h = plot_h_w_ratio*usable_w
    plottable_w = plottable_h/plot_h_w_ratio

    # Now we determine the size of the axes, as a percentage of the current figure size
    dx_perc_h = (plottable_h*d_x/(d_x + d_z))/f_h
    dz_perc_h = (plottable_h*d_z/(d_x + d_z))/f_h
    dy_perc_w = (plottable_w*d_y/(d_y + d_z))/f_w
    dz_perc_w = (plottable_w*d_z/(d_y + d_z))/f_w

    # Now we determine position of axes as percentage of current figure size
    v_0_p = buffer/f_h
    h_0_p = buffer/f_w
    v_1_p = 2*buffer/f_h + dz_perc_h
    h_1_p = 2*buffer/f_w + dy_perc_w

    # Now we specify the rectanges for each axes
    z_proj_rect = (h_0_p, v_1_p, dy_perc_w, dx_perc_h)
    x_proj_rect = (h_0_p, v_0_p, dy_perc_w, dz_perc_h)
    y_proj_rect = (h_1_p, v_1_p, dz_perc_w, dx_perc_h)
    cmap_rect = (h_1_p, v_0_p, dz_perc_w, dz_perc_h)

    # Now we add the axes, adding the title while it is convenient
    z_proj_axes = f.add_axes(z_proj_rect)
    if title is not None:
        plt.title(title, color=textcolor)
    x_proj_axes = f.add_axes(x_proj_rect)
    y_proj_axes = f.add_axes(y_proj_rect)
    if cmap_im is not None:
        cmap_axes = f.add_axes(cmap_rect)

    # Make sure the axes don't change aspect ratio when we scale the figure
    z_proj_axes.set_aspect('equal')
    x_proj_axes.set_aspect('equal')
    y_proj_axes.set_aspect('equal')
    if cmap_im is not None:
        cmap_axes.set_aspect('equal')

    # Get rid of units on the projection axes

    z_proj_axes.axes.get_xaxis().set_visible(False)
    z_proj_axes.axes.get_yaxis().set_visible(False)

    x_proj_axes.axes.get_xaxis().set_visible(False)
    x_proj_axes.axes.get_yaxis().set_visible(False)

    y_proj_axes.axes.get_xaxis().set_visible(False)
    y_proj_axes.axes.get_yaxis().set_visible(False)

    # Now we show the projections
    z_proj_axes.imshow(z_proj)
    x_proj_axes.imshow(np.moveaxis(x_proj, 0, 1))
    y_proj_axes.imshow(y_proj)

    # Now we add the colormap
    if cmap_im is not None:
        if cmap_extent is not None:
            a_ratio = np.abs(cmap_extent[1] - cmap_extent[0])/np.abs(cmap_extent[3] - cmap_extent[2])
            cmap_axes.imshow(cmap_im, extent=cmap_extent, aspect=a_ratio)
        else:
            cmap_axes.imshow(cmap_im)


        cmap_axes.get_yaxis().set_tick_params(color=textcolor, labelcolor=textcolor)
        cmap_axes.get_xaxis().set_tick_params(color=textcolor, labelcolor=textcolor)

        # Add labels to cmap if needed
        if cmap_xlabel is not None:
            plt.xlabel(cmap_xlabel, color=textcolor)
        if cmap_ylabel is not None:
            plt.ylabel(cmap_ylabel, color=textcolor)


