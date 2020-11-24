""" Utilities for viewing volumes.  """

import copy
import importlib
import pathlib
from typing import Sequence, Union, Tuple

#import imageio
import importlib
import matplotlib.animation
import matplotlib.colors
import matplotlib.cm
import matplotlib.figure
import matplotlib.pyplot as plt
import matplotlib.transforms

#if importlib.util.find_spec('moveipy'):
import moviepy.editor as editor
#else:
#    print('Unable to import moviepy.  Minor functionality will not be available.')

import numpy as np


from janelia_core.dataprocessing.dataset import ROI
from janelia_core.visualization.custom_color_maps import MultiParamCMap
from janelia_core.visualization.custom_color_maps import visualize_two_param_map
from janelia_core.visualization.image_generation import rgb_3d_max_project


def comb_movies(movie_paths: Sequence[pathlib.Path], save_path: pathlib.Path,
                fig_size: Sequence[int]=[21, 12], facecolor: Tuple[float] =(0,0,0),
                fps: int=10, bitrate=-1, ):
    """ Given a set of movies, combines them by putting their frames side-by-side.


    Args:

        movie_paths: Paths to movies to combined

        save_path: The path to save the combined video to.

    """

    clips = [editor.VideoFileClip(movie_path) for movie_path in movie_paths]
    final_clip = editor.clips_array([clips])
    final_clip.write_videofile(save_path)


def make_rgb_three_ch_z_plane_movie(z_imgs, save_path: str, fps: int = 10,
                                    title: str = None,
                                    cmaps: list = None,
                                    figsize: Sequence[int] = [7, 12],
                                    facecolor: Union[float] = (0, 0, 0),
                                    text_color: Union[float] = (1.0, 1.0, 1.0),
                                    bitrate=-1, one_index_z_plane: bool = False):

    CLR_BAR_H = .1
    C_MAP_H_BUFFER = .05
    C_MAP_V_BUFFER = .05
    C_MAP_WIDTH = .33 - 2*C_MAP_H_BUFFER

    # Define a helper function
    def update_image(z, images, im_h, title_h, title_s):
        if one_index_z_plane:
            z_title = z + 1
        else:
            z_title = z

        im_h.set_data(images[z])
        title_h.set_text(title_s + 'z = ' + str(z_title))
        return im_h,

    # Get the first frame of the video
    frame0 = z_imgs[0]
    n_z_planes = len(z_imgs)

    # Setup the basic figure for plotting, showing the first frame
    fig = plt.figure(figsize=figsize, facecolor=facecolor)
    if cmaps is not None:
        ax_position = [0, .1, 1.0, 1.0 - CLR_BAR_H]
    else:
        ax_position = [0, 0, 1.0, 1.0]

    z_ax = fig.add_axes(ax_position)
    z_im = z_ax.imshow(frame0)

    # Setup the title
    if title is not None:
        title_str = title + ', '
    else:
        title_str = ''
    title_h = plt.title(title_str, color=text_color)

    # Show colormaps if we are suppose to
    if cmaps is not None:
        cmap_h = 1 - ax_position[-1] - C_MAP_V_BUFFER
        # If cmaps is not None, we expect it to be of length 3
        for c_i, cmap in enumerate(cmaps):
            cur_start = c_i*.33 + C_MAP_H_BUFFER
            cmap_ax = fig.add_axes([cur_start, C_MAP_V_BUFFER, C_MAP_WIDTH, cmap_h])
            cmap_im = np.zeros([1, 1000, 3])
            cmap_im[:, :, c_i] = np.linspace(0, 1, 1000)
            cmap_ax.imshow(cmap_im, aspect='auto')
            cmap_ax.axes.get_xaxis().set_tick_params(color=text_color, labelcolor=text_color)
            plt.xticks([0, 1000], labels=["{:3.3f}".format(cmap['dark_vl']),"{:3.3f}".format(cmap['bright_vl'])])
            plt.yticks([])
            plt.xlabel(cmap['label'], color=text_color)

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


def make_rgb_z_plane_movie(z_imgs: Sequence[np.ndarray], save_path: str,
                           title: str = None, fps: int = 10,
                           cmap: MultiParamCMap = None, cmap_param_strs: Sequence[str] = None,
                           cmap_param_vls: Sequence[Sequence] = None,
                           figsize: Sequence[int] = [7, 12],
                           facecolor: Union[float] = (0, 0, 0),
                           text_color: Union[float] = (1.0, 1.0, 1.0),
                           bitrate=-1, one_index_z_plane: bool = False,
                           ax_position = None):
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

        one_index_z_plane: True if z-plane numbers should start at 1 instead of 0 when generating titles for each plane

        ax_position: Position of axes for plotting images.
    """

    # Define a helper function
    def update_image(z, images, im_h, title_h, title_s):
        if one_index_z_plane:
            z_title = z + 1
        else:
            z_title = z

        im_h.set_data(images[z])
        title_h.set_text(title_s + 'z = ' + str(z_title))
        return im_h,

    # Get the first frame of the video
    frame0 = z_imgs[40]
    n_z_planes = len(z_imgs)

    print('n_z_planes: ' + str(n_z_planes))
    print('plane_0_shape: ' + str(z_imgs[0].shape))

    # Setup the basic figure for plotting, showing the first frame
    fig = plt.figure(figsize=figsize, facecolor=facecolor)
    if ax_position is None:
        z_ax = plt.subplot()
    else:
        z_ax = fig.add_axes(ax_position)
    z_im = z_ax.imshow(frame0)

    # Setup the title
    if title is not None:
        title_str = title + ', '
    else:
        title_str = ''
    title_h = plt.title(title_str, color=text_color)

    # Show the color map if we are suppose to
    if cmap is not None:
        #cmap_im = plt.subplot2grid([10, 10], [9, 8], 2, 2)
        cmap_im = fig.add_axes([.8, .0, 0.2, 0.2])
        if cmap_param_vls is not None:
            cmap_p0_vls = cmap_param_vls[0]
            cmap_p1_vls = cmap_param_vls[1]
        else:
            cmap_p0_vls = None
            cmap_p1_vls = None
        visualize_two_param_map(cmap=cmap, plot_ax=cmap_im, p0_vls=cmap_p0_vls, p1_vls=cmap_p1_vls)
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
                       clim_percs: Sequence[float] = [.1, 99.9],
                       one_index_z_plane: bool = False) -> matplotlib.transforms.Bbox:
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

        one_index_z_plane: True if z-plane numbers should start with 1 in the frame titles

    Returns:

        ax_pos: The bounding box for the axes used to show frames

    """

    # Define a helper function
    def update_image(z, image, im_h, title_h, title_s):
        im_h.set_data(np.squeeze(image[z, :, :]))
        z_n = z

        if one_index_z_plane:
            z_n = z_n + 1

        title_h.set_text(title_s + 'z = ' + str(z_n))
        return im_h,

    # If clim is None, calculate color limits
    if clim is None:
        p_lower = np.nanpercentile(volume, clim_percs[0])
        p_upper = np.nanpercentile(volume, clim_percs[1])
        c_range = np.max(np.abs([p_lower, p_upper]))
        clim = (-c_range, c_range)

    # Set any nan values to 0
    volume[np.isnan(volume)] = 0

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

    ax_pos = z_im.axes.get_position()

    # Close the figure
    plt.close(fig)

    return ax_pos


def signed_max_proj(vol: np.ndarray, dim: int):
    """ Performs a signed max projection of a volume.

    By "signed max project" we mean the find the entry with largest absolute value along the projection
    and then return the original (signed) value.

    This function will ignore nan values.

    Args:
        vol: The volume.

        dim: The dimension to project along.

    Returns:

        proj: The projected volume
    """

    vol_copy = copy.deepcopy(vol)
    vol_copy[np.isnan(vol)] = 0.0

    max_inds = np.expand_dims(np.argmax(np.abs(vol_copy), axis=dim), axis=dim)
    return np.squeeze(np.take_along_axis(vol, max_inds, axis=dim))


def visualize_rgb_max_project(vol: np.ndarray, dim_m: np.ndarray = None, cmap_im: np.ndarray = None,
                              overlays: Sequence[np.ndarray] = None, cmap_extent: Sequence[float] = None,
                              cmap_xlabel: str = None, cmap_ylabel: str = None,
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
            y_dim

    Args:
        vol: The volume to generate the max projection of. Should be 4 dimensional, with the last dimension containing
        RGB values.  Dimensions are assumed to be in the convention [x, y, z],

        dim_m: A scalar multiplier for each dimension in the order x, y, z to account for aspect ratios.  If None,
        a value of [1, 1, 1] will be used.

        cmap_im: An optional image of an colormap to include

        overlays: If provided, overlays[0] is an image to overlay the z-projection, and overlays[1] and [2] and images
        to overlay the x and y projections.  These overlays should be of the same dimensions as the projections.

        cmap_extent: Values to associate the the image of the colormap in the form of [left, right, bottom, top].
        Note that colormap will be plotted so colors associated with the smallest parameter values appear in the
        bottom left of the colormap image.

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

    if dim_m is None:
        dim_m = np.ones(3)

    # Get volume dimensions
    d_x, d_y, d_z, _ = vol.shape
    print('d_x: ' + str(d_x))
    print('d_y: ' + str(d_y))
    print('d_z: ' + str(d_z))

    # Apply aspect ratio correction
    d_x = d_x*dim_m[0]
    d_y = d_y*dim_m[1]
    d_z = d_z*dim_m[2]

    xy_aspect_ratio = dim_m[0]/dim_m[1]
    xz_aspect_ratio = dim_m[0]/dim_m[2]
    yz_aspect_ratio = dim_m[1]/dim_m[2]

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

    # Now we specify the rectangles for each axes
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

    # Get rid of units on the projection axes

    z_proj_axes.axes.get_xaxis().set_visible(False)
    z_proj_axes.axes.get_yaxis().set_visible(False)

    x_proj_axes.axes.get_xaxis().set_visible(False)
    x_proj_axes.axes.get_yaxis().set_visible(False)

    y_proj_axes.axes.get_xaxis().set_visible(False)
    y_proj_axes.axes.get_yaxis().set_visible(False)

    # Now we show the projections
    z_proj_axes.imshow(z_proj, aspect=xy_aspect_ratio)
    x_proj_axes.imshow(np.flipud(np.moveaxis(x_proj, 0, 1)), aspect=1/yz_aspect_ratio)
    y_proj_axes.imshow(y_proj, aspect=xz_aspect_ratio)

    # Now we add the overlays
    z_proj_axes.imshow(overlays[0], aspect=xy_aspect_ratio)
    x_proj_axes.imshow(overlays[1], aspect=1/yz_aspect_ratio)
    y_proj_axes.imshow(overlays[2], aspect=xz_aspect_ratio)

    # Now we add the colormap
    if cmap_im is not None:
        if cmap_extent is not None:
            a_ratio = np.abs(cmap_extent[1] - cmap_extent[0])/np.abs(cmap_extent[3] - cmap_extent[2])
            cmap_axes.imshow(cmap_im, extent=cmap_extent, aspect=a_ratio, origin='lower')
        else:
            cmap_axes.imshow(cmap_im)


        cmap_axes.get_yaxis().set_tick_params(color=textcolor, labelcolor=textcolor)
        cmap_axes.get_xaxis().set_tick_params(color=textcolor, labelcolor=textcolor)

        # Add labels to cmap if needed
        if cmap_xlabel is not None:
            plt.xlabel(cmap_xlabel, color=textcolor)
        if cmap_ylabel is not None:
            plt.ylabel(cmap_ylabel, color=textcolor)


def visualize_projs(horz_projs:  Sequence[np.ndarray], sag_projs: Sequence[np.ndarray],
                    cor_projs: Sequence[np.ndarray], cmaps: Sequence[matplotlib.colors.Colormap],
                    clims: Union[None, Sequence[Union[None, tuple]]], dim_m: Sequence[float] = None,
                    title: str = None, plot_cmap: bool = False, f: matplotlib.figure.Figure = None,
                    buffer: float = .6, tgt_h: float = 3.0, facecolor: Sequence[float] = (0, 0, 0),
                    textcolor: Sequence[float] = (1, 1, 1)):

    """ Visualizes horizontal, sagital and coronal projections of the same volume of data.

    This function will generate a figure with the following layout:

            y_dim                z_dim
        --------------        -----------
        |               |    |          |
        |               |    | sag_proj |    ^
        |               |    |          |    |
        |  horz-proj    |    |          |   x_dim
        |               |    |          |    |
        |               |    |          |
        |               |    |          |
        ------------          -----------

        ----------------
      ^ |               |     ----------
      | | coronal-proj  |    |    cmap  |
  z_dim |               |    |          |
      | |               |     ----------
        ----------------

    Multiple images can be shown, overlayed on eachother.  If a colormap is shown, it will
    be the colormap for the topmost image.

    Args:
        horz_projs: horz_projs[i] is the i^th horizontal projection image. Images will be overlaid one on top of the
        other, with the i+1th image over the ith image.  Each image shape should be [x_dim, y_dim], where x_dim
        and y_dim refer to the figure above.

        sag_projs: sag_projs[i] is the i^th sagital projection image.  Each image should be of shape [x_dim, z_dim].

        cor_projs: cor_projs[i] is the i^th coronal projection image.  Each image should be of shape [y_dim, z_dim]

        cmaps: cmaps[i] is the colormap for plotting the i^th image in each of the projections.

        clims: clims[i] is a tuple of color limits to use when plotting the i^th image.  clims[i] can also be
        None, meaning no color limits will be explicitly passed to the plotting command for the i^th images.
        If clims is None, then no color limits will be explicitly passed for all images.

        dim_m: A scalar multiplier for each dimension in the order x, y, z to account for aspect ratios.  If None,
        a value of [1, 1, 1] will be used.

        title: A string to use as the title for the figure. If None, no title will be created.

        f: Figure to plot into.  If not provided, one will be created

        buffer: The amount of space to put around plots in inches.

        tgt_h: The height of the figure that will be created in inches if f is None.

        facecolor: The color of the figure background, if we are creating a figure

        textcolor: The color to plot text in
    """

    if dim_m is None:
        dim_m = np.ones(3)

    if clims is None:
        clims = [None]*len(horz_projs)

    # Get volume dimensions
    d_x, d_y = horz_projs[0].shape
    d_z = sag_projs[0].shape[1]

    # Apply aspect ratio corrections
    d_x = d_x*dim_m[0]
    d_y = d_y*dim_m[1]
    d_z = d_z*dim_m[2]

    xy_aspect_ratio = dim_m[0]/dim_m[1]
    xz_aspect_ratio = dim_m[0]/dim_m[2]
    yz_aspect_ratio = dim_m[1]/dim_m[2]

    # Create the figure if we need to
    if f is None:
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
    if plot_cmap:
        cmap_axes = f.add_axes(cmap_rect)

    # Make sure the axes don't change aspect ratio when we scale the figure
    z_proj_axes.set_aspect('equal')
    x_proj_axes.set_aspect('equal')
    y_proj_axes.set_aspect('equal')
    if plot_cmap:
        cmap_axes.set_aspect('equal')

    # Get rid of units on the projection axes

    z_proj_axes.axes.get_xaxis().set_visible(False)
    z_proj_axes.axes.get_yaxis().set_visible(False)

    x_proj_axes.axes.get_xaxis().set_visible(False)
    x_proj_axes.axes.get_yaxis().set_visible(False)

    y_proj_axes.axes.get_xaxis().set_visible(False)
    y_proj_axes.axes.get_yaxis().set_visible(False)

    # Now we show the projections
    for (z_proj, y_proj, x_proj, cmap, clim) in zip(horz_projs, sag_projs, cor_projs, cmaps, clims):
        c_im = z_proj_axes.imshow(z_proj, cmap=cmap, clim=clim, aspect=xy_aspect_ratio)
        x_proj_axes.imshow(np.moveaxis(x_proj, 0, 1), cmap=cmap, clim=clim, aspect=1/xz_aspect_ratio)
        y_proj_axes.imshow(y_proj, cmap=cmap, clim=clim, aspect=yz_aspect_ratio)

    # Now we show the colormap if we are suppose to
    if plot_cmap:
        plt.colorbar(mappable=c_im, cax=cmap_axes)

        cmap_axes.get_yaxis().set_tick_params(color=textcolor, labelcolor=textcolor)
        cmap_axes.get_xaxis().set_tick_params(color=textcolor, labelcolor=textcolor)


def gen_composite_roi_vol(rois: Sequence[ROI], weights: np.ndarray, vol_shape: Sequence[int],
                          verbose: bool = False):

    """ Generates a volume for visualization of a composite ROI.

    Args:
        rois: A list of rois to form the composite ROI out of.

        weights: A list of weights for each roi in rois.

        vol_shape: The shape of the volume to generate.  The indices of ROIS should index into this volume.

        verbose: True if updates on progress should be printed

    Returns:

        comp_roi_vol: The volume with the composite ROI in it.

    """

    w_sum = np.zeros(vol_shape)
    cnts = np.zeros(vol_shape)

    n_rois = len(rois)
    for r_i, roi in enumerate(rois):
        inds = roi.list_all_voxel_inds()
        w = roi.list_all_weights()

        w_sum[inds] = w_sum[inds] + weights[r_i]*w
        cnts[inds] += 1

        if verbose and r_i % 10000 == 0:
            print('Completed processing ' + str(r_i) + ' of ' + str(n_rois) + ' rois.')

    comp_roi_vol = w_sum/cnts
    comp_roi_vol[np.where(cnts == 0)] = np.nan

    return comp_roi_vol