# pylint: disable=invalid-name
'''
This module is where pieces of the figures are put together. Data is
compbined with maps and skewts to provide the final product.
'''

import copy
import gc
import os

import matplotlib.pyplot as plt
import numpy as np

from adb_graphics.datahandler import gribfile
from adb_graphics.datahandler import gribdata
import adb_graphics.errors as errors
from adb_graphics.figures import maps
from adb_graphics.figures import skewt

AIRPORTS = 'static/Airports_locs.txt'

def add_obs_panel(ax, model_name, obs_file, proj_info, short_name, tile):

    # pylint: disable=too-many-arguments
    ''' Plot observation data provided by the obs_file
    path using the assigned projection. '''

    gribobs = gribfile.GribFile(filename=obs_file)
    ax.axis('on')
    field = gribdata.fieldData(
        ds=gribobs.contents,
        fhr=0,
        level='obs',
        model='obs',
        short_name=short_name,
        )
    map_fields = maps.MapFields(main_field=field)
    m = maps.Map(
        airport_fn=AIRPORTS,
        ax=ax,
        grid_info=copy.deepcopy(proj_info),
        model='obs',
        tile=tile,
        )
    dm = maps.MultiPanelDataMap(
        map_fields=map_fields,
        map_=m,
        member='OBS',
        model_name=model_name,
        )

    # Draw the map
    dm.draw(show=True)

def parallel_maps(cla, fhr, ds, level, model, spec, variable, workdir,
                  tile='full', ds2=None):

    # pylint: disable=too-many-arguments,too-many-locals
    # pylint: disable=too-many-branches,too-many-statements

    '''
    Function that creates plan-view maps, either a single panel, or
    multipanel for a forecast ensemble. Can be used in parallel.

    Input:

      fhr        forecast hour
      ds         xarray dataset from the grib file
      level      the vertical level of the variable to be plotted
                 corresponding to a key in the specs file
      model      model name: rap, hrrr, hrrre, rrfs, rtma
      spec       the dictionary of specifications for the given variable
                 and level
      variable   the name of the variable section in the specs file
      workdir    output directory

    Optional:
      tile       the label of the tile being plotted
    '''

    fig, axes = set_figure(cla.model_name, cla.graphic_type, tile)

    # set last_panel to send into DataMap for colorbar control
    last_panel = False

    # Declare the type of object depending on graphic type
    map_classes = {
        "enspanel": maps.MultiPanelDataMap,
        "diff": maps.DiffMap,
        }
    map_class = map_classes.get(cla.graphic_type, maps.DataMap)

    has_det = getattr(cla, 'has_det', True)
    last_drawn_cf = None
    last_drawn_dm = None
    colorbar_drawn = False

    for index, current_ax in enumerate(axes):

        if current_ax is axes[-1]:
            last_panel = True
        mem = None
        mem_label = None
        if cla.graphic_type == 'enspanel':
            # Panel 0: title and logo in top-left
            if index == 0:
                current_ax.axis('off')
                mem = 0
            # Panel 8: observation panel in bottom-left (if provided)
            elif index == 8:
                obs_fp = getattr(cla, 'obs_file_path', None)
                if not (spec.get('include_obs', False) and obs_fp and os.path.exists(obs_fp)):
                    current_ax.axis('off')
                    continue
                mem = 0
            # Panel 4: deterministic member in middle-left (if provided)
            elif index == 4:
                if not has_det:
                    current_ax.axis('off')
                    continue
                mem = 0
                mem_label = 'DET'
            else:
                # Map remaining 9 panels to ensemble members 1..9
                # Panels 1, 2, 3 -> ens members 1, 2, 3
                # Panels 5, 6, 7 -> ens members 4, 5, 6
                # Panels 9, 10, 11 -> ens members 7, 8, 9
                ens_num = index if index < 4 else (index - 1 if index < 8 else index - 2)
                # If deterministic member is loaded at slice 0, member N is at slice N.
                # If deterministic is absent, member N is at slice N - 1.
                mem = ens_num if has_det else (ens_num - 1)
                mem_label = str(ens_num)

            # If member index exceeds available ensemble slices, blank panel and skip
            if index != 0 and mem is not None:
                ens_dim_size = ds.sizes.get('ens_mem', None)
                if ens_dim_size is not None and mem >= ens_dim_size:
                    current_ax.axis('off')
                    continue

        # Object to be plotted on the map in filled contours.
        field = gribdata.fieldData(
            ds=ds,
            fhr=fhr,
            filetype=cla.file_type,
            level=level,
            member=mem,
            model=model,
            short_name=variable,
            config=cla.specs['file']
            )

        try:
            field.field
        except errors.GribReadError:
            print(f'Cannot find grib2 variable for {variable} at {level}. Skipping.')
            return

        if cla.graphic_type == "diff":

            field2 = gribdata.fieldData(
                ds=ds2,
                fhr=fhr,
                filetype=cla.file_type,
                level=level,
                member=mem,
                model=model,
                short_name=variable,
                config=cla.specs['file']
                )

            try:
                field2.field
            except errors.GribReadError:
                print((f'Cannot find grib2 variable for {variable} at {level} in',
                       '2nd set of files. Skipping.'))
                return

            field.data = field.values() - field2.values()


        map_fields = maps.MapFields(
            fields_spec=spec,
            main_field=field,
            map_type=cla.graphic_type,
            model=model,
            tile=tile,
            )

        # Generate a map object
        m = maps.Map(
            airport_fn=AIRPORTS,
            ax=current_ax,
            grid_info=field.grid_info(),
            model=model,
            plot_airports=spec.get('plot_airports', True),
            tile=tile,
            )

        # Send all objects (map_field, contours, hatches) to a DataMap object
        dm = map_class(
            map_fields=map_fields,
            map_=m,
            member=mem_label if cla.graphic_type == 'enspanel' else mem,
            model_name=cla.model_name,
            last_panel=last_panel,
            )

        # Draw the map
        if cla.graphic_type == 'enspanel':
            if index == 0:
                dm.title()
                dm.add_logo(current_ax)
            elif index == 8:
                if spec.get('include_obs', False) and getattr(cla, 'obs_file_path', None):
                    # Add observation panel to lower left. Currently only
                    # supported for composite reflectivity.
                    try:
                        add_obs_panel(
                            ax=axes[8],
                            model_name=cla.model_name,
                            obs_file=cla.obs_file_path,
                            proj_info=copy.deepcopy(field.grid_info()),
                            short_name=variable,
                            tile=tile,
                            )
                    except Exception as err:
                        print(f"Warning: Failed to plot observation panel: {err}")
                        axes[8].axis('off')
            else:
                last_drawn_cf = dm.draw(show=True)
                last_drawn_dm = dm
                if last_panel:
                    colorbar_drawn = True
        else:
            dm.draw(show=True)

    # Ensure colorbar is rendered for enspanel if the last panel was skipped
    if cla.graphic_type == 'enspanel' and not colorbar_drawn and last_drawn_dm is not None and last_drawn_cf is not None:
        cax = plt.axes([0.0, 0.0, 1.0, 0.2])
        last_drawn_dm._colorbar(ax=cax, cc=last_drawn_cf)
        cax.axis('off')

    # Build the output path
    png_file = f'{variable}_{tile}_{level}_f{fhr:03d}.png'
    png_file = png_file.replace("__", "_")
    png_path = os.path.join(workdir, png_file)

    print('*' * 120)
    print(f"Creating image file: {png_path}")
    print('*' * 120)

    # Save the png file to disk
    plt.savefig(
        png_path,
        bbox_inches='tight',
        dpi=cla.img_res,
        format='png',
        orientation='landscape',
        pil_kwargs={'optimize': True},
        )

    fig.clear()
    # Clear the current axes.
    plt.cla()
    # Clear the current figure.
    plt.clf()
    # Closes all the figure windows.
    plt.close('all')
    del field
    del m
    gc.collect()

def parallel_skewt(cla, fhr, ds, site, workdir):

    '''
    Function that creates a single SkewT plot. Can be used in parallel.
    Input:

      cla        command line arguments Namespace object
      ds         the XArray dataset
      fhr        the forecast hour integer
      site       the string representation of the site from the sites file
      workdir    output directory
    '''

    skew = skewt.SkewTDiagram(
        ds=ds,
        fhr=fhr,
        filetype=cla.file_type,
        loc=site,
        max_plev=cla.max_plev,
        model_name=cla.model_name,
        )
    skew.create_diagram()
    outfile = f"{skew.site_code}_{skew.site_num}_skewt_f{fhr:03d}.png"
    png_path = os.path.join(workdir, outfile)

    print('*' * 80)
    print(f"Creating image file: {png_path}")
    print('*' * 80)

    # pylint: disable=duplicate-code
    plt.savefig(
        png_path,
        bbox_inches='tight',
        dpi=cla.img_res,
        format='png',
        orientation='landscape',
        )

    start_time = cla.start_time.strftime('%Y%m%d%H')
    csvfile = f"{skew.site_code}.{skew.site_num}.skewt.{start_time}_f{fhr:03d}.csv"
    csv_path = os.path.join(workdir, csvfile)
    print('*' * 80)
    print(f"Creating csv file: {csv_path}")
    print('*' * 80)
    skew.create_csv(csv_path)

    plt.close()

def set_figure(model_name, graphic_type, tile):

    ''' Create the figure and subplots appropriate for the model and
    graphics type. Return the figure handle and list of axes. '''

    if model_name == "HRRR-HI":
        inches = 12.2
    else:
        inches = 10

    # Settings for a default single map
    x_aspect = 1
    y_aspect = 1
    nrows = 1
    ncols = 1

    if graphic_type == 'enspanel':
        nrows = 3
        ncols = 4
        inches = 20
        # Most rough-square subdomains can use the 0.8 y_aspect
        y_aspect = 0.8
        x_aspect = 1
        if tile in ['full', 'NW']:
            # Horizontal rectangle subdomains, and CONUS need more
            # squashed horizontal rectangles
            y_aspect = 0.5
        if tile in ['SE']:
            # Vertical rectangle subdomains can use a bit more height
            # than the others
            y_aspect = 0.95

    fig, ax = plt.subplots(nrows, ncols,
                           figsize=(x_aspect*inches, y_aspect*inches),
                           sharex=True,
                           sharey=True,
                           )
    # Flatten the 2D array and number panel axes from top left to bottom right
    # sequentially
    ax = ax.flatten() if isinstance(ax, np.ndarray) else [ax]

    return fig, ax
