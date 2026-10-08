import re
import os
import numpy as np
import yt
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import WMAP9
from scipy.interpolate import CubicSpline

# Returns a list of all outputs
def find_outputs(folder='.'):
    outputs=np.array([int(name[-5:]) for name in os.listdir(folder) if name[:8]=='output_0'])
    outputs.sort()
    return outputs

# Function to point from directory of run to the info file of an output
def info_file(output,folder='.'):
    filename="{1}/output_{0}/info_{0}.txt".format(str(output).zfill(5),folder)
    print("Opening",filename)
    return "{1}/output_{0}/info_{0}.txt".format(str(output).zfill(5),folder)

def safe_float_convert(s):
    """
    Convert string s to float.
    If s is invalid due to missing 'E' in exponent (e.g. 1.23-149),
    return 0.0 instead.
    """
    try:
        return float(s)
    except ValueError:
        # Try to detect malformed exponent like 0.724106766732118-149
        m = re.match(r'^([+-]?\d*\.?\d*)([+-]\d+)$', s)
        if m:
            # malformed, treat as zero
            return 0.0
        else:
            # re-raise if it's some other error
            raise

import re

def parse_line(line: str) -> tuple:
    """
    Parse a metadata line of the format:
        #<idx>  <name> [<ncomp>] [<unit>] <anything>

    Any characters after the closing bracket are discarded.

    Parameters
    ----------
    line : str
        A single metadata line, e.g. "#9  rg_scale 1 [pc]  (MIN(rbondi,racc))"

    Returns
    -------
    tuple : (idx, name, ncomp, unit)
        idx   (int  | None) – the number following '#'
        name  (str  | None) – the field name
        ncomp (int)         – number of components/columns, defaults to 1
        unit  (str  | None) – the string inside brackets
    """
    pattern = r'^\s*#(\d+)\s+(\w+)(?:\s+(\d+))?(?:\s+\[([^\]]+)\])?'
    match = re.match(pattern, line)

    if not match:
        return None, None, 1, None

    return (
        int(match.group(1)),
        match.group(2),
        int(match.group(3)) if match.group(3) is not None else 1,
        match.group(4),
    )


def parse_file_with_multiple_blocks(filename):
    all_data = []
    col_names = None
    col_units = None
    col_ncomps = None

    # Unified pattern: # <idx> <name> <ncomp> [<unit>]
    # \S+ captures the name as a single whitespace-delimited token,
    # and the following \d+ captures the component count separately.
    col_pattern = re.compile(r'^#\s*(\d+)\s+(\S+)\s+(\d+)\s*(?:\[(.*?)\])?')

    with open(filename) as f:
        current_data = []
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                idx, name, ncomp, unit = parse_line(line)
                idx = int(idx) - 1
                ncomp = int(ncomp)
                if col_names is None:
                    col_names = []
                    col_units = []
                    col_ncomps = []
                while len(col_names) <= idx:
                    col_names.append(None)
                    col_units.append(None)
                    col_ncomps.append(None)
                col_names[idx] = name.strip()
                col_units[idx] = unit.strip() if unit else None
                col_ncomps[idx] = ncomp
            elif line == '':
                if current_data:
                    all_data.extend(current_data)
                    current_data = []
            else:
                parts = line.split()
                if col_names:
                    #print(col_names,col_ncomps)
                    total_cols_required = sum(col_ncomps)
                    if len(parts) >= total_cols_required:
                        current_data.append(parts)
        if current_data:
            all_data.extend(current_data)

    if not all_data:
        raise ValueError("No data read from the file.")

    # Convert to numpy array with object dtype for initial conversion
    n_cols = sum(col_ncomps)
    data_array = np.empty((len(all_data), n_cols), dtype=object)
    for i, row in enumerate(all_data):
        for j in range(n_cols):
            data_array[i, j] = row[j]

    # Convert all entries with safe_float_convert, except first col if int
    # The first column (assumed nstep) is index 0 of col_names (at position 0)
    # We will reconstruct fields by taking slices according to col_ncomps
    # First column is int (nstep), rest floats

    # Identify index of the "nstep" column in col_names (assumed index 0)
    nstep_pos = 0

    # Convert all data elements to float safely
    for i in range(data_array.shape[0]):
        for j in range(data_array.shape[1]):
            data_array[i, j] = safe_float_convert(data_array[i, j])

    # Now convert the whole array to float
    data_array = data_array.astype(float)

    # Extract the "nstep" array as int64
    nstep_arr = data_array[:, nstep_pos].astype(np.int64)

    # Build astropy Table columns
    table = Table()
    col_start = 0
    for i, (name, unit, ncomp) in enumerate(zip(col_names, col_units, col_ncomps)):
        if name is None:
            col_start += ncomp
            continue
        if ncomp == 1:
            col_data = data_array[:, col_start]
        else:
            col_data = data_array[:, col_start:col_start + ncomp]
        col_start += ncomp

        if unit:
            try:
                col_unit = u.Unit(unit)
            except Exception:
                col_unit = None
        else:
            col_unit = None

        table[name] = col_data * col_unit if col_unit else col_data

    key_col = col_names[nstep_pos]
    last_index_map = {}
    for i, val in enumerate(table[key_col]):
        last_index_map[val] = i
    last_indices = sorted(last_index_map.values())

    table = table[last_indices]
    table.sort(key_col)

    return table


class SinkDataLoader:
    def __init__(self, df, sinkid=1, create_file=False,verbose=False):
        """
        datafolder: base folder path (string)
        sinkid: integer sink id, formatted with zero-padding in filename
        create_file: if True, parse raw file and create cached preprocessed file;
                     if False, try to load cached file if exists

        On creation, loads data into self.table (an astropy Table)
        """
        self.datafolder = df
        self.sinkid = sinkid
        self.create_file = create_file
        self.verbose=verbose

        # Construct filenames
        # Raw file: df+'/SINK/sink_00001.txt'
        fname_raw = os.path.join(self.datafolder, 'SINK', f'sink_{sinkid:05d}.txt')
        # Preprocessed cached file: df+'/Data/sink_00001.fits'
        data_dir = os.path.join(self.datafolder, 'Data')
        os.makedirs(data_dir, exist_ok=True)
        fname_cached = os.path.join(data_dir, f'sink_{sinkid:05d}.fits')

        self.raw_filename = fname_raw
        self.cached_filename = fname_cached

        if create_file:
            # Parse raw file and save preprocessed
            if verbose:
                print(f"Parsing raw file {fname_raw} ...")
            self.data = parse_file_with_multiple_blocks(fname_raw)
            if self.data['time'][0]<0:
                self.compute_cosmo_time()
            if verbose:
                print(f"Saving preprocessed table to {fname_cached} ...")
            self.data.write(fname_cached, format='fits', overwrite=True)
        else:
            # Try loading cached file, else fallback to parsing
            if os.path.exists(fname_cached):
                if self.verbose:
                    print(f"Loading preprocessed table from {fname_cached} ...")
                self.data= Table.read(fname_cached, format='fits')
            else:

                print(f"Preprocessed file not found, parsing raw file {fname_raw} ...")
                self.data = parse_file_with_multiple_blocks(fname_raw)
                if self.data['time'][0]<0:
                    self.compute_cosmo_time()
                print(f"Saving preprocessed table to {fname_cached} ...")

                self.data.write(fname_cached, format='fits', overwrite=True)
    def compute_cosmo_time(self,add_z=True):
        if self.verbose:
            print("Cosmo run! Recalculating time from expansion.")
        try:
            outputs=find_outputs(self.df)[0]
            self.ds=yt.load(info_file(outputs,self.df))
            cosmo=FlatLambdaCDM(H0=100*self.ds.hubble_constant,Om0=self.ds.omega_matter)
            if self.verbose:
                print("Using FlatLambdaCDM")
        except:
            cosmo=WMAP9
            if self.verbose:
                print("Using WMAP9")
        if add_z:
            self.data.add_column(name='z',col=(1/self.data['aexp']-1))
        time=cosmo.age(self.data['z']).value
        self.data.replace_column(name='time',col=time)
        self.data['time'].unit='Gyr'

    def get_orbit_in_galaxy(self,mt):
        """
        Instance method to add `gal_position` and `gal_r` columns to self.data,
        computed relative to spline-interpolated galactic positions from mt.

        Parameters
        ----------
        mt : object
        Provides mt.data['aexp'] and method mt.get_branch(branch_id, field)
        """
        self.mt=mt
        data = self.data  # Access the object's data attribute

        # Create mask for rows where aexp <= max in mt data
        aexp_max = mt.data['aexp'].max()
        mask = data['aexp'] <= aexp_max

        nrows = len(data)
        # Initialize new columns with NaNs
        gal_position = np.full((nrows, 3), np.nan, dtype=float)

        # Prepare spline arrays from mt (reverse order)
        agal = mt.get_branch(1, 'aexp')[::-1]
        x_gal = mt.get_branch(1, 'x')[::-1]
        y_gal = mt.get_branch(1, 'y')[::-1]
        z_gal = mt.get_branch(1, 'z')[::-1]

        spline_x = CubicSpline(agal, x_gal)
        spline_y = CubicSpline(agal, y_gal)
        spline_z = CubicSpline(agal, z_gal)

        # Evaluate splines at masked aexp values
        aexp_masked = data['aexp'][mask]
        gal_x = spline_x(aexp_masked)
        gal_y = spline_y(aexp_masked)
        gal_z = spline_z(aexp_masked)

        # Original positions (assume Nx3 array)
        original_pos = np.array(data['position'])

        # Relative positions
        relative_pos = original_pos[mask] - np.column_stack([gal_x, gal_y, gal_z])

        # Fill nan arrays with calculated values in masked positions
        gal_position[mask] = relative_pos

        # Add new columns to self.data
        data['gal_position'] = gal_position
        return

    def magnitude(self, vector_col_name):
        """
        Compute and return the magnitude (norm) of a 3-vector column in self.data,
        preserving units.

        Parameters
        ----------
        vector_col_name : str
        The name of the 3-vector column in self.data. This column is expected to be
        an array-like quantity with shape (N, 3) and astropy.units.

        Returns
        -------
        mag : astropy.units.Quantity
        1D array of magnitudes with correct units.
        """
        # Extract the vector column (Nx3)
        vec = self.data[vector_col_name]

        # If the column has units, preserve them; otherwise treat as unitless
        unit = vec.unit if hasattr(vec, 'unit') else None

        # Convert to Quantity array if not already
        vec_q = vec if unit else vec * u.dimensionless_unscaled

        # Compute norm along the last axis (x,y,z components)
        mag = np.linalg.norm(vec_q, axis=-1)

        # Ensure magnitude has units
        if unit and not hasattr(mag, 'unit'):
            mag = mag * unit

        return mag
