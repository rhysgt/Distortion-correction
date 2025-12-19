import skimage                   
from multiprocessing import Pool 
import concurrent.futures
import matplotlib.pyplot as plt 
import numpy as np 
import skimage
import pathlib
import os
import re
from pathlib import Path
from PIL import Image as PILImage
from scipy.ndimage import map_coordinates 

from src.classProjector import Projector
from src.classImage import Image
from src.classRegion import Region


def SetGlobalHorizontalShift(im0, im1, l):
    """
    Compares two consecutive images and sets 
    the position of the right image 
    """
 
    I0c = im0.pix[:,-l:]
    I1c = im1.pix[:, :l]
    shift,_,_ = skimage.registration.phase_cross_correlation( I0c, I1c, upsample_factor=100 ) 
    ti = shift[0] 
    tj = im0.pix.shape[1] - I0c.shape[1]  + shift[1] 
     
    im1.SetCoordinates(  im0.tx + ti  ,  im0.ty + tj )
     
    
def SetGlobalVerticalShift(im0, im1, l):
    """
    Compares two consecutives images and sets 
    the position of the bottom image 
    """
    
    I0c  =  im0.pix[-l:,:]   
    I1c  =  im1.pix[:l, :]   
    
    shift,_,_ = skimage.registration.phase_cross_correlation( I0c, I1c, upsample_factor=100 ) 
    ti = im1.pix.shape[0] - I1c.shape[0] + shift[0]
    tj = shift[1] 
    
    im1.SetCoordinates( im0.tx + ti,  im0.ty + tj )
    
def ProgressBar(percent):
    width = 40 
    left = width * percent // 100
    right = width - left
    
    tags = "█" * int(np.round(left))
    spaces = " " * int(np.round(right))
    percents = f"{percent:.0f}%"
    
    print("\r[", tags, spaces, "]", percents, sep="", end="", flush=True)
    
    

class Grid:
    """
    Totality of regions 
    Related to the distorted images to be stitched 
    """
    def __init__(self,nImages,overlap,shape,images,regions):
        """
        shape : (nx,ny) : number of images in x and y directions 
        overlap : (ox,oy): size of the overlap in x and y directions in % 
        imfiles : list of the image files 
        ATTENTION: ONLY ONE CONSIDERED ORDER (SNAKE BY ROWS - RIGHT & DOWN)
        """
        self.sx = shape[0]
        self.sy = shape[1]
        self.images = images 
        self.regions = regions 
        self.nx = nImages[0]
        self.ny = nImages[1]
        self.ox = overlap[0]
        self.oy = overlap[1] 
    
    def BuildInterp(self, method = 'cubic-spline'):
        """ 
        Builds interpolation scheme for all the images 
        """
        for im in self.images:
            im.BuildInterp(method)

    def GaussianFilter(self, sigma):
        """
        Filters all the images of the grid 
        """
        for im in self.images:
            im.GaussianFilter(sigma) 
            
    def LoadImages(self):
        """ 
        Loads the images of the mosaic
        """
        for im in self.images:
            im.Load() 
            
    def Connectivity(self,conn):
        """
        Sets the vector that selects the modes and translation in the 
        toral Degree of Freedom vector
        """
        self.conn = conn
    
    @staticmethod
    def parse_field_filename(filename):
        """
        Parse tile position from filename format: Field_YY_XX.tif
        
        Parameters
        ----------
        filename : str
            Filename to parse (e.g., 'Field_12_18.tif' = row 12, column 18)
        
        Returns
        -------
        tuple or None
            (y, x) position as integers (0-indexed), or None if pattern doesn't match
        """
        match = re.match(r'^(?:Field)_(\d+)_(\d+)\.(?:tif|tiff)$', filename, flags=re.IGNORECASE)
        if match:
            y, x = int(match.group(1)), int(match.group(2))
            return (y, x)
        return None
    
    @staticmethod
    def auto_detect_grid_config(directory):
        """
        Automatically detect grid configuration from Field_YY_XX.tif files.
        
        Parameters
        ----------
        directory : str or Path
            Directory containing the tile images
        
        Returns
        -------
        dict
            Configuration dictionary with grid dimensions and tile info
        
        Raises
        ------
        ValueError
            If no valid Field_YY_XX.tif files found
        """
        directory = Path(directory)
        
        # Find all matching files
        tiles = []
        for fname in os.listdir(directory):
            pos = Grid.parse_field_filename(fname)
            if pos is not None:
                tiles.append((pos[0], pos[1], fname))
        
        if not tiles:
            raise ValueError(f"No files matching Field_YY_XX.tif pattern found in {directory}")
        
        # Sort by position (y, then x)
        tiles.sort(key=lambda t: (t[0], t[1]))
        
        # Determine grid dimensions
        y_positions = set(t[0] for t in tiles)
        x_positions = set(t[1] for t in tiles)
        
        ny = len(y_positions)
        nx = len(x_positions)
        
        # Get image dimensions from first file
        first_file = directory / tiles[0][2]
        with PILImage.open(first_file) as img:
            sx, sy = img.size
        
        print(f"Auto-detected grid configuration:")
        print(f"  Grid size: {nx} x {ny} tiles")
        print(f"  Image size: {sx} x {sy} pixels")
        print(f"  Total tiles found: {len(tiles)}")
        
        return {
            'nx': nx,
            'ny': ny,
            'sx': sx,
            'sy': sy,
            'tiles': tiles,
            'extension': '.tif'
        }
    
    @classmethod
    def CreateGrid(cls, input_param, images=None):
        """
        Creates the grid object representing the image mosaic.
        
        Parameters
        ----------
        input_param : dict
            Dictionary of configuration parameters
        images : list, optional
            Pre-loaded images (default: None)
        
        Returns
        -------
        Grid
            Grid object containing images and regions
        """
        if 'tiles' not in input_param:
            config = cls.auto_detect_grid_config(input_param['dire'])
            input_param.update(config)
        
        # Extract parameters
        nx = input_param['nx']  # number of columns
        ny = input_param['ny']  # number of rows
        nrows = input_param.get('nex', ny)
        ncols = input_param.get('ney', nx)
        
        ox = input_param['ox']
        oy = input_param['oy']
        sigma_gaussian = input_param['sigma_gaussian']
        interpolation = input_param['interpolation']
        sx = input_param['sx']
        sy = input_param['sy']
        dire = input_param['dire']
        
        tiles_dict = {(t[0], t[1]): t[2] for t in input_param['tiles']}
        
        # Load and prepare images in row-major order
        first_time = images is None
        if first_time:
            images = [None] * (nrows * ncols)

        idx = 0
        for row in range(nrows):
            for col in range(ncols):
                if first_time:
                    filename = tiles_dict.get((row, col))
                    if filename is None:
                        raise ValueError(f"Missing tile at position ({row}, {col}). Expected Field_{row}_{col}.tif")
                    
                    images[idx] = Image(dire + filename)
                    tx = col * np.floor(sy - sy * oy / 100)  # x offset per column
                    ty = row * np.floor(sx - sx * ox / 100)  # y offset per row
                    images[idx].SetCoordinates(ty, tx)
                    images[idx].SetIndices(row, col)

                images[idx].Load()
                images[idx].GaussianFilter(sigma=sigma_gaussian)
                images[idx].BuildInterp(method=interpolation)
                idx += 1

        # Normalize coordinates to start from (0, 0)
        tx0, ty0 = images[0].tx, images[0].ty
        for im in images:
            im.SetCoordinates(im.tx - tx0, im.ty - ty0)

        # Build regions
        regions = []
        for row in range(nrows):
            for col in range(ncols - 1):
                left = row * ncols + col
                right = row * ncols + col + 1
                regions.append(Region((images[left], images[right]), (left, right), 'v'))
        
        for col in range(ncols):
            for row in range(nrows - 1):
                top = row * ncols + col
                bottom = (row + 1) * ncols + col
                regions.append(Region((images[top], images[bottom]), (top, bottom), 'h'))
        
        grid = cls((nx, ny), (ox, oy), (sx, sy), images, regions)
        return grid 
            
    
    def ExportTile(self, file):
        """ Writing Tile Configuration file for FIJI 
            Warning !! Inverted x and y axis for FIJI """ 
        with open(file, 'w') as f:
            f.write('# Define the number of dimensions we are working on\n')
            f.write('dim = 2\n')
            f.write('\n# Define the image coordinates\n')  
            for im in self.images:
                f.write(pathlib.Path(im.fname).name+"; ; "+"(%f,%f)"%(im.ty,im.tx)+"\n")


    def ReadTile(self, file):
        """ Reading Tile configuration File from FIJI  
            Warning !! Inverted x and y axis for FIJI """      
        im_file_names = [pathlib.Path(im.fname).name for im in self.images ] 
        tile = open(file)  
        line = tile.readline()
        while line!='# Define the image coordinates\n':
            line = tile.readline()
        line = tile.readline()        
        while len(line)!=0 : 
            linec  = line.split(';')
            # Reading image 
            im_file_name   = linec[0]
            coords = linec[2][2:-2].split(',')
            tx = float(coords[1])
            ty = float(coords[0])
            index_im  = im_file_names.index(im_file_name)
            self.images[index_im].SetCoordinates(tx,ty)
            line   = tile.readline()
 
    
    def PlotImageBoxes(self, origin=(0,0),eps=(0,0), color='green'):
        """
        Plots the overlapping regions 
        """
        # A = np.zeros( ( int(self.sx*self.nx - (self.nx-1)*np.floor(self.sx*self.ox/100))+eps[0], 
        #                 int(self.sy*self.ny - (self.ny-1)*np.floor(self.sy*self.oy/100))+eps[1]  ) )
        # plt.imshow(A)
        for im in self.images:
            im.PlotBox(origin,color)  

    def _process_blending_region(self, im, camc, xs_flat, ys_flat, origin, windowExt, 
                                 sx_ims, sy_ims, sx_max, sy_max, 
                                 ims, ws, is_overlap_fn, weight_fn, interpolation_order):
        """Helper for linear blending (reduces code duplication)"""
        im_tx, im_ty = im.tx, im.ty
        
        imin = max(0, int(np.floor(im_tx + origin[0] - windowExt[0])))
        imax = min(sx_ims - 1, int(np.ceil(im_tx + origin[0] + self.sx + windowExt[0])))
        jmin = max(0, int(np.floor(im_ty + origin[1] - windowExt[1])))
        jmax = min(sy_ims - 1, int(np.ceil(im_ty + origin[1] + self.sy + windowExt[1])))
        
        # Early exit if no overlap
        if imin > imax or jmin > jmax:
            return
        
        mask = (xs_flat >= imin) & (xs_flat <= imax) & (ys_flat >= jmin) & (ys_flat <= jmax)
        idx_window = np.where(mask)[0]
        
        if len(idx_window) == 0:
            return
        
        u, v = camc.P(xs_flat[idx_window] - origin[0] - im_tx,
                      ys_flat[idx_window] - origin[1] - im_ty)
        
        fov_mask = (u >= 0) & (u <= sx_max) & (v >= 0) & (v <= sy_max)
        idx_fov = idx_window[fov_mask]
        
        if len(idx_fov) == 0:
            return
        
        u_fov, v_fov = u[fov_mask], v[fov_mask]
        
        # Separate overlapping and non-overlapping regions
        mask_overlap = is_overlap_fn(u_fov, v_fov)

        # Use cached spline-filtered pixels when available to avoid per-call prefiltering
        if interpolation_order > 1:
            pix_src = im.pix_prefiltered if getattr(im, "pix_prefiltered", None) is not None else im.pix
        else:
            pix_src = im.pix

        # Build weights in one pass: 1 for non-overlap, blend weight for overlap
        weights = np.ones_like(u_fov)
        if np.any(mask_overlap):
            weights[mask_overlap] = weight_fn(u_fov[mask_overlap], v_fov[mask_overlap])

        # Single interpolation call for both overlap/non-overlap
        sampled = map_coordinates(
            pix_src,
            [u_fov, v_fov],
            order=interpolation_order,
            mode='constant',
            cval=0,
            prefilter=False,
        )

        ws[idx_fov] += weights
        ims[idx_fov] += sampled * weights
    
    def StitchImages(self, cam=None, origin=(0,0), eps=(0,0), 
                     interpolation_order=3, fusion_mode='linear blending'):
        """
        Stitch the grid into one image 
        """
        
        camc = None 
        if cam is None: 
            camc = Projector( np.array([0,0]), 0, 0, [0], [0], 1, 1 ) # Identity projector  
        else: 
            camc = cam 
            
        for r in self.regions: 
            r.SetBounds(epsilon=0) 

        # Ensure each image has a cached spline-filtered version for fast map_coordinates
        if interpolation_order > 1:
            for im in self.images:
                if getattr(im, "pix_prefiltered", None) is None:
                    im.BuildSplinePrefilter(order = interpolation_order)
        
        # Compute fused image size once
        sx_ims = int(self.sx*self.nx - (self.nx-1)*np.floor(self.sx*self.ox/100))+eps[0]
        sy_ims = int(self.sy*self.ny - (self.ny-1)*np.floor(self.sy*self.oy/100))+eps[1]
        
        # Pre-compute coordinate grids once (shared by both fusion modes)
        xs, ys = np.meshgrid(np.arange(sx_ims), np.arange(sy_ims), indexing='ij')
        xs_flat = xs.ravel()
        ys_flat = ys.ravel()
        
        ims = np.zeros(sx_ims*sy_ims)
        ws = np.zeros(sx_ims*sy_ims)
        windowExt = np.array([10, 10])
        
        if fusion_mode == 'linear blending':
            # Cache boundary values to avoid repeated computation
            sx_max = self.sx - 1
            sy_max = self.sy - 1
            
            print('Fusing images (linear blending):')
            for k, r in enumerate(self.regions):
                ProgressBar(100 * (k+1) / len(self.regions))
                if r.type == 'v':
                    # Vertical regions: blending on v coordinate
                    r_hy = r.hy  # Extract to avoid lambda closure issues
                    
                    # Left image im0 (decreasing weight with v)
                    self._process_blending_region(
                        r.im0, camc, xs_flat, ys_flat, origin, windowExt, 
                        sx_ims, sy_ims, sx_max, sy_max, ims, ws,
                        is_overlap_fn=lambda u, v, hy=r_hy, sy=sy_max: v > sy - hy,
                        weight_fn=lambda u, v, hy=r_hy, sy=sy_max: (v - sy) / (-hy),
                        interpolation_order=interpolation_order
                    )
                    
                    # Right image im1 (increasing weight with v)
                    self._process_blending_region(
                        r.im1, camc, xs_flat, ys_flat, origin, windowExt,
                        sx_ims, sy_ims, sx_max, sy_max, ims, ws,
                        is_overlap_fn=lambda u, v, hy=r_hy: v <= hy,
                        weight_fn=lambda u, v, hy=r_hy: v / hy,
                        interpolation_order=interpolation_order
                    )
                    
                elif r.type == 'h':
                    # Horizontal regions: blending on u coordinate
                    r_hx = r.hx  # Extract to avoid lambda closure issues
                    
                    # Top image im0 (decreasing weight with u)
                    self._process_blending_region(
                        r.im0, camc, xs_flat, ys_flat, origin, windowExt,
                        sx_ims, sy_ims, sx_max, sy_max, ims, ws,
                        is_overlap_fn=lambda u, v, hx=r_hx, sx=sx_max: u > sx - hx,
                        weight_fn=lambda u, v, hx=r_hx, sx=sx_max: (u - sx) / (-hx),
                        interpolation_order=interpolation_order
                    )
                    
                    # Bottom image im1 (increasing weight with u)
                    self._process_blending_region(
                        r.im1, camc, xs_flat, ys_flat, origin, windowExt,
                        sx_ims, sy_ims, sx_max, sy_max, ims, ws,
                        is_overlap_fn=lambda u, v, hx=r_hx: u <= hx,
                        weight_fn=lambda u, v, hx=r_hx: u / hx,
                        interpolation_order=interpolation_order
                    )
                else:
                    raise ValueError('Unknown region type')
            
            ws[ws == 0] = 1
            ims = ims / ws
            print('\n')
            return ims.reshape((sx_ims, sy_ims)) 

        else:
            raise ValueError('Unknown blending method')
    
    def local_assembly(self,r,cam):
        """
        Redefined function for the parallelization 
        """
        return r.GetOps(cam)
    
    def GetOps(self,cam):
        """
        Returns the Gauss-Newton side members 
        """
        import os
        nd   = len(cam.p)  # Number of distortion parameters 
        nImages = len(self.images)
        ndof = nd + 2*nImages
        H   = np.zeros((ndof,ndof))
        b   = np.zeros(ndof)
        res_tot = [None]*len(self.regions)
        arnd = np.arange(nd)
        # Precompute connectivity per region to avoid repeated concat
        region_conn = [np.concatenate((arnd, self.conn[r.index_im0,:], self.conn[r.index_im1,:]))
                       for r in self.regions]

        max_workers = min(len(self.regions), max(1, os.cpu_count() or 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i, (r, result) in enumerate(zip(self.regions, executor.map(self.local_assembly, self.regions, [cam]*len(self.regions)))):
                Hl, bl, resl = result
                rep = region_conn[i]
                repk = np.ix_(rep, rep)
                H[repk] += Hl
                b[rep]  += bl
                res_tot[i] = resl
        return H, b, res_tot   
    
    def PlotResidualMap(self,res_list, epsImage = 10):
        """
        Plot the difference between neighboring images 
        on their overlapping regions in order 
        to reveal the effect of the distortion 
        """
        # Size of the fused image 
        sx_ims = int(self.sx*self.nx - (self.nx-1)*np.floor(self.sx*self.ox/100) + epsImage )
        sy_ims = int(self.sy*self.ny - (self.ny-1)*np.floor(self.sy*self.oy/100) + epsImage )  
        # Creating empty image and the average weights 
        R = np.zeros((sx_ims,sy_ims)) 
        for i,r in enumerate(self.regions):
            shape = (r.xmax - r.xmin, r.ymax - r.ymin)
            X = r.pgx.reshape(shape)
            Y = r.pgy.reshape(shape)
            R[X,Y] = res_list[i].reshape(shape)
        plt.imshow(R,cmap='RdBu')

 
    
    def RunGN(self,cam):
        return 
    
    def SetPairShift(self,cam,overlap):
        for r in self.regions:
            r.SetPairShift(cam, overlap) 
            
    
    def SetTranslations(self, alphax=0.8, alphay = 0.8 ):
        """
        Finds the translation between the images 
        SHould be improved 
        Gives less accurate results than Fiji 
        """
        # alphax is parameter of reducing the 
        # size of the overlapping regions in order 
        
        lx = int( np.floor(alphax * self.sx * self.ox / 100) ) 
        ly = int( np.floor(alphay * self.sy * self.oy / 100) )  
        
        # Process rows: compare left-right neighbors
        for row in range(self.ny):
            for col in range(self.nx - 1):
                idx_left = row * self.nx + col
                idx_right = row * self.nx + col + 1
                SetGlobalHorizontalShift(self.images[idx_left], self.images[idx_right], ly)
        
        # Process columns: compare top-bottom neighbors
        for col in range(self.nx):
            for row in range(self.ny - 1):
                idx_top = row * self.nx + col
                idx_bottom = (row + 1) * self.nx + col
                SetGlobalVerticalShift(self.images[idx_top], self.images[idx_bottom], lx)
                    

