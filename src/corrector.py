from classRegion import Region 
from classImage import Image
from classGrid import Grid
from classProjector import Projector
import numpy as np  
import scipy as sp
import os
import re
from pathlib import Path
from PIL import Image as PILImage
 

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
        Configuration dictionary with:
        - 'nx': number of tiles in x direction
        - 'ny': number of tiles in y direction
        - 'sx': image width in pixels
        - 'sy': image height in pixels
        - 'tiles': list of (y, x, filename) tuples sorted by position
        - 'extension': '.tif'
    
    Raises
    ------
    ValueError
        If no valid Field_YY_XX.tif files found or if images have inconsistent sizes
    """
    directory = Path(directory)
    
    # Find all matching files
    tiles = []
    for fname in os.listdir(directory):
        pos = parse_field_filename(fname)
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


def read_fiji_log(filepath):
    """Read correlation scores from Fiji stitching log file.
    
    Parameters
    ----------
    filepath : str
        Path to the Fiji log file
    
    Returns
    -------
    np.ndarray
        Array of correlation coefficients from overlapping regions
    """
    scores = []
    with open(filepath) as f:
        for line in f:
            if '=' in line and '(' in line:
                # Extract value between '=' and '('
                start = line.index('=') + 1
                end = line.index('(', start)
                scores.append(float(line[start:end].strip()))
    return np.array(scores)
    


def CreateGrid(input_param, images=None):
    """
    Creates the grid object representing the image mosaic.
    
    Parameters
    ----------
    input_param : dict
        Dictionary of configuration parameters
    images : list, optional
        Pre-loaded images (default: None)
    Note: Always uses auto-detected Field_YY_XX.tif tiles in input_param['dire'].
    
    Returns
    -------
    Grid
        Grid object containing images and regions
    """

    if 'tiles' not in input_param:
        config = auto_detect_grid_config(input_param['dire'])
        input_param.update(config)
    
    # Extract parameters
    nx = input_param['nx']  # number of columns in full grid
    ny = input_param['ny']  # number of rows in full grid
    
    # Default to processing full grid: nex=rows, ney=cols
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

    # Build regions: horizontal (left-right) and vertical (top-bottom) overlaps
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
    
    grid = Grid((nx,ny),(ox,oy),(sx,sy),images,regions)
    return grid 
            
 
def DistortionAdjustment(input_param, cam, images, epsilon=0): 
    """Identify the distortion function using Gauss-Newton optimization.
    
    Parameters
    ----------
    input_param : dict
        Configuration parameters
    cam : Projector or None
        Initial projector/camera model
    images : list or None
        Pre-loaded images
    epsilon : float, optional
        Cropping parameter for reducing overlapping regions (default: 0)
    
    Returns
    -------
    tuple
        (cam, images, grid, res_tot): Updated camera model, images, grid, and residuals
    """
    # Create grid (auto-detects tiles if not already loaded)
    grid = CreateGrid(input_param, images=images)
    
    # Extract parameters
    subsampling = input_param['subsampling']
    sx = input_param['sx']
    sy = input_param['sy']
    Niter = input_param['Niter']
    tol = input_param['tol']
    modes = input_param['modes']
    mx = input_param['mx']
    my = input_param['my']
    d0 = input_param.get('d0')
    if d0 is None:
        d0 = np.zeros(len(mx) + len(my))
    
    im_list = grid.images
    r_list = grid.regions
    
    # Track residuals across iterations
    Res = np.ones((len(r_list), Niter)) * np.nan
    
    # Initialize camera and preallocate parameter vector
    first_time = (images is None)
    if first_time:
        cam = Projector(d0, sx/2, sy/2, mx, my, sx, sy)
        if modes in ('t', 't+d'):
            # Vectorized initialization: [d0, tx_0, ty_0, ..., tx_n, ty_n]
            p = np.concatenate([d0, np.array([[im.tx, im.ty] for im in im_list]).ravel()])
        elif modes == 'd':
            p = d0.copy()
        else:
            raise ValueError(f"Unknown mode: {modes}. Must be 't', 'd', or 't+d'.")
    else:
        if modes in ('t', 't+d'):
            p = np.concatenate([cam.p, np.array([[im.tx, im.ty] for im in im_list]).ravel()])
        else:
            p = cam.p.copy()
    
    nd = len(cam.p)
    conn = np.arange(2 * len(im_list)).reshape((-1, 2)) + nd
    grid.Connectivity(conn)
    
    # Select parameters to optimize based on mode (compute once)
    if modes == 't':
        rep = grid.conn[:, :].ravel()
    elif modes == 't+d':
        rep = np.r_[np.arange(nd), grid.conn[:, :].ravel()]
    elif modes == 'd':
        rep = np.arange(nd)
    
    # Gauss-Newton optimization iterations
    print('--GN')
    
    for ik in range(Niter):
        # Update overlapping region positions
        for r in r_list:
            r.SetBounds(epsilon)
            r.IntegrationPts(s=subsampling)
        
        H, b, res_tot = grid.GetOps(cam)
        
        # Solve for parameter updates on selected subset
        Hk = H[np.ix_(rep, rep)]
        bk = b[rep]
        dp = np.linalg.solve(Hk, bk)
        p[rep] += dp
        
        # Update image translations and distortion parameters
        for i, im in enumerate(im_list):
            im.SetCoordinates(p[conn[i, 0]], p[conn[i, 1]])
        if modes in ('t+d', 'd'):
            cam.p = p[:nd]
        
        # Check convergence
        err = np.linalg.norm(dp) / np.linalg.norm(p)

        residual_stds = np.array([np.std(res_tot[i]) for i in range(len(r_list))])
        Res[:, ik] = residual_stds
        print(f"Iter # {ik + 1:2d} | dp/p={err:1.2e} | "
                f"mean_std={np.mean(residual_stds):.2f}, "
                f"max_std={np.max(residual_stds):.2f}")
        
        if err < tol:
            break
        
    return cam, images, grid, res_tot  


def DistortionAdjustement_Multiscale(parameters, cam0=None, images0=None, epsilon=0):
    """Multi-scale distortion adjustment procedure.
    
    Processes the mosaic at multiple resolution scales (coarse to fine)
    to improve convergence and robustness.
    
    Parameters
    ----------
    parameters : dict
        Configuration parameters with scale-specific settings:
        - interpolation_scales, subsampling_scales, sigma_gauss_scales, Niter_scales
    cam0 : Projector, optional
        Initial camera/projector model
    images0 : list, optional
        Pre-loaded images
    epsilon : float, optional
        Cropping parameter for reducing overlapping regions (default: 0)
    
    Returns
    -------
    tuple
        (cam, images, grid, res_tot): Optimized camera model, images, grid, and residuals
    """
    cam, images = cam0, images0
    
    for i, (interp, subsample, sigma, niter) in enumerate(zip(
        parameters['interpolation_scales'],
        parameters['subsampling_scales'],
        parameters['sigma_gauss_scales'],
        parameters['Niter_scales']
    )):
        print(f'*********** SCALE {i + 1} ***********')
        parameters['interpolation'] = interp
        parameters['subsampling'] = subsample
        parameters['sigma_gaussian'] = sigma
        parameters['Niter'] = niter
        
        cam, images, grid, res_tot = DistortionAdjustment(parameters, cam, images, epsilon)
    
    return cam, images, grid, res_tot 




def GetCamFromData(X,Y,Px,Py,xc,yc,mx,my,CameraFunc):
    """
    Returns the Camera model that fits the 
    best the Field (Px,Py) defined on the data points 
    (X,Y)
    """
    
    def residual(p,x,f):
        cam = CameraFunc(p, xc, yc, mx, my)
        Mx,My = cam.P(x[:,0],x[:,1])
        model = np.r_[Mx,My]
        return f-model 

    def Jac_residual(p,x,f):
        cam = CameraFunc(p, xc, yc, mx, my)
        dMx,dMy =  cam.dPdp(x[:,0],x[:,1])
        return -np.vstack((dMx,dMy)) 
    
    
    p0 = np.zeros(len(mx)+len(my)) # Initial start for parameters 
    result = sp.optimize.least_squares(residual,
                                       p0,  
                                       args=(np.c_[X,Y], np.r_[Px, Py] ),
                                       jac= Jac_residual )
 
    
    
    p = result['x']
    cam = CameraFunc(p, xc, yc, mx, my)
    return cam 


def GetCamFromOneImage(images,rois,tx,ty,cam,Niter,tol):
    """
    images: center, right, left, up, down
    rois: overlapping regions 
    proj: used projector 
    tx: image translations in x  (size 4)  # Attention tx<0
    ty: image translations in y  (size 4)  # Attention ty<0
    """
    
    """
    Setting the data points for the four regions 
    """
    xtot = []
    ytot = [] 
    # plt.figure() 
    # plt.imshow(images[0].pix, cmap='gray')
    for i in range(4):
        x1d = np.arange(rois[i][0,0],rois[i][1,0])
        y1d = np.arange(rois[i][0,1],rois[i][1,1])
        X,Y = np.meshgrid(x1d,y1d,indexing='ij') 
        xtot.append(X.ravel())
        ytot.append(Y.ravel())
    #     plt.plot(xtot[i],ytot[i],'.',markersize=2)
    # raise ValueError('Stop')
    p = cam.p 
    m = len(cam.p)
    for i in range(4):
        p = np.r_[p,tx[i],ty[i]]
    f1 = images[0] # Reference image is the central image
    # GN iterations 
    for ik in range(Niter):
        H   = np.zeros((len(p),len(p)))
        b   = np.zeros(len(p))
        res_tot = [None]*4 
        # Loop over the overlapping regions 
        for i in range(4):
            f2 = images[i+1]
            x = xtot[i]
            y = ytot[i]
            pgu1, pgv1   = cam.P(x,y) 
            pgu2, pgv2   = cam.P(x+p[m+2*i],y+p[m+2*i+1])  
            f1p            = f1.Interp(pgu1, pgv1)
            df1dx, df1dy   = f1.InterpGrad(pgu1,pgv1)  
            f2p            = f2.Interp(pgu2, pgv2)
            df2dx, df2dy   = f2.InterpGrad(pgu2,pgv2)   
            # Jacobian of projector with respect to the distortion parameters 
            Jpu1, Jpv1       = cam.dPdp(x,y)
            
            Jxu1      = np.zeros((len(x),8))
            Jyv1      = np.zeros((len(x),8))
            Jpu1 = np.c_[Jpu1,Jxu1]
            Jpv1 = np.c_[Jpv1,Jyv1]
            
            Jxu2          = np.zeros((len(x),8))
            Jyv2          = np.zeros((len(x),8))
            dudx, dudy, dvdx, dvdy = cam.dPdX(x+p[m+2*i],y+p[m+2*i+1])  
            Jxu2[:,2*i]   = dudx 
            Jyv2[:,2*i+1] = dvdy 
            
            Jpu2, Jpv2       = cam.dPdp(x+p[m+2*i],y+p[m+2*i+1])
            Jpu2 = np.c_[Jpu2, Jxu2 ]
            Jpv2 = np.c_[Jpv2, Jyv2 ]            
            
            
            Jp1  = np.concatenate((Jpu1,Jpv1))
            Jp2  = np.concatenate((Jpu2,Jpv2))
            n = len(x)
            df1  =  sp.sparse.dia_matrix((np.vstack((df1dx, df1dy)), np.array([0,-n])), shape=(2*n,n))
            df2  =  sp.sparse.dia_matrix((np.vstack((df2dx, df2dy)), np.array([0,-n])), shape=(2*n,n))      
            M    =  df1.T.dot(Jp1) - df2.T.dot(Jp2)  
            H    += M.T.dot(M) 
            res  = f1p - f2p 
            res_tot[i] = res 
            b    -= M.T.dot(res) 
        dp = np.linalg.solve(H,b)
        p  += dp 
        cam.p = p[:m] 
        err = np.linalg.norm(dp)/np.linalg.norm(p)
        print("Iter # %2d | s1=%2.2f , s2=%2.2f, s3=%2.2f, s4=%2.2f | dp/p=%1.2e" 
              % (ik + 1, np.std(res_tot[0]),np.std(res_tot[1]),np.std(res_tot[2]),np.std(res_tot[3]), err))
        if err < tol:
            break
    return cam
    