from classGrid import Grid
from classProjector import Projector
import numpy as np  
import scipy as sp


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
    

def DistortionAdjustment(input_param, cam, images, grid=None, epsilon=0):
    """Identify the distortion function using Gauss-Newton optimization.
    
    Parameters
    ----------
    input_param : dict
        Configuration parameters
    cam : Projector or None
        Initial projector/camera model
    images : list or None
        Pre-loaded images
    grid : Grid, optional
        Pre-created grid (default: None, will auto-create)
    epsilon : float, optional
        Cropping parameter for reducing overlapping regions (default: 0)
    
    Returns
    -------
    tuple
        (cam, images, grid, res_tot): Updated camera model, images, grid, and residuals
    """
    # Create grid if not provided (auto-detects tiles if not already loaded)
    if grid is None:
        grid = Grid.CreateGrid(input_param, images=images)
    
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


def DistortionAdjustment_Multiscale(parameters, cam0=None, images0=None, grid0=None, epsilon=0):
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
    grid0 : Grid, optional
        Pre-created grid (default: None, will auto-create)
    epsilon : float, optional
        Cropping parameter for reducing overlapping regions (default: 0)
    
    Returns
    -------
    tuple
        (cam, images, grid, res_tot): Optimized camera model, images, grid, and residuals
    """
    cam, images, grid = cam0, images0, grid0
    
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
        
        cam, images, grid, res_tot = DistortionAdjustment(parameters, cam, images, grid, epsilon)
    
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
    