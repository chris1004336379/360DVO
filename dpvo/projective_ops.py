import torch

MIN_DEPTH = 0.2

def extract_intrinsics(intrinsics):
    return intrinsics[...,None,None,:].unbind(dim=-1)

def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float())

    return torch.stack([x, y], dim=-1)


def iproj(patches, intrinsics):
    """ inverse projection """
    x, y, d = patches.unbind(dim=2)
    fx, fy, cx, cy = intrinsics[...,None,None].unbind(dim=2)

    lon = (x - cx) / fx
    lat = (y - cy) / fy

    xn = torch.div(torch.cos(lat) * torch.sin(lon), d)
    yn = torch.div(-torch.sin(lat), d)
    zn = torch.div(torch.cos(lat) * torch.cos(lon), d)
    X = torch.stack([xn, yn, zn, torch.ones_like(xn)], dim=-1)
    return X
 

def proj(X, intrinsics, depth=False):
    """ projection """

    X, Y, Z, W = X.unbind(dim=-1)
    fx, fy, cx, cy = intrinsics[...,None,None].unbind(dim=2)

    # d = 0.01 * torch.ones_like(Z)
    # d[Z > 0.01] = 1.0 / Z[Z > 0.01]
    # d = torch.ones_like(Z)
    # d[Z.abs() > 0.1] = 1.0 / Z[Z.abs() > 0.1]

    d = 1.0 / torch.sqrt(torch.square(X) + torch.square(Y) + torch.square(Z))
    x = fx * torch.arctan(X / Z) + cx
    y = -fy * torch.arcsin(d * Y) + cy

    if depth:
        return torch.stack([x, y, d], dim=-1)

    return torch.stack([x, y], dim=-1)


def transform(poses, patches, intrinsics, ii, jj, kk, depth=False, valid=False, jacobian=False, tonly=False):
    """ projective transform """

    # backproject
    X0 = iproj(patches[:,kk], intrinsics[:,ii])

    # transform
    Gij = poses[:, jj] * poses[:, ii].inv()

    if tonly:
        Gij[...,3:] = torch.as_tensor([0,0,0,1], device=Gij.device)

    X1 = Gij[:,:,None,None] * X0

    # project
    x1 = proj(X1, intrinsics[:,jj], depth)


    if jacobian:
        p = X1.shape[2]
        X, Y, Z, W = X1[...,p//2 ,p//2,:].unbind(dim=-1)
        x, y, d0 = patches[:,kk].unbind(dim=-1)
        o = torch.zeros_like(W)
        i = torch.zeros_like(W)

        fx, fy, cx, cy = intrinsics[:,jj].unbind(dim=-1)

        d = 1.0 / torch.sqrt(torch.square(X) + torch.square(Y) + torch.square(Z))
        d2 = d * d
        dxz = 1.0 / torch.sqrt(torch.square(X) + torch.square(Z))
        dxz2 = dxz * dxz

        lon = (x - cx) / fx
        lat = (y - cy) / fy

        Ja = torch.stack([
            W,  o,  o,  o,  Z, -Y,
            o,  W,  o, -Z,  o,  X, 
            o,  o,  W,  Y, -X,  o,
            o,  o,  o,  o,  o,  o,
        ], dim=-1).view(1, len(ii), 4, 6)
        
        Jp = torch.stack([
            fx*Z*dxz2, o, -fx*X*dxz2, o,
            fy*d2*X*Y*dxz, -fy*d2/dxz, fy*d2*Y*Z*dxz, o,
        ], dim=-1).view(1, len(ii), 2, 4)

        Jq = torch.stack([
            -torch.cos(lat) * torch.sin(lon) / (d0 * d0),
            torch.sin(lat) / (d0 * d0), 
            -torch.cos(lat) * torch.cos(lon) / (d0 * d0),
            o,
        ], dim=-1).view(1, len(ii), 1, 4)

        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:,:,None].adjT(Jj)
        
        Jz = torch.matmul(Jp, Gij.matrix())
        Jd = torch.matmul(Jz, Jq)

        return x1, (d > 0.2).float(), (Ji, Jj, Jd)

    if valid:
        return x1, (X1[...,2] > 0.2).float()
        
    return x1

def point_cloud(poses, patches, intrinsics, ix):
    """ generate point cloud from patches """
    return poses[:,ix,None,None].inv() * iproj(patches, intrinsics[:,ix])


def flow_mag(poses, patches, intrinsics, ii, jj, kk, beta=0.3):
    """ projective transform """

    coords0 = transform(poses, patches, intrinsics, ii, ii, kk)
    coords1 = transform(poses, patches, intrinsics, ii, jj, kk, tonly=False)
    coords2 = transform(poses, patches, intrinsics, ii, jj, kk, tonly=True)

    flow1 = (coords1 - coords0).norm(dim=-1)
    flow2 = (coords2 - coords0).norm(dim=-1)

    return beta * flow1 + (1-beta) * flow2
