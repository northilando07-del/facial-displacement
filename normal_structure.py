"""Region-local line evidence, constrained patch graphs and tangent smoothing.

Polarity denotes a ridge/valley in the *shading residual of one light*.
It is never treated as illumination-independent ground-truth geometry.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw


SEMANTIC_NAMES = ['invalid', 'forehead', 'left_temple', 'right_temple',
    'left_eye', 'right_eye', 'nose', 'left_alar', 'right_alar',
    'left_nasolabial', 'right_nasolabial', 'left_cheek', 'right_cheek',
    'philtrum', 'upper_lip', 'chin']
_ALLOWED_NAMES = [('forehead','left_temple'), ('forehead','right_temple'),
    ('forehead','left_eye'), ('forehead','right_eye'), ('forehead','nose'),
    ('left_eye','left_temple'), ('right_eye','right_temple'),
    ('left_eye','nose'), ('right_eye','nose'),
    ('left_eye','left_cheek'), ('right_eye','right_cheek'),
    ('left_eye','left_alar'), ('right_eye','right_alar'),
    ('nose','left_alar'), ('nose','right_alar'), ('nose','philtrum'),
    ('left_alar','left_nasolabial'), ('right_alar','right_nasolabial'),
    ('left_alar','left_cheek'), ('right_alar','right_cheek'),
    ('left_nasolabial','left_cheek'), ('right_nasolabial','right_cheek'),
    ('left_nasolabial','upper_lip'), ('right_nasolabial','upper_lip'),
    ('left_nasolabial','chin'), ('right_nasolabial','chin'),
    ('left_cheek','left_temple'), ('right_cheek','right_temple'),
    ('left_cheek','chin'), ('right_cheek','chin'),
    ('philtrum','upper_lip'), ('upper_lip','chin')]
ALLOWED = {tuple(sorted((SEMANTIC_NAMES.index(a), SEMANTIC_NAMES.index(b))))
           for a,b in _ALLOWED_NAMES}


def semantic_regions(shape, landmarks, valid):
    """Landmark Voronoi regions: a transparent heuristic, not face parsing."""
    lm = np.asarray(landmarks)
    height = lm[8,1] - lm[27,1]
    eye_l, eye_r = lm[36:42].mean(0), lm[42:48].mean(0)
    seeds = [lm[27]+[0,-.4*height], (lm[0]+lm[17])/2,
        (lm[16]+lm[26])/2, eye_l+[0,.06*height], eye_r+[0,.06*height],
        (lm[27]+lm[30])/2, lm[31], lm[35], (lm[31]+lm[48])/2,
        (lm[35]+lm[54])/2, (lm[3]+eye_l)/2, (lm[13]+eye_r)/2,
        (lm[33]+lm[51])/2, (lm[50]+lm[52])/2, (lm[8]+lm[57])/2]
    yy,xx = np.indices(shape, dtype=np.float32)
    best = np.full(shape, np.inf, np.float32)
    labels = np.zeros(shape, np.int16)
    for index, (x,y) in enumerate(seeds,1):
        dist = (xx-x)**2+(yy-y)**2
        take = dist < best
        labels[take] = index
        best[take] = dist[take]
    labels[~valid] = 0
    return labels


def line_features(field, scales=(.9,1.8,3.4)):
    """Scale-normalized Hessian line response, with sign-stable eigenvectors."""
    field = np.asarray(field, np.float32)
    response = np.zeros_like(field)
    tangent = np.zeros((*field.shape,2), np.float32)
    polarity = np.zeros(field.shape, np.int8)
    width = np.ones_like(field)
    stability = np.zeros_like(field)
    previous = None
    for sigma in scales:
        a = ndi.gaussian_filter(field, sigma, order=(0,2))*sigma**2
        b = ndi.gaussian_filter(field, sigma, order=(1,1))*sigma**2
        c = ndi.gaussian_filter(field, sigma, order=(2,0))*sigma**2
        disc = np.sqrt((a-c)**2+4*b*b)
        plus, minus = (a+c+disc)*.5, (a+c-disc)*.5
        choose_plus = np.abs(plus) >= np.abs(minus)
        dominant = np.where(choose_plus, plus, minus)
        small = np.where(choose_plus, minus, plus)
        anisotropy = np.clip(1-np.abs(small)/(np.abs(dominant)+1e-9),0,1)
        strength = np.abs(dominant)*anisotropy**2
        phi = .5*np.arctan2(2*b,a-c)+np.where(choose_plus,0,np.pi/2)
        tx,ty = -np.sin(phi),np.cos(phi)
        signs = np.where(tx < 0,-1.,1.)
        tx,ty = tx*signs,ty*signs
        sign = -np.sign(dominant).astype(np.int8)
        if previous is not None:
            stability += (previous == sign)*anisotropy
        previous = sign
        take = strength > response
        response[take] = strength[take]
        tangent[take,0], tangent[take,1] = tx[take],ty[take]
        polarity[take] = sign[take]
        width[take] = sigma
    stability /= max(len(scales)-1,1)
    return dict(response=response,tangent=tangent,polarity=polarity,
                scale=width,stability=stability)


def direction_weights(tangent, support, valid, normals=None):
    tx,ty = tangent[...,0],tangent[...,1]
    wh = (.015+.5*(tx[:,1:]**2+tx[:,:-1]**2))
    wv = (.015+.5*(ty[1:]**2+ty[:-1]**2))
    wh *= np.sum(tangent[:,1:]*tangent[:,:-1],-1)**2
    wv *= np.sum(tangent[1:]*tangent[:-1],-1)**2
    wh *= np.sqrt(support[:,1:]*support[:,:-1])*(valid[:,1:]&valid[:,:-1])
    wv *= np.sqrt(support[1:]*support[:-1])*(valid[1:]&valid[:-1])
    if normals is not None:
        wh *= np.sum(normals[:,1:]*normals[:,:-1],-1) > .97
        wv *= np.sum(normals[1:]*normals[:-1],-1) > .97
    return wh.astype(np.float32),wv.astype(np.float32)


def smooth_along(field, wh, wv, iterations=5, amount=.7):
    """Anchored anisotropic diffusion. No wraparound, blur or hole crossing."""
    original = np.asarray(field,np.float32)
    result = original.copy()
    denom = np.ones(original.shape[:2],np.float32)
    denom[:,1:] += amount*wh; denom[:,:-1] += amount*wh
    denom[1:] += amount*wv; denom[:-1] += amount*wv
    if original.ndim == 3:
        wh,wv,denom = wh[...,None],wv[...,None],denom[...,None]
    for _ in range(iterations):
        update = original.copy()
        update[:,1:] += amount*wh*result[:,:-1]
        update[:,:-1] += amount*wh*result[:,1:]
        update[1:] += amount*wv*result[:-1]
        update[:-1] += amount*wv*result[1:]
        result = update/denom
    return result


def compatible_edge(a,b,labels,semantics,response,threshold,valid,adjacent,
                    max_distance,boundary_limit):
    """Return a graph edge score, or None when any hard constraint fails."""
    delta = np.asarray(b['center'])-a['center']
    distance = float(np.linalg.norm(delta))
    if distance < .5 or distance > max_distance or a['polarity'] != b['polarity']:
        return None
    cross = a['region_id'] != b['region_id']
    if cross and (tuple(sorted((a['region_id'],b['region_id']))) not in adjacent
                  or max(a['boundary_distance'],b['boundary_distance']) > boundary_limit):
        return None
    sa,sb = a['semantic_id'],b['semantic_id']
    if sa != sb and tuple(sorted((sa,sb))) not in ALLOWED:
        return None
    ta,tb = np.asarray(a['tangent_dir']),np.asarray(b['tangent_dir'])
    orientation = abs(float(ta@tb))
    forward = min(abs(float(ta@delta)),abs(float(tb@delta)))/distance
    strength = min(a['strength'],b['strength'])/max(a['strength'],b['strength'],1e-9)
    scale_ratio = max(a['scale'],b['scale'])/max(min(a['scale'],b['scale']),1e-9)
    if orientation < .78 or forward < .65 or strength < .20 or scale_ratio > 3.9:
        return None
    count = max(5,int(np.ceil(distance))+1)
    points = np.linspace(a['center'],b['center'],count)
    xx = np.clip(np.rint(points[:,0]).astype(int),0,valid.shape[1]-1)
    yy = np.clip(np.rint(points[:,1]).astype(int),0,valid.shape[0]-1)
    if not valid[yy,xx].all():
        return None
    region_path = labels[yy,xx]
    if not np.isin(region_path,[a['region_id'],b['region_id']]).all():
        return None
    semantic_path = semantics[yy,xx]
    if not np.isin(semantic_path,[sa,sb]).all():
        return None
    supported = response[yy,xx] > threshold[yy,xx]*.25
    gaps = np.flatnonzero(np.diff(np.r_[True,supported,True]))
    max_gap = int(np.max(gaps[1::2]-gaps[::2])) if len(gaps) else 0
    if supported.mean() < .58 or max_gap*distance/count > boundary_limit*.5:
        return None
    score = .30*orientation+.25*forward+.15*strength+.15+.15*float(supported.mean())-.10*distance/max_distance
    return float(score) if score >= .69 else None


def connect_structures(field,confidence,valid,labels,semantics,noise,
                       normals=None,step=8,enhancement=.20,
                       line_scales=(.9,1.8,3.4),support_radius=3.):
    """Detect within mesh regions, link neighboring regions, then enhance chains."""
    if step < 1 or support_radius <= 0 or not line_scales or min(line_scales) <= 0:
        raise ValueError('Structure scales, step and support radius must be positive.')
    feature = line_features(field, scales=line_scales)
    response,tangent = feature['response'],feature['tangent']
    threshold = np.full(field.shape,max(float(noise)*.3,1e-5),np.float32)
    for label,sl in enumerate(ndi.find_objects(np.where(valid,labels+1,0)),0):
        if sl is None:
            continue
        inside = valid[sl] & (labels[sl] == label)
        values = response[sl][inside]
        if values.size:
            threshold[sl][inside] = max(float(noise)*.3,float(np.quantile(values,.64)),1e-5)
    seed = valid & (confidence > .025) & (response > threshold) & (feature['stability'] > .22)
    boundary = np.zeros(field.shape,bool)
    bh = (labels[:,1:] != labels[:,:-1]) & valid[:,1:] & valid[:,:-1]
    bv = (labels[1:] != labels[:-1]) & valid[1:] & valid[:-1]
    boundary[:,1:] |= bh; boundary[:,:-1] |= bh
    boundary[1:] |= bv; boundary[:-1] |= bv
    distance_boundary = ndi.distance_transform_edt(~boundary)
    adjacent = set()
    for a,b in [(labels[:,1:][bh],labels[:,:-1][bh]),(labels[1:][bv],labels[:-1][bv])]:
        if a.size:
            pairs = np.unique(np.sort(np.column_stack([a,b]),axis=1),axis=0)
            adjacent.update(map(tuple,pairs.tolist()))
    yy,xx = np.nonzero(seed)
    nodes = []
    if len(xx):
        regions_count = max(int(labels.max())+1,1)
        tiles = (yy//step)*((field.shape[1]+step-1)//step)+xx//step
        keys = ((tiles*regions_count+labels[yy,xx])*len(SEMANTIC_NAMES)+semantics[yy,xx])*2+(feature['polarity'][yy,xx]>0)
        order = np.argsort(keys,kind='stable')
        cuts = np.r_[0,np.flatnonzero(np.diff(keys[order]))+1,len(order)]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            ids = order[lo:hi]
            if len(ids) < 3:
                continue
            ys,xs = yy[ids],xx[ids]
            weights = response[ys,xs]*confidence[ys,xs]+1e-10
            center = [float(np.average(xs,weights=weights)),float(np.average(ys,weights=weights))]
            tv = tangent[ys,xs].copy()
            tv *= np.where(tv@tv[0]<0,-1,1)[:,None]
            tv = np.average(tv,axis=0,weights=weights)
            tv /= max(np.linalg.norm(tv),1e-8)
            nodes.append(dict(id=len(nodes),region_id=int(labels[ys[0],xs[0]]),
                semantic_id=int(semantics[ys[0],xs[0]]),center=center,
                tangent_dir=tv.tolist(),grad_dir=[float(tv[1]),float(-tv[0])],
                polarity=int(feature['polarity'][ys[0],xs[0]]),
                strength=float(np.average(response[ys,xs],weights=weights)),
                scale=float(np.average(feature['scale'][ys,xs],weights=weights)),
                confidence=float(np.average(confidence[ys,xs],weights=weights)),
                boundary_distance=float(distance_boundary[ys,xs].min()),pixels=len(ids)))
    edges = []
    if len(nodes)>1:
        centers = np.asarray([n['center'] for n in nodes])
        for i,j in sorted(cKDTree(centers).query_pairs(step*2.1)):
            score = compatible_edge(nodes[i],nodes[j],labels,semantics,response,
                                    threshold,valid,adjacent,step*2.1,step*1.5)
            if score is not None:
                edges.append(dict(a=i,b=j,score=score,cross_region=nodes[i]['region_id']!=nodes[j]['region_id']))
    degree = np.zeros(len(nodes),int)
    selected = []
    parent = np.arange(len(nodes))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i=parent[i]
        return i
    for edge in sorted(edges,key=lambda e:-e['score']):
        a,b=edge['a'],edge['b']
        if degree[a]>=3 or degree[b]>=3:
            continue
        selected.append(edge);degree[a]+=1;degree[b]+=1
        parent[root(a)]=root(b)
    groups = {}
    for i in range(len(nodes)):
        groups.setdefault(root(i),[]).append(i)
    chains = [ids for ids in groups.values() if len(ids)>=3]
    linked_ids = {i for chain in chains for i in chain}
    canvas = Image.new('L',(field.shape[1],field.shape[0]),0)
    draw = ImageDraw.Draw(canvas)
    for e in selected:
        if e['a'] in linked_ids:
            draw.line([tuple(nodes[e['a']]['center']),tuple(nodes[e['b']]['center'])],fill=255,width=2)
    line = np.asarray(canvas)>0
    if line.any():
        distance = ndi.distance_transform_edt(~line)
        support = np.exp(-.5*(distance/support_radius)**2).astype(np.float32)
        support[distance>support_radius*8/3] = 0
    else:
        support = np.zeros(field.shape,np.float32)
    support *= valid
    wh,wv = direction_weights(tangent,support,valid,normals)
    smoothed = smooth_along(field,wh,wv)
    enhanced = smoothed*(1+enhancement*support)
    enhanced[~valid]=0
    new_conf = np.clip(confidence*(.65+.65*support),0,1).astype(np.float32)*valid
    graph = dict(nodes=nodes,edges=selected,chains=chains,
        stats=dict(nodes=len(nodes),edges=len(selected),chains=len(chains),
                   linked_nodes=len(linked_ids),isolated_nodes=len(nodes)-len(linked_ids),
                   cross_region_edges=sum(e['cross_region'] for e in selected),
                   chain_support_pixels=int((support>.2).sum())),
        semantic_names=SEMANTIC_NAMES,allowed_semantic_pairs=_ALLOWED_NAMES,
        polarity_note='Per-light residual ridge/valley; not proven physical groove/peak.',
        constraints='Adjacent mesh regions, boundary proximity, polarity, tangent, scale, strength, semantic whitelist and supported path.')
    return dict(residual=enhanced.astype(np.float32),confidence=new_conf,
                support=support,tangent=tangent,response=response,graph=graph)
