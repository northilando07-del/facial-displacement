"""Blender: actual high-to-low normal bake and matched clay renders."""
import argparse
import json
import math
from pathlib import Path
import sys
import struct
import zlib
import numpy as np
import bpy
from mathutils import Vector


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def material(name):
    result=bpy.data.materials.new(name);result.use_nodes=True
    shader=result.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value=(.38,.32,.27,1)
    shader.inputs['Roughness'].default_value=.62
    shader.inputs['Specular IOR Level'].default_value=.25
    return result


def make_object(name,data,report,rest=False,only=None):
    names=data['material_names'].tolist()
    chosen=np.ones(len(data['faces']),bool)
    if only is not None:
        chosen=data['material_ids']==names.index(only)
    else:
        for i,label in enumerate(names):
            if any(word in label.lower() for word in ('eyebrow','eyelash','tear')):
                chosen &= data['material_ids']!=i
    faces=data['faces'][chosen]
    rotation=np.asarray(report['camera']['rotation'])
    center=np.asarray(report['camera']['center'])
    vertices=(data['rest' if rest else 'vertices']-center)@rotation.T*.1
    normals=data['base_normals' if rest else 'normals']@rotation.T
    mesh=bpy.data.meshes.new(name)
    mesh.from_pydata(vertices.tolist(),[],faces.tolist());mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.collection.objects.link(obj)
    uv=mesh.uv_layers.new(name='UVMap')
    uv.data.foreach_set('uv',data['corner_uv'][chosen].reshape(-1))
    mids=data['material_ids'][chosen]
    slots={}
    for mid in np.unique(mids):
        mat=material(names[int(mid)]+' / '+name)
        slots[int(mid)]=len(mesh.materials);mesh.materials.append(mat)
    for polygon,mid in zip(mesh.polygons,mids):
        polygon.material_index=slots[int(mid)];polygon.use_smooth=True
    mesh.normals_split_custom_set(normals[faces].reshape(-1,3).tolist())
    obj['source_material_slots']=json.dumps({names[i]:slot for i,slot in slots.items()})
    obj['geometry_role']='Undisplaced original geometry' if rest else 'Actual displaced geometry'
    return obj


def prepare_scene(report,cfg):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene=bpy.context.scene;scene.render.engine='CYCLES'
    scene.cycles.samples=int(cfg.get('render_samples',32))
    scene.cycles.seed=77;scene.cycles.use_denoising=True
    backend='CPU'
    preferences=bpy.context.preferences.addons['cycles'].preferences
    for name in ('OPTIX','CUDA'):
        try:
            preferences.compute_device_type=name;preferences.get_devices()
            if any(d.type!='CPU' for d in preferences.devices):
                for device in preferences.devices:device.use=device.type!='CPU'
                scene.cycles.device='GPU';backend=name;break
        except (ValueError,TypeError,RuntimeError):pass
    width,height=report['screen_size'];size=int(cfg.get('render_size',1000))
    scene.render.resolution_y=size;scene.render.resolution_x=round(size*width/height)
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.render.image_settings.color_depth='8';scene.view_settings.view_transform='AgX'
    world=bpy.data.worlds.new('Neutral studio');world.use_nodes=True;scene.world=world
    world.node_tree.nodes['Background'].inputs['Color'].default_value=(.22,.25,.3,1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value=.18
    scale=report['camera']['scale'];tx,ty=report['camera']['translation']
    x=(width*.5-tx)*.1/scale;y=(ty-height*.5)*.1/scale
    target=Vector((x,y,0))
    bpy.ops.object.camera_add(location=(x,y,6))
    camera=bpy.context.object;camera.name='Calibrated portrait camera'
    camera.rotation_euler=(0,0,0);camera.data.type='ORTHO'
    camera.data.ortho_scale=height*.1/scale;scene.camera=camera
    def area(name,position,power,size):
        data=bpy.data.lights.new(name,'AREA');data.energy=power;data.shape='DISK';data.size=size
        obj=bpy.data.objects.new(name,data);bpy.context.collection.objects.link(obj)
        obj.location=position;obj.rotation_euler=(target-obj.location).to_track_quat('-Z','Y').to_euler()
        return obj
    key=area('Key',(-3,2.5,4),450,1.4)
    area('Fill',(2.5,1,4),50,3);area('Rim',(1.5,3,-2),170,2)
    return scene,camera,key,target,backend


def save_rgb(image,path,depth='16'):
    # Normal colors are numerical data and must bypass display transforms.
    width,height=image.size
    pixels=np.empty(width*height*4,np.float32);image.pixels.foreach_get(pixels)
    rgb=np.flipud(pixels.reshape(height,width,4))[:,:,:3]
    bits=int(depth);maximum=(1<<bits)-1
    encoded=np.rint(np.clip(rgb,0,1)*maximum).astype('>u2' if bits==16 else np.uint8)
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    # Stream rows so an 8K RGB16 export does not duplicate the entire raw PNG.
    compressor=zlib.compressobj(6)
    with Path(path).open('wb') as stream:
        stream.write(b'\x89PNG\r\n\x1a\n')
        stream.write(chunk(b'IHDR',struct.pack('>IIBBBBB',width,height,bits,2,0,0,0)))
        for row in encoded:
            data=compressor.compress(b'\0'+row.tobytes())
            if data:stream.write(chunk(b'IDAT',data))
        stream.write(chunk(b'IDAT',compressor.flush()))
        stream.write(chunk(b'IEND',b''))
    return rgb


def attach_normal(obj,material_name,image):
    slots=json.loads(obj['source_material_slots'])
    mat=obj.data.materials[slots[material_name]]
    nodes,links=mat.node_tree.nodes,mat.node_tree.links
    texture=nodes.new('ShaderNodeTexImage');texture.image=image;texture.location=(-550,0)
    normal=nodes.new('ShaderNodeNormalMap');normal.space='TANGENT';normal.uv_map='UVMap'
    normal.inputs['Strength'].default_value=1.;normal.location=(-250,0)
    links.new(texture.outputs['Color'],normal.inputs['Color'])
    links.new(normal.outputs['Normal'],nodes.get('Principled BSDF').inputs['Normal'])
    return texture


def bake_normals(root,report,cfg,low_data,high_data,low_display):
    size=int(cfg.get('bake_resolution',8192));results=[]
    names=read(Path(report['evidence_directory'])/'run_config.json')['skin_materials']
    scene=bpy.context.scene
    for name in names:
        directory=root/'maps'/name;directory.mkdir(parents=True,exist_ok=True)
        low=make_object('Bake target '+name,low_data,report,rest=True,only=name)
        high=make_object('Bake source '+name,high_data,report,only=name)
        image=bpy.data.images.new('Baked actual high geometry '+name,width=size,height=size,alpha=False,float_buffer=True)
        image.colorspace_settings.name='Non-Color'
        neutral=np.empty((size,size,4),np.float32);neutral[:]=(.5,.5,1,1)
        image.pixels.foreach_set(neutral.ravel())
        del neutral
        for mat in low.data.materials:
            node=mat.node_tree.nodes.new('ShaderNodeTexImage');node.image=image
            mat.node_tree.nodes.active=node
        bpy.ops.object.select_all(action='DESELECT')
        low.select_set(True);high.select_set(True);bpy.context.view_layer.objects.active=low
        extrusion=.1*max(1.,report['max_displacement_mm']*2)/report['millimeters_per_unit']
        scene.render.bake.use_clear=False
        if report['moved_final_vertices']>0:
            bpy.ops.object.bake(type='NORMAL',normal_space='TANGENT',use_selected_to_active=True,
                cage_extrusion=extrusion,max_ray_distance=extrusion*3,margin=16)
        # Bake filtering blends vectors at some UV boundaries and can leave
        # blue values slightly above one. Normalize vectors before encoding;
        # this preserves their directions and avoids clipping numeric data.
        pixels=np.empty(size*size*4,np.float32);image.pixels.foreach_get(pixels)
        pixels=pixels.reshape(-1,4);filtered_vectors=0
        for start in range(0,len(pixels),size*128):
            block=pixels[start:start+size*128]
            vectors=block[:,:3]*2-1
            lengths=np.linalg.norm(vectors,axis=1)
            filtered_vectors+=int((np.abs(lengths-1)>1e-4).sum())
            if np.any(lengths<1e-6):
                raise RuntimeError('Bake returned a zero-length tangent vector.')
            block[:,:3]=(vectors/lengths[:,None]+1)*.5
        image.pixels.foreach_set(pixels.ravel())
        del pixels,block,vectors,lengths
        rgb=save_rgb(image,directory/'normal_opengl16.png')
        save_rgb(image,directory/'normal_preview8.png',depth='8')
        np.save(directory/'normal_baked_float32.npy',rgb)
        changed_texels=0
        for start in range(0,size,128):
            changed_texels+=int((np.linalg.norm(rgb[start:start+128]-np.array([.5,.5,1],np.float32),axis=2)>.001).sum())
        del rgb
        result=dict(material=name,resolution=size,changed_texels=changed_texels,
            bake='Cycles selected-to-active from actual displaced geometry' if report['moved_final_vertices'] else 'Analytic neutral map: zero displacement',
            tangent_space='Blender native tangent normal bake',
            renormalized_filtered_vectors=filtered_vectors,
            raw_png_bit_depth=16,cage_extrusion_scene_units=extrusion)
        if result['changed_texels']<100 and report['moved_final_vertices']>0:
            raise RuntimeError('High-to-low normal bake is unexpectedly neutral.')
        print('NORMAL_BAKE',json.dumps(result),flush=True);results.append(result)
        baked=bpy.data.images.load(str(directory/'normal_opengl16.png'),check_existing=False)
        baked.colorspace_settings.name='Non-Color';baked.pack()
        attach_normal(low_display,name,baked)
        bpy.data.objects.remove(low,do_unlink=True);bpy.data.objects.remove(high,do_unlink=True)
        bpy.data.images.remove(image)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reuse-geometry-renders',action='store_true',
                        help='Keep existing base/high renders while validating a changed bake.')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    root=args.output.resolve();out=root/'scene';out.mkdir(parents=True,exist_ok=True)
    report=read(root/'report.json');cfg=read(root/'run_config.json')
    structured=cfg.get('structure',{}).get('enabled',False)
    scene_file=out/('displacement88.blend' if structured else 'displacement77.blend')
    low_data=load(root/'structure/original.npz' if structured else root/'levels/level_00.npz')
    high_data=load(root/'head_displaced.npz')
    scene,camera,key,target,backend=prepare_scene(report,cfg)
    base=make_object('01 Original - no normal map',low_data,report,rest=True)
    high=make_object('02 Displaced high mesh - no normal map',high_data,report)
    baked=make_object('03 Original mesh - baked high normal',low_data,report,rest=True)
    objects={'base':base,'geometry':high,'baked':baked}
    if structured:
        corrected=make_object('04 Structure corrected - before details',load(root/'structure/coarse_04.npz'),report)
        objects={'base':base,'structure':corrected,'geometry':high,'baked':baked}
    for obj in objects.values():obj.hide_render=True
    baked_maps=bake_normals(root,report,cfg,low_data,high_data,baked)
    # Save an inspectable scene before rendering, with real geometry selected.
    high.hide_render=False
    bpy.ops.object.select_all(action='DESELECT')
    high.select_set(True);bpy.context.view_layer.objects.active=high
    scene['workflow']='Document 88: 16 / 4 / 1 cells, then residual detail subdivision' if structured else 'Document 77: residual displacement'
    scene['evidence_directory']=report['evidence_directory']
    scene['max_displacement_mm']=report['max_displacement_mm']
    scene['unit_provenance']=report['unit_provenance']
    scene['display_note']='02 is actual geometry with no normal texture; 03 is the original mesh with the baked normal.'
    bpy.ops.wm.save_as_mainfile(filepath=str(scene_file))
    rendered=[]
    for view in ('left','right','oblique'):
        key.location.x=3 if view=='right' else -3
        key.rotation_euler=(target-key.location).to_track_quat('-Z','Y').to_euler()
        angle=math.radians(30) if view=='oblique' else 0.
        camera.location=target+Vector((6*math.sin(angle),0,6*math.cos(angle)))
        # Portrait geometry uses world Y as up; avoid track-quaternion roll
        # toward world Z when orbiting away from the frontal singularity.
        camera.rotation_euler=(0.,angle,0.)
        for state,obj in objects.items():
            for other in objects.values():other.hide_render=other!=obj
            name=view+'_'+state+'.png';scene.render.filepath=str(out/name)
            if args.reuse_geometry_renders and state!='baked' and (out/name).is_file():
                rendered.append(name)
                continue
            bpy.ops.render.render(write_still=True);rendered.append(name)
            print('GEOMETRY_RENDER_COMPLETE',name,flush=True)
    camera.location=target+Vector((0,0,6));camera.rotation_euler=(0,0,0)
    key.location.x=-3;key.rotation_euler=(target-key.location).to_track_quat('-Z','Y').to_euler()
    for obj in objects.values():
        obj.hide_render=obj!=high;obj.hide_set(obj!=high);obj.select_set(obj==high)
    bpy.context.view_layer.objects.active=high
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type=='VIEW_3D':area.spaces.active.region_3d.view_perspective='CAMERA'
    bpy.ops.wm.save_as_mainfile(filepath=str(scene_file))
    verification=dict(render_device=backend,samples=scene.cycles.samples,
        image_size=[scene.render.resolution_x,scene.render.resolution_y],rendered=rendered,
        high_vertices=len(high.data.vertices),high_triangles=len(high.data.polygons),
        base_vertices=len(base.data.vertices),baked_vertices=len(baked.data.vertices),
        high_has_normal_texture=any(n.type=='NORMAL_MAP' for m in high.data.materials for n in m.node_tree.nodes),
        baked_maps=baked_maps,baked_strength=1.,scene=str(scene_file))
    if verification['high_has_normal_texture']:
        raise RuntimeError('Geometry validation render must not use a normal map.')
    (out/'render_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
    report['rendering']=verification;report['status']='render_complete'
    (root/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('GEOMETRY_SCENE_COMPLETE',json.dumps(verification),flush=True)


if __name__=='__main__':main()
