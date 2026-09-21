"""OBJ + selected reference images -> fresh bound JSON -> layered normal workflow."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--bindings-only',action='store_true')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8-sig'))
    references=[str(Path(p).resolve()) for p in cfg.get('references',[])]
    if not references:raise ValueError('Select at least one reference image.')
    mesh=Path(cfg['mesh']).resolve()
    if mesh.suffix.lower()!='.obj' or not mesh.is_file():raise ValueError('Select an existing OBJ mesh.')
    for ref in references:
        if not Path(ref).is_file():raise FileNotFoundError(ref)
    settings=cfg.get('automatic_inputs',{})
    detector_python=settings.get('detector_python')
    soap_root=settings.get('soap_root')
    if not detector_python or not soap_root:
        raise ValueError('automatic_inputs.detector_python and automatic_inputs.soap_root are required.')
    python=Path(detector_python)
    soap=Path(soap_root)
    if not python.is_file():raise FileNotFoundError('Face detector Python environment is unavailable: '+str(python))
    if not (soap/'headlab_geometry.py').is_file():raise FileNotFoundError('Source head-selection module unavailable: '+str(soap))
    out=Path(cfg['output']).resolve();out.mkdir(parents=True,exist_ok=True)
    identity=dict(mesh=digest(mesh),references=[digest(p) for p in references],paths=references,
        code=[digest(ROOT/p) for p in ('auto_bind_model.py','detect_upgrade_references.py','prepare_inputs.py','normal_geometry.py')],
        head_selection=digest(soap/'headlab_geometry.py'))
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    auto=out/'auto_inputs'/key;auto.mkdir(parents=True,exist_ok=True)
    manifest=auto/'selected_references.json';write(manifest,references)
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    if settings.get('torch_home'):
        env['TORCH_HOME']=str(settings['torch_home'])
    print('Detecting reference landmarks...',flush=True)
    subprocess.run([str(python),'-u',str(ROOT/'detect_upgrade_references.py'),'--paths-json',str(manifest),'--output',str(auto/'reference_detection.json')],cwd=ROOT,env=env,check=True)
    binding=auto/'source_binding.json'
    if not binding.exists():
        print('Detecting and binding source-model landmarks...',flush=True)
        subprocess.run([str(python),'-u',str(ROOT/'auto_bind_model.py'),'--mesh',str(mesh),'--output',str(auto),'--soap-root',str(soap)],cwd=ROOT,env=env,check=True)
    source=json.loads(binding.read_text(encoding='utf-8'));records=json.loads((auto/'reference_detection.json').read_text(encoding='utf-8'))
    if source['mesh_sha256']!=digest(mesh):raise ValueError('Model changed during input preparation.')
    if [digest(p) for p in references]!=identity['references']:raise ValueError('Reference changed during input preparation.')
    observation=dict(mesh_sha256=source['mesh_sha256'],reference_sha256=records[0]['sha256'],
        source_indices=source['source_indices'],source_barycentric=source['source_barycentric'],
        target_original_pixels=records[0]['landmarks'],original_image_size=records[0]['size'],
        provenance=source['method'],pixel_snap_max=source['pixel_snap_max'])
    write(auto/'calibration_observations.json',observation);write(auto/'input_identity.json',identity)
    cfg.update(mesh=str(mesh),references=references,reference_detections=str(auto/'reference_detection.json'),
        calibration_observations=str(auto/'calibration_observations.json'),source_rotation=source['source_rotation'],
        skin_materials=source['skin_materials'])
    cfg.pop('soap_job',None);cfg.pop('max_references',None)
    # Do not silently inherit bindings or material names from a different asset.
    write(auto/'prepared_config.json',cfg)
    if args.bindings_only:
        write(out/'prepared_config.json',cfg)
        print('INPUT_BINDINGS_COMPLETE',str(out/'prepared_config.json'),flush=True)
        return
    print('Input JSON ready. Starting normal layers.',flush=True)
    subprocess.run([sys.executable,'-u',str(ROOT/'run_layered.py'),'--config',str(auto/'prepared_config.json')],cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
