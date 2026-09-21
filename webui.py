"""Local normal workflow UI. Standard library only. No task starts automatically."""
import argparse
import json
import mimetypes
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT=Path(__file__).resolve().parent
CONFIG=ROOT/'displacement.json'
EXAMPLE_CONFIG=ROOT/'displacement.example.json'
LOCK=threading.RLock()
STATE={'status':'idle','log':'','output':'','started':None,'exit_code':None}
PROCESS=None


def load_config():
    if CONFIG.is_file():
        return json.loads(CONFIG.read_text(encoding='utf-8-sig'))
    if not EXAMPLE_CONFIG.is_file():
        raise FileNotFoundError('Missing displacement.json and displacement.example.json')
    cfg=json.loads(EXAMPLE_CONFIG.read_text(encoding='utf-8-sig'))
    CONFIG.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
    return cfg


def validate(cfg):
    if not isinstance(cfg,dict):raise ValueError('配置必须是对象')
    from displacement_geometry import settings
    cfg['displacement']=settings(cfg.get('displacement'))
    from displacement_structure import structure_settings
    cfg['structure']=structure_settings(cfg.get('structure'))
    for name in ('mesh','output'):
        if not isinstance(cfg.get(name),str) or not cfg[name].strip():raise ValueError('请填写 '+name)
    if not Path(cfg['output']).resolve().is_relative_to(ROOT):raise ValueError('输出目录必须位于当前位移工作流目录内')
    def number(value,lo,hi,name):
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not lo<=value<=hi:
            raise ValueError(name+' 超出允许范围')
    number(cfg.get('normal_strength',.8),0,4,'总强度')
    from normal_semantic_layers import semantic_settings, SEMANTIC_NAMES
    cfg['semantic_layers']=semantic_settings(cfg.get('semantic_layers'))
    names=SEMANTIC_NAMES if cfg['semantic_layers']['enabled'] else ('low','mid','high','lips')
    for name in names:number(cfg['layers']['gains'][name],0,4,name)
    p=cfg['protection']
    for name,lo,hi in [('brow_width_scale',.1,2),('pore_strength',0,1),('pore_max_angle',.1,3),('donor_distance',1,200)]:number(p[name],lo,hi,name)
    if type(p['fill_pores']) is not bool:raise ValueError('毛孔开关必须是布尔值')
    if cfg['semantic_layers']['enabled']:p['fill_pores']=False
    if cfg['texture_resolution'] not in (1024,2048,4096,8192):raise ValueError('不支持的贴图尺寸')
    cfg.setdefault('bake_resolution',8192)
    if type(cfg['bake_resolution']) is not int or cfg['bake_resolution'] not in (1024,2048,4096,8192):
        raise ValueError('位移与烘焙法线尺寸必须为 1024、2048、4096 或 8192')
    number(cfg['virtual_surface_samples'],0,8000000,'采样数')
    from normal_geometry_field import geometry_settings
    cfg['geometry']=geometry_settings(cfg.get('geometry'))
    from normal_reference_fusion import fusion_settings
    cfg['reference_fusion']=fusion_settings(cfg.get('reference_fusion'))
    cfg['comparison_previews']=False
    references=cfg.get('references')
    if not isinstance(references,list) or not references or any(not isinstance(p,str) or not p.strip() for p in references):
        raise ValueError('请选择至少一张参考图')
    if len(references)>16:raise ValueError('一次最多选择 16 张参考图')
    if len(set(references))!=len(references):raise ValueError('参考图不能重复')
    if cfg['structure']['enabled'] and len(references)<3:
        raise ValueError('脸部大形精修需要至少三张同视角、不同光源方向的参考图')
    return cfg


def run_job(cfg):
    global PROCESS
    try:
        folder=ROOT/'output/webui_jobs'/time.strftime('%Y%m%d_%H%M%S')
        folder.mkdir(parents=True,exist_ok=True)
        path=folder/'config.json';path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
        import os
        env=dict(os.environ,PYTHONIOENCODING='utf-8',OPENBLAS_NUM_THREADS='1')
        with LOCK:
            PROCESS=subprocess.Popen([sys.executable,'-u',str(ROOT/'run_displacement.py'),'--config',str(path)],
                cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env)
        with (folder/'run.log').open('w',encoding='utf-8') as log:
            for line in PROCESS.stdout:
                log.write(line);log.flush()
                with LOCK:STATE['log']=(STATE['log']+line)[-60000:]
        code=PROCESS.wait()
        with LOCK:STATE.update(status='complete' if code==0 else 'failed',exit_code=code)
    except Exception as e:
        with LOCK:STATE.update(status='failed',log=STATE['log']+'\n'+str(e))
    finally:
        with LOCK:PROCESS=None


class Handler(BaseHTTPRequestHandler):
    def reply(self,data,code=200):
        payload=json.dumps(data,ensure_ascii=False).encode()
        self.send_response(code);self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
    def do_GET(self):
        path=unquote(urlparse(self.path).path)
        if path=='/api/config':return self.reply(load_config())
        if path=='/api/status':
            with LOCK:return self.reply(dict(STATE))
        if path=='/api/results':
            with LOCK:output=STATE['output'];complete=STATE['status']=='complete'
            if not output or not complete:return self.reply([])
            base=Path(output).resolve()
            files=list(base.glob('maps/*/normal_preview8.png'))+list(base.glob('maps/*/displacement_preview.png'))
            files.extend(base.glob('maps/*/vector_displacement_preview.png'))
            files.extend(base.glob('scene/*comparison.png'))
            geometry_titles={'geometry/mid_preview.png':'中频高度（诊断）',
                'geometry/confidence.png':'烘焙支持范围',
                'geometry/photo_confidence.png':'多图光度置信度',
                'geometry/weight_primary.png':'主参考回退权重',
                'geometry/fold_preview.png':'Fold 大褶皱高度',
                'geometry/wrinkle_preview.png':'Wrinkle 表情纹高度',
                'geometry/fine_preview.png':'Fine 细纹高度',
                'geometry/fold_confirmed_support.png':'大褶皱多图确认范围',
                'geometry/fold_weight_structure.png':'大褶皱结构拟合权重'}
            files.extend(base/name for name in geometry_titles if (base/name).is_file())
            if (base/'synthesized_pores_mask.png').is_file():files.append(base/'synthesized_pores_mask.png')
            titles={'normal_preview8.png':'烘焙法线', 'displacement_preview.png':'细节法向位移',
                    'vector_displacement_preview.png':'总向量位移 XYZ'}
            return self.reply([{'path':f.relative_to(base).as_posix(),'title':geometry_titles.get(
                f.relative_to(base).as_posix(), f.parent.name+' / '+titles.get(f.name,f.stem))} for f in files])
        if path.startswith('/result/'):
            with LOCK:output=STATE['output']
            if not output:return self.send_error(404)
            base=Path(output).resolve();file=(base/path[len('/result/'):]).resolve()
            if not file.is_relative_to(base) or not file.is_file() or file.suffix.lower() not in ('.png','.json','.md','.obj','.npz','.npy','.blend'):return self.send_error(404)
        elif path=='/':file=ROOT/'webui/displacement.html'
        else:return self.send_error(404)
        content=file.read_bytes();self.send_response(200);self.send_header('Content-Type',mimetypes.guess_type(str(file))[0] or 'application/octet-stream');self.send_header('Content-Length',str(len(content)));self.end_headers();self.wfile.write(content)
    def do_POST(self):
        # Only requests from this local page may change configuration or run jobs.
        if self.headers.get('X-Normal-UI')!='1':return self.reply({'error':'只接受本地界面请求'},403)
        origin=self.headers.get('Origin')
        if origin and origin not in ('http://'+self.headers.get('Host',''),):return self.reply({'error':'来源不匹配'},403)
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=100000:raise ValueError('请求长度无效')
            data=json.loads(self.rfile.read(length))
            if self.path=='/api/pick':
                kind=data.get('kind')
                if kind not in ('mesh','references','output'):raise ValueError('未知选择类型')
                result=subprocess.run([sys.executable,str(ROOT/'pick_inputs.py'),kind],
                    cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
                if result.returncode:raise ValueError('无法打开文件选择器；请直接填写路径。')
                return self.reply(json.loads(result.stdout))
            if self.path not in ('/api/save','/api/run'):return self.send_error(404)
            cfg=validate(data)
            with LOCK:
                if STATE['status']=='running':return self.reply({'error':'任务正在运行，请等待完成'},409)
                CONFIG.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
                if self.path=='/api/run':
                    for key in ('mesh',):
                        if not Path(cfg[key]).exists():raise ValueError('路径不存在：'+cfg[key])
                    for image in cfg['references']:
                        if not Path(image).is_file():raise ValueError('参考图不存在：'+image)
                    STATE.update(status='running',log='',output=str(Path(cfg['output']).resolve()),started=time.time(),exit_code=None)
                    threading.Thread(target=run_job,args=(cfg,),daemon=True).start()
            self.reply({'ok':True})
        except (ValueError,KeyError,TypeError,OSError) as e:self.reply({'error':str(e)},400)
    def log_message(self,*args):pass


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=7861);parser.add_argument('--open',action='store_true');args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    url=f'http://127.0.0.1:{args.port}';print('Normal workflow UI: '+url,flush=True)
    if args.open:webbrowser.open(url)
    server.serve_forever()
