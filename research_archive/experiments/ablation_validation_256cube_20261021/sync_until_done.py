"""One-time background result copy; exits after completion or a worker failure."""
from pathlib import Path
from datetime import datetime,timezone
from html.parser import HTMLParser
from urllib.parse import urlsplit,unquote
import hashlib,html,json,os,re,subprocess,time,zipfile
ROOT=Path(__file__).resolve().parent
HOST='eias-member.eitech.edu.cn'
REMOTE='/docsystem/niuke/lvyuehang/fm3d_validation_256cube_20261021'
PY='/docsystem/niuke/lvyuehang/conda_envs/lyh_flowoct/bin/python'
ARCHIVE=ROOT.parent.parent/'artifacts/validation_256cube_progress.zip'
FLAGS=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0


def call(args,source=None):
    result=subprocess.run(args,capture_output=True,text=True,encoding='utf-8',errors='replace',
                          timeout=180,creationflags=FLAGS,input=source)
    if result.returncode:raise RuntimeError((result.stderr or result.stdout)[-2000:])
    return result.stdout.strip()


def ssh(script):
    if script=='peek256.py':
        # Sending read-only status code over stdin also works before file upload.
        code=(ROOT/script).read_text(encoding='utf-8')
        code=code.replace('r=Path(__file__).resolve().parent','r=Path('+repr(REMOTE)+')')
        return call(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,PY,'-'],code)
    return call(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,PY,REMOTE+'/'+script])


def status(value):
    value['utc']=datetime.now(timezone.utc).isoformat()
    p=ROOT/'sync_status.json';temporary=p.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temporary.replace(p)
    print(json.dumps(value,ensure_ascii=False),flush=True)
    page=ROOT/'reports/index.html'
    if page.exists():
        document=page.read_text(encoding='utf-8')
        state=value['state']
        if state=='running':
            message='已确认完成 '+str(len(value['completed']))+'/6 组。'
            message+=' '.join(x['arm']+'：'+str(x['updates'])+'/400' for x in value['active'])
        elif state=='complete':message='六组均已完成，35份预测及本地报告检查通过。'
        elif state=='failed':message='服务器会话已结束，最终审计尚未通过。请查看 sync_status.json。'
        else:message='服务器连接暂不可用，正在退避重连；当前显示最近已确认的结果。'
        banner='<!-- LOCAL_SYNC --><p class="note">'+html.escape(message)+'</p><!-- /LOCAL_SYNC -->'
        if '<!-- LOCAL_SYNC -->' in document:
            document=re.sub(r'<!-- LOCAL_SYNC -->.*?<!-- /LOCAL_SYNC -->',banner,document,flags=re.S)
        else:document=document.replace('</h1>','</h1>'+banner,1)
        if state in ('running','sync_retry') and 'http-equiv="refresh"' not in document:
            document=document.replace('<meta charset="utf-8">','<meta charset="utf-8"><meta http-equiv="refresh" content="60">',1)
        if state in ('complete','failed'):
            document=re.sub(r'<meta http-equiv="refresh"[^>]*>','',document)
        page.write_text(document,encoding='utf-8')


class Links(HTMLParser):
    def __init__(self):super().__init__();self.missing=[];self.count=0
    def handle_starttag(self,tag,attrs):
        for key,value in attrs:
            if key not in ('src','href') or not value or value.startswith('#'):continue
            parsed=urlsplit(value)
            if parsed.scheme or parsed.netloc:continue
            self.count+=1
            if not (ROOT/'reports'/unquote(parsed.path)).exists():self.missing.append(value)


def main():
    previous=None
    failures=0
    while True:
        try:
            remote=json.loads(ssh('peek256.py'))
            failures=0
            key=(tuple(remote['completed']),bool(remote['audit']),remote['worker_alive'])
            if key!=previous:
                manifest=json.loads(ssh('mirror256.py'))
                call(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',HOST+':'+REMOTE+'/mirror256.zip',str(ARCHIVE)])
                assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()==manifest['sha256']
                with zipfile.ZipFile(ARCHIVE) as z:
                    assert all((ROOT/name).resolve().is_relative_to(ROOT.resolve()) for name in z.namelist())
                    # The status and currently executing local helper belong to this host.
                    for item in z.infolist():
                        if item.filename not in ('sync_status.json','sync_until_done.py'):
                            z.extract(item,ROOT)
                previous=key
            if remote['audit']:
                audit=remote['audit'];assert audit['status']=='passed' and audit['checked_3d_predictions']==35
                assert len(remote['completed'])==6
                parser=Links();parser.feed((ROOT/'reports/index.html').read_text(encoding='utf-8'))
                assert not parser.missing,parser.missing
                from PIL import Image
                for p in (ROOT/'reports').glob('*_256.png'):assert Image.open(p).size==(256,256),p
                result=dict(status='passed',html_links_checked=parser.count,missing_links=[],
                            server_audit=audit,archive_sha256=manifest['sha256'])
                (ROOT/'verification/local_report_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
                status(dict(state='complete',completed=remote['completed'],verified_predictions=35))
                return
            if not remote['worker_alive']:
                status(dict(state='failed',remote=remote))
                return
            status(dict(state='running',completed=remote['completed'],active=remote['active']))
        except Exception as error:
            failures+=1
            status(dict(state='sync_retry',error=str(error)))
        time.sleep(min(300,45*2**min(failures,3)))


if __name__=='__main__':main()
