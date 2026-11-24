"""Single authorized new candidate; historical Cls/M3 are references, not queued training."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml
from pycocotools.coco import COCO
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils.ops import xyxy2xywh
from ultralytics.utils.torch_utils import unwrap_model
from scripts.eval_paper2 import evaluate as coco_evaluate, ap, ar100
from rebuilt_model import ROOT, STAGE, PUBLIC, NAMES, build_models, as_o2m, file_sha, state_sha

DATA = Path('/SVRDD/SVRDD_YOLO_READY')
DATA_YAML = STAGE / 'svrdd7.absolute.yaml'
INITIAL = STAGE / 'initialization/candidate_initial_seed42.pt'
RUN_NAME = 'P2-PKI2Cls-TaskResidualASFF_SVRDD7_o2m_100E_seed42_rebuilt_20261101'
PROTOCOL = dict(task='detect', mode='train', model=str(INITIAL), data=str(DATA_YAML),
                epochs=100, batch=32, imgsz=640, device='0', workers=4, seed=42,
                optimizer='MuSGD', lr0=0.01, lrf=0.01, momentum=0.9, weight_decay=0.0005,
                nbs=64, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.0,
                deterministic=True, amp=True, rect=False, cos_lr=False, close_mosaic=10,
                mosaic=1.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
                hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1,
                scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
                box=7.5, cls=0.5, dfl=1.5, val=True, split='val', conf=0.001,
                iou=0.7, max_det=300, save=True, save_period=5, plots=False,
                cache=False, fraction=1.0, multi_scale=0.0, compile=False,
                pretrained=True, resume=False, patience=1000000000, verbose=True,
                project=str(STAGE/'runs'), name=RUN_NAME, exist_ok=False)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    temporary.replace(path)


def status(phase, **fields):
    dump(STAGE/'status.json', dict(time=now(), phase=phase, pid=os.getpid(), **fields))


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    model = ckpt.get('ema') or ckpt.get('model')
    assert model is not None
    return model.float()


def source_hashes():
    tracked = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files'], text=True).splitlines()
    paths = [ROOT/name for name in tracked] + list((STAGE/'source').glob('*.py'))
    return {str(p): file_sha(p) for p in paths if p.is_file()}


def prepare():
    if INITIAL.exists():
        raise FileExistsError('Initial file already exists; do not overwrite initialized tensors.')
    torch.set_num_threads(2)
    for split in ['train', 'val']:
        for kind in ['images', 'labels']:
            target = DATA/kind/split
            link = STAGE/'data_view'/kind/split
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                assert link.resolve() == target.resolve()
            else:
                link.symlink_to(target, target_is_directory=True)
    for split, count in [('train',6000),('val',1000)]:
        images=sorted((STAGE/'data_view/images'/split).glob('*.jpg'))
        assert len(images)==count
        (STAGE/(split+'.txt')).write_text('\n'.join(str(p) for p in images)+'\n')
    dataset = {'path': str(STAGE), 'train': 'train.txt', 'val': 'val.txt',
               'nc': 7, 'names': NAMES}
    DATA_YAML.write_text(yaml.safe_dump(dataset, sort_keys=False), encoding='utf-8')
    native, cls, candidate = build_models()
    initial_dir = INITIAL.parent
    initial_dir.mkdir(exist_ok=True)
    for name, model in [('native7_initial_seed42_rebuilt', native), ('cls_initial_seed42_rebuilt', cls),
                        ('candidate_initial_seed42', candidate)]:
        torch.save({'model': model.cpu(), 'ema': None, 'epoch': -1, 'optimizer': None,
                    'train_args': PROTOCOL, 'date': now()}, initial_dir/(name+'.pt'))
    public_model = __import__('ultralytics').YOLO(str(PUBLIC)).model.float()
    public_state = public_model.state_dict()
    inherited = {k:v for k,v in native.state_dict().items() if k in public_state and v.shape==public_state[k].shape}
    from scripts.paper2_common import o2m_state_sha256
    receipt = {'time': now(), 'reconstructed': True, 'native7_state_sha256': state_sha(native),
               'native7_o2m_state_sha256': o2m_state_sha256(native), 'cls_state_sha256': state_sha(cls),
               'candidate_state_sha256': state_sha(candidate), 'public_sha256': file_sha(PUBLIC),
               'parameters': {'native7': sum(p.numel() for p in native.parameters()),
                              'Cls': sum(p.numel() for p in cls.parameters()),
                              'candidate': sum(p.numel() for p in candidate.parameters())},
               'PKI_parameters': sum(p.numel() for p in cls.model[-1].paper2_pki2.parameters()),
               'ASFF_parameters': sum(p.numel() for p in candidate.model[-1].asff.parameters()),
               'public_compatible_state_numel': sum(v.numel() for v in inherited.values()),
               'gate_initial_values': candidate.model[-1].task_gate_bias.sigmoid().detach().tolist(),
               'initial_files': {p.name:file_sha(p) for p in initial_dir.glob('*.pt')},
               'historical_exact_initialization_claim': False}
    dump(STAGE/'initialization.json', receipt)
    dump(STAGE/'protocol.json', PROTOCOL)
    dump(STAGE/'historical_references.json', {
        'comparison_status': 'historical references; reconstructed implementation and initialization are separately identified',
        'Cls': {'AP50':70.209,'AP75':45.789,'AP50_95':43.849,
                'best_sha256':'a3a81d00ce69c4be062cdc055a32e2b5c8bdc3cc5b4932fb09ec1bd91b91da20'},
        'M3': {'AP50':69.963,'AP75':45.705,'AP50_95':43.608,
               'best_sha256':'a50a26f79cf9aa76c5b5ff21b28efc60e554d66eaa666b68d7da281d39d83c16'},
        'baseline_training_queued': False})
    dump(STAGE/'source_manifest.json', source_hashes())
    print(json.dumps(receipt,indent=2),flush=True)


class TaskValidator(DetectionValidator):
    def __call__(self, trainer=None, model=None):
        if trainer is None:
            model = as_o2m(load_model(model) if isinstance(model,(str,Path)) else model)
        return super().__call__(trainer=trainer, model=model)

    def init_metrics(self, model):
        super().init_metrics(model)
        if self.training:
            self.box_head = unwrap_model(model).model[-1]
        self.end2end = False

    def postprocess(self, preds):
        if self.training and self.box_head.end2end:
            decoded = self.box_head._inference(preds[1]['one2many'])
            boxes = xyxy2xywh(decoded[:,:4].transpose(1,2)).transpose(1,2)
            preds = torch.cat([boxes, decoded[:,4:]],1)
        return super().postprocess(preds)

    def get_stats(self):
        result = super().get_stats()
        values = self.metrics.box.all_ap
        result['metrics/AP75(B)'] = float(values[:,5].mean()) if len(values) else 0.0
        return result


class TaskTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        assert self.data['nc'] == 7
        return load_model(INITIAL)

    def get_validator(self):
        self.loss_names = ('box_loss','cls_loss','dfl_loss')
        return TaskValidator(self.test_loader,save_dir=self.save_dir,args=copy.copy(self.args),_callbacks=self.callbacks)

    def preprocess_batch(self,batch):
        result = super().preprocess_batch(batch)
        path = self.save_dir/'first_batch_receipt.json'
        if not path.exists():
            import hashlib
            fingerprint = lambda x: hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            receipt={'images':[Path(p).name for p in result['im_file']],
                     'img_sha256':fingerprint(result['img']), 'cls_sha256':fingerprint(result['cls']),
                     'boxes_sha256':fingerprint(result['bboxes']), 'batch_idx_sha256':fingerprint(result['batch_idx']),
                     'img_shape':list(result['img'].shape)}
            if self.save_dir==STAGE/'runs'/RUN_NAME:
                ready=json.loads((STAGE/'preflight/READY.json').read_text())
                expected=json.loads((Path(ready['attempt'])/'loader/first_batch_receipt.json').read_text())
                assert receipt==expected,'Formal first batch differs from verified seed42 preflight'
            dump(path,receipt)
        return result

    def save_model(self):
        super().save_model()
        path=self.save_dir/'checkpoint_epoch_map.json'
        result=json.loads(path.read_text()) if path.exists() else {}
        result['last']=self.epoch+1
        if self.best_fitness==self.fitness: result['best']=self.epoch+1
        dump(path,result)

    def _handle_nan_recovery(self,epoch):
        if not torch.isfinite(self.loss).all() or not torch.isfinite(self.tloss).all():
            raise FloatingPointError('Non-finite loss: stop without automatic restart.')
        return False


def epoch_start(trainer):
    trainer.model.model[-1].asff.capture=True


def epoch_end(trainer):
    if not trainer.validator.training: return
    h=unwrap_model(trainer.model).model[-1]
    row={'epoch':trainer.epoch+1,'gates_cls_reg':h.task_gate_bias.sigmoid().detach().cpu().tolist(),
         'PKI_alpha_P4_P5':h.paper2_pki2.alpha.detach().cpu().tolist(),
         'first_batch_features':h.asff.observation,'metrics':trainer.metrics,
         'next_epoch_o2m':trainer.model.criterion.o2m,'time':now()}
    with (trainer.save_dir/'gate_trace.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    status('TRAINING',epoch=trainer.epoch+1,total_epochs=100,run=str(trainer.save_dir),metrics=trainer.metrics)


def train_start(trainer):
    expected=json.loads((STAGE/'initialization.json').read_text())['candidate_state_sha256']
    assert state_sha(unwrap_model(trainer.model))==expected
    assert type(trainer.optimizer).__name__=='MuSGD'
    assert trainer.epochs==100 and trainer.args.seed==42 and trainer.batch_size==32
    assert trainer.accumulate==2 and trainer.args.nbs==64
    torch.cuda.reset_peak_memory_stats()
    dump(trainer.save_dir/'runtime_verified.json',{'time':now(),'initial_state_sha256':expected,
         'epochs':100,'seed':42,'batch':32,'optimizer':'MuSGD','baseline_queued':False,
         'muon':getattr(trainer.optimizer,'muon',None),'sgd':getattr(trainer.optimizer,'sgd',None)})
    status('RUNNING_VERIFIED',run=str(trainer.save_dir),seed=42,total_epochs=100)


def train():
    ready=json.loads((STAGE/'preflight/READY.json').read_text())
    assert ready['passed'] and ready['initial_sha256']==file_sha(INITIAL)
    assert ready['gpu']['training_O2M_validation']
    assert ready['data_yaml_sha256']==file_sha(DATA_YAML)
    assert json.loads((STAGE/'protocol.json').read_text())==PROTOCOL
    expected=json.loads((STAGE/'source_manifest.json').read_text())
    assert source_hashes()==expected,'Source changed after preflight'
    assert not (STAGE/'runs'/RUN_NAME).exists(),'Refuse duplicate or overwrite'
    trainer=TaskTrainer(overrides=PROTOCOL)
    trainer.add_callback('on_train_start',train_start)
    trainer.add_callback('on_train_epoch_start',epoch_start)
    trainer.add_callback('on_fit_epoch_end',epoch_end)
    started=time.monotonic()
    trainer.train()
    dump(STAGE/'training_resources.json',{'elapsed_seconds':time.monotonic()-started,
        'peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_GiB':torch.cuda.max_memory_reserved()/2**30})
    assert source_hashes()==expected
    with (trainer.save_dir/'results.csv').open() as f: rows=[{k.strip():v for k,v in row.items()} for row in csv.DictReader(f,skipinitialspace=True)]
    assert len(rows)==100 and int(rows[-1]['epoch'])==100
    status('TRAINING_COMPLETE',run=str(trainer.save_dir),epochs=100)


def evaluate_checkpoint(checkpoint,output,data_yaml=DATA_YAML,batch=32):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    model=load_model(checkpoint)
    parameters=sum(p.numel() for p in model.parameters())
    validator=TaskValidator(args=dict(task='detect',data=str(data_yaml),imgsz=640,batch=batch,
        workers=4,device='0',conf=0.001,iou=0.7,max_det=300,rect=True,save_json=True,
        plots=False,verbose=False,half=False,split='val'),save_dir=output)
    native=validator(model=model)
    gt=COCO(str(DATA/'annotations/instances_val.json'))
    image_ids={Path(v['file_name']).stem:k for k,v in gt.imgs.items()}
    categories={v['name']:k for k,v in gt.cats.items()}
    assert set(categories)==set(NAMES)
    predictions=[]
    for p in validator.jdict:
        index=int(p['category_id'])-1
        assert 0<=index<7 and Path(p['file_name']).stem in image_ids
        predictions.append(dict(image_id=image_ids[Path(p['file_name']).stem],category_id=categories[NAMES[index]],
                                bbox=p['bbox'],score=p['score']))
    dump(output/'predictions.coco.json',predictions)
    ev=coco_evaluate(gt,predictions,[categories[n] for n in NAMES])
    # Standalone full evaluation only. Preflight subset tests do not publish these metrics.
    main={'AP50':ap(ev,iou=0.5),'AP75':ap(ev,iou=0.75),'AP50_95':ap(ev),
          'AP_small':ap(ev,'small'),'AP_medium':ap(ev,'medium'),'AP_large':ap(ev,'large'),
          'AR100':ar100(ev),'P':float(validator.metrics.box.mp),'Recall':float(validator.metrics.box.mr)}
    classes=[]
    present=list(validator.metrics.box.ap_class_index)
    for i,name in enumerate(NAMES):
        e=coco_evaluate(gt,predictions,[categories[name]])
        j=present.index(i) if i in present else None
        classes.append({'class':name,'AP50':ap(e,iou=.5),'AP75':ap(e,iou=.75),'AP50_95':ap(e),
                        'AR100':ar100(e),'P':float(validator.metrics.box.p[j]) if j is not None else None,
                        'Recall':float(validator.metrics.box.r[j]) if j is not None else None})
    result={'checkpoint':str(checkpoint),'sha256':file_sha(checkpoint),'main':main,'classes':classes,
            'training_parameters':parameters,'evaluation':'O2M + class-aware NMS; COCO maxDets100; Val only',
            'native_metrics':native}
    dump(output/'metrics.json',result)
    return result


def finalize():
    result={}
    for point in ['best','last']:
        result[point]=evaluate_checkpoint(STAGE/'runs'/RUN_NAME/'weights'/(point+'.pt'),STAGE/'completion'/point)
    dump(STAGE/'completion/summary.json',result)
    dump(STAGE/'completion/artifact_sha256.json',
         {str(p):file_sha(p) for p in (STAGE/'runs'/RUN_NAME).rglob('*') if p.is_file()})
    status('COMPLETED',run=str(STAGE/'runs'/RUN_NAME),results=str(STAGE/'completion/summary.json'))
    print(json.dumps({k:v['main'] for k,v in result.items()},indent=2),flush=True)


if __name__=='__main__':
    import sys
    torch.set_num_threads(2)
    {'prepare':prepare,'train':train,'finalize':finalize}[sys.argv[1]]()
