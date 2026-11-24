"""COCO Val selection: AP50:95, then AP50, then the earlier epoch; FP32 validation."""
import contextlib
import copy
import io
import json
from datetime import datetime
from pathlib import Path
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import experiment as parent
import ultralytics.engine.trainer as engine
from ultralytics.utils.torch_utils import unwrap_model


class COCOValidator(parent.TaskValidator):
    def __call__(self, trainer=None, model=None):
        # BaseValidator derives validation dtype from trainer.amp; training AMP is restored immediately.
        old_amp = trainer.amp if trainer is not None else None
        if trainer is not None:
            trainer.amp = False
        try:
            result = super().__call__(trainer=trainer, model=model)
        finally:
            if trainer is not None:
                trainer.amp = old_amp
        result.update({f'COCO/{k}': v for k, v in self.coco_main.items()})
        result['fitness'] = self.coco_main['AP50_95']
        return result

    def init_metrics(self, model):
        super().init_metrics(model)
        if not hasattr(self, 'ground_truth'):
            self.ground_truth = COCO(str(parent.DATA / 'annotations/instances_val.json'))
            self.image_ids = {Path(v['file_name']).stem: k for k,v in self.ground_truth.imgs.items()}
            self.categories = {v['name']: k for k,v in self.ground_truth.cats.items()}
            assert len(self.image_ids) == 1000 and set(self.categories) == set(parent.NAMES)
        self.evaluated_ids = set()
        self.coco_main = None

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        self.evaluated_ids.update(self.image_ids[Path(p).stem] for p in batch['im_file'])

    def get_stats(self):
        result = super().get_stats()
        predictions = []
        for p in self.jdict:
            index = int(p['category_id']) - 1
            assert 0 <= index < 7
            predictions.append(dict(image_id=self.image_ids[Path(p['file_name']).stem],
                category_id=self.categories[parent.NAMES[index]], bbox=p['bbox'], score=p['score']))
        with contextlib.redirect_stdout(io.StringIO()):
            if predictions:
                detections = self.ground_truth.loadRes(predictions)
            else:
                detections = COCO()
                detections.dataset = {**self.ground_truth.dataset, 'annotations': []}
                detections.createIndex()
            ev = COCOeval(self.ground_truth, detections, 'bbox')
            ev.params.imgIds = sorted(self.evaluated_ids)
            ev.params.catIds = [self.categories[n] for n in parent.NAMES]
            ev.params.maxDets = [1, 10, 100]
            ev.evaluate()
            ev.accumulate()
        self.coco_main = dict(AP50_95=parent.ap(ev), AP50=parent.ap(ev, iou=.5),
                              AP75=parent.ap(ev, iou=.75), AR100=parent.ar100(ev))
        assert all(v is not None and 0 <= v <= 1 for v in self.coco_main.values())
        return result


class COCOSelectionTrainer(parent.TaskTrainer):
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        if mode == 'val':
            batch_size = int(self.batch_size)
        return super().get_dataloader(dataset_path, batch_size, rank, mode)

    def get_validator(self):
        self.loss_names = ('box_loss', 'cls_loss', 'dfl_loss')
        args = copy.copy(self.args)
        args.save_json = True
        return COCOValidator(self.test_loader, save_dir=self.save_dir, args=args, _callbacks=self.callbacks)

    def validate(self):
        assert self.world_size <= 1, 'This frozen protocol is single GPU.'
        metrics = self.validator(self)
        fitness = metrics.pop('fitness')
        key = (self.validator.coco_main['AP50_95'], self.validator.coco_main['AP50'])
        best_key = getattr(self, '_best_coco_key', None)
        self._new_coco_best = best_key is None or key > best_key
        if self._new_coco_best:
            self._best_coco_key = key
            self.best_fitness = fitness
        return metrics, fitness

    def save_model(self):
        # Preserve native checkpoint contents; change only BEST selection and add its actual epoch map.
        buffer = io.BytesIO()
        torch.save(dict(epoch=self.epoch, best_fitness=self.best_fitness, model=None,
            ema=copy.deepcopy(unwrap_model(self.ema.ema)).half(), updates=self.ema.updates,
            optimizer=engine.convert_optimizer_state_dict_to_fp16(copy.deepcopy(self.optimizer.state_dict())),
            scaler=self.scaler.state_dict(), train_args=vars(self.args),
            train_metrics={**self.metrics, 'fitness': self.fitness}, train_results=self.read_results_csv(),
            date=datetime.now().isoformat(), version=engine.__version__,
            git=dict(root=str(engine.GIT.root), branch=engine.GIT.branch,
                     commit=engine.GIT.commit, origin=engine.GIT.origin),
            license='AGPL-3.0 (https://ultralytics.com/license)', docs='https://docs.ultralytics.com'), buffer)
        serialized = buffer.getvalue()
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last.write_bytes(serialized)
        if self._new_coco_best:
            self.best.write_bytes(serialized)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            (self.wdir / f'epoch{self.epoch}.pt').write_bytes(serialized)
        path = self.save_dir / 'checkpoint_epoch_map.json'
        mapping = json.loads(path.read_text()) if path.exists() else {}
        mapping['last'] = self.epoch + 1
        if self._new_coco_best:
            mapping['best'] = self.epoch + 1
        parent.dump(path, mapping)
