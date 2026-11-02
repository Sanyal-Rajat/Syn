"""Optional transport objective on the existing, unchanged-capacity UNet."""
import torch

from .lidardm_adapter import LidarDMAdapter
from ..ot.config import OTConfig
from ..ot.constraints import physical_loss, pose_temporal_loss
from ..ot.features import conditional_ot, features, mass_ot, return_mass, select
from ..ot.reference import load_reference


class OTLidarDMAdapter(LidarDMAdapter):
    def __init__(self, config, sensor_meta):
        super().__init__(config, sensor_meta)
        self.ot = OTConfig(**config['ot'])
        data, self.reference_metadata = load_reference(self.ot.reference, config['manifest'], self.ot)
        for key in ('features', 'bins', 'centers', 'mass'):
            self.register_buffer(f'reference_{key}', torch.as_tensor(data[key]), persistent=False)

    def forward(self, batch):
        result = super().forward(batch, return_prediction=True)
        prediction = result.pop('prediction')
        with torch.autocast(device_type=prediction['range'].device.type, enabled=False):
            prediction = {key: value.float() for key, value in prediction.items()}
            cot = uot = prediction['range'].sum() * 0
            unsupported = []
            if self.ot.conditional_weight:
                vectors, bins, _ = features(prediction, self.sensor_meta, self.ot, batch['range'])
                source = select(vectors, bins, batch['valid'].reshape(-1).float(), self.ot.max_points)
                target = {int(key): self.reference_features[self.reference_bins == key].float()
                          for key in self.reference_bins.unique().tolist()}
                cot, unsupported = conditional_ot(source, target, self.ot)
            if self.ot.mass_weight:
                centers, mass = return_mass(prediction, self.sensor_meta, self.ot)
                uot = mass_ot(mass, self.reference_mass.float(), centers, self.ot)
            physics = physical_loss(prediction['range'], prediction['valid'], batch['range'], batch['valid'],
                                     self.sensor_meta['max_range_m'], self.config['geometry_tolerance_m'])
            temporal, available = pose_temporal_loss(prediction['range'], batch, self.sensor_meta,
                self.config['geometry_tolerance_m']) if self.config['temporal_weight'] else (cot * 0, False)
            regularization = (prediction['range'] / self.sensor_meta['max_range_m']).square().mean()
            result['loss'] = result['loss'] + self.ot.conditional_weight * cot + self.ot.mass_weight * uot
            result['loss'] = result['loss'] + self.ot.physics_weight * physics
            if available:
                result['loss'] = result['loss'] + self.config['temporal_weight'] * (
                    temporal / self.sensor_meta['max_range_m'] - result['temporal'])
            result['loss'] = result['loss'] + self.ot.regularization_weight * regularization
        result.update(conditional_ot=cot, unbalanced_ot=uot, physical=physics, pose_temporal=temporal,
                      regularization=regularization, unsupported_bins=cot.new_tensor(len(unsupported)),
                      pose_supported=cot.new_tensor(float(available)))
        return result
