import copy
import logging

import torch
from torch import nn

from inclearn.lib import factory

from .classifiers import (Classifier, CosineClassifier, DomainClassifier, ETFCls)
from .trans_classifiers import TransClassifiers

# logger = logging.getLogger(__name__)
from inclearn.lib.logger import LOGGER

logger = LOGGER.LOGGER


class BasicNet(nn.Module):

    def __init__(self, backbone_type, convnet_kwargs={}, classifier_kwargs={}, postprocessor_kwargs={}, device=None,
                 return_features=False, extract_no_act=False, classifier_no_act=False, ddp=False, all_args=None, ):
        super(BasicNet, self).__init__()

        if postprocessor_kwargs.get("type") is None:
            self.post_processor = None
        else:
            raise NotImplementedError(
                "Unknown postprocessor {}.".format(postprocessor_kwargs["type"])
            )
        logger.info("Post processor is: {}".format(self.post_processor))

        if 'trans' in backbone_type.lower() or 'adapter' in backbone_type.lower() or 'deit' in backbone_type.lower():
            self.trans = True
        else:
            self.trans = False
        self.backbone = factory.get_backbone(backbone_type, all_args=all_args, **convnet_kwargs)

        logger.info(f'classifier type: {classifier_kwargs}')
        if "type" not in classifier_kwargs:
            raise ValueError("Specify a classifier!", classifier_kwargs)
        if classifier_kwargs["type"] == "fc":
            out_dim = self.backbone.out_dim if hasattr(self.backbone, 'out_dim') else self.backbone.num_features
            self.classifier = Classifier(out_dim, device=device, **classifier_kwargs)
        elif classifier_kwargs["type"] == "cosine":
            out_dim = self.backbone.out_dim if hasattr(self.backbone, 'out_dim') else self.backbone.num_features
            self.classifier = CosineClassifier(out_dim, device=device, **classifier_kwargs)
        elif classifier_kwargs["type"] == "etf":
            out_dim = self.backbone.out_dim if hasattr(self.backbone, 'out_dim') else self.backbone.num_features
            self.classifier = ETFCls(out_dim, **classifier_kwargs)
        elif classifier_kwargs["type"].lower() == "transclassifier":
            self.classifier = TransClassifiers(self.features_dim, device='cuda', ddp=True, **classifier_kwargs)
        else:
            raise ValueError("Unknown classifier type {}.".format(classifier_kwargs["type"]))

        self.return_features = return_features
        self.extract_no_act = extract_no_act
        self.classifier_no_act = classifier_no_act
        self.device = device

        self.domain_classifier = None

        if self.extract_no_act:
            logger.info("Features will be extracted without the last ReLU.")
        if self.classifier_no_act:
            logger.info("No ReLU will be applied on features before feeding the classifier.")
        if not ddp:
            self.to(self.device)
        else:
            self.to(torch.device('cuda'))

    def on_task_end(self):
        if isinstance(self.classifier, nn.Module):
            self.classifier.on_task_end()
        if isinstance(self.post_processor, nn.Module):
            self.post_processor.on_task_end()

    def on_epoch_end(self):
        if isinstance(self.classifier, nn.Module):
            self.classifier.on_epoch_end()
        if isinstance(self.post_processor, nn.Module):
            self.post_processor.on_epoch_end()

    def forward(self, x, feat=False, only_feat=False, classifier_forward_args={}, *args, **kwargs):

        outputs = self.backbone(x, *args, **kwargs)
        outputs_each_block = kwargs.get('output_each_block', False)
        if self.trans:
            if outputs_each_block:
                block_outputs = outputs[1]
                outputs = outputs[0]
            if feat:
                if outputs_each_block:
                    return self.classifier(outputs, **classifier_forward_args), outputs, block_outputs
                else:
                    return self.classifier(outputs, **classifier_forward_args), outputs
            elif only_feat:
                if outputs_each_block:
                    return outputs, block_outputs
                else:
                    return outputs
            if outputs_each_block:
                return self.classifier(outputs, **classifier_forward_args), block_outputs
            else:
                return self.classifier(outputs, **classifier_forward_args)
        else:
            if hasattr(self, "classifier_no_act") and self.classifier_no_act:
                selected_features = outputs["raw_features"]
            else:
                selected_features = outputs["features"]

            clf_outputs = self.classifier(selected_features, **classifier_forward_args)
            outputs.update(clf_outputs)

            return outputs

    def post_process(self, x):
        if self.post_processor is None:
            return x
        return self.post_processor(x)

    @property
    def features_dim(self):
        if self.trans:
            return self.backbone.num_features
        else:
            return self.backbone.out_dim

    def add_classes(self, n_classes):
        self.classifier.add_classes(n_classes)

    def add_imprinted_classes(self, class_indexes, inc_dataset, **kwargs):
        if hasattr(self.classifier, "add_imprinted_classes"):
            self.classifier.add_imprinted_classes(class_indexes, inc_dataset, self, **kwargs)

    def add_custom_weights(self, weights, **kwargs):
        self.classifier.add_custom_weights(weights, **kwargs)

    def extract(self, x, APG: nn.Module = None, *args, **kwargs):
        if APG:
            # low_level_feat = self.backbone(x, return_immediate_feat=True)[1].mean(dim=1).unsqueeze(1)
            origin_feat = self.backbone(x, return_immediate_feat=True, break_immediate=True)
            low_level_feat = origin_feat[1][:, 0].unsqueeze(1)
            prompts = APG(low_level_feat)
            outputs_final = self.backbone(origin_feat[1], extra_tokens=prompts, shortcut=True)

            augmented_feat = outputs_final

            low_level_feat = low_level_feat.squeeze(1)
            if self.classifier.pre_MLP:
                outputs_final = self.classifier(augmented_feat, only_MLP_features=True)  # we now have MLPs in the cls
                # low_level_feat = origin_feat[0]  # we needs the feature after the normalization layer.
                # outputs_final = (outputs_final,)  # to co-operate with the next line.

            outputs = (outputs_final, low_level_feat, augmented_feat)
        else:
            origin_feat = self.backbone(x, return_immediate_feat=True, break_immediate=True)
            low_level_feat = origin_feat[1][:, 0].unsqueeze(1)
            outputs_final = self.backbone(origin_feat[1], shortcut=True)
            outputs = (outputs_final, low_level_feat, outputs_final)
        if self.trans:
            return outputs
        if self.extract_no_act:
            return outputs["raw_features"]
        return outputs["features"]

    def predict_rotations(self, inputs):
        if self.rotations_predictor is None:
            raise ValueError("Enable the rotations predictor.")
        return self.rotations_predictor(self.backbone(inputs)["features"])

    def freeze(self, trainable=False, model="all"):
        if model == "all":
            model = self
        elif model == "backbones":
            model = self.backbone
        elif model == "classifier":
            model = self.classifier
        else:
            assert False, model

        if not isinstance(model, nn.Module):
            return self

        for param in model.parameters():
            param.requires_grad = trainable
        if hasattr(self, "gradcam_hook") and self.gradcam_hook and model == "backbones":
            for param in self.backbone.last_conv.parameters():
                param.requires_grad = True

        if not trainable:
            model.eval()
        else:
            model.train()

        return self

    def get_group_parameters(self, backbone=True):
        if backbone:
            groups = {"backbones": self.backbone.parameters()}
        else:
            groups = {}

        if isinstance(self.post_processor, FactorScalar):
            groups["postprocessing"] = self.post_processor.parameters()
        if hasattr(self.classifier, "new_weights"):
            groups["new_weights"] = self.classifier.new_weights
        if hasattr(self.classifier, "old_weights"):
            groups["old_weights"] = self.classifier.old_weights
        if hasattr(self.backbone, "last_block"):
            groups["last_block"] = self.backbone.last_block.parameters()
        if hasattr(self.classifier, "_negative_weights"
                   ) and isinstance(self.classifier._negative_weights, nn.Parameter):
            groups["neg_weights"] = self.classifier._negative_weights

        return groups

    def copy(self):
        return copy.deepcopy(self)

    @property
    def n_classes(self):
        return self.classifier.n_classes
