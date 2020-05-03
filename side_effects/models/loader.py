import json
from collections import OrderedDict
import torch
from side_effects.trainer import Trainer


def rename_state_dict_keys(source, keys_to_remove=None):
    state_dict = torch.load(source, map_location="cpu")
    print(state_dict)
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        print(key)
        if key not in keys_to_remove:
            new_key = key.split("__")[-1]
            new_state_dict[new_key] = value

    torch.save(new_state_dict, source)


def load_pretrained_model(model_path, task_id):
    config = json.load(open(f"{model_path}/{task_id}_params.json"))
    model_params = config["model_params"]
    model = Trainer(network_params=model_params["network_params"])
    checkpoint = torch.load(f"{model_path}/{task_id}.checkpoint.pth", map_location="cpu")
    model.model.load_state_dict(checkpoint['net'])
    for param in model.model.parameters():
        param.requires_grad = False
    for p in model.model.drug_feature_extractor.parameters():
        print(p.requires_grad)
    return model
#t
#
#     keys_to_remove = []
#     if delete_layers == 'last':
#         model.model.classifier.net = nn.Sequential(*list(model.model.classifier.net.children())[:-2])
#         keys_to_remove.extend(["classifier.net.1.weight", "classifier.net.1.bias"])
#     rename_state_dict_keys(f"{directory}/weights.json", keys_to_remove)
#     model.load(f"{directory}/weights.json")
#     model.model.eval()
#     return model
#
#
# def _resnet(arch, block, layers, pretrained, progress, **kwargs):
#     model = ResNet(block, layers, **kwargs)
#     if pretrained:
#         state_dict = load_state_dict_from_url(model_urls[arch],
#                                               progress=progress)
#         model.load_state_dict(state_dict)
#     return model




if __name__ == '__main__':
    load_pretrained_model("/home/rogia/Téléchargements/cmap_pretrain", "L1000_bmnddi_2bb10187")

    pass

    # from gensim.test.utils import datapath
    # from gensim.models import KeyedVectors
    #
    # h = KeyedVectors.load_word2vec_format(datapath("/media/rogia/ROGIA/BioWordVec_PubMed_MIMICIII_d200.vec.bin"),
    #                                       binary=True, limit=1000000, datatype=np.float32)
    #
    # # check if the label belongs to the vocab
    # print(h.index2word)
    #
    # print(h.most_similar(positive=["femur", "fracture"]))
    #
    # # Get the vector associated to this word
    # print(h.get_vector('blood pressure'))
