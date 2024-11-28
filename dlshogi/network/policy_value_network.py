import torch
import torch.nn as nn
import re

def policy_value_network(network, add_sigmoid=False):
    # wideresnet10 and resnet10_swish are treated specially because there are published models
    if network == 'wideresnet10':
        from dlshogi.network.policy_value_network_wideresnet10 import PolicyValueNetwork
    elif network == 'resnet10_swish':
        from dlshogi.network.policy_value_network_resnet10_swish import PolicyValueNetwork
    elif network == "ABN":
        from dlshogi.network.policy_value_network_resnet10_swish_att import PolicyValueNetwork
    elif network == "ABN_map27":
        from dlshogi.network.policy_value_network_resnet10_swish_att_27 import PolicyValueNetwork
    elif network == "ABN_1":
        from dlshogi.network.policy_value_network_resnet1_swish_att import PolicyValueNetwork
    elif network == "ABN_Multi":
        from dlshogi.network.policy_value_network_resnet10_swish_multi_att import PolicyValueNetwork
    elif network == "ABN_1_value_27":
        from dlshogi.network.value_network_resnet1_swish_att_27 import PolicyValueNetwork
        
    elif "ABN_27map" in network:
        from dlshogi.network.policy_value_network_resnet_swish_27att import PolicyValueNetwork
        
    elif network[:6] == 'resnet':
        from dlshogi.network.policy_value_network_resnet import PolicyValueNetwork
    elif network[:5] == 'senet':
        from dlshogi.network.policy_value_network_senet import PolicyValueNetwork
    else:
        # user defined network
        names = network.split('.')
        if len(names) == 1:
            PolicyValueNetwork = globals()[names[0]]
        else:
            from importlib import import_module
            PolicyValueNetwork = getattr(import_module('.'.join(names[:-1])), names[-1])

    if add_sigmoid:
        class PolicyValueNetworkAddSigmoid(PolicyValueNetwork):
            def __init__(self, *args, **kwargs):
                super(PolicyValueNetworkAddSigmoid, self).__init__(*args, **kwargs)

            def forward(self, x1, x2):
                y1, y2, a1, a2 = super(PolicyValueNetworkAddSigmoid, self).forward(x1, x2) #abn用に追加
                return y1, torch.sigmoid(y2), a1, a2
            
            # def forward(self, x1, x2):
            #     y1, y2 = super(PolicyValueNetworkAddSigmoid, self).forward(x1, x2) #resnet10_swish
            #     return y1, torch.sigmoid(y2)

        PolicyValueNetwork = PolicyValueNetworkAddSigmoid

    if network in [ 'wideresnet10', 'resnet10_swish']:
        return PolicyValueNetwork()
    elif "ABN_27map" in network:
        # 正規表現パターンに基づいて blocks と k を抽出する
        pattern = r'_(\d+)_(\d+)'
        match = re.search(pattern, network)
        if match:
            blocks = int(match.group(1))
            k = int(match.group(2))
            return PolicyValueNetwork(resnet_blocks=blocks, k=k)
        else:
            return PolicyValueNetwork()
    
    elif network[:6] == 'resnet' or network[:5] == 'senet':
        m = re.match('^(resnet|senet)(\d+)(x\d+){0,1}(_fcl\d+){0,1}(_reduction\d+){0,1}(_.+){0,1}$', network)

        # blocks
        blocks = int(m[2])

        # channels
        if m[3] is None:
            channels = { 10: 192, 15: 224, 20: 256 }[blocks]
        else:
            channels = int(m[3][1:])

        # fcl
        if m[4] is None:
            fcl = 256
        else:
            fcl = int(m[4][4:])

        # activation
        if m[6] is None:
            activation = nn.ReLU()
        else:
            activation = { '_relu': nn.ReLU(), '_swish': nn.SiLU(), '_swish_att': nn.SiLU()}[m[6]]

        if m[1] == 'resnet':
            return PolicyValueNetwork(blocks=blocks, channels=channels, activation=activation, fcl=fcl)
        else: # senet
            # reduction
            if m[5] is None:
                reduction = 8
            else:
                reduction = int(m[5][10:])
            return PolicyValueNetwork(blocks=blocks, channels=channels, activation=activation, fcl=fcl, reduction=reduction)
    else:
        return PolicyValueNetwork()
