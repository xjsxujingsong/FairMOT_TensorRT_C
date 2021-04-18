This is C++ version of [FairMOT](https://github.com/ifzhang/FairMOT) using TensorRT in Windows. The default model has been replaced with dlav0 to ignore the dependency of DCN. I have trained this model in the CrowdHuman dataset. If you would like to use the original DCN model, please refer to another repository [FairMOT_TensorRT](https://github.com/xjsxujingsong/FairMOT_TensorRT).


## How to Run
export onnx model file in [FairMOT](https://github.com/ifzhang/FairMOT) by adding the following code in line 479 in "/src/lib/models/networks/pose_dla_conv.py", and put it to the folder "models"

```
    z = {}
    for head in self.heads:
        z[head] = self.__getattr__(head)(y[-1])

    hm = z["hm"]
    wh = z["wh"]
    reg = z["reg"]
    hm = F.sigmoid(hm)
    hm_pool = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)


    id_feature = z['id']
    id_feature = F.normalize(id_feature, dim=1)
    id_feature = id_feature.permute(0, 2, 3, 1).contiguous() #switch id dim
    return [hm, wh, reg, hm_pool, id_feature]
```


## Acknowledgement

- Kalman Filter is borrowed from [DeepSort](https://github.com/bitzy/DeepSort), [deep_sort] https://github.com/apennisi/deep_sort
- [FairMOT]（https://github.com/ifzhang/FairMOT）