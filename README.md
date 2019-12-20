# Accelerate HF-Net using Openvino

In this project, we adapted the source code of hfnet for openvino tensorflow support that can be successfully convert to openvino model. After converted to openvino, the fps is increased from 6 to 22.

The original code is [Here](https://github.com/ethz-asl/hfnet)

# How to:

1. Donwload the trained weights from [here](https://drive.google.com/drive/folders/1B2jSg_H5BSXjNq8iAFis1aQ4vt2FAv-y?usp=sharing) and convert the graph for inference.

```
cd hfnet/hfnet
python3 export_model.py ../../hfnet_ckpt/config.yaml ../../ --exper_name ../../hfnet_ckpt
```
And then you will get a folder `saved_model` which contain the whole inference graph and weights.

2. Convert the `saved_model` to openvino format.

* You needs to install the master branch of [OpenVINO](https://github.com/opencv/dldt)

```
python3 opt/intel/openvino/deployment_tools/model_optimizer/mo.py --saved_model_dir ./saved_models
```

and then you will get the `saved_model.bin` and `saved_model.xml`.

if you are using OpenVINO 2019R3, you may failed to convert the model, here is the [solution](https://github.com/opencv/dldt/issues/344).

**here is my converted model [xml](https://drive.google.com/file/d/1dVTk8AURVvH8fBsMkziSNYqIcUIWvcV_/view?usp=sharing), [bin](https://drive.google.com/file/d/1DPp4j4p3ytxZGZGtTMQUE1AF-ZbLL4q_/view?usp=sharing)**

3. testing the model:

after done all steps before, you can run ```python3 tf_camera_demo.py``` for using tensorflow get the HFNet outputs or run ```python3 vino_camera_demo.py``` for using Openvino get the HFNet ouputs.

Feel free to open any issue if you need to help:)

![Image text](https://raw.githubusercontent.com/ahuizxc/HFNet_VINO/master/demo_vino.gif)
