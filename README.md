视频链接：https://www.bilibili.com/video/BV1R6goeME1w/?vd_source=5fd403a80632dfe3f92e876b9131ccc0

摘  要

高动态范围成像和目标检测是计算机视觉领域的两个关键技术。HDRI能够捕捉和显示场景中的更多亮度细节，显著提升图像质量；目标检测技术则致力于在图像或视频中识别和定位特定目标。将HDRI与先进的目标检测算法如YoloV5结合，能够在复杂光照条件下实现更加精确的目标检测。这篇摘要将探讨基于YoloV5的高动态范围成像的目标检测方法及其在不同应用场景中的优势。

HDRI技术通过多重曝光或其他方式捕捉场景中的丰富亮度信息，生成包含更多细节的图像。传统成像技术在高对比度场景中容易丢失亮部和暗部细节，而HDRI通过合成多张不同曝光的图像，克服了这一问题。HDRI的核心过程包括多重曝光图像采集、图像合成和色调映射。多重曝光图像采集是通过不同曝光时间拍摄多张图像，图像合成使用算法将这些图像合成一张HDR图像，色调映射则将HDR图像映射到显示设备能够表现的亮度范围内，以便于观察和处理。

目标检测技术旨在从图像或视频中识别并定位特定目标，如行人、车辆、动物等。YoloV5是目标检测领域的一个重要算法，它将目标检测视为一个回归问题，通过单次前向传播直接预测目标的类别和位置。与传统算法相比，YoloV5在速度和精度上都有显著提升，特别适合实时应用。YoloV5的核心包括网络结构设计、损失函数定义和多尺度检测。
在复杂光照条件下（如强光、背光、低光等），传统的目标检测算法容易出现漏检和误检现象。将HDRI技术与YoloV5目标检测算法结合，可以显著提高目标检测的鲁棒性和准确性。通过HDRI捕捉更多亮度细节，生成高质量的图像输入YoloV5模型，使得在极端光照条件下目标的细节更加清晰，从而提升检测效果。

关键词：HDRI，深度学习；卷积神经网络；目标检测，YOLOv5

第一章 绪  论

1.1 研究背景

高动态范围成像（HDRI）技术在计算机视觉和摄影领域的兴起，是为了克服传统成像技术在高对比度场景下的局限性。HDRI通过捕捉和显示更广泛的亮度细节，提高了图像质量，解决了传统成像技术中常见的亮部过曝和暗部欠曝问题。

研究基于深度学习的目标检测算法具有重要的理论和实践意义。目标检测是计算机视觉领域的一项关键任务，它在许多应用中起着重要作用，包括智能监控、自动驾驶、目标行为分析等。基于深度学习的目标检测算法通过学习大量的数据，并利用卷积神经网络（CNN）等深度学习模型自动提取特征，能够更精确地识别目标。研究基于深度学习的目标检测算法的背景是深度学习技术的快速发展和广泛应用。深度学习模型在计算机视觉任务中取得了突破性的成果，如在ImageNet图像分类挑战赛中超越人类。在目标检测领域，基于深度学习的方法不仅在检测准确率上有明显优势，而且还具有较高的计算效率。目标检测算法的核心是从复杂的场景中准确地提取出目标,以实现计算机对人类的理解和识别。目前,传统的目标检测算法主要是基于特征提取和分类算法来实现,其缺点在于容易受到光照、背景、拍摄角度等因素的影响,导致检测效果不稳定。而深度学习技术则能够对图像数据进行端到端的训练,从而实现对目标的准确检测。

因此,本文将基于深度学习技术,结合HDRI对目标检测算法进行研究和探讨以提高目标检测算法的准确率和稳定性,为实现智能交通、安全监控等领域提供更好的支持和应用。

1.2 研究现状

HDRI通过多重曝光技术获取场景中的不同亮度信息，再通过图像合成和色调映射等技术生成高动态范围图像。传统的低动态范围（LDR）图像由于感光元件的限制，只能记录有限的亮度范围，而HDRI克服了这一限制，能够捕捉和显示更广泛的亮度信息。HDRI技术在近年来取得了显著进展，从基础理论到应用研究都展现出了强劲的发展势头。随着深度学习和硬件技术的不断进步，HDRI在各个领域的应用前景广阔。未来，HDRI技术将继续朝着更高质量、更高效率和更广泛应用的方向发展，为计算机视觉和图像处理领域带来更多创新和突破。

目标检测是计算机视觉的一个重要研究方向,其目的是精确识别给定图像中特定目标物体的类别和位置。近年来,深度卷积神经网络(Deep Convolutional Neural Networks,DCNN)所具有的特征学习和迁移学习能力,在目标检测算法特征提取、图像表达、分类与识别等方面取得了显著进展。近些年来，目标目标检测是计算机视觉领域的一个重要研究方向、基础任务和研究热点，结合YOLO深度学习算法进行的相关研究也日渐成熟。YOLO将目标检测概括为一个回归问题，实现端到端的训练和检测，由于其良好的速度-精度平衡，近几年一直处于目标检测领域的领先地位，被成功地研究、改进和应用到众多不同领域。

1.3 研究意义

HDRI技术在各个领域的应用不断拓展。例如，在医疗影像中，HDRI用于提高X射线和CT图像的对比度，帮助医生更准确地诊断病情。在虚拟现实（VR）和增强现实（AR）中，HDRI提升了视觉体验的真实感和沉浸感。此外，HDRI在电影制作、视频游戏、摄影等领域也有广泛应用。

目标检测作为计算机视觉领域的一个重要分支，近年来得到了广泛的关注和快速的发展。它不仅在学术界引起了广泛的研究兴趣，在工业界也得到了大量的应用。目标检测的研究意义不仅体现在技术的进步上，更在于它在各个领域中的广泛应用及其带来的社会和经济效益。

目标检测技术的发展，得益于深度学习特别是卷积神经网络（CNN）的突破性进展。经典的目标检测算法如R-CNN、Fast R-CNN、Faster R-CNN以及YOLO系列和SSD等，不仅提高了检测精度，还显著提升了检测速度。这些算法的提出和改进，为目标检测技术的发展奠定了坚实的基础。

1.4 应用场景

自动驾驶：在自动驾驶中，车辆需要在不同光照条件下准确识别道路上的行人、车辆和交通标志。结合HDRI的YoloV5能够在强光和低光环境下提高检测精度，确保行车安全。
安防监控：在安防监控中，夜间或复杂光照条件下的监控视频质量较差，目标检测效果不理想。结合HDRI的监控系统能够在各种光照条件下提升检测效果，提高公共安全水平。
无人机巡检：无人机在巡检过程中，常常会遇到强光、阴影等复杂光照条件。结合HDRI的目标检测系统能够在这些条件下准确检测目标，提高巡检效率和准确性。












第二章 相关理论及模型介绍

2.1结合的方法

2.1.1HDRI图像生成

（1）多重曝光图像采集

为了生成高动态范围图像，首先需要采集多张具有不同曝光时间的图像。通常使用HDR相机进行拍摄，可以设置多个不同的曝光时间，如低曝光、中等曝光和高曝光。低曝光图像能够捕捉场景中的高亮度细节，中等曝光图像捕捉中等亮度细节，而高曝光图像捕捉低亮度细节。这些图像共同覆盖了更广的亮度范围。

（2） 图像合成

    通过HDR合成算法，将多张不同曝光的图像合成一张HDR图像。这些算法通常利用每张图像中曝光最佳的部分，综合生成一张细节丰富的HDR图像。合成过程中需要考虑图像对齐，以避免由于相机抖动或物体移动导致的图像偏移。常用的HDR合成算法包括加权平均法、对数曝光融合法等。
    
（3） 色调映射

HDR图像虽然包含了丰富的亮度信息，但由于显示设备的动态范围有限，需要通过色调映射将HDR图像映射到低动态范围（LDR）。色调映射算法能够在保留亮度细节的同时，调整图像对比度，使其适合显示和进一步处理。常见的色调映射算法包括局部色调映射和全局色调映射。

2.1.2目标检测模型训练

（1）数据集准备

    为了训练一个能够处理HDR图像的目标检测模型，需要准备包含HDR图像的数据集。这些数据集可以通过上述HDRI图像生成步骤获得，也可以使用已有的HDR图像数据集。为了增强模型的泛化能力，可以进行数据增强，如模拟不同光照条件下的图像变化，生成更多样化的训练样本。
    
（2）模型训练

    使用YoloV5算法进行模型训练。YoloV5是一种先进的目标检测算法，能够在速度和精度之间取得良好的平衡。训练过程中，可以将HDR图像或处理后的LDR图像输入模型，使模型学会在不同光照条件下检测目标。需要调整训练超参数，如学习率、批量大小等，以获得最佳的训练效果。
    
（3）模型优化

    根据HDR图像的特点，对YoloV5模型进行优化。可以考虑引入注意力机制，使模型能够关注图像中的关键区域，提升检测性能。此外，可以调整网络结构，使其更适应处理HDR图像。例如，增加网络的深度或宽度，以捕捉更多的图像细节；或者在网络中加入多尺度特征提取模块，提高对不同大小目标的检测能力。

2.1.3模型部署与测试

（1）模型部署

    训练好的YoloV5模型可以部署到实际应用环境中。部署过程中需要考虑计算资源和实时性要求，可以选择在高性能服务器或嵌入式设备上运行模型。为了保证实时性，可以进行模型压缩和加速，如使用模型剪枝、量化等技术。
    
（2）性能测试

    在实际应用场景中测试模型性能，验证其在不同光照条件下的检测效果。测试过程中可以采集一系列具有不同光照条件的图像或视频，评估模型的检测准确性和鲁棒性。通过对比实验，验证HDRI技术在提升目标检测性能方面的效果。

2.2YOLO系列算法

YOLO 是一种基于卷积神经网络的实时目标检测算法，其主要思想是将目标检测问题看作是一个回归问题，同时在一张图像上直接预测多个目标的位置和类别。YOLO 算法的实现过程可以分为以下几个步骤：

（1）输入预处理：将原始图像调整为固定大小，并将像素值缩放到 0-1 之间。

（2）CNN特征提取：使用卷积神经网络对输入图像进行特征提取，通过多个卷积层和池化层获取高层次的语义特征。

（3）目标分类：在最后的卷积层后添加一个全连接层，用于预测图像中每个目标的类别。

（4）目标定位：将最后的卷积层输出的特征图分成一个 S×S 的网格，每个网格预测 N 个边界框以及每个框对应的置信度得分。对于每个框，还需要预其对应目标的坐标偏移量。由于每个框可能会跨越多个网格，因此对于每个框需要预测的是其相对于当前网格左上角的位置和大小。

（5）损失函数：定义损失函数来度量目标检测结果的准确性。YOLO 算法采用了两个部分的损失函数：定位误差和分类误差。

（6）非极大值抑制：对预测的边界框进行筛选，使用非极大值抑制来消除冗余的框，保留最终的检测结果。YOLO 算法采用的是端到端的训练方式，可以直接从图像中预测目标的类别和位置，具有非常高的检测速度。对于复杂背景和目标的遮挡等情况都有较好的鲁棒性，并且 YOLO 算法可以使用不同尺度的特征图来检测不同大小的目标，具有更好的尺度适应能力。

YOLOv5的优点：首先是检测速度快，因为YOLOv5采用了一种端到端的方式，先提取区域，然后再检测物体，其次是准确率高，YOLOv5采用了多尺度特征，在计算流程中，每个特征都被考虑到，这有效地提高了检测精度；另外YOLOv5还采用了模型重用和自适应采样技术，使模型参数更加有效，以达到更高的检测准确率。YOLOv5的第三大优点是其模型参数较小，YOLOv5的训练过程也更容易，可以在短时间内训练出更精确的模型，而不需要大量的计算资源。总而言之，YOLOv5具有准确率高，检测速度快，模型参数小，容易训练等优点，是目前最受欢迎的物体检测算法之一。

2.3HDRI系列算法

（1）拍摄照片并记录曝光值

（2）对齐图片

（3）计算crf并合成图片




第三章 实验环境与数据集

3.1 目标检测以及HDRI环境的安装

本实验基于Pycharm编辑软件，及Python环境进行程序实验。需要安装一些基本环境所需要的库文件。 为了能够运行YOLOv5项目，我们需要安装一些必要的Python库。这些库的列表可以requirements.txt文件中找到。我们可以使用以下命令来安装这些库：
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

3.2.数据集

在网上找的公开数据集，然后进行HDRI图像生成。数据包括常见的80类别，包括各种车辆（如飞机、自行车、船、公共汽车、汽车、摩托车和火车）、家居用品（如瓶子、椅子、餐桌、盆栽、沙发和电视/显示器）以及动物（如鸟、猫、牛、狗、马、羊）和人。数据集是一个被广泛应用于计算机视觉领域的数据集，该数据集包含了大量的日常场景，其中每个场景都包含了丰富的上下文信息，有助于深入理解各种对象的语义。每一张图像都进行了详细的标注，包括边界框、分割掩码等。
数据集部分图片展示：


第四章部分主体代码

4.1文件：`train.py`

这个文件主要用于模型的训练，包含了数据加载、模型定义、训练循环等功能。

4.1.1. 导入必要的库和模块：
    ```python
    import argparse
    import os
    import random
    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    ```
4.1.2. 定义训练函数：
    ```python
    def train(hyp):
        # 数据加载和预处理
        ...
        # 模型定义
        ...
        # 训练循环
        ...
        for epoch in range(start_epoch, epochs):
            # 训练代码
            ...
    ```
4.1.3. 解析命令行参数：
    ```python
    def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=300)
        parser.add_argument('--batch-size', type=int, default=16)
        ...
        opt = parser.parse_args()
        return opt
    ```

4.1.4. 主函数：
    ```python
    if __name__ == "__main__":
        opt = parse_opt()
        train(opt)
    ```

4.2文件：`val.py`
这个文件主要用于模型的验证，包含了验证数据的加载、模型评估等功能。
4.2.1. 导入必要的库和模块：
    ```python
    import argparse
    import json
    import os
    import torch
    from pathlib import Path
    ```

4.2.2. 定义验证函数：
    ```python
    def validate(data_loader, model):
        # 数据加载和预处理
        ...
        # 模型验证
        ...
        for i, (images, targets) in enumerate(data_loader):
            # 验证代码
            ...
    ```

4.2.3. 解析命令行参数：
    ```python
    def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default='path/to/weights')
        parser.add_argument('--data', type=str, default='path/to/data')
        ...
        opt = parser.parse_args()
        return opt
    ```

4.2.4. 主函数：
    ```python
    if __name__ == "__main__":
        opt = parse_opt()
        validate(opt)
    ```

4.3文件：`benchmarks.py`

这个文件主要用于模型的基准测试，包含了模型导出、性能评估等功能。
4.3.1. 导入必要的库和模块：
    ```python
    import argparse
    import time
    import torch
    from pathlib import Path
    ```

4.3.2. 定义测试函数：
    ```python
    def test(weights, imgsz, device):
        # 模型加载
        ...
        # 数据加载和预处理
        ...
        # 性能测试
        ...
        for i, (images, targets) in enumerate(data_loader):
            # 测试代码
            ...
    ```

4.3.3. 解析命令行参数：
    ```python
    def parse_opt():
        parser = argparse.ArgumentParser()
        parser.add_argument('--weights', type=str, default='path/to/weights')
        parser.add_argument('--imgsz', type=int, default=640)
        ...
        opt = parser.parse_args()
        return opt
    ```

4.3.4. 主函数：
    ```python
    if __name__ == "__main__":
        opt = parse_opt()
        test(opt)
    ```
4.4 hdri.py:
对拍摄的图像进行处理
总结
        -  hdri.py  同时拍摄的多张图片并记录其对应的曝光时间，计算crf曲线并通过其合成。
- `train.py` 用于模型训练，包含数据加载、模型定义和训练循环。
- `val.py` 用于模型验证，包含验证数据的加载和模型评估。
- `benchmarks.py` 用于模型的基准测试和性能评估。
第五章 实验验证与结果分析
5.1 结合HDRI的YOLOv5训练结果性能分析
为了验证我们模型的性能，我们可以在一个独立的测试集上进行评估。这个测试集应该包含各种各样的目标图像，以确保我们的评估结果具有普遍性。我们可以计算模型在这个测试集上的精确度、召回率和F1分数，并将其与训练结果进行比较。如果这两组结果相近，那么我们可以认为模型具有良好的泛化能力。
5.1.1 常规曲线 P-R curve、P curve、R curve
(1) P curve
当判定概率超过置信度阈值时，各个类别识别的准确率。当置信度越大时，类别检测越准确，但是这样就有可能漏掉一些判定概率较低的真实样本。
如图4.1所示，本次训练模型在置信度达到0.93时，mask和no-mask的预测准度都达到了100%。


图5.1 YOLOv5 训练下的目标模型的置信度阈值 - 准确率曲线图
(2) R curve
当置信度越小的时候，类别检测的越全面（不容易被漏掉，但容易误判）。


图5.2 YOLOv5训练下的目标模型的置信度阈值 – 召回率曲线图
如图5.2所示，本次训练模型在置信度达到0.98的时候，recall曲线回归到0。
(3)P-R curve
如图5.3所示，P-R curve曲线图在recall趋近于1.0时，此时预测准确率则最低。当recall趋近于0时，预测准确率最高为1.0。

图5.3 YOLOv5训练下的目标模型的精确率和召回率的关系图
5.1.2Results


图5.4loss functions 曲线图
如图5.4所示，图中各参数的定义为：
•横坐标代表的是训练轮数(epoch)；
•box_loss是预测框与标定框之间的误差（CIoU），越小定位得越准；
•obj_loss是计算网络的置信度，越小判定为目标的能力越准；
•cls_loss是计算锚框与对应的标定分类是否正确，越小分类得越准；
•mAP_0.5:0.95（mAP@[0.5:0.95]）是 表示在不同IoU阈值（从0.5到0.95，步长0.05）（0.5、0.55、0.6、0.65、0.7、0.75、0.8、0.85、0.9、0.95）上的平均mAP；
•mAP_0.5是表示阈值大于0.5的平均mAP。
5.2结果验证





图5.5测试照片
本研究使用了4张目标图片进行测试。在测试图像中，对于明显的目标图像，系统正确识别了，正确率达到了100%，预测效果很好。

第六章 结论
本文提出了一种基于YOLOv5的目标检测系统。鉴于目标检测在许多领域的重要性，如公共安全、交通管理和智能监控，我们设计了这个系统以提高目标检测的精度和速度。本文的主要内容分为三个部分。
第一部分介绍了目标检测的基础知识。我们首先解释了目标检测的定义，并介绍了目标检测常用的数据集以及它们在目标检测中的重要性。然后，我们描述了衡量目标检测性能的两个关键指标：精度和速度，并从Precision, Recall, F1 score，P-R curve以及AP等多个方面进行了详细的解释。
第二部分对YOLOv5目标检测网络进行了详细的解读。我们选择YOLOv5作为目标检测系统的基础，是因为它在精度和速度上均优于其他神经网络算法，且易于部署。我们详细解读了YOLOv5的网络结构，特别是它如何通过Mosaic数据增强、自适应锚框计算、自适应图片缩放等技术，以提高目标检测的精度和速度。
总的来说，我们提出的结合HDRI基于YOLOv5的目标检测系统，不仅能够以高精度和高速度进行目标检测，还具有良好的灵活性和易于部署的优点。无论是在公共安全、交通管理还是智能监控等领域，这个系统都能发挥重要的作用。将来，我们还计划对系统进行进一步优化，以处理更多的复杂场景，并提高系统的稳定性和可靠性。

第六章 分工贡献

钟意伟：实验环境的搭建与数据集收集
王一童、郑志强：模型的创建、训练、测试
杨成莹：实验验证与结果分析






参考文献
[1]何珺.工信部倡议:发挥人工智能赋能效用抗击疫情[J].今日制造与升级,2020(Z1):42-43.
[2]Redmon J，Divvala S，Girshick R, et al. You only look once: Unified, real-time object detection [C]//Proceedings of the IEEE conference on computer vision and pattern recognition.2016:779-788.
[3]Liu W, Anguelov D, Erhan D, et al. Ssd: Single shot multibox detector[C]//European conference on computer vision. Springer, Cham, 2016:21-37.
[4]石绍鹏，王军.基于离线标定的全景视频快速拼接算法设计[J].激光杂志，2021,42(01): 113-117.
[5]李康顺,李凯,张文生.一种基于改进BP神经网络的PCA人脸识别算法[J/OL].计算机应用与软件, 2014,1(1):14-16[2017-03-14].
[6]Sun Y, Liang D, Wang X, et al. DeepID3: Face Recognition with Very Deep Neural Networks: 10.48550/arXiv.1502.00873[P]. 2015.
[7]GUO He-fei,LU Jian-feng,DONG Zhong-wen.A face recognition based on improved LBP feature ［J］.Modern Electronics Technique,2015(4):98-101.
[8]王欣然，田启川，张东.人脸口罩佩戴检测研究综述[J].计算机工程与应用,2022,58(10):13-26.


