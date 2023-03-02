from build_utils.layers import *
from build_utils.parse_config import *

ONNX_EXPORT = False

# 传进来的是列表
def create_modules(modules_defs: list, img_size):
    """
    Constructs module list of layer blocks from module configuration in module_defs
    :param modules_defs: 通过.cfg文件解析得到的每个层结构的列表
    :param img_size:
    :return:
    """

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size
    # 删除解析cfg列表中的第一个配置(对应[net]的配置)
    modules_defs.pop(0)  # cfg training hyperparams (unused)
    # output_filters用来记录搭建每个模块时他所输出的特征矩阵的channel,其中的3对应传入的RGB图像的通道数目
    output_filters = [3]  # input channels
    # 搭建网络的时候会依次将每个模块传入module_list
    module_list = nn.ModuleList()
    # 统计哪些特征层的输出会被后续的层使用到(可能是特征融合，也可能是拼接)，就是cfg文件里面的-2，,-3这些东西
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    # 遍历搭建每个层结构
    # 返回索引和模块对应的信息，遍历每一个模块
    for i, mdef in enumerate(modules_defs):
        # 模块可能包含多个层结构，放到sequential中
        modules = nn.Sequential()

        # 判断模块的类型
        if mdef["type"] == "convolutional":
            # 取出层结构里面的具体操作
            bn = mdef["batch_normalize"]  # 1 or 0 / use or not
            filters = mdef["filters"]
            k = mdef["size"]  # kernel size
            # 此代码的每个convolutional都有stride，不用管后面一个参数
            stride = mdef["stride"] if "stride" in mdef else (mdef['stride_y'], mdef["stride_x"])
            # 如果k是整数，就添加一个Conv2d，
            if isinstance(k, int):
                modules.add_module("Conv2d", nn.Conv2d(in_channels=output_filters[-1], #output_filters[-1]表示最后一个传入的特征矩阵的channel
                                                       out_channels=filters,
                                                       kernel_size=k,
                                                       stride=stride,
                                                       padding=k // 2 if mdef["pad"] else 0,
                                                       bias=not bn))
            else:
                raise TypeError("conv2d filter size must be int type.")

            if bn:
                # 添加BN层，参数对应上一层的输出通道个数
                modules.add_module("BatchNorm2d", nn.BatchNorm2d(filters))
            else:
                # 如果该卷积操作没有bn层，意味着该层为yolo的predictor。yolov3spp除了最后三个预测层，其他地方卷积都有BN层。
                # 后面还要使用，所以通过touts记录下来
                routs.append(i)  # detection output (goes into yolo layer)

            if mdef["activation"] == "leaky":
                modules.add_module("activation", nn.LeakyReLU(0.1, inplace=True))
            else:
                pass
        # yolov3spp没这个结构
        elif mdef["type"] == "BatchNorm2d":
            pass

        elif mdef["type"] == "maxpool":
            k = mdef["size"]  # kernel size
            stride = mdef["stride"]
            modules = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)

        elif mdef["type"] == "upsample":
            # 判断是否要到处ONNX模型，先忽略
            if ONNX_EXPORT:  # explicitly state size, avoid scale_factor
                g = (yolo_index + 1) * 2 / 32  # gain
                modules = nn.Upsample(size=tuple(int(x * g) for x in img_size))
            else:
                modules = nn.Upsample(scale_factor=mdef["stride"])
        # layers为一个值表示指向某一层的输出，为多个值的时候表示拼接多层输出
        elif mdef["type"] == "route":  # [-2],  [-1,-3,-5,-6], [-1, 61]
            # 提取layers参数
            layers = mdef["layers"]
            # filters用来记录当前层输出特征矩阵的channel，多层的时候maxpool会用到，有个sum求和函数用来记录这一层的输出总数。，单层的时候残差链接会用到
            # if l > 0，就需要对l进行+1，因为output_filters刚开始有一个3，所以索引为0的位置所对应的channel并不是第一个模块的输出特征矩阵的channel，第一个模块的输出特征矩阵的channel应该在索引为1的地方
            # 当L<0的时候是倒着数的，则没关系
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            # routs用来记录用到那些输出层，这里的i表示routs 的索引
            # extend表示把两个list连起来，append表示在列表尾部追加元素
            routs.extend([i + l if l < 0 else l for l in layers])
            # 通过FeatureConcat来创建模块
            modules = FeatureConcat(layers=layers)

        elif mdef["type"] == "shortcut":
            layers = mdef["from"]
            # 获取shortcut模块上一层模块的输出特征的channel
            filters = output_filters[-1]
            # routs.extend([i + l if l < 0 else l for l in layers])
            # 有shortcut要使用前面的输出，因此要记录使用的是那一层的输出
            # shortcut的layers里面只有一个值，直接获取他的值,i+layer[0]获取到需要融合的特征层
            routs.append(i + layers[0])
            #  weight="weights_type" in mdef没用到，不用管
            modules = WeightedFeatureFusion(layers=layers, weight="weights_type" in mdef)

        elif mdef["type"] == "yolo":
            yolo_index += 1  # 记录是第几个yolo_layer [0, 1, 2],初始的时候为-1
            stride = [32, 16, 8]  # 预测特征层对应原图的缩放比例
            # anchors=mdef["anchors"][mdef["mask"]]对应当前预测特征层中所使用的anchors
            # nc表示预测目标类别个数
            # img_size不用管
            # stride，对应哪个预测特征层就传入相应的stride
            print(mdef["anchors"].shape)
            modules = YOLOLayer(anchors=mdef["anchors"][mdef["mask"]],  # anchor list
                                nc=mdef["classes"],  # number of classes
                                img_size=img_size,
                                stride=stride[yolo_index])

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            # 对yololayer的前一层进行初始化
            try:
                j = -1
                # bias: shape(255,) 索引0对应Sequential中的Conv2d
                # view: shape(3, 85)
                # module_list[j][0]，j=-1表示获取上一个模块的的信息，0表示取卷积层的bias，因为前面子啊搭建卷积的时候都是使用sequential，在预测的时候只有一个卷积，就是索引位置为0的模块
                b = module_list[j][0].bias.view(modules.na, -1)
                b.data[:, 4] += -4.5  # obj
                b.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            except Exception as e:
                print('WARNING: smart bias initialization failure.', e)
        else:
            print("Warning: Unrecognized Layer Type: " + mdef["type"])

        # Register module list and number of output filters
        # 将刚刚遍历得到的每个模块放到 module_list中，module_list就是nn.ModuleList。
        module_list.append(modules)
        # 添加filters，只有shortcut层和route,卷积层有，但是这些结构的特征层的通道会发生变化，所以会重新赋值。
        output_filters.append(filters)

    # 一个列表里面全部存放false
    routs_binary = [False] * len(modules_defs)
    # 遍历之前构建的routs
    for i in routs:
        # 需要用到的索引处令为true
        routs_binary[i] = True
    return module_list, routs_binary


class YOLOLayer(nn.Module):
    """
    对YOLO的预测predict输出进行处理
    """
    def __init__(self, anchors, nc, img_size, stride):
        super(YOLOLayer, self).__init__()
        # 传入的anchors是numpy格式，转为tensor
        self.anchors = torch.Tensor(anchors)
        self.stride = stride  # layer stride 特征图上一步对应原图上的步距 [32, 16, 8],根据对应原图的硕放比例来自定定
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        # 针对每个anchor预测多少个参数
        self.no = nc + 5  # number of outputs (85: x, y, w, h, obj, cls1, ...)
        # self.nx, self.ny，self.ng分别对应所使用的特征层的高度和宽度，grid的size(简单初始化为0，后面再重新赋值)
        self.nx, self.ny, self.ng = 0, 0, (0, 0)  # initialize number of x, y gridpoints
        # 将anchors大小缩放到grid尺度，因为传入的anchor都是针对原图上的尺度
        self.anchor_vec = self.anchors / self.stride
        #  self.anchor_vec.view(1, self.na, 1, 1, 2)里面的参数分别对应batch_size, na, grid_h, grid_w, wh,
        # 值为1的维度对应的值不是固定值，后续操作可根据broadcast广播机制自动扩充
        self.anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2)
        # self.grid正向传播的时候会重新赋值
        self.grid = None

        # 不用管
        if ONNX_EXPORT:
            self.training = False
            self.create_grids((img_size[1] // stride, img_size[0] // stride))  # number x, y grid points

    def create_grids(self, ng=(13, 13), device="cpu"):
        """
        更新grids信息并生成新的grids参数
        :param ng: 特征图大小
        :param device:
        :return:
        """
        self.nx, self.ny = ng
        self.ng = torch.tensor(ng, dtype=torch.float)

        # build xy offsets 构建每个cell处的anchor的xy偏移量(在feature map上的)
        if not self.training:  # 训练模式不需要回归到最终预测boxes
            # torch.meshgrid得到每一个grid左上角的坐标nx表示左上角的x方向的坐标，ny表示左上角的y方向的坐标
            yv, xv = torch.meshgrid([torch.arange(self.ny, device=device),
                                     torch.arange(self.nx, device=device)])
            # batch_size, na, grid_h, grid_w, wh。将x，y坐标组合到一块。通过view更改形状和前面self.anchor_vec一样形式，同理这里的1,1会通过广播机制自己调整
            self.grid = torch.stack((xv, yv), 2).view((1, 1, self.ny, self.nx, 2)).float()

        if self.anchor_vec.device != device:
            self.anchor_vec = self.anchor_vec.to(device)
            self.anchor_wh = self.anchor_wh.to(device)

    # p对应预测的参数
    def forward(self, p):
        # 为false不用管
        if ONNX_EXPORT:
            bs = 1  # batch size
        else:
            # bs, _, ny, nx四个参数分别对应 batch_size, predict_param(255), grid(13), grid(13)
            bs, _, ny, nx = p.shape  # batch_size, predict_param(255), grid(13), grid(13)
            # 判断self.nx, self.ny是否等于当前输入特征矩阵的nx, ny，入过不等于说明grid size是发生变化的，需要重新生成，或者为none,第一次的时候也需要重新生成
            if (self.nx, self.ny) != (nx, ny) or self.grid is None:  # fix no grid bug
                self.create_grids((nx, ny), p.device)

        # view: (batch_size, 255, 13, 13) -> (batch_size, 3, 85, 13, 13)
        # permute: (batch_size, 3, 85, 13, 13) -> (batch_size, 3, 13, 13, 85)
        # [bs, anchor, grid, grid, xywh + obj + classes]。permute会改变原有参数的排列顺序，在内存中就不在连续，通过contiguous让其变成内存连续的变量
        p = p.view(bs, self.na, self.no, self.ny, self.nx).permute(0, 1, 3, 4, 2).contiguous()  # prediction

        if self.training:
            return p
        elif ONNX_EXPORT:
            # Avoid broadcasting for ANE operations
            m = self.na * self.nx * self.ny  # 3*
            ng = 1. / self.ng.repeat(m, 1)
            grid = self.grid.repeat(1, self.na, 1, 1, 1).view(m, 2)
            anchor_wh = self.anchor_wh.repeat(1, 1, self.nx, self.ny, 1).view(m, 2) * ng

            p = p.view(m, self.no)
            # xy = torch.sigmoid(p[:, 0:2]) + grid  # x, y
            # wh = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            # p_cls = torch.sigmoid(p[:, 4:5]) if self.nc == 1 else \
            #     torch.sigmoid(p[:, 5:self.no]) * torch.sigmoid(p[:, 4:5])  # conf
            p[:, :2] = (torch.sigmoid(p[:, 0:2]) + grid) * ng  # x, y
            p[:, 2:4] = torch.exp(p[:, 2:4]) * anchor_wh  # width, height
            p[:, 4:] = torch.sigmoid(p[:, 4:])
            p[:, 5:] = p[:, 5:self.no] * p[:, 4:5]
            return p
        else:  # inference
            # [bs, anchor, grid, grid, xywh + obj + classes]
            io = p.clone()  # inference output
            # 将所有预测的x，y的偏移量通过sigmoid函数处理在加上grid参数
            # io[..., :2]表示取最后一维的前两个参数，几x和y，得到预测值的偏移量(经过sigmoid处理)，省略号表示除了最后一维的所有维度
            io[..., :2] = torch.sigmoid(io[..., :2]) + self.grid  # xy 计算在feature map上的xy坐标
            # io[..., 2:4]取最后一维索引为2和3的数值，即对应宽高的缩放(左必右开)，对应[bs, anchor, grid, grid, xywh + obj + classes]中w和h。
            io[..., 2:4] = torch.exp(io[..., 2:4]) * self.anchor_wh  # wh yolo method 计算在feature map上的wh
            # 同理将xywh四个参数乘上缩放比例，映射回原图尺寸
            io[..., :4] *= self.stride  # 换算映射回原图尺度
            # 将obj + classes这些参数全部经过sigmoid
            torch.sigmoid_(io[..., 4:])
            # 重新reshape回去，self.no表示针对每个anchor预测的参数个数，Coco数据集就是85.这里还会返回预测器原始的输出
            return io.view(bs, -1, self.no), p  # view [1, 3, 13, 13, 85] as [1, 507, 85]


class Darknet(nn.Module):
    """
    YOLOv3 spp object detection model
    """
    # img_size=(416, 416)制定训练图像的尺寸，在训练的时候不起任何作用，只在到处onnx格式模型的时候才起作用。verbose表示子啊实例化模型的时候是否需要打印模型信息
    def __init__(self, cfg, img_size=(416, 416), verbose=False):
        super(Darknet, self).__init__()
        # 这里传入的img_size只在导出ONNX模型时起作用
        self.input_size = [img_size] * 2 if isinstance(img_size, int) else img_size
        # 通过parse_model_cfg解析网络对应的.cfg文件，将cfg里面的每个结构解析成列表中每一个元素，每个元素以字典的形式存在
        self.module_defs = parse_model_cfg(cfg)
        # 根据解析的网络结构一层一层去搭建。通过create_modules传入解析好的列表搭建yolov3spp网络。img_size同样只在onnx模型起作用，训练不起作用
        self.module_list, self.routs = create_modules(self.module_defs, img_size)
        # 获取所有YOLOLayer层的索引
        self.yolo_layers = get_yolo_layers(self)

        # 打印下模型的信息，如果verbose为True则打印详细信息
        self.info(verbose) if not ONNX_EXPORT else None  # print model description

    def forward(self, x, verbose=False):
        return self.forward_once(x, verbose=verbose)

    # 表示输入的训练数据，打包好的一个个batch
    def forward_once(self, x, verbose=False):
        # yolo_out收集每个yolo_layer层的输出
        # out收集每个模块的输出
        yolo_out, out = [], []
        # 默认为false
        if verbose:
            print('0', x.shape)
            str = ""

        # 遍历搭建好的整个网络
        for i, module in enumerate(self.module_list):
            # 获取每个模块的name
            name = module.__class__.__name__
            if name in ["WeightedFeatureFusion", "FeatureConcat"]:  # sum, concat
                if verbose:
                    l = [i - 1] + module.layers  # layers
                    sh = [list(x.shape)] + [list(out[i].shape) for i in module.layers]  # shapes
                    str = ' >> ' + ' + '.join(['layer %g %s' % x for x in zip(l, sh)])
                # 将x和收集的每一层的输出传到module中
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name == "YOLOLayer":
                yolo_out.append(module(x))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)
            # 保存每一个模块的输出，先进行self.routs[i]判断，为true才保存在out中，否者放到空列表中
            out.append(x if self.routs[i] else [])
            if verbose:
                print('%g/%g %s -' % (i, len(self.module_list), name), list(x.shape), str)
                str = ''

        # 训练模式直接返回上面收集的yolo_out。
        if self.training:  # train
            return yolo_out
        elif ONNX_EXPORT:  # export
            # x = [torch.cat(x, 0) for x in zip(*yolo_out)]
            # return x[0], torch.cat(x[1:3], 1)  # scores, boxes: 3780x80, 3780x4
            p = torch.cat(yolo_out, dim=0)

            # # 根据objectness虑除低概率目标
            # mask = torch.nonzero(torch.gt(p[:, 4], 0.1), as_tuple=False).squeeze(1)
            # # onnx不支持超过一维的索引（pytorch太灵活了）
            # # p = p[mask]
            # p = torch.index_select(p, dim=0, index=mask)
            #
            # # 虑除小面积目标，w > 2 and h > 2 pixel
            # # ONNX暂不支持bitwise_and和all操作
            # mask_s = torch.gt(p[:, 2], 2./self.input_size[0]) & torch.gt(p[:, 3], 2./self.input_size[1])
            # mask_s = torch.nonzero(mask_s, as_tuple=False).squeeze(1)
            # p = torch.index_select(p, dim=0, index=mask_s)  # width-height 虑除小目标
            #
            # if mask_s.numel() == 0:
            #     return torch.empty([0, 85])

            return p
        # 如果是测试和验证的时候则通过非关键字传，因为在验证和测试的时候会返回两个值，一个是io view之后的值，一个是p
        else:  # inference or test
            x, p = zip(*yolo_out)  # inference output, training output
            x = torch.cat(x, 1)  # cat yolo outputs

            return x, p

    def info(self, verbose=False):
        """
        打印模型的信息
        :param verbose:
        :return:
        """
        torch_utils.model_info(self, verbose)


def get_yolo_layers(self):
    """
    获取网络中三个"YOLOLayer"模块对应的索引
    :param self:
    :return:
    """
    return [i for i, m in enumerate(self.module_list) if m.__class__.__name__ == 'YOLOLayer']  # [89, 101, 113]



