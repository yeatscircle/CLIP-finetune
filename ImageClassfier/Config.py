import torchvision.transforms as transforms
from model import processor
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 6666,
    'dataset_dir': "C:\\dats_tec\\archive",
    'n_epochs': 10,
    'batch_size': 64,
    'learning_rate': 3e-4,
    'weight_decay': 1e-5,
    'early_stop': 100,
    'clip_flag': True,
    'save_path': './models/model.ckpt',
    'resnet_save_path': './models/resnet_model.ckpt',
    'num_workers': 12
}

# 一般情况下，我们不会在验证集和测试集上做数据扩增
# 我们只需要将图片裁剪成同样的大小并装换成Tensor就行
test_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 当然，我们也可以再测试集中对数据进行扩增（对同样本的不同装换）
#  - 用训练数据的装化方法（train_tfm）去对测试集数据进行转化，产出扩增样本
#  - 对同个照片的不同样本分别进行预测
#  - 最后可以用soft vote / hard vote 等集成方法输出最后的预测
train_tfm = transforms.Compose([
    # 图片裁剪 (height = width = 224)
    transforms.Resize((224, 224)),
    # TODO:在这部分还可以增加一些图片处理的操作
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
    # ToTensor() 放在所有处理的最后
    transforms.ToTensor(),
])


def labels_to_text(label):
    return 'A photo of ' + label


 