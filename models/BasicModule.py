from torch.nn import Module
import time

class BasicModule(Module):
    """
    封装了nn.Module，主要提供save和load两个方法。
    """
    def __init__(self):
        super(BasicModule, self).__init__()
        self.model_name = self.__class__.__name__

    def save(self, name=None):
        """
        保存模型，默认使用"模型名字+时间.pth"作为文件名，\n
        如ALexNet_20190710_235729.pth\n
        :param name: 模型的名称
        :return: name of the saved file
        """
        prefix = 'checkpoints/' + self.model_name +"_"
        if name is None:
            name = time.strftime(prefix + "%Y%m%d_%H%M%S.pth", time.localtime())

        torch.save(self.state_dict(), name)
        return name

    def load(self, path):
        """
        加载指定路径的模型 \n
        :param path: 模型的路径
        :return: None
        """
        self.load_state_dict(torch.load(path))


