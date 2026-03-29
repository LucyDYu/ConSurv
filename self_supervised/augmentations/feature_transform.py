from copy import deepcopy
import torch
import torch.nn as nn
from torchvision import transforms

class RandomGaussianNoise(nn.Module):
    """
    Adds random Gaussian noise to a tensor.
    The noise is sampled from a normal distribution with a mean of 0 and a user-defined standard deviation.
    """
    def __init__(self, std=0.1):
        """
        Args:
            std (float): The standard deviation of the Gaussian noise to be added.
        """
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the noise to the input tensor.
        """
        if not self.training or self.std <= 0:
            return x
        
        noise = torch.randn_like(x) * self.std
        return x + noise

class FeatureTransform:
    """
    A transform pipeline for tensor features, inspired by SimCLRTransform.
    Applies a series of non-learnable augmentations to feature vectors.
    """
    def __init__(self, noise_std: float = 0.1, mask_p: float = 0.1):
        """
        Args:
            noise_std (float): Standard deviation of the Gaussian noise to add. Set to 0 to disable.
            mask_p (float): Probability of an element to be zeroed. Set to 0 to disable.
        """
        self.transform = transforms.Compose(
            [
                RandomGaussianNoise(std=noise_std),
            ]
        )

    def __call__(self, x: torch.Tensor, single_view: bool = False):
        """
        Applies augmentations to a dictionary of multimodal WSI data to create ONE augmented view.

        Args:
            **kwargs: The dictionary of data (e.g., {'x_path': tensor, 'x_omic1': tensor, ...}).
        
        Returns:
            A new dictionary containing the augmented view.
        """
        if single_view:
            return self.transform(x)
        else:
            # Applying the same transform instance twice will produce different results
            # due to the random nature of the augmentations.
            view1 = self.transform(x)
            view2 = self.transform(x)
            return view1, view2 
        


    def transform_wsi(self, full_data):
        list_flag=False
        if len(full_data) == 1:
            # data is a list containing a single data sample
            full_data = full_data[0]
            list_flag=True
        data_aug=deepcopy(full_data)

        data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, *_ = data_aug
        data_WSI = self.transform(data_WSI)
        data_omic1 = self.transform(data_omic1)
        data_omic2 = self.transform(data_omic2)
        data_omic3 = self.transform(data_omic3)
        data_omic4 = self.transform(data_omic4)
        data_omic5 = self.transform(data_omic5)
        data_omic6 = self.transform(data_omic6)
        data_aug = (data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c, *_)

        if list_flag:
            return [data_aug]
        else:
            return data_aug
        
        
        
    