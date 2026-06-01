import torch


def gram_matrix(feature):
    batch_size, channels, height, width = feature.size()
    feature = feature.view(batch_size, channels, height * width)
    gram = torch.bmm(feature, feature.transpose(1, 2))
    return gram / (channels * height * width)


def total_variation_loss(image):
    loss_h = torch.mean(torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]))
    loss_w = torch.mean(torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]))
    return loss_h + loss_w
