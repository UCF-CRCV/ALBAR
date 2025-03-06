import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t


# Ft model loading function.
def load_model(arch='swin_t', saved_model_file=None, num_classes=400, kin_pretrained=False):
    if arch == 'swin_t':
        model = swin3d_t(weights='DEFAULT' if kin_pretrained else None)
    else:
        print(f'Architecture {arch} invalid for model. Try \'swin_t\'.')
        return
    
    if num_classes != 400:
        model.head = nn.Linear(768, num_classes)

    # Load in saved model.
    if saved_model_file:
        try:
            saved_dict = torch.load(saved_model_file)
            model.load_state_dict(saved_dict['model_state_dict'], strict=True)
            print(f'model loaded from {saved_model_file} successfully!')
        except:
            print(f'Error loading model from {saved_model_file}.')
            print(f'model freshly initialized! Pretrained: {kin_pretrained}')
    else:
        print(f'model freshly initialized! Pretrained: {kin_pretrained}')

    return model


if __name__ == '__main__':
    model = load_model(arch='swin_t', num_classes=101, kin_pretrained=True)
    print(model)
    model.eval()
    model.cuda()
    inputs = torch.rand((4, 3, 16, 224, 224)).cuda()

    with torch.no_grad():
        output, feat = model(inputs)

    print(f'Output shape is: {output.shape}')
    print(f'Feature shape is: {feat.shape}')
