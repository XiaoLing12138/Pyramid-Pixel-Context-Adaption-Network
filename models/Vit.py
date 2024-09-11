from timm.models.vision_transformer import vit_base_patch16_224_in21k as create_model

def Vit(num_classes=3):
    return create_model(num_classes=num_classes)
