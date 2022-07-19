# CLIP

CLIP in Towhee is built on top of the [official implementation](https://github.com/openai/CLIP).

Available model names:
- clip_vit_b16
- clip_vit_b32 (support multilingual)
- clip_resnet_r50
- clip_resnet_r101

## Code Example

- Create model
```python
from towhee.models import clip

# Create CLIP model with parameters
model = clip.create_model(
    embed_dim=512, image_resolution=4,
    vision_layers=12, vision_width=768, vision_patch_size=2,
    context_length=77, vocab_size=49408, transformer_width=512,
    transformer_heads=8, transformer_layers=12
    )

# Create CLIP model with model name (no-pretrain)
model = clip.create_model(model_name='clip_vit_b32', pretrained=False)

# Load pretrained model with model name
model = clip.create_model(model_name='clip_vit_b32', pretrained=True)
```

- Encode image
```python
import torch

dummy_img = torch.rand(1, 3, 224, 224)
img_features = model.encode_image(dummy_img)
```

- Encode text
```python
# Tokenized input
text = torch.randint(high=49408, size=(1, 77), dtype=torch.int32)
text_features = model.encode_text(text)

# String input
text = ['test']
text_features = model.encode_text(text)

# Multilingual only for supported models
text_chinese = ['测试']
text_features = model.encode_text(text, multilingual=True)
```

- Calculate similarities
```python
img = torch.rand(1, 3, 224, 224)
text = ['test']
logits_per_img, logits_per_text = model(img, text)

# Multilingual only for supported models
text = ['测试']
logits_per_img, logits_per_text = model(img, text, multilingual=True)
```