from transformers import CLIPModel, CLIPProcessor
import torch.nn as nn

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
for param in model.parameters():
    param.requires_grad = False


class CLIPClassifier(nn.Module):
    def __init__(self, clip_model=model, num_classes=100):
        super().__init__()
        self.clip = clip_model
        self.classifier = nn.Linear(self.clip.config.vision_config.hidden_size, num_classes)

    def forward(self, pixel_values):
        # clip有两个部分,一个NLP部分一个是Vision部分
        outputs = self.clip.vision_model(pixel_values=pixel_values)
        # 这是Vision_model的结果在经过最终的池化过程后得到的固定长度的向量,一般形式为(batch_size,hidden_layers)
        pooled_output = outputs.pooler_output
        # 进行了一个分类操作
        logits = self.classifier(pooled_output)
        return logits
