"""模型加载与推理模块"""
import os
import re
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor


class ModelWorker:
    def __init__(self, model_path, device="cuda", dtype=torch.bfloat16):
        self.device = device
        self.dtype = dtype
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._model_path = model_path

    def load(self):
        if self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_path, trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(
            self._model_path, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            self._model_path,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

    @torch.no_grad()
    def predict(self, image, prompt, max_new_tokens=2048, temperature=0.7):
        self.load()
        if image.mode != "RGB":
            image = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode="hybrid",
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=False,
        )

        answer = response[0] if isinstance(response, tuple) else response
        return {"answer": answer}

    @staticmethod
    def parse_boxes(answer, image_width, image_height):
        """解析模型输出的 <box><x1><y1><x2><y2></box> 格式坐标"""
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append(
                {
                    "x1": x1 / 1000.0 * image_width,
                    "y1": y1 / 1000.0 * image_height,
                    "x2": x2 / 1000.0 * image_width,
                    "y2": y2 / 1000.0 * image_height,
                }
            )
        return boxes

    def cleanup(self):
        """清理 GPU 显存"""
        if self.model is not None:
            del self.model
            self.model = None
        self.tokenizer = None
        self.processor = None
        torch.cuda.empty_cache()
