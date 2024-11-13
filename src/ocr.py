# ocr.py

from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ucaslcl/GOT-OCR2_0", trust_remote_code=True)
model = AutoModel.from_pretrained("ucaslcl/GOT-OCR2_0", trust_remote_code=True, device_map='cuda')

def detecting_number_plate_ocr(image_path, model=model, tokenizer=tokenizer):
    res = model.chat(tokenizer, image_path, ocr_type='ocr')
    return res
