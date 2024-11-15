from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("ucaslcl/GOT-OCR2_0", trust_remote_code=True)
model = AutoModel.from_pretrained("ucaslcl/GOT-OCR2_0",trust_remote_code=True, device_map='cuda')

def detecting_number_plate(image_path,model,tokenizer):
    res = model.chat(tokenizer,image_path,ocr_type = 'ocr')
    return res

# image = 'overspeeding_vehicles_ss/overspeeding_vehicle_13.png'
# res = model.chat(tokenizer,image, ocr_type = 'ocr')
# print(res)
