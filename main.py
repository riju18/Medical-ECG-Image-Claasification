import json
import os
import numpy as np
from PIL import Image
import onnxruntime as rt
import cv2

EXPORT_MODEL_VERSION = 1


class ONNXModel:
    def __init__(self, model_dir) -> None:
        with open("signature.json", "r") as f:
            self.signature = json.load(f)
        self.model_file = self.signature.get("filename")
        if not os.path.isfile(self.model_file):
            raise FileNotFoundError(f"Model file does not exist")
        self.signature_inputs = self.signature.get("inputs")
        self.signature_outputs = self.signature.get("outputs")
        self.session = None
        if "Image" not in self.signature_inputs:
            raise ValueError("ONNX model doesn't have 'Image' input! Check signature.json, and please report issue to Lobe.")
        version = self.signature.get("export_model_version")
        if version is None or version != EXPORT_MODEL_VERSION:
            print(
                f"There has been a change to the model format. Please use a model with a signature 'export_model_version' that matches {EXPORT_MODEL_VERSION}."
            )

    def load(self) -> None:
        self.session = rt.InferenceSession(path_or_bytes=self.model_file)

    def predict(self, image: Image.Image) -> dict:
        img = self.process_image(image, self.signature_inputs.get("Image").get("shape"))
        fetches = [(key, value.get("name")) for key, value in self.signature_outputs.items()]
        feed = {self.signature_inputs.get("Image").get("name"): [img]}
        outputs = self.session.run(output_names=[name for (_, name) in fetches], input_feed=feed)
        return self.process_output(fetches, outputs)

    def process_image(self, image: Image.Image, input_shape: list) -> np.ndarray:
        width, height = image.size
        if image.mode != "RGB":
            image = image.convert("RGB")
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            image = image.crop((left, top, right, bottom))
        input_width, input_height = input_shape[1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))

        image = np.asarray(image) / 255.0
        return image.astype(np.float32)

    def process_output(self, fetches: dict, outputs: dict) -> dict:
        out_keys = ["label", "confidence"]
        results = {}
        for i, (key, _) in enumerate(fetches):
            val = outputs[i].tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        confs = results["Confidences"]
        labels = self.signature.get("classes").get("Label")
        output = [dict(zip(out_keys, group)) for group in zip(labels, confs)]
        sorted_output = {"predictions": sorted(output, key=lambda k: k["confidence"], reverse=True)}
        return sorted_output


if __name__ == "__main__":
    model_dir = os.path.join(os.getcwd())
    image_file = "ECG-test/Normal Person test/Normal(120).jpg"
    image = Image.open(image_file)
    get_image = cv2.imread(image_file)
    model = ONNXModel(model_dir=model_dir)
    model.load()
    outputs = model.predict(image)
    result = sorted(outputs["predictions"], key=lambda i: i["confidence"], reverse=True)[0]
    print(f"accuracy_of_all_class: {outputs}")
    cv2.imshow('{}-{}%'.format(result["label"], str(round(result["confidence"] * 100, 2))), get_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
